from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer


# ============================================================
# Commercial-friendly baseline for residue-level metal binding
# site prediction from (sequence, metal_type) -> binary labels.
#
# Expected CSV schema:
#   sequence      : amino-acid sequence (e.g. "ACDEFGHIK")
#   metal         : metal name/code (e.g. "CU", "copper", "gold")
#   labels        : binary string same length as sequence (e.g. "001010000")
# Optional columns:
#   split         : train/val/test
#   group_id      : protein/cluster id for split leakage control
#   sample_weight : float
#
# Recommended: build this CSV from BioLiP2 + MetalDB yourself.
# Do NOT reuse LMetalSite code or weights in commercial software.
# ============================================================

AA_VOCAB = set("ACDEFGHIKLMNPQRSTVWYUXBJZO")
DEFAULT_METALS = [
    "ZN", "CA", "MG", "MN", "FE", "CU", "CO", "NI", "AU", "CD",
    "HG", "PB", "K", "NA", "MO", "V", "W"
]


@dataclass
class TrainConfig:
    csv_path: str = "metal_binding_dataset_from_CU.csv"
    model_name: str = "facebook/esm2_t12_35M_UR50D"
    output_dir: str = "./outputs_metal_site"
    max_length: int = 1024
    batch_size: int = 4
    num_workers: int = 2
    lr_head: float = 2e-4
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-2
    epochs: int = 15
    warmup_ratio: float = 0.1
    grad_accum_steps: int = 1
    grad_clip: float = 1.0
    metal_emb_dim: int = 64
    hidden_dim: int = 256
    nhead: int = 8
    num_layers: int = 3
    dropout: float = 0.2
    freeze_backbone_epochs: int = 1
    pos_weight: float = 6.0
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True
    use_bfloat16: bool = True
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    group_col: Optional[str] = "group_id"
    use_predefined_split: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MetalVocab:
    def __init__(self, metals: List[str]):
        normed = []
        seen = set()
        for m in metals:
            nm = self.normalize(m)
            if nm not in seen:
                seen.add(nm)
                normed.append(nm)
        self.idx_to_metal = ["<UNK>"] + normed
        self.metal_to_idx = {m: i for i, m in enumerate(self.idx_to_metal)}

    @staticmethod
    def normalize(metal: str) -> str:
        m = str(metal).strip().upper()
        aliases = {
            "COPPER": "CU", "GOLD": "AU", "ZINC": "ZN", "CALCIUM": "CA",
            "MAGNESIUM": "MG", "MANGANESE": "MN", "IRON": "FE", "COBALT": "CO",
            "NICKEL": "NI", "CADMIUM": "CD", "MERCURY": "HG", "LEAD": "PB",
            "POTASSIUM": "K", "SODIUM": "NA", "MOLYBDENUM": "MO", "VANADIUM": "V",
            "TUNGSTEN": "W",
        }
        return aliases.get(m, m)

    def encode(self, metal: str) -> int:
        m = self.normalize(metal)
        return self.metal_to_idx.get(m, 0)

    def __len__(self) -> int:
        return len(self.idx_to_metal)


def validate_row(seq: str, labels: str) -> bool:
    if not isinstance(seq, str) or not isinstance(labels, str):
        return False
    seq = seq.strip().upper()
    labels = labels.strip()
    if len(seq) == 0 or len(seq) != len(labels):
        return False
    if any(ch not in AA_VOCAB for ch in seq):
        return False
    if any(ch not in {"0", "1"} for ch in labels):
        return False
    return True


def build_split(df: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
    if cfg.use_predefined_split and "split" in df.columns:
        return df

    df = df.copy()
    rng = np.random.default_rng(cfg.seed)

    if cfg.group_col and cfg.group_col in df.columns:
        groups = df[cfg.group_col].fillna(df.index.astype(str)).astype(str).unique().tolist()
        rng.shuffle(groups)
        n = len(groups)
        n_train = int(n * cfg.train_split)
        n_val = int(n * cfg.val_split)
        train_g = set(groups[:n_train])
        val_g = set(groups[n_train:n_train + n_val])
        def assign(g: str) -> str:
            if g in train_g:
                return "train"
            if g in val_g:
                return "val"
            return "test"
        df["split"] = df[cfg.group_col].fillna(df.index.astype(str)).astype(str).map(assign)
    else:
        idx = np.arange(len(df))
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(n * cfg.train_split)
        n_val = int(n * cfg.val_split)
        split = np.array(["test"] * n, dtype=object)
        split[idx[:n_train]] = "train"
        split[idx[n_train:n_train + n_val]] = "val"
        df["split"] = split
    return df


class MetalBindingDataset(Dataset):
    def __init__(self, df: pd.DataFrame, metal_vocab: MetalVocab):
        self.df = df.reset_index(drop=True)
        self.metal_vocab = metal_vocab

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        seq = row["sequence"].strip().upper()
        labels = torch.tensor([int(x) for x in row["labels"].strip()], dtype=torch.float32)
        metal_id = self.metal_vocab.encode(row["metal"])
        sample_weight = float(row["sample_weight"]) if "sample_weight" in row and pd.notna(row["sample_weight"]) else 1.0
        return {
            "sequence": seq,
            "labels": labels,
            "metal_id": metal_id,
            "sample_weight": sample_weight,
        }


class Collator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        seqs = [b["sequence"] for b in batch]
        spaced = [" ".join(list(s)) for s in seqs]
        tok = self.tokenizer(
            spaced,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        B, T = tok["input_ids"].shape
        labels = torch.full((B, T), -100.0, dtype=torch.float32)
        residue_mask = torch.zeros((B, T), dtype=torch.bool)

        for i, sample in enumerate(batch):
            seq_len = min(len(sample["sequence"]), self.max_length - 2)
            labels[i, 1:1 + seq_len] = sample["labels"][:seq_len]
            residue_mask[i, 1:1 + seq_len] = True

        metal_ids = torch.tensor([b["metal_id"] for b in batch], dtype=torch.long)
        sample_weights = torch.tensor([b["sample_weight"] for b in batch], dtype=torch.float32)

        tok["labels"] = labels
        tok["residue_mask"] = residue_mask
        tok["metal_ids"] = metal_ids
        tok["sample_weights"] = sample_weights
        return tok


class ResidueMetalSiteModel(nn.Module):
    def __init__(self, cfg: TrainConfig, num_metals: int):
        super().__init__()
        self.cfg = cfg
        self.backbone = AutoModel.from_pretrained(cfg.model_name)
        backbone_dim = self.backbone.config.hidden_size

        self.metal_emb = nn.Embedding(num_metals, cfg.metal_emb_dim)
        self.input_proj = nn.Linear(backbone_dim + cfg.metal_emb_dim, cfg.hidden_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.nhead,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        metal_ids: torch.Tensor,
    ) -> torch.Tensor:
        x = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        metal = self.metal_emb(metal_ids)[:, None, :].expand(x.size(0), x.size(1), -1)
        h = self.input_proj(torch.cat([x, metal], dim=-1))

        key_padding_mask = ~attention_mask.bool()
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)
        logits = self.classifier(h).squeeze(-1)
        return logits


@torch.no_grad()
def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, residue_mask: torch.Tensor) -> Dict[str, float]:
    valid = residue_mask & (labels >= 0)
    if valid.sum().item() == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "acc": 0.0}

    probs = torch.sigmoid(logits[valid])
    preds = (probs >= 0.5).long()
    gold = labels[valid].long()

    tp = ((preds == 1) & (gold == 1)).sum().item()
    fp = ((preds == 1) & (gold == 0)).sum().item()
    fn = ((preds == 0) & (gold == 1)).sum().item()
    tn = ((preds == 0) & (gold == 0)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1, "acc": acc}


def masked_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    residue_mask: torch.Tensor,
    sample_weights: torch.Tensor,
    pos_weight: float,
) -> torch.Tensor:
    valid = residue_mask & (labels >= 0)
    element_loss = F.binary_cross_entropy_with_logits(
        logits,
        torch.clamp(labels, min=0.0),
        reduction="none",
        pos_weight=torch.tensor(pos_weight, device=logits.device),
    )
    element_loss = element_loss * valid.float()
    # Per-sample weighting
    per_sample = element_loss.sum(dim=1) / valid.float().sum(dim=1).clamp_min(1.0)
    loss = (per_sample * sample_weights).mean()
    return loss


def save_checkpoint(path: str, model: nn.Module, cfg: TrainConfig, metal_vocab: MetalVocab) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "config": asdict(cfg),
        "metal_vocab": metal_vocab.idx_to_metal,
    }
    torch.save(payload, path)


def train_epoch(model, loader, optimizer, scheduler, scaler, device, cfg):
    model.train()
    total_loss = 0.0
    total_steps = 0

    if hasattr(model.backbone, "requires_grad_"):
        pass

    autocast_dtype = torch.bfloat16 if (cfg.use_bfloat16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(cfg.amp and device.startswith("cuda"))):
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                metal_ids=batch["metal_ids"],
            )
            loss = masked_bce_loss(
                logits,
                batch["labels"],
                batch["residue_mask"],
                batch["sample_weights"],
                cfg.pos_weight,
            ) / cfg.grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % cfg.grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * cfg.grad_accum_steps
        total_steps += 1

    return total_loss / max(total_steps, 1)


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()
    total_loss = 0.0
    total_steps = 0
    all_logits, all_labels, all_masks = [], [], []

    autocast_dtype = torch.bfloat16 if (cfg.use_bfloat16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(cfg.amp and device.startswith("cuda"))):
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                metal_ids=batch["metal_ids"],
            )
            loss = masked_bce_loss(
                logits,
                batch["labels"],
                batch["residue_mask"],
                batch["sample_weights"],
                cfg.pos_weight,
            )
        total_loss += loss.item()
        total_steps += 1
        all_logits.append(logits.detach().cpu())
        all_labels.append(batch["labels"].detach().cpu())
        all_masks.append(batch["residue_mask"].detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    masks = torch.cat(all_masks, dim=0)
    metrics = compute_metrics(logits, labels, masks)
    metrics["loss"] = total_loss / max(total_steps, 1)
    return metrics


def make_optimizer_and_scheduler(model: ResidueMetalSiteModel, cfg: TrainConfig, num_training_steps: int):
    backbone_params = list(model.backbone.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg.lr_backbone},
            {"params": head_params, "lr": cfg.lr_head},
        ],
        weight_decay=cfg.weight_decay,
    )

    warmup_steps = int(cfg.warmup_ratio * num_training_steps)

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, num_training_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def load_and_prepare_dataframe(cfg: TrainConfig) -> Tuple[pd.DataFrame, MetalVocab]:
    df = pd.read_csv(cfg.csv_path)
    required = {"sequence", "metal", "labels"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["sequence"] = df["sequence"].astype(str).str.upper().str.replace(r"\s+", "", regex=True)
    df["labels"] = df["labels"].astype(str).str.replace(r"\s+", "", regex=True)
    valid_mask = [validate_row(s, y) for s, y in zip(df["sequence"], df["labels"])]
    df = df.loc[valid_mask].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No valid rows after validation.")

    metals = sorted({MetalVocab.normalize(m) for m in df["metal"].dropna().unique().tolist()} | set(DEFAULT_METALS))
    metal_vocab = MetalVocab(metals)
    df = build_split(df, cfg)
    return df, metal_vocab


def set_backbone_trainability(model: ResidueMetalSiteModel, trainable: bool) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = trainable


def run_training(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    df, metal_vocab = load_and_prepare_dataframe(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    print(f"Rows: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(f"Metals: {metal_vocab.idx_to_metal}")

    collator = Collator(tokenizer, cfg.max_length)
    train_loader = DataLoader(
        MetalBindingDataset(train_df, metal_vocab),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )
    val_loader = DataLoader(
        MetalBindingDataset(val_df, metal_vocab),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )
    test_loader = DataLoader(
        MetalBindingDataset(test_df, metal_vocab),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )

    model = ResidueMetalSiteModel(cfg, num_metals=len(metal_vocab)).to(cfg.device)
    num_updates_per_epoch = max(1, math.ceil(len(train_loader) / cfg.grad_accum_steps))
    optimizer, scheduler = make_optimizer_and_scheduler(model, cfg, cfg.epochs * num_updates_per_epoch)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and cfg.device.startswith("cuda") and not (cfg.use_bfloat16 and torch.cuda.is_bf16_supported())))

    best_val_f1 = -1.0
    best_path = os.path.join(cfg.output_dir, "best_model.pt")

    for epoch in range(cfg.epochs):
        train_backbone = epoch >= cfg.freeze_backbone_epochs
        set_backbone_trainability(model, train_backbone)

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, cfg.device, cfg)
        val_metrics = evaluate(model, val_loader, cfg.device, cfg)

        print(
            f"Epoch {epoch+1:02d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_prec={val_metrics['precision']:.4f} | "
            f"val_rec={val_metrics['recall']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            save_checkpoint(best_path, model, cfg, metal_vocab)
            print(f"Saved best checkpoint to {best_path}")

    checkpoint = torch.load(best_path, map_location=cfg.device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(model, test_loader, cfg.device, cfg)
    print("Test metrics:", test_metrics)


@torch.no_grad()
def predict_sites(
    model_ckpt: str,
    sequences: List[str],
    metals: List[str],
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    max_length: int = 1024,
) -> List[Dict]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_ckpt, map_location=device)
    cfg_dict = ckpt["config"]
    if model_name is not None:
        cfg_dict["model_name"] = model_name
    cfg = TrainConfig(**cfg_dict)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    metal_vocab = MetalVocab(ckpt["metal_vocab"][1:])
    model = ResidueMetalSiteModel(cfg, num_metals=len(metal_vocab)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    collator = Collator(tokenizer, max_length=max_length)
    pseudo_batch = []
    for s, m in zip(sequences, metals):
        pseudo_batch.append({
            "sequence": s.strip().upper(),
            "labels": torch.zeros(len(s.strip()), dtype=torch.float32),
            "metal_id": metal_vocab.encode(m),
            "sample_weight": 1.0,
        })
    batch = collator(pseudo_batch)
    batch = {k: v.to(device) for k, v in batch.items()}
    logits = model(batch["input_ids"], batch["attention_mask"], batch["metal_ids"])
    probs = torch.sigmoid(logits).cpu()

    outputs = []
    for i, seq in enumerate(sequences):
        seq = seq.strip().upper()
        n = min(len(seq), max_length - 2)
        residue_probs = probs[i, 1:1 + n].tolist()
        residue_preds = [1 if p >= 0.5 else 0 for p in residue_probs]
        outputs.append({
            "sequence": seq,
            "metal": metals[i],
            "binding_probabilities": residue_probs,
            "binding_predictions": residue_preds,
        })
    return outputs


if __name__ == "__main__":
    cfg = TrainConfig(
        csv_path=os.environ.get("CSV_PATH", "metal_binding_dataset_from_CU.csv"),
        output_dir=os.environ.get("OUTPUT_DIR", "./outputs_metal_site"),
        model_name=os.environ.get("MODEL_NAME", "facebook/esm2_t12_35M_UR50D"),
        batch_size=int(os.environ.get("BATCH_SIZE", 4)),
        epochs=int(os.environ.get("EPOCHS", 15)),
        max_length=int(os.environ.get("MAX_LENGTH", 1024)),
    )
    print("Training config:")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")
    run_training(cfg)
