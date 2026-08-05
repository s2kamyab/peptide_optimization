
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_I = {a: i for i, a in enumerate(AA)}
I_TO_AA = {i: a for i, a in enumerate(AA)}
SEQ_LEN = 10
VOCAB = 20


@dataclass
class ModelConfig:
    hidden_size: int = 64
    latent_dim: int = 64
    n_layers: int = 2
    dropout: float = 0.0
    decoder_conditioning: str = "concat_z_at_every_decoder_step"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_peptide(x: object) -> Optional[str]:
    pep = str(x).strip().upper()
    if len(pep) != SEQ_LEN:
        return None
    if any(ch not in AA_TO_I for ch in pep):
        return None
    return pep


def onehot_encode_peptides(peptides: List[str]) -> torch.Tensor:
    x = torch.zeros((len(peptides), SEQ_LEN, VOCAB), dtype=torch.float32)
    for n, pep in enumerate(peptides):
        pep2 = clean_peptide(pep)
        if pep2 is None:
            raise ValueError(f"Invalid peptide: {pep!r}")
        for t, ch in enumerate(pep2):
            x[n, t, AA_TO_I[ch]] = 1.0
    return x


class PeptideCSVIterable(IterableDataset):
    """Stream length-10 peptide windows from CSV shards without loading all shards into memory."""

    def __init__(
        self,
        files: List[str],
        peptide_col: str = "peptide_len10",
        chunksize: int = 8192,
        shuffle_files: bool = True,
        deduplicate_within_epoch: bool = False,
    ):
        self.files = list(files)
        self.peptide_col = peptide_col
        self.chunksize = int(chunksize)
        self.shuffle_files = bool(shuffle_files)
        self.deduplicate_within_epoch = bool(deduplicate_within_epoch)

    def __iter__(self) -> Iterable[torch.Tensor]:
        files = list(self.files)
        if self.shuffle_files:
            random.shuffle(files)

        seen = set()
        for csv_path in files:
            for chunk in pd.read_csv(csv_path, usecols=[self.peptide_col], chunksize=self.chunksize):
                peptides = []
                for value in chunk[self.peptide_col].tolist():
                    pep = clean_peptide(value)
                    if pep is None:
                        continue
                    if self.deduplicate_within_epoch and pep in seen:
                        continue
                    seen.add(pep)
                    peptides.append(pep)
                if peptides:
                    yield onehot_encode_peptides(peptides)


class GRUEncoder(nn.Module):
    def __init__(self, hidden_size: int, latent_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Linear(VOCAB, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.to_mu = nn.Linear(hidden_size, latent_dim)
        self.to_logvar = nn.Linear(hidden_size, latent_dim)

    def forward(self, x_onehot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.in_proj(x_onehot)
        out_top, h_n = self.gru(h)
        h_last = h_n[-1]
        mu = self.to_mu(h_last)
        logvar = self.to_logvar(h_last).clamp(min=-8.0, max=8.0)
        return mu, logvar, out_top


class LatentConditionedGRUDecoder(nn.Module):
    """GRU decoder where z is concatenated to the token embedding at every step."""

    def __init__(self, hidden_size: int, latent_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.token_embed = nn.Linear(VOCAB, hidden_size)
        self.z_to_h = nn.Linear(latent_dim, n_layers * hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size + latent_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.to_logits = nn.Linear(hidden_size, VOCAB)

    def forward(self, z: torch.Tensor, x_onehot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = z.size(0)
        x_shift = torch.zeros_like(x_onehot)
        x_shift[:, 1:, :] = x_onehot[:, :-1, :]
        emb = self.token_embed(x_shift)
        z_repeat = z.unsqueeze(1).expand(-1, x_onehot.size(1), -1)
        decoder_input = torch.cat([emb, z_repeat], dim=-1)
        h0 = self.z_to_h(z).view(self.gru.num_layers, batch_size, self.gru.hidden_size)
        out_top, _ = self.gru(decoder_input, h0)
        logits = self.to_logits(out_top)
        return logits, out_top


class GRUVAE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.enc = GRUEncoder(cfg.hidden_size, cfg.latent_dim, cfg.n_layers, cfg.dropout)
        self.dec = LatentConditionedGRUDecoder(cfg.hidden_size, cfg.latent_dim, cfg.n_layers, cfg.dropout)

    def reparam(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x_onehot: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar, enc_out = self.enc(x_onehot)
        z = self.reparam(mu, logvar)
        logits, dec_out = self.dec(z, x_onehot)
        return {"mu": mu, "logvar": logvar, "z": z, "logits": logits, "enc_out": enc_out, "dec_out": dec_out}


def vae_loss_no_flow(x_onehot: torch.Tensor, out: Dict[str, torch.Tensor], beta: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    target = x_onehot.argmax(dim=-1)
    recon = F.cross_entropy(out["logits"].reshape(-1, VOCAB), target.reshape(-1), reduction="mean")
    mu, logvar = out["mu"], out["logvar"]
    kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean().clamp(min=0.0)
    loss = recon + float(beta) * kl
    pred = out["logits"].argmax(dim=-1)
    token_acc = (pred == target).float().mean()
    return loss, {
        "loss": float(loss.detach().cpu()),
        "recon": float(recon.detach().cpu()),
        "kl": float(kl.detach().cpu()),
        "token_acc": float(token_acc.detach().cpu()),
    }


def kl_beta_for_epoch(epoch: int, kl_beta: float, warmup_epochs: int) -> float:
    return min(float(kl_beta), float(kl_beta) * float(epoch) / max(1, int(warmup_epochs)))


def append_history(path: str, row: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: ModelConfig,
    args: argparse.Namespace,
    epoch: int,
    metrics: Dict[str, float],
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "checkpoint_type": "pretrained_gru_vae_latent_conditioned_no_flow",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": asdict(cfg),
        "args": vars(args),
        "epoch": int(epoch),
        "metrics": metrics,
        "aa": AA,
        "seq_len": SEQ_LEN,
        "decoder_latent_conditioning": cfg.decoder_conditioning,
    }
    torch.save(payload, path)
    with open(path + ".json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in payload.items() if k not in {"model_state_dict", "optimizer_state_dict"}}, f, indent=2)


def load_training_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, cfg: ModelConfig, device: torch.device, load_optimizer: bool = True) -> int:
    ckpt = torch.load(path, map_location=device)
    saved_cfg = ckpt.get("model_config", {})
    current_cfg = asdict(cfg)
    mismatches = {
        k: (saved_cfg.get(k), current_cfg.get(k))
        for k in ["hidden_size", "latent_dim", "n_layers", "dropout"]
        if saved_cfg.get(k) != current_cfg.get(k)
    }
    if mismatches:
        raise ValueError(f"Checkpoint architecture mismatch: {mismatches}")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    if load_optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
    epoch = int(ckpt.get("epoch", 0))
    print(f"Resumed {path} from epoch {epoch}")
    return epoch + 1


def find_pretrain_files(parts_dir: str, file_pattern: str) -> List[str]:
    if file_pattern == "auto":
        patterns = [
            "metalpdb_ALL_chain_mapped_len10_high_confidence_part_*.csv",
            "metalpdb_binding_windows_len10_part_*.csv",
            "*chain_mapped*len10*part_*.csv",
            "*len10*part_*.csv",
            "*.csv",
        ]
    else:
        patterns = [file_pattern]

    for pat in patterns:
        full = os.path.join(parts_dir, pat)
        files = sorted(glob.glob(full))
        files = [p for p in files if "failures" not in os.path.basename(p).lower()]
        if files:
            print(f"Found {len(files)} files using pattern: {full}")
            return files

    raise FileNotFoundError(f"No pretraining CSV shards found in {parts_dir!r} with pattern={file_pattern!r}")


def train(args: argparse.Namespace) -> str:
    set_seed(args.seed)
    device = torch.device(args.device)
    cfg = ModelConfig(args.hidden_size, args.latent_dim, args.n_layers, args.dropout)

    files = find_pretrain_files(args.parts_dir, args.file_pattern)
    dataset = PeptideCSVIterable(
        files=files,
        peptide_col=args.peptide_col,
        chunksize=args.chunksize,
        shuffle_files=True,
        deduplicate_within_epoch=args.deduplicate,
    )
    loader = DataLoader(dataset, batch_size=None, num_workers=0)

    model = GRUVAE(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ckpt_name = f"pretrained_gru_vae_latent_conditioned_h{cfg.hidden_size}_z{cfg.latent_dim}.pt"
    out_path = os.path.join(args.out_dir, ckpt_name)
    history_path = args.history_csv or os.path.join(args.out_dir, "pretraining_history.csv")

    start_epoch = 1
    resume_path = args.resume_checkpoint or out_path
    if not args.no_resume and os.path.exists(resume_path):
        start_epoch = load_training_checkpoint(resume_path, model, optimizer, cfg, device, load_optimizer=not args.no_resume_optimizer)
    elif args.resume_checkpoint and not os.path.exists(args.resume_checkpoint):
        raise FileNotFoundError(args.resume_checkpoint)

    if args.no_resume and os.path.exists(history_path) and not args.append_history:
        os.remove(history_path)

    print(f"Pretraining latent-conditioned GRU-VAE H{cfg.hidden_size}/Z{cfg.latent_dim} on {len(files)} all-metal shards.")
    print(f"Output checkpoint: {out_path}")
    print(f"History CSV: {history_path}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        beta = kl_beta_for_epoch(epoch, args.kl_beta, args.kl_warmup_epochs)
        sums = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "token_acc": 0.0}
        n_batches = 0
        last_out = None

        for x in loader:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss, metrics = vae_loss_no_flow(x, out, beta=beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            last_out = out
            n_batches += 1
            for k in sums:
                sums[k] += metrics[k]

            if args.log_every and n_batches % args.log_every == 0:
                avg = {k: v / max(1, n_batches) for k, v in sums.items()}
                print(f"epoch={epoch} batch={n_batches} beta={beta:.6f} loss={avg['loss']:.4f} recon={avg['recon']:.4f} kl={avg['kl']:.6f} acc={avg['token_acc']:.4f}")

        metrics = {k: v / max(1, n_batches) for k, v in sums.items()}
        if last_out is not None:
            diagnostics = {
                "mu_abs_mean": float(last_out["mu"].abs().mean().detach().cpu()),
                "mu_std": float(last_out["mu"].std().detach().cpu()),
                "logvar_mean": float(last_out["logvar"].mean().detach().cpu()),
                "logvar_std": float(last_out["logvar"].std().detach().cpu()),
            }
        else:
            diagnostics = {"mu_abs_mean": np.nan, "mu_std": np.nan, "logvar_mean": np.nan, "logvar_std": np.nan}

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "epoch": epoch,
            "n_batches": n_batches,
            "beta": beta,
            **metrics,
            **diagnostics,
            "lr": optimizer.param_groups[0]["lr"],
            "hidden_size": cfg.hidden_size,
            "latent_dim": cfg.latent_dim,
            "n_layers": cfg.n_layers,
            "dropout": cfg.dropout,
            "decoder_conditioning": cfg.decoder_conditioning,
        }
        append_history(history_path, row)
        save_checkpoint(out_path, model, optimizer, cfg, args, epoch, {**metrics, **diagnostics})
        print(f"epoch={epoch} metrics={metrics} diagnostics={diagnostics}")

    print(f"Saved pretrained checkpoint: {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pretrain larger latent-conditioned GRU-VAE on all-metal MetalPDB length-10 peptide windows.")
    p.add_argument("--parts-dir", default="metalpdb_all_metals_chain_mapped_len10_high_confidence_parts/parts")
    p.add_argument("--file-pattern", default="auto", help="CSV shard glob under --parts-dir. Use 'auto' for common shard names.")
    p.add_argument("--peptide-col", default="peptide_len10")
    p.add_argument("--out-dir", default="transfer_gru_vae_checkpoints_h64_z64_latent_conditioned_high_confidence_dataset")
    p.add_argument("--history-csv", default=None)
    p.add_argument("--append-history", action="store_true")
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--chunksize", type=int, default=8192)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--kl-beta", type=float, default=1e-4)
    p.add_argument("--kl-warmup-epochs", type=int, default=30)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--resume-checkpoint", default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--no-resume-optimizer", action="store_true")
    p.add_argument("--deduplicate", action="store_true")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train(args)
