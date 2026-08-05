
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
from datetime import datetime
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, TensorDataset

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_I = {a: i for i, a in enumerate(AA)}
I_TO_AA = {i: a for i, a in enumerate(AA)}
SEQ_LEN = 10
VOCAB = 20

@dataclass
class ModelConfig:
    hidden_size: int = 32
    latent_dim: int = 32
    n_layers: int = 2
    dropout: float = 0.0


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
        for t, ch in enumerate(pep):
            x[n, t, AA_TO_I[ch]] = 1.0
    return x


def decode_logits_to_peptides(logits: torch.Tensor) -> List[str]:
    idx = logits.argmax(dim=-1).detach().cpu().tolist()
    return ["".join(I_TO_AA[int(i)] for i in row) for row in idx]


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
        x = self.in_proj(x_onehot)
        out_top, h_n = self.gru(x)
        h_last = h_n[-1]
        mu = self.to_mu(h_last)
        logvar = self.to_logvar(h_last).clamp(min=-8.0, max=8.0)
        return mu, logvar, out_top


class GRUDecoder(nn.Module):
    def __init__(self, hidden_size: int, latent_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.token_embed = nn.Linear(VOCAB, hidden_size)
        self.z_to_h = nn.Linear(latent_dim, n_layers * hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
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
        h0 = self.z_to_h(z).view(self.gru.num_layers, batch_size, self.gru.hidden_size)
        out_top, _ = self.gru(emb, h0)
        logits = self.to_logits(out_top)
        return logits, out_top


class GRUVAE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.enc = GRUEncoder(cfg.hidden_size, cfg.latent_dim, cfg.n_layers, cfg.dropout)
        self.dec = GRUDecoder(cfg.hidden_size, cfg.latent_dim, cfg.n_layers, cfg.dropout)

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
    # kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean().clamp(min=0.0)
    kl_per_sample = -0.5 * torch.sum(
        1.0 + logvar - mu.pow(2) - logvar.exp(),
        dim=-1,
    )
    kl = kl_per_sample.mean().clamp(min=0.0)
    loss = recon + beta * kl
    pred = out["logits"].argmax(dim=-1)
    token_acc = (pred == target).float().mean()
    return loss, {"loss": float(loss.detach().cpu()), "recon": float(recon.detach().cpu()), "kl": float(kl.detach().cpu()), "token_acc": float(token_acc.detach().cpu())}


def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, cfg: ModelConfig, args: argparse.Namespace, epoch: int, metrics: Dict[str, float], extra: Optional[Dict] = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": asdict(cfg),
        "args": vars(args),
        "epoch": epoch,
        "metrics": metrics,
        "aa": AA,
        "seq_len": SEQ_LEN,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    with open(path + ".json", "w", encoding="utf-8") as f:
        json.dump({"model_config": asdict(cfg), "epoch": epoch, "metrics": metrics}, f, indent=2)


def append_training_history_csv(path: str, row: Dict[str, float]) -> None:
    """Append one epoch-level training summary to a CSV file.

    The file is created with a header if it does not exist. This makes resumed
    runs easy to track because each completed epoch is added as one new row.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path)
    pd.DataFrame([row]).to_csv(path, mode="a", header=write_header, index=False)


def load_pretrained_vae_weights(model: nn.Module, checkpoint_path: str, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    current = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in current and current[k].shape == v.shape}
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    skipped = sorted(set(state) - set(compatible))
    print(f"Loaded {len(compatible)} compatible tensors from {checkpoint_path}")
    if skipped:
        print(f"Skipped incompatible tensors: {skipped[:12]}{' ...' if len(skipped) > 12 else ''}")
    if missing:
        print(f"Missing tensors initialized randomly: {missing}")
    if unexpected:
        print(f"Unexpected tensors ignored: {unexpected}")


def load_training_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: ModelConfig,
    device: torch.device,
    load_optimizer: bool = True,
) -> int:
    """Load a full training checkpoint and return the next epoch number.

    The checkpoint must match the current architecture. Optimizer state is loaded
    by default so AdamW momentum/variance terms continue from the previous run.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    saved_cfg = ckpt.get("model_config")
    current_cfg = asdict(cfg)
    if saved_cfg is not None:
        mismatches = {
            k: (saved_cfg.get(k), current_cfg.get(k))
            for k in current_cfg
            if saved_cfg.get(k) != current_cfg.get(k)
        }
        if mismatches:
            raise ValueError(
                "Checkpoint architecture does not match current arguments. "
                f"Mismatches: {mismatches}. Use the same --hidden-size, "
                "--latent-dim, --n-layers, and --dropout values, or start "
                "fresh with --no-resume / a different --out-dir."
            )

    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    if load_optimizer and "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            # Make sure optimizer state tensors are on the selected device after loading.
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(device)
            print("Loaded optimizer state from checkpoint.")
        except Exception as exc:
            print(f"Warning: could not load optimizer state: {exc}")
            print("Continuing with a freshly initialized optimizer.")

    saved_epoch = int(ckpt.get("epoch", 0))
    next_epoch = saved_epoch + 1
    print(f"Resumed checkpoint: {checkpoint_path}")
    print(f"Checkpoint epoch: {saved_epoch}; next epoch: {next_epoch}")
    if "metrics" in ckpt:
        print(f"Checkpoint metrics: {ckpt['metrics']}")
    return next_epoch


def kl_beta_for_epoch(epoch: int, kl_beta: float, warmup_epochs: int) -> float:
    return min(kl_beta, kl_beta * epoch / max(1, warmup_epochs))


class PeptideCSVIterable(IterableDataset):
    """Stream peptide_len10 values from many CSV shards using pandas chunks."""
    def __init__(self, files: List[str], peptide_col: str = "peptide_len10", chunksize: int = 8192, shuffle_files: bool = True, deduplicate_within_epoch: bool = False):
        self.files = list(files)
        self.peptide_col = peptide_col
        self.chunksize = chunksize
        self.shuffle_files = shuffle_files
        self.deduplicate_within_epoch = deduplicate_within_epoch

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


def find_pretrain_files(parts_dir: str) -> List[str]:
    pattern = os.path.join(parts_dir,"metalpdb_ALL_chain_mapped_len10_high_confidence_part_*.csv")# "metalpdb_binding_windows_len10_part_*.csv")
    files = sorted(glob.glob(pattern))
    files = [p for p in files if "failures" not in os.path.basename(p)]
    if not files:
        raise FileNotFoundError(f"No pretraining files found with pattern: {pattern}")
    return files


def train(args: argparse.Namespace) -> str:
    set_seed(args.seed)
    device = torch.device(args.device)
    cfg = ModelConfig(hidden_size=args.hidden_size, latent_dim=args.latent_dim, n_layers=args.n_layers, dropout=args.dropout)
    files = find_pretrain_files(args.parts_dir)
    dataset = PeptideCSVIterable(files, peptide_col=args.peptide_col, chunksize=args.chunksize, shuffle_files=True, deduplicate_within_epoch=args.deduplicate)
    loader = DataLoader(dataset, batch_size=None, num_workers=0)

    model = GRUVAE(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    out_path = os.path.join(args.out_dir, "pretrained_gru_vae_no_flow_h32_z32.pt")
    history_path = args.history_csv or os.path.join(args.out_dir, "pretraining_history.csv")

    start_epoch = 1
    resume_path = args.resume_checkpoint or out_path
    if not args.no_resume and os.path.exists(resume_path):
        start_epoch = load_training_checkpoint(
            checkpoint_path=resume_path,
            model=model,
            optimizer=optimizer,
            cfg=cfg,
            device=device,
            load_optimizer=not args.no_resume_optimizer,
        )
    elif args.resume_checkpoint and not os.path.exists(args.resume_checkpoint):
        raise FileNotFoundError(f"Requested --resume-checkpoint was not found: {args.resume_checkpoint}")

    if args.no_resume and start_epoch == 1 and os.path.exists(history_path) and not args.append_history:
        os.remove(history_path)
        print(f"Removed existing history CSV because --no-resume was used: {history_path}")

    print(f"Training history CSV: {history_path}")

    if start_epoch > args.epochs:
        print(
            f"Checkpoint is already at epoch {start_epoch - 1}, "
            f"which is >= requested --epochs {args.epochs}. Nothing to train."
        )
        return out_path

    print(f"Pretraining GRU-VAE on {len(files)} all-metal CSV shards")
    print(f"Training epochs: {start_epoch} to {args.epochs}")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        sums = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "token_acc": 0.0}
        n_batches = 0
        beta = kl_beta_for_epoch(epoch, args.kl_beta, args.kl_warmup_epochs)
        for x in loader:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss, metrics = vae_loss_no_flow(x, out, beta=beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            n_batches += 1
            for key in sums:
                sums[key] += metrics[key]
            if args.log_every and n_batches % args.log_every == 0:
                avg = {k: v / n_batches for k, v in sums.items()}
                print(f"epoch={epoch} batch={n_batches} beta={beta:.5f} loss={avg['loss']:.4f} recon={avg['recon']:.4f} kl={avg['kl']:.8f} acc={avg['token_acc']:.4f}")
        with torch.no_grad():
            mu_abs_mean = out["mu"].abs().mean().item()
            mu_std = out["mu"].std().item()
            logvar_mean = out["logvar"].mean().item()
            logvar_std = out["logvar"].std().item()
            print(
                "mu_abs_mean=", mu_abs_mean,
                "mu_std=", mu_std,
                "logvar_mean=", logvar_mean,
                "logvar_std=", logvar_std,
            )
        epoch_metrics = {k: v / max(1, n_batches) for k, v in sums.items()}
        print(f"epoch={epoch} metrics={epoch_metrics}")

        history_row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "epoch": epoch,
            "n_batches": n_batches,
            "beta": beta,
            "loss": epoch_metrics["loss"],
            "recon": epoch_metrics["recon"],
            "kl": epoch_metrics["kl"],
            "token_acc": epoch_metrics["token_acc"],
            "mu_abs_mean": mu_abs_mean,
            "mu_std": mu_std,
            "logvar_mean": logvar_mean,
            "logvar_std": logvar_std,
            "lr": optimizer.param_groups[0].get("lr", float("nan")),
            "hidden_size": cfg.hidden_size,
            "latent_dim": cfg.latent_dim,
            "n_layers": cfg.n_layers,
            "dropout": cfg.dropout,
        }
        append_training_history_csv(history_path, history_row)
        print(f"Appended epoch {epoch} to history CSV: {history_path}")

        save_checkpoint(out_path, model, optimizer, cfg, args, epoch, epoch_metrics)
    print(f"Saved pretrained checkpoint: {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pretrain compact H32/Z32 GRU-VAE on all-metal MetalPDB length-10 peptide windows.")
    p.add_argument("--parts-dir", default="metalpdb_all_metals_chain_mapped_len10_high_confidence_parts/parts")
    p.add_argument("--peptide-col", default="peptide_len10")
    p.add_argument("--out-dir", default="transfer_gru_vae_checkpoints_h32_z32_high_confidence_dataset")
    p.add_argument("--history-csv", default=None, help="Optional CSV path for epoch-level training history. Defaults to <out-dir>/pretraining_history.csv.")
    p.add_argument("--append-history", action="store_true", help="When starting from scratch with --no-resume, append to an existing history CSV instead of deleting it.")
    p.add_argument("--hidden-size", type=int, default=32)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--chunksize", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--kl-beta", type=float, default=0.01)
    p.add_argument("--kl-warmup-epochs", type=int, default=5)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--resume-checkpoint", default=None, help="Optional checkpoint path to resume from. Defaults to the output checkpoint path if it exists.")
    p.add_argument("--no-resume", action="store_true", help="Start from scratch even if a checkpoint already exists.")
    p.add_argument("--no-resume-optimizer", action="store_true", help="Resume model weights but reinitialize the optimizer state.")
    p.add_argument("--deduplicate", action="store_true")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train(args)
