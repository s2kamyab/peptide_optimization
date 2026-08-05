
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
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
    kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean().clamp(min=0.0)
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


def kl_beta_for_epoch(epoch: int, kl_beta: float, warmup_epochs: int) -> float:
    return min(kl_beta, kl_beta * epoch / max(1, warmup_epochs))


class PlanarFlow(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim) * 0.02)
        self.u = nn.Parameter(torch.randn(dim) * 0.02)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a = z @ self.w + self.b
        h = torch.tanh(a)
        z_new = z + h.unsqueeze(-1) * self.u
        psi = (1.0 - h.pow(2)).unsqueeze(-1) * self.w.unsqueeze(0)
        det_term = 1.0 + (psi * self.u.unsqueeze(0)).sum(dim=-1)
        logabsdet = torch.log(torch.abs(det_term) + 1e-8)
        return z_new, logabsdet

class FlowSequence(nn.Module):
    def __init__(self, dim: int, n_flows: int):
        super().__init__()
        self.flows = nn.ModuleList([PlanarFlow(dim) for _ in range(n_flows)])

    def forward(self, z0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = z0
        sum_logdet = torch.zeros(z0.size(0), device=z0.device)
        for flow in self.flows:
            z, logdet = flow(z)
            sum_logdet = sum_logdet + logdet
        return z, sum_logdet

class FlowFineTuneModel(nn.Module):
    def __init__(self, cfg: ModelConfig, n_flows: int):
        super().__init__()
        self.cfg = cfg
        self.enc = GRUEncoder(cfg.hidden_size, cfg.latent_dim, cfg.n_layers, cfg.dropout)
        self.dec = GRUDecoder(cfg.hidden_size, cfg.latent_dim, cfg.n_layers, cfg.dropout)
        self.flow = FlowSequence(cfg.latent_dim, n_flows)
        self.score_head = nn.Sequential(nn.Linear(cfg.latent_dim, cfg.hidden_size), nn.SiLU(), nn.Linear(cfg.hidden_size, 1))

    def reparam(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x_onehot: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar, enc_out = self.enc(x_onehot)
        z0 = self.reparam(mu, logvar)
        zK, sum_logdet = self.flow(z0)
        logits, dec_out = self.dec(zK, x_onehot)
        score_pred = self.score_head(zK).squeeze(-1)
        return {"mu": mu, "logvar": logvar, "z0": z0, "zK": zK, "sum_logdet": sum_logdet, "logits": logits, "score_pred": score_pred, "enc_out": enc_out, "dec_out": dec_out}


def vae_loss_with_flow(x_onehot: torch.Tensor, out: Dict[str, torch.Tensor], beta: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    target = x_onehot.argmax(dim=-1)
    recon = F.cross_entropy(out["logits"].reshape(-1, VOCAB), target.reshape(-1), reduction="mean")
    mu, logvar, z0, zK, sum_logdet = out["mu"], out["logvar"], out["z0"], out["zK"], out["sum_logdet"]
    log2pi = math.log(2.0 * math.pi)
    var = torch.exp(logvar)
    log_q0 = -0.5 * ((((z0 - mu) ** 2) / var) + logvar + log2pi).sum(dim=-1)
    log_p_zK = -0.5 * (zK.pow(2) + log2pi).sum(dim=-1)
    kl = (log_q0 - sum_logdet - log_p_zK).mean().clamp(min=0.0)
    loss = recon + beta * kl
    pred = out["logits"].argmax(dim=-1)
    token_acc = (pred == target).float().mean()
    return loss, {"loss": float(loss.detach().cpu()), "recon": float(recon.detach().cpu()), "kl": float(kl.detach().cpu()), "token_acc": float(token_acc.detach().cpu())}



def straight_through_argmax_onehot(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Return hard one-hot tokens in the forward pass with softmax gradients backward."""
    tau = max(float(temperature), 1e-6)
    probs = torch.softmax(logits / tau, dim=-1)
    hard_idx = probs.argmax(dim=-1)
    hard = F.one_hot(hard_idx, num_classes=VOCAB).to(dtype=probs.dtype)
    return hard + probs - probs.detach()


def latent_roundtrip_consistency(
    model: "FlowFineTuneModel",
    x_onehot: torch.Tensor,
    out: Dict[str, torch.Tensor],
    temperature: float,
    cosine_weight: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Differentiable cycle consistency aligned with the BO search space.

    BO later uses the deterministic after-flow representation flow(mu). Therefore
    the round-trip path is:

        x -> mu -> flow(mu)=z_ref -> free-running decoder -> hard-ST tokens
          -> encoder -> mu_rt -> flow(mu_rt)=z_rt

    The decoder receives an all-zero teacher-forcing tensor so the original
    peptide tokens are not leaked into the cycle-consistency path.
    """
    z_ref, _ = model.flow(out["mu"])
    x_dummy = torch.zeros_like(x_onehot)
    rt_logits, _ = model.dec(z_ref, x_dummy)
    x_rt_st = straight_through_argmax_onehot(rt_logits, temperature=temperature)
    mu_rt, _, _ = model.enc(x_rt_st)
    z_rt, _ = model.flow(mu_rt)

    per_sample_l2 = torch.linalg.norm(z_ref - z_rt, dim=-1)
    per_sample_cos = F.cosine_similarity(z_ref, z_rt, dim=-1, eps=1e-8)
    l2_mean = per_sample_l2.mean()
    cos_mean = per_sample_cos.mean()
    cos_loss = (1.0 - per_sample_cos).mean()
    rt_loss = l2_mean + float(cosine_weight) * cos_loss

    return rt_loss, {
        "roundtrip_loss": rt_loss,
        "roundtrip_l2": l2_mean,
        "roundtrip_cosine": cos_mean,
    }


def load_cu_dataset(csv_path: str, peptide_col: str, score_col: str) -> Tuple[torch.Tensor, torch.Tensor]:
    df = pd.read_csv(csv_path)
    peptides, scores = [], []
    for pep_raw, score_raw in zip(df[peptide_col].tolist(), df[score_col].tolist()):
        pep = clean_peptide(pep_raw)
        if pep is None or pd.isna(score_raw):
            continue
        peptides.append(pep)
        scores.append(float(score_raw))
    if not peptides:
        raise ValueError(f"No valid scored Cu peptides found in {csv_path}")
    return onehot_encode_peptides(peptides), torch.tensor(scores, dtype=torch.float32)


def build_optimizer(model: FlowFineTuneModel, args: argparse.Namespace) -> torch.optim.Optimizer:
    vae_params = list(model.enc.parameters()) + list(model.dec.parameters())
    new_params = list(model.flow.parameters()) + list(model.score_head.parameters())
    return torch.optim.AdamW([
        {"params": vae_params, "lr": args.vae_lr, "name": "pretrained_gru_vae"},
        {"params": new_params, "lr": args.flow_lr, "name": "flow_and_score_head"},
    ], weight_decay=args.weight_decay)


def append_history_row(path: str, row: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exists = os.path.exists(path)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not exists, index=False)


def train(args: argparse.Namespace) -> str:
    set_seed(args.seed)
    device = torch.device(args.device)
    cfg = ModelConfig(
        hidden_size=args.hidden_size,
        latent_dim=args.latent_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )
    x_all, y_all = load_cu_dataset(args.cu_csv, args.peptide_col, args.score_col)
    n = x_all.size(0)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
    n_val = max(1, int(round(args.val_frac * n)))
    train_idx, val_idx = perm[n_val:], perm[:n_val]
    train_loader = DataLoader(
        TensorDataset(x_all[train_idx], y_all[train_idx]),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(x_all[val_idx], y_all[val_idx]),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = FlowFineTuneModel(cfg, n_flows=args.n_flows).to(device)
    load_pretrained_vae_weights(model, args.init_checkpoint, device)
    optimizer = build_optimizer(model, args)

    best_total_path = os.path.join(
        args.out_dir,
        "best_total_loss_h32_z32_with_flow_roundtrip.pt",
    )
    best_roundtrip_path = os.path.join(
        args.out_dir,
        "best_roundtrip_h32_z32_with_flow_roundtrip.pt",
    )
    last_epoch_path = os.path.join(
        args.out_dir,
        "last_epoch_h32_z32_with_flow_roundtrip.pt",
    )
    history_path = os.path.join(args.out_dir, "training_history_roundtrip.csv")
    checkpoint_summary_path = os.path.join(args.out_dir, "checkpoint_summary.csv")

    if os.path.exists(history_path) and not args.append_history:
        os.remove(history_path)
    if os.path.exists(checkpoint_summary_path) and not args.append_history:
        os.remove(checkpoint_summary_path)

    best_val = float("inf")
    best_val_epoch = 0
    best_roundtrip_l2 = float("inf")
    best_roundtrip_epoch = 0
    print(f"Fine-tuning H32/Z32 model with normalizing flow on {len(train_idx)} Cu rows; validating on {len(val_idx)} rows")
    print(f"Learning rates: GRU-VAE={args.vae_lr}, flow+score_head={args.flow_lr}")
    print(
        f"Round-trip: weight={args.roundtrip_loss_weight}, "
        f"cosine_weight={args.roundtrip_cosine_weight}, "
        f"temperature={args.roundtrip_temperature}"
    )

    for epoch in range(1, args.epochs + 1):
        beta = kl_beta_for_epoch(epoch, args.kl_beta, args.kl_warmup_epochs)
        model.train()
        sums = {
            "loss": 0.0,
            "recon": 0.0,
            "kl": 0.0,
            "score_mse": 0.0,
            "token_acc": 0.0,
            "roundtrip_loss": 0.0,
            "roundtrip_l2": 0.0,
            "roundtrip_cosine": 0.0,
        }
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss_vae, metrics = vae_loss_with_flow(x, out, beta)
            score_mse = F.mse_loss(out["score_pred"], y)
            rt_loss, rt_metrics = latent_roundtrip_consistency(
                model,
                x,
                out,
                temperature=args.roundtrip_temperature,
                cosine_weight=args.roundtrip_cosine_weight,
            )
            loss = (
                loss_vae
                + args.score_loss_weight * score_mse
                + args.roundtrip_loss_weight * rt_loss
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            n_batches += 1
            sums["loss"] += float(loss.detach().cpu())
            sums["recon"] += metrics["recon"]
            sums["kl"] += metrics["kl"]
            sums["score_mse"] += float(score_mse.detach().cpu())
            sums["token_acc"] += metrics["token_acc"]
            for key in ["roundtrip_loss", "roundtrip_l2", "roundtrip_cosine"]:
                sums[key] += float(rt_metrics[key].detach().cpu())

        train_metrics = {k: v / max(1, n_batches) for k, v in sums.items()}
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            beta=args.kl_beta,
            score_loss_weight=args.score_loss_weight,
            roundtrip_loss_weight=args.roundtrip_loss_weight,
            roundtrip_cosine_weight=args.roundtrip_cosine_weight,
            roundtrip_temperature=args.roundtrip_temperature,
        )

        row: Dict[str, object] = {
            "epoch": epoch,
            "beta": beta,
            "hidden_size": args.hidden_size,
            "latent_dim": args.latent_dim,
            "vae_lr": args.vae_lr,
            "flow_lr": args.flow_lr,
            "kl_beta": args.kl_beta,
            "score_loss_weight": args.score_loss_weight,
            "roundtrip_loss_weight": args.roundtrip_loss_weight,
            "roundtrip_cosine_weight": args.roundtrip_cosine_weight,
            "roundtrip_temperature": args.roundtrip_temperature,
        }
        for k, v in train_metrics.items():
            row[f"train_{k}"] = v
        for k, v in val_metrics.items():
            row[f"val_{k}"] = v
        append_history_row(history_path, row)

        print(f"epoch={epoch} train={train_metrics} val={val_metrics}")

        common_extra = {
            "n_flows": args.n_flows,
            "roundtrip_loss_weight": args.roundtrip_loss_weight,
            "roundtrip_definition": "x->mu->flow(mu)->free_decode_ST_argmax->enc->flow(mu_rt)",
        }

        # 1) Save the checkpoint with the best validation total loss.
        if val_metrics["loss"] <= best_val:
            best_val = val_metrics["loss"]
            best_val_epoch = epoch
            save_checkpoint(
                best_total_path,
                model,
                optimizer,
                cfg,
                args,
                epoch,
                val_metrics,
                extra={
                    **common_extra,
                    "checkpoint_selection": "minimum_validation_total_loss",
                },
            )

        # 2) Save the checkpoint with the lowest validation round-trip L2.
        # This is the main geometry-focused checkpoint for latent-space BO.
        if val_metrics["roundtrip_l2"] <= best_roundtrip_l2:
            best_roundtrip_l2 = val_metrics["roundtrip_l2"]
            best_roundtrip_epoch = epoch
            save_checkpoint(
                best_roundtrip_path,
                model,
                optimizer,
                cfg,
                args,
                epoch,
                val_metrics,
                extra={
                    **common_extra,
                    "checkpoint_selection": "minimum_validation_roundtrip_l2",
                },
            )

        # 3) Always overwrite a last-epoch checkpoint so the final training state
        # is available even when it is not selected by either validation criterion.
        save_checkpoint(
            last_epoch_path,
            model,
            optimizer,
            cfg,
            args,
            epoch,
            val_metrics,
            extra={
                **common_extra,
                "checkpoint_selection": "last_completed_epoch",
            },
        )

    # Save a compact comparison table for downstream BO checkpoint selection.
    summary_rows = []
    for selection, path in [
        ("best_total_loss", best_total_path),
        ("best_roundtrip_l2", best_roundtrip_path),
        ("last_epoch", last_epoch_path),
    ]:
        ckpt = torch.load(path, map_location="cpu")
        metrics = ckpt.get("metrics", {})
        summary_rows.append({
            "selection": selection,
            "checkpoint_path": path,
            "epoch": int(ckpt.get("epoch", -1)),
            "val_loss": metrics.get("loss", float("nan")),
            "val_recon": metrics.get("recon", float("nan")),
            "val_kl": metrics.get("kl", float("nan")),
            "val_score_mse": metrics.get("score_mse", float("nan")),
            "val_token_acc": metrics.get("token_acc", float("nan")),
            "val_roundtrip_loss": metrics.get("roundtrip_loss", float("nan")),
            "val_roundtrip_l2": metrics.get("roundtrip_l2", float("nan")),
            "val_roundtrip_cosine": metrics.get("roundtrip_cosine", float("nan")),
        })

    pd.DataFrame(summary_rows).to_csv(checkpoint_summary_path, index=False)

    print(f"Saved best-total-loss checkpoint: {best_total_path}")
    print(f"  epoch={best_val_epoch}, val_loss={best_val:.6f}")
    print(f"Saved best-roundtrip checkpoint: {best_roundtrip_path}")
    print(
        f"  epoch={best_roundtrip_epoch}, "
        f"val_roundtrip_l2={best_roundtrip_l2:.6f}"
    )
    print(f"Saved last-epoch checkpoint: {last_epoch_path}")
    print(f"Saved checkpoint comparison: {checkpoint_summary_path}")
    print(f"Saved training history: {history_path}")

    # Return the geometry-focused checkpoint because the downstream target is
    # latent-space Bayesian optimization.
    return best_roundtrip_path


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    beta,
    score_loss_weight,
    roundtrip_loss_weight,
    roundtrip_cosine_weight,
    roundtrip_temperature,
):
    model.eval()
    sums = {
        "loss": 0.0,
        "recon": 0.0,
        "kl": 0.0,
        "score_mse": 0.0,
        "token_acc": 0.0,
        "roundtrip_loss": 0.0,
        "roundtrip_l2": 0.0,
        "roundtrip_cosine": 0.0,
    }
    n_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_vae, metrics = vae_loss_with_flow(x, out, beta)
        score_mse = F.mse_loss(out["score_pred"], y)
        rt_loss, rt_metrics = latent_roundtrip_consistency(
            model,
            x,
            out,
            temperature=roundtrip_temperature,
            cosine_weight=roundtrip_cosine_weight,
        )
        loss = loss_vae + score_loss_weight * score_mse + roundtrip_loss_weight * rt_loss

        n_batches += 1
        sums["loss"] += float(loss.cpu())
        sums["recon"] += metrics["recon"]
        sums["kl"] += metrics["kl"]
        sums["score_mse"] += float(score_mse.cpu())
        sums["token_acc"] += metrics["token_acc"]
        for key in ["roundtrip_loss", "roundtrip_l2", "roundtrip_cosine"]:
            sums[key] += float(rt_metrics[key].cpu())
    return {k: v / max(1, n_batches) for k, v in sums.items()}


def parse_args():
    p=argparse.ArgumentParser(description="Fine-tune compact H32/Z32 pretrained GRU-VAE on Cu with normalizing flow, stronger latent round-trip consistency, and separate validation-loss / round-trip / last-epoch checkpoints.")
    p.add_argument("--init-checkpoint", default="transfer_gru_vae_checkpoints_h32_z32_high_confidence_dataset/pretrained_gru_vae_no_flow_h32_z32.pt")#"transfer_gru_vae_checkpoints_h32_z32/pretrained_gru_vae_no_flow_h32_z32.pt")
    p.add_argument("--cu-csv", default="metalpdb_CU_chain_mapped_len10_high_confidence_blackbox_scored_ranked.csv")
    p.add_argument("--peptide-col", default="peptide_len10"); p.add_argument("--score-col", default="final_score")
    p.add_argument("--out-dir", default="transfer_gru_vae_flow_checkpoints_h32_z32_roundtrip_rt005_high_confidence_data")
    p.add_argument("--hidden-size", type=int, default=32); p.add_argument("--latent-dim", type=int, default=32); p.add_argument("--n-layers", type=int, default=2); p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--n-flows", type=int, default=2)
    p.add_argument("--epochs", type=int, default=200); p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--vae-lr", type=float, default=1e-5, help="Small LR for pretrained GRU encoder/decoder.")
    p.add_argument("--flow-lr", type=float, default=1e-4, help="Larger LR for new normalizing flow and score head.")
    p.add_argument("--weight-decay", type=float, default=1e-4); p.add_argument("--kl-beta", type=float, default=0.01); p.add_argument("--kl-warmup-epochs", type=int, default=5)
    p.add_argument("--score-loss-weight", type=float, default=0.2); p.add_argument("--grad-clip", type=float, default=5.0); p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--roundtrip-loss-weight", type=float, default=0.05, help="Weight of deterministic after-flow latent round-trip consistency loss. Default 0.05 balances the round-trip contribution more closely with the KL contribution; suggested ablation: 0.0, 0.01, 0.05, 0.1.")
    p.add_argument("--roundtrip-cosine-weight", type=float, default=0.1, help="Cosine-direction penalty inside the round-trip loss.")
    p.add_argument("--roundtrip-temperature", type=float, default=1.0, help="Softmax temperature used by straight-through argmax decoding in the differentiable round-trip path.")
    p.add_argument("--append-history", action="store_true", help="Append to an existing round-trip training history CSV instead of replacing it.")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

if __name__ == "__main__":
    args=parse_args(); os.makedirs(args.out_dir, exist_ok=True); train(args)
