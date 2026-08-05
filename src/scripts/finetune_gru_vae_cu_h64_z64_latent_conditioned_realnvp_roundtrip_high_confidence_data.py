
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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


@dataclass
class RealNVPConfig:
    n_layers: int = 4
    hidden_dim: int = 128
    max_scale: float = 1.5


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
        h = self.in_proj(x_onehot)
        out_top, h_n = self.gru(h)
        h_last = h_n[-1]
        mu = self.to_mu(h_last)
        logvar = self.to_logvar(h_last).clamp(min=-8.0, max=8.0)
        return mu, logvar, out_top


class LatentConditionedGRUDecoder(nn.Module):
    """z is concatenated to the token embedding at every recurrent step."""

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
        dec_input = torch.cat([emb, z_repeat], dim=-1)
        h0 = self.z_to_h(z).view(self.gru.num_layers, batch_size, self.gru.hidden_size)
        out_top, _ = self.gru(dec_input, h0)
        logits = self.to_logits(out_top)
        return logits, out_top

    def step(self, z: torch.Tensor, current_token: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.token_embed(current_token)
        z_step = z.unsqueeze(1)
        dec_input = torch.cat([emb, z_step], dim=-1)
        out_step, h = self.gru(dec_input, h)
        logits = self.to_logits(out_step[:, -1, :])
        return logits, h

    def initial_hidden(self, z: torch.Tensor) -> torch.Tensor:
        return self.z_to_h(z).view(self.gru.num_layers, z.size(0), self.gru.hidden_size)


class AffineCouplingBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, mask: torch.Tensor, max_scale: float = 1.5):
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.max_scale = float(max_scale)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_masked = x * self.mask
        s_t = self.net(x_masked)
        s, t = s_t.chunk(2, dim=-1)
        s = torch.tanh(s) * self.max_scale
        inv_mask = 1.0 - self.mask
        y = x_masked + inv_mask * (x * torch.exp(s) + t)
        logdet = (inv_mask * s).sum(dim=-1)
        return y, logdet

    def inverse(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y_masked = y * self.mask
        s_t = self.net(y_masked)
        s, t = s_t.chunk(2, dim=-1)
        s = torch.tanh(s) * self.max_scale
        inv_mask = 1.0 - self.mask
        x = y_masked + inv_mask * ((y - t) * torch.exp(-s))
        logdet = -(inv_mask * s).sum(dim=-1)
        return x, logdet


class RealNVPFlow(nn.Module):
    def __init__(self, dim: int, n_layers: int, hidden_dim: int, max_scale: float = 1.5):
        super().__init__()
        masks = []
        for layer in range(int(n_layers)):
            mask = torch.zeros(dim)
            mask[layer % 2 :: 2] = 1.0
            masks.append(mask)
        self.blocks = nn.ModuleList(
            [AffineCouplingBlock(dim, hidden_dim, mask, max_scale=max_scale) for mask in masks]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = x
        logdet_total = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
        for block in self.blocks:
            h, logdet = block(h)
            logdet_total = logdet_total + logdet
        return h, logdet_total

    def inverse(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = y
        logdet_total = torch.zeros(y.size(0), dtype=y.dtype, device=y.device)
        for block in reversed(self.blocks):
            h, logdet = block.inverse(h)
            logdet_total = logdet_total + logdet
        return h, logdet_total


class FlowFineTuneModel(nn.Module):
    def __init__(self, cfg: ModelConfig, flow_cfg: RealNVPConfig):
        super().__init__()
        self.cfg = cfg
        self.flow_cfg = flow_cfg
        self.enc = GRUEncoder(cfg.hidden_size, cfg.latent_dim, cfg.n_layers, cfg.dropout)
        self.dec = LatentConditionedGRUDecoder(cfg.hidden_size, cfg.latent_dim, cfg.n_layers, cfg.dropout)
        self.flow = RealNVPFlow(cfg.latent_dim, flow_cfg.n_layers, flow_cfg.hidden_dim, flow_cfg.max_scale)
        self.score_head = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_size),
            nn.SiLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.SiLU(),
            nn.Linear(cfg.hidden_size, 1),
        )

    def reparam(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x_onehot: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar, enc_out = self.enc(x_onehot)
        z0 = self.reparam(mu, logvar)
        zK, sum_logdet = self.flow(z0)
        logits, dec_out = self.dec(zK, x_onehot)
        score_pred = self.score_head(zK).squeeze(-1)
        return {
            "mu": mu,
            "logvar": logvar,
            "z0": z0,
            "zK": zK,
            "sum_logdet": sum_logdet,
            "logits": logits,
            "score_pred": score_pred,
            "enc_out": enc_out,
            "dec_out": dec_out,
        }


def kl_beta_for_epoch(epoch: int, kl_beta: float, warmup_epochs: int) -> float:
    return min(float(kl_beta), float(kl_beta) * float(epoch) / max(1, int(warmup_epochs)))


def vae_loss_with_flow(x_onehot: torch.Tensor, out: Dict[str, torch.Tensor], beta: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    target = x_onehot.argmax(dim=-1)
    recon = F.cross_entropy(out["logits"].reshape(-1, VOCAB), target.reshape(-1), reduction="mean")
    mu, logvar, z0, zK, sum_logdet = out["mu"], out["logvar"], out["z0"], out["zK"], out["sum_logdet"]

    log2pi = math.log(2.0 * math.pi)
    var = torch.exp(logvar)
    log_q0 = -0.5 * ((((z0 - mu) ** 2) / var) + logvar + log2pi).sum(dim=-1)
    log_p_zK = -0.5 * (zK.pow(2) + log2pi).sum(dim=-1)
    kl = (log_q0 - sum_logdet - log_p_zK).mean().clamp(min=0.0)

    loss = recon + float(beta) * kl
    pred = out["logits"].argmax(dim=-1)
    token_acc = (pred == target).float().mean()
    return loss, {
        "loss": float(loss.detach().cpu()),
        "recon": float(recon.detach().cpu()),
        "kl": float(kl.detach().cpu()),
        "token_acc": float(token_acc.detach().cpu()),
    }


def straight_through_argmax_onehot(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    tau = max(float(temperature), 1e-6)
    probs = torch.softmax(logits / tau, dim=-1)
    hard_idx = probs.argmax(dim=-1)
    hard = F.one_hot(hard_idx, num_classes=VOCAB).to(dtype=probs.dtype)
    return hard + probs - probs.detach()


def autoregressive_free_decode_st(model: FlowFineTuneModel, z: torch.Tensor, temperature: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable free-running decode using straight-through argmax feedback."""
    batch_size = z.size(0)
    h = model.dec.initial_hidden(z)
    current_token = torch.zeros(batch_size, 1, VOCAB, dtype=z.dtype, device=z.device)
    logits_steps = []
    hard_steps = []
    for _ in range(SEQ_LEN):
        logits, h = model.dec.step(z, current_token, h)
        x_st = straight_through_argmax_onehot(logits, temperature=temperature)
        logits_steps.append(logits)
        hard_steps.append(x_st)
        current_token = x_st.unsqueeze(1)
    return torch.stack(logits_steps, dim=1), torch.stack(hard_steps, dim=1)


def autoregressive_free_decode_argmax(model: FlowFineTuneModel, z: torch.Tensor, temperature: float = 1.0) -> List[str]:
    model.eval()
    batch_size = z.size(0)
    h = model.dec.initial_hidden(z)
    current_token = torch.zeros(batch_size, 1, VOCAB, dtype=z.dtype, device=z.device)
    out_tokens = [[] for _ in range(batch_size)]
    for _ in range(SEQ_LEN):
        logits, h = model.dec.step(z, current_token, h)
        idx = torch.softmax(logits / max(float(temperature), 1e-6), dim=-1).argmax(dim=-1)
        current_token = F.one_hot(idx, num_classes=VOCAB).to(dtype=z.dtype).unsqueeze(1)
        for b in range(batch_size):
            out_tokens[b].append(I_TO_AA[int(idx[b].item())])
    return ["".join(row) for row in out_tokens]


def latent_roundtrip_consistency(
    model: FlowFineTuneModel,
    x_onehot: torch.Tensor,
    out: Dict[str, torch.Tensor],
    temperature: float,
    cosine_weight: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """BO-aligned cycle: x -> mu -> RealNVP(mu)=zK -> free AR decode -> enc -> RealNVP(mu_rt)."""
    z_ref, _ = model.flow(out["mu"])
    _, x_rt_st = autoregressive_free_decode_st(model, z_ref, temperature=temperature)
    mu_rt, _, _ = model.enc(x_rt_st)
    z_rt, _ = model.flow(mu_rt)

    per_sample_l2 = torch.linalg.norm(z_ref - z_rt, dim=-1)
    per_sample_cos = F.cosine_similarity(z_ref, z_rt, dim=-1, eps=1e-8)
    l2_mean = per_sample_l2.mean()
    cos_mean = per_sample_cos.mean()
    rt_loss = l2_mean + float(cosine_weight) * (1.0 - per_sample_cos).mean()
    return rt_loss, {"roundtrip_loss": rt_loss, "roundtrip_l2": l2_mean, "roundtrip_cosine": cos_mean}


def load_cu_dataset(csv_path: str, peptide_col: str, score_col: str) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
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
    return peptides, onehot_encode_peptides(peptides), torch.tensor(scores, dtype=torch.float32)


def load_pretrained_vae_weights(model: FlowFineTuneModel, checkpoint_path: str, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    current = model.state_dict()
    compatible = {}
    skipped = []
    for k, v in state.items():
        if k in current and current[k].shape == v.shape:
            compatible[k] = v
        else:
            skipped.append(k)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    score_missing = [m for m in missing if m.startswith("score_head.") or m.startswith("flow.")]
    non_new_missing = [m for m in missing if not (m.startswith("score_head.") or m.startswith("flow."))]
    if non_new_missing:
        raise RuntimeError(f"Pretrained checkpoint did not initialize required encoder/decoder tensors: {non_new_missing}")
    print(f"Loaded {len(compatible)} compatible tensors from {checkpoint_path}")
    if skipped:
        print(f"Skipped incompatible tensors: {skipped[:12]}{' ...' if len(skipped) > 12 else ''}")
    if score_missing:
        print(f"New fine-tuning tensors initialized randomly: {score_missing[:12]}{' ...' if len(score_missing) > 12 else ''}")
    if unexpected:
        print(f"Unexpected tensors ignored: {unexpected}")


def build_optimizer(model: FlowFineTuneModel, args: argparse.Namespace) -> torch.optim.Optimizer:
    vae_params = list(model.enc.parameters()) + list(model.dec.parameters())
    flow_params = list(model.flow.parameters())
    score_params = list(model.score_head.parameters())
    return torch.optim.AdamW(
        [
            {"params": vae_params, "lr": args.vae_lr, "name": "pretrained_latent_conditioned_gru_vae"},
            {"params": flow_params, "lr": args.flow_lr, "name": "realnvp_flow"},
            {"params": score_params, "lr": args.score_head_lr, "name": "score_head"},
        ],
        weight_decay=args.weight_decay,
    )


def append_history_row(path: str, row: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: ModelConfig,
    flow_cfg: RealNVPConfig,
    args: argparse.Namespace,
    epoch: int,
    metrics: Dict[str, float],
    extra: Optional[Dict] = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "checkpoint_type": "cu_gru_vae_latent_conditioned_realnvp_roundtrip",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": asdict(cfg),
        "flow_config": asdict(flow_cfg),
        "args": vars(args),
        "epoch": int(epoch),
        "metrics": metrics,
        "aa": AA,
        "seq_len": SEQ_LEN,
        "decoder_latent_conditioning": cfg.decoder_conditioning,
        "flow_type": "RealNVP",
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    with open(path + ".json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in payload.items() if k not in {"model_state_dict", "optimizer_state_dict"}}, f, indent=2)


@torch.no_grad()
def evaluate(model, loader, device, beta, score_loss_weight, roundtrip_loss_weight, roundtrip_cosine_weight, roundtrip_temperature):
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
        "free_decode_unique_fraction": 0.0,
        "free_decode_dominant_fraction": 0.0,
    }
    n_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_vae, metrics = vae_loss_with_flow(x, out, beta)
        score_mse = F.mse_loss(out["score_pred"], y)
        rt_loss, rt_metrics = latent_roundtrip_consistency(
            model, x, out, temperature=roundtrip_temperature, cosine_weight=roundtrip_cosine_weight
        )
        loss = loss_vae + score_loss_weight * score_mse + roundtrip_loss_weight * rt_loss

        z_ref, _ = model.flow(out["mu"])
        decoded = autoregressive_free_decode_argmax(model, z_ref, temperature=1.0)
        counts = pd.Series(decoded).value_counts()
        unique_frac = float(len(counts) / max(1, len(decoded)))
        dominant_frac = float(counts.iloc[0] / max(1, len(decoded))) if len(counts) else 0.0

        n_batches += 1
        sums["loss"] += float(loss.cpu())
        sums["recon"] += metrics["recon"]
        sums["kl"] += metrics["kl"]
        sums["score_mse"] += float(score_mse.cpu())
        sums["token_acc"] += metrics["token_acc"]
        sums["roundtrip_loss"] += float(rt_metrics["roundtrip_loss"].cpu())
        sums["roundtrip_l2"] += float(rt_metrics["roundtrip_l2"].cpu())
        sums["roundtrip_cosine"] += float(rt_metrics["roundtrip_cosine"].cpu())
        sums["free_decode_unique_fraction"] += unique_frac
        sums["free_decode_dominant_fraction"] += dominant_frac
    return {k: v / max(1, n_batches) for k, v in sums.items()}


def train(args: argparse.Namespace) -> str:
    set_seed(args.seed)
    device = torch.device(args.device)
    cfg = ModelConfig(args.hidden_size, args.latent_dim, args.n_layers, args.dropout)
    flow_cfg = RealNVPConfig(args.flow_layers, args.flow_hidden_dim, args.flow_max_scale)

    peptides, x_all, y_all = load_cu_dataset(args.cu_csv, args.peptide_col, args.score_col)
    n = x_all.size(0)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
    n_val = max(1, int(round(args.val_frac * n)))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_loader = DataLoader(TensorDataset(x_all[train_idx], y_all[train_idx]), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_all[val_idx], y_all[val_idx]), batch_size=args.batch_size, shuffle=False)

    model = FlowFineTuneModel(cfg, flow_cfg).to(device)
    load_pretrained_vae_weights(model, args.init_checkpoint, device)
    optimizer = build_optimizer(model, args)

    best_total_path = os.path.join(args.out_dir, f"best_total_loss_h{cfg.hidden_size}_z{cfg.latent_dim}_latent_conditioned_realnvp_roundtrip.pt")
    best_roundtrip_path = os.path.join(args.out_dir, f"best_roundtrip_h{cfg.hidden_size}_z{cfg.latent_dim}_latent_conditioned_realnvp_roundtrip.pt")
    best_score_path = os.path.join(args.out_dir, f"best_score_mse_h{cfg.hidden_size}_z{cfg.latent_dim}_latent_conditioned_realnvp_roundtrip.pt")
    last_epoch_path = os.path.join(args.out_dir, f"last_epoch_h{cfg.hidden_size}_z{cfg.latent_dim}_latent_conditioned_realnvp_roundtrip.pt")
    history_path = os.path.join(args.out_dir, "training_history_roundtrip.csv")
    checkpoint_summary_path = os.path.join(args.out_dir, "checkpoint_summary.csv")

    if os.path.exists(history_path) and not args.append_history:
        os.remove(history_path)
    if os.path.exists(checkpoint_summary_path) and not args.append_history:
        os.remove(checkpoint_summary_path)

    best_val = float("inf")
    best_roundtrip_l2 = float("inf")
    best_score_mse = float("inf")
    best_val_epoch = best_roundtrip_epoch = best_score_epoch = 0

    print(f"Fine-tuning H{cfg.hidden_size}/Z{cfg.latent_dim} latent-conditioned GRU-VAE + RealNVP on {len(train_idx)} Cu rows; validating on {len(val_idx)} rows")
    print(f"RealNVP: layers={flow_cfg.n_layers}, hidden_dim={flow_cfg.hidden_dim}, max_scale={flow_cfg.max_scale}")
    print(f"Learning rates: GRU-VAE={args.vae_lr}, RealNVP={args.flow_lr}, score_head={args.score_head_lr}")

    for epoch in range(1, args.epochs + 1):
        beta = kl_beta_for_epoch(epoch, args.kl_beta, args.kl_warmup_epochs)
        model.train()
        sums = {k: 0.0 for k in ["loss", "recon", "kl", "score_mse", "token_acc", "roundtrip_loss", "roundtrip_l2", "roundtrip_cosine"]}
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)

            out = model(x)
            loss_vae, metrics = vae_loss_with_flow(x, out, beta)
            score_mse = F.mse_loss(out["score_pred"], y)
            rt_loss, rt_metrics = latent_roundtrip_consistency(
                model, x, out, temperature=args.roundtrip_temperature, cosine_weight=args.roundtrip_cosine_weight
            )
            loss = loss_vae + args.score_loss_weight * score_mse + args.roundtrip_loss_weight * rt_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            n_batches += 1
            sums["loss"] += float(loss.detach().cpu())
            sums["recon"] += metrics["recon"]
            sums["kl"] += metrics["kl"]
            sums["score_mse"] += float(score_mse.detach().cpu())
            sums["token_acc"] += metrics["token_acc"]
            sums["roundtrip_loss"] += float(rt_metrics["roundtrip_loss"].detach().cpu())
            sums["roundtrip_l2"] += float(rt_metrics["roundtrip_l2"].detach().cpu())
            sums["roundtrip_cosine"] += float(rt_metrics["roundtrip_cosine"].detach().cpu())

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
            "n_layers": args.n_layers,
            "flow_type": "RealNVP",
            "flow_layers": args.flow_layers,
            "flow_hidden_dim": args.flow_hidden_dim,
            "vae_lr": args.vae_lr,
            "flow_lr": args.flow_lr,
            "score_head_lr": args.score_head_lr,
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
            "roundtrip_loss_weight": args.roundtrip_loss_weight,
            "roundtrip_definition": "x->mu->RealNVP(mu)->autoregressive_free_decode_ST->enc->RealNVP(mu_rt)",
            "score_col": args.score_col,
            "cu_csv": args.cu_csv,
        }

        if val_metrics["loss"] <= best_val:
            best_val = val_metrics["loss"]
            best_val_epoch = epoch
            save_checkpoint(best_total_path, model, optimizer, cfg, flow_cfg, args, epoch, val_metrics, {**common_extra, "checkpoint_selection": "minimum_validation_total_loss"})

        if val_metrics["roundtrip_l2"] <= best_roundtrip_l2:
            best_roundtrip_l2 = val_metrics["roundtrip_l2"]
            best_roundtrip_epoch = epoch
            save_checkpoint(best_roundtrip_path, model, optimizer, cfg, flow_cfg, args, epoch, val_metrics, {**common_extra, "checkpoint_selection": "minimum_validation_roundtrip_l2"})

        if val_metrics["score_mse"] <= best_score_mse:
            best_score_mse = val_metrics["score_mse"]
            best_score_epoch = epoch
            save_checkpoint(best_score_path, model, optimizer, cfg, flow_cfg, args, epoch, val_metrics, {**common_extra, "checkpoint_selection": "minimum_validation_score_mse"})

        save_checkpoint(last_epoch_path, model, optimizer, cfg, flow_cfg, args, epoch, val_metrics, {**common_extra, "checkpoint_selection": "last_completed_epoch"})

    summary_rows = []
    for selection, path in [
        ("best_total_loss", best_total_path),
        ("best_roundtrip_l2", best_roundtrip_path),
        ("best_score_mse", best_score_path),
        ("last_epoch", last_epoch_path),
    ]:
        ckpt = torch.load(path, map_location="cpu")
        m = ckpt.get("metrics", {})
        summary_rows.append({
            "selection": selection,
            "checkpoint_path": path,
            "epoch": int(ckpt.get("epoch", -1)),
            "val_loss": m.get("loss", float("nan")),
            "val_recon": m.get("recon", float("nan")),
            "val_kl": m.get("kl", float("nan")),
            "val_score_mse": m.get("score_mse", float("nan")),
            "val_token_acc": m.get("token_acc", float("nan")),
            "val_roundtrip_l2": m.get("roundtrip_l2", float("nan")),
            "val_roundtrip_cosine": m.get("roundtrip_cosine", float("nan")),
            "val_free_decode_unique_fraction": m.get("free_decode_unique_fraction", float("nan")),
            "val_free_decode_dominant_fraction": m.get("free_decode_dominant_fraction", float("nan")),
        })
    pd.DataFrame(summary_rows).to_csv(checkpoint_summary_path, index=False)

    print(f"Saved best-total-loss checkpoint: {best_total_path} epoch={best_val_epoch}")
    print(f"Saved best-roundtrip checkpoint: {best_roundtrip_path} epoch={best_roundtrip_epoch}")
    print(f"Saved best-score-MSE checkpoint: {best_score_path} epoch={best_score_epoch}")
    print(f"Saved checkpoint comparison: {checkpoint_summary_path}")
    print(f"Saved training history: {history_path}")
    return best_roundtrip_path


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune larger latent-conditioned GRU-VAE with RealNVP on Cu, using score head and BO-aligned round-trip.")
    p.add_argument("--init-checkpoint", default="transfer_gru_vae_checkpoints_h64_z64_latent_conditioned_high_confidence_dataset/pretrained_gru_vae_latent_conditioned_h64_z64.pt")
    p.add_argument("--cu-csv", default="metalpdb_CU_chain_mapped_len10_high_confidence_blackbox_scored_ranked.csv")
    p.add_argument("--peptide-col", default="peptide_len10")
    p.add_argument("--score-col", default="final_score")
    p.add_argument("--out-dir", default="transfer_gru_vae_realnvp_checkpoints_h64_z64_latent_conditioned_high_confidence_data")
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--flow-layers", type=int, default=4)
    p.add_argument("--flow-hidden-dim", type=int, default=128)
    p.add_argument("--flow-max-scale", type=float, default=1.5)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--vae-lr", type=float, default=1e-5)
    p.add_argument("--flow-lr", type=float, default=1e-4)
    p.add_argument("--score-head-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--kl-beta", type=float, default=0.01)
    p.add_argument("--kl-warmup-epochs", type=int, default=5)
    p.add_argument("--score-loss-weight", type=float, default=0.2)
    p.add_argument("--roundtrip-loss-weight", type=float, default=0.05)
    p.add_argument("--roundtrip-cosine-weight", type=float, default=0.1)
    p.add_argument("--roundtrip-temperature", type=float, default=1.0)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--append-history", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train(args)
