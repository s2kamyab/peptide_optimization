from __future__ import annotations

"""
Fine-tune a pretrained latent-conditioned GRU-VAE with a latent diffusion module
instead of a RealNVP normalizing flow.

This script combines:
  1) The H64/Z64 latent-conditioned GRU-VAE architecture used in
     finetune_gru_vae_cu_h64_z64_latent_conditioned_realnvp_roundtrip_high_confidence_data.py
  2) The latent diffusion / DDIM inversion machinery used in
     finetune_cu_peptide_latent_diffusion_search_cost_sampling_h32_z32.py

Result:
  GRU-VAE encoder/decoder + latent diffusion over standardized encoder-mu space.

Default behavior:
  - Load pretrained H64/Z64 GRU-VAE encoder/decoder.
  - Freeze the VAE by default.
  - Compute latent_mean and latent_std from encoder mu on the Cu dataset.
  - Train a latent diffusion denoiser on standardized mu.
  - Train an auxiliary score head on standardized mu.
  - Export DDIM-inverted epsilon coordinates for later BO.

BO path after training:
  peptide -> encoder_mu -> standardize -> DDIM inversion -> epsilon
  epsilon -> DDIM sample -> unstandardize -> decoder latent -> peptide
"""

import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path


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
class DiffusionConfig:
    hidden_dim: int = 128
    time_dim: int = 32
    n_blocks: int = 4
    train_steps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 2e-2


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


def levenshtein_edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + int(ca != cb),
                )
            )
        previous = current
    return int(previous[-1])


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

    def initial_hidden(self, z: torch.Tensor) -> torch.Tensor:
        return self.z_to_h(z).view(self.gru.num_layers, z.size(0), self.gru.hidden_size)

    def step(self, z: torch.Tensor, current_token: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.token_embed(current_token)
        z_step = z.unsqueeze(1)
        dec_input = torch.cat([emb, z_step], dim=-1)
        out_step, h = self.gru(dec_input, h)
        logits = self.to_logits(out_step[:, -1, :])
        return logits, h


class GRUVAE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.enc = GRUEncoder(cfg.hidden_size, cfg.latent_dim, cfg.n_layers, cfg.dropout)
        self.dec = LatentConditionedGRUDecoder(cfg.hidden_size, cfg.latent_dim, cfg.n_layers, cfg.dropout)

    def reparam(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, x_onehot: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar, enc_out = self.enc(x_onehot)
        z = self.reparam(mu, logvar)
        logits, dec_out = self.dec(z, x_onehot)
        return {"mu": mu, "logvar": logvar, "z": z, "logits": logits, "enc_out": enc_out, "dec_out": dec_out}


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time_dim must be even")
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        exponent = -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
        frequencies = torch.exp(exponent)
        angles = t.float().unsqueeze(-1) * frequencies.unsqueeze(0)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class ResidualDenoiserBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.silu(self.fc1(h))
        h = self.fc2(h)
        return x + h


class LatentDenoiser(nn.Module):
    def __init__(self, latent_dim: int, cfg: DiffusionConfig):
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(cfg.time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.time_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.in_proj = nn.Linear(latent_dim, cfg.hidden_dim)
        self.blocks = nn.ModuleList([ResidualDenoiserBlock(cfg.hidden_dim) for _ in range(cfg.n_blocks)])
        self.out_norm = nn.LayerNorm(cfg.hidden_dim)
        self.out_proj = nn.Linear(cfg.hidden_dim, latent_dim)

    def forward(self, h_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(h_t)
        h = h + self.time_mlp(self.time_embedding(t))
        for block in self.blocks:
            h = block(h)
        h = F.silu(self.out_norm(h))
        return self.out_proj(h)


class LatentDiffusion(nn.Module):
    def __init__(self, latent_dim: int, cfg: DiffusionConfig):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.cfg = cfg
        self.denoiser = LatentDenoiser(latent_dim, cfg)
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.train_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def q_sample(self, h0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha_bar_t = self.alpha_bars[t].unsqueeze(-1)
        return torch.sqrt(alpha_bar_t) * h0 + torch.sqrt(1.0 - alpha_bar_t) * noise

    def training_loss(self, h0: torch.Tensor) -> torch.Tensor:
        batch_size = h0.size(0)
        t = torch.randint(low=0, high=self.cfg.train_steps, size=(batch_size,), device=h0.device)
        noise = torch.randn_like(h0)
        h_t = self.q_sample(h0, t, noise)
        pred_noise = self.denoiser(h_t, t)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def ddim_sample(self, base_noise: torch.Tensor, inference_steps: int = 20) -> torch.Tensor:
        self.eval()
        n_train = self.cfg.train_steps
        inference_steps = min(max(int(inference_steps), 1), n_train)
        timesteps = torch.linspace(n_train - 1, 0, inference_steps, device=base_noise.device).round().long()
        timesteps = torch.unique_consecutive(timesteps)
        h = base_noise
        for step_index, t_scalar in enumerate(timesteps):
            t = torch.full((h.size(0),), int(t_scalar.item()), device=h.device, dtype=torch.long)
            pred_noise = self.denoiser(h, t)
            alpha_bar_t = self.alpha_bars[t_scalar]
            pred_h0 = (h - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
            if step_index == len(timesteps) - 1:
                h = pred_h0
                break
            t_prev = timesteps[step_index + 1]
            alpha_bar_prev = self.alpha_bars[t_prev]
            h = torch.sqrt(alpha_bar_prev) * pred_h0 + torch.sqrt(1.0 - alpha_bar_prev) * pred_noise
        return h

    @torch.no_grad()
    def ddim_invert(self, h0: torch.Tensor, inference_steps: int = 20) -> torch.Tensor:
        self.eval()
        n_train = self.cfg.train_steps
        inference_steps = min(max(int(inference_steps), 1), n_train)
        timesteps = torch.linspace(0, n_train - 1, inference_steps, device=h0.device).round().long()
        timesteps = torch.unique_consecutive(timesteps)
        h = h0
        for step_index, t_scalar in enumerate(timesteps[:-1]):
            t = torch.full((h.size(0),), int(t_scalar.item()), device=h.device, dtype=torch.long)
            pred_noise = self.denoiser(h, t)
            alpha_bar_t = self.alpha_bars[t_scalar]
            pred_h0 = (h - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
            t_next = timesteps[step_index + 1]
            alpha_bar_next = self.alpha_bars[t_next]
            h = torch.sqrt(alpha_bar_next) * pred_h0 + torch.sqrt(1.0 - alpha_bar_next) * pred_noise
        return h


class ScoreHead(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)


def load_cu_dataset(csv_path: str, peptide_col: str, score_col: str) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    df = pd.read_csv(csv_path)
    if peptide_col not in df.columns:
        # Backward-compatible fallback for older files.
        for alt in ["sequence", "peptide", "peptide_len10"]:
            if alt in df.columns:
                peptide_col = alt
                break
    if score_col not in df.columns:
        raise ValueError(f"Score column {score_col!r} not found in {csv_path}. Use a blackbox-scored CSV with final_score.")
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


def load_pretrained_vae_weights(vae: GRUVAE, checkpoint_path: str, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("vae_state_dict", ckpt))
    current = vae.state_dict()
    compatible = {}
    skipped = []
    for k, v in state.items():
        # Support both direct enc./dec. keys and model wrappers containing enc./dec.
        if k in current and current[k].shape == v.shape:
            compatible[k] = v
        elif k.startswith("enc.") or k.startswith("dec."):
            skipped.append(k)
        else:
            skipped.append(k)
    missing, unexpected = vae.load_state_dict(compatible, strict=False)
    non_new_missing = [m for m in missing if m.startswith("enc.") or m.startswith("dec.")]
    if non_new_missing:
        raise RuntimeError(
            "Pretrained checkpoint did not initialize required GRU-VAE tensors. "
            f"Missing examples: {non_new_missing[:12]}"
        )
    print(f"Loaded {len(compatible)} compatible GRU-VAE tensors from {checkpoint_path}")
    if skipped:
        print(f"Skipped non-GRU-VAE/incompatible tensors: {skipped[:12]}{' ...' if len(skipped) > 12 else ''}")
    if unexpected:
        print(f"Unexpected tensors ignored: {unexpected}")


def maybe_load_diffusion_weights(diffusion: LatentDiffusion, checkpoint_path: Optional[str], device: torch.device) -> None:
    if not checkpoint_path:
        return
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("diffusion_state_dict", ckpt.get("model_state_dict", ckpt))
    current = diffusion.state_dict()
    compatible = {k: v for k, v in state.items() if k in current and current[k].shape == v.shape}
    if not compatible:
        print(f"No compatible diffusion tensors found in {checkpoint_path}; diffusion initialized randomly.")
        return
    diffusion.load_state_dict(compatible, strict=False)
    print(f"Loaded {len(compatible)} compatible diffusion tensors from {checkpoint_path}")


@torch.no_grad()
def compute_latent_stats(vae: GRUVAE, x_all: torch.Tensor, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    vae.eval()
    mus = []
    for start in range(0, x_all.size(0), batch_size):
        x = x_all[start : start + batch_size].to(device)
        mu, _, _ = vae.enc(x)
        mus.append(mu.detach())
    mu_all = torch.cat(mus, dim=0)
    latent_mean = mu_all.mean(dim=0)
    latent_std = mu_all.std(dim=0).clamp(min=1e-6)
    return latent_mean, latent_std


def standardize_mu(mu: torch.Tensor, latent_mean: torch.Tensor, latent_std: torch.Tensor) -> torch.Tensor:
    return (mu - latent_mean.to(mu.device)) / latent_std.to(mu.device).clamp(min=1e-6)


def unstandardize_latent(h: torch.Tensor, latent_mean: torch.Tensor, latent_std: torch.Tensor) -> torch.Tensor:
    return h * latent_std.to(h.device).clamp(min=1e-6) + latent_mean.to(h.device)


def pearson_corr_safe(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    if a.numel() < 2 or float(a.std()) < 1e-12 or float(b.std()) < 1e-12:
        return float("nan")
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1].item())


def make_loader(x: torch.Tensor, y: torch.Tensor, indices: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(x[indices], y[indices], indices), batch_size=batch_size, shuffle=shuffle)


def append_history_row(path: str, row: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def json_safe(obj):
    """Recursively convert tensors, numpy scalars, paths, and argparse-like values to JSON-safe objects."""
    if torch.is_tensor(obj):
        arr = obj.detach().cpu()
        if arr.numel() == 1:
            return float(arr.item())
        return arr.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if hasattr(obj, "__fspath__"):
        return os.fspath(obj)
    return obj


def vae_reconstruction_loss(x_onehot: torch.Tensor, logits: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    target = x_onehot.argmax(dim=-1)
    recon = F.cross_entropy(logits.reshape(-1, VOCAB), target.reshape(-1), reduction="mean")
    pred = logits.argmax(dim=-1)
    token_acc = (pred == target).float().mean()
    return recon, {"recon": float(recon.detach().cpu()), "token_acc": float(token_acc.detach().cpu())}


def decode_scheduled_sampling(vae: GRUVAE, z: torch.Tensor, x_true: torch.Tensor, teacher_forcing_ratio: float) -> torch.Tensor:
    batch_size = z.size(0)
    h = vae.dec.initial_hidden(z)
    current_token = torch.zeros(batch_size, 1, VOCAB, dtype=z.dtype, device=z.device)
    logits_steps: List[torch.Tensor] = []
    for t in range(SEQ_LEN):
        logits, h = vae.dec.step(z, current_token, h)
        logits_steps.append(logits.unsqueeze(1))
        if t == SEQ_LEN - 1:
            continue
        predicted_idx = logits.argmax(dim=-1)
        predicted_token = F.one_hot(predicted_idx, num_classes=VOCAB).to(dtype=z.dtype)
        true_token = x_true[:, t, :].to(dtype=z.dtype)
        use_teacher = torch.rand(batch_size, 1, device=z.device) < float(teacher_forcing_ratio)
        next_token = torch.where(use_teacher, true_token, predicted_token)
        current_token = next_token.unsqueeze(1)
    return torch.cat(logits_steps, dim=1)


@torch.no_grad()
def autoregressive_decode(vae: GRUVAE, z: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, List[str]]:
    vae.eval()
    batch_size = z.size(0)
    h = vae.dec.initial_hidden(z)
    current_token = torch.zeros(batch_size, 1, VOCAB, dtype=z.dtype, device=z.device)
    logits_steps: List[torch.Tensor] = []
    token_steps: List[torch.Tensor] = []
    tau = max(float(temperature), 1e-6)
    for _ in range(SEQ_LEN):
        logits, h = vae.dec.step(z, current_token, h)
        logits = logits / tau
        token_idx = logits.argmax(dim=-1)
        logits_steps.append(logits.unsqueeze(1))
        token_steps.append(token_idx.unsqueeze(1))
        current_token = F.one_hot(token_idx, num_classes=VOCAB).to(dtype=z.dtype).unsqueeze(1)
    logits_all = torch.cat(logits_steps, dim=1)
    token_idx_all = torch.cat(token_steps, dim=1)
    peptides = ["".join(I_TO_AA[int(i)] for i in row) for row in token_idx_all.detach().cpu().tolist()]
    return logits_all, peptides


@torch.no_grad()
def diffusion_inversion_metrics(
    vae: GRUVAE,
    diffusion: LatentDiffusion,
    x_onehot: torch.Tensor,
    source_peptides: List[str],
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    ddim_steps: int,
    temperature: float,
    project_epsilon_to_sphere: bool,
) -> Dict[str, float]:
    vae.eval()
    diffusion.eval()
    mu, _, _ = vae.enc(x_onehot)
    h0 = standardize_mu(mu, latent_mean, latent_std)
    epsilon = diffusion.ddim_invert(h0, inference_steps=ddim_steps)
    if project_epsilon_to_sphere:
        radius = math.sqrt(vae.cfg.latent_dim)
        epsilon = epsilon / epsilon.norm(dim=-1, keepdim=True).clamp(min=1e-8) * radius
    h_back = diffusion.ddim_sample(epsilon, inference_steps=ddim_steps)
    mu_back = unstandardize_latent(h_back, latent_mean, latent_std)
    _, decoded = autoregressive_decode(vae, mu_back, temperature=temperature)
    edits = [levenshtein_edit_distance(src, dec) for src, dec in zip(source_peptides, decoded)]
    h_l2 = torch.linalg.norm(h0 - h_back, dim=-1)
    h_cos = F.cosine_similarity(h0, h_back, dim=-1, eps=1e-8)
    eps_norm = torch.linalg.norm(epsilon, dim=-1)
    counts: Dict[str, int] = {}
    for pep in decoded:
        counts[pep] = counts.get(pep, 0) + 1
    return {
        "ddim_inversion_h_l2_mean": float(h_l2.mean().cpu()),
        "ddim_inversion_h_l2_median": float(h_l2.median().cpu()),
        "ddim_inversion_h_cosine_mean": float(h_cos.mean().cpu()),
        "ddim_inversion_epsilon_norm_mean": float(eps_norm.mean().cpu()),
        "ddim_inversion_epsilon_norm_std": float(eps_norm.std().cpu()),
        "ddim_decode_unique_fraction": float(len(set(decoded)) / max(len(decoded), 1)),
        "ddim_decode_dominant_fraction": float(max(counts.values()) / max(len(decoded), 1) if counts else float("nan")),
        "ddim_source_to_decode_edit_mean": float(np.mean(edits)),
        "ddim_source_to_decode_edit_median": float(np.median(edits)),
    }


@torch.no_grad()
def local_epsilon_smoothness_metrics(
    vae: GRUVAE,
    diffusion: LatentDiffusion,
    x_onehot: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    ddim_steps: int,
    temperature: float,
    n_subset: int,
    n_neighbors: int,
    noise_std: float,
    seed: int,
    renormalize_to_sphere: bool,
) -> Dict[str, float]:
    vae.eval()
    diffusion.eval()
    n = x_onehot.size(0)
    n_subset = min(int(n_subset), n)
    if n_subset <= 0 or n_neighbors <= 0:
        return {"local_epsilon_edit_mean": float("nan"), "local_epsilon_identical_fraction": float("nan")}
    rng = np.random.default_rng(seed)
    subset_idx = rng.choice(n, size=n_subset, replace=False).tolist()
    x = x_onehot[subset_idx]
    mu, _, _ = vae.enc(x)
    h0 = standardize_mu(mu, latent_mean, latent_std)
    eps_center = diffusion.ddim_invert(h0, inference_steps=ddim_steps)
    radius = math.sqrt(vae.cfg.latent_dim)
    if renormalize_to_sphere:
        eps_center = eps_center / eps_center.norm(dim=-1, keepdim=True).clamp(min=1e-8) * radius
    h_center = diffusion.ddim_sample(eps_center, inference_steps=ddim_steps)
    z_center = unstandardize_latent(h_center, latent_mean, latent_std)
    _, p_center = autoregressive_decode(vae, z_center, temperature=temperature)
    torch_gen = torch.Generator(device="cpu")
    torch_gen.manual_seed(seed)
    edit_values: List[int] = []
    identical_flags: List[float] = []
    all_neighbors: List[str] = []
    for i in range(n_subset):
        eps_i = eps_center[i : i + 1]
        p_i = p_center[i]
        for _ in range(int(n_neighbors)):
            noise_cpu = torch.randn(eps_i.shape, generator=torch_gen, dtype=torch.float32, device="cpu") * float(noise_std)
            eps_neighbor = eps_i + noise_cpu.to(eps_i.device, dtype=eps_i.dtype)
            if renormalize_to_sphere:
                eps_neighbor = eps_neighbor / eps_neighbor.norm(dim=-1, keepdim=True).clamp(min=1e-8) * radius
            h_neighbor = diffusion.ddim_sample(eps_neighbor, inference_steps=ddim_steps)
            z_neighbor = unstandardize_latent(h_neighbor, latent_mean, latent_std)
            _, p_neighbor_list = autoregressive_decode(vae, z_neighbor, temperature=temperature)
            p_neighbor = p_neighbor_list[0]
            edit = levenshtein_edit_distance(p_i, p_neighbor)
            edit_values.append(edit)
            identical_flags.append(float(edit == 0))
            all_neighbors.append(p_neighbor)
    return {
        "local_epsilon_subset_size": float(n_subset),
        "local_epsilon_neighbors_per_sequence": float(n_neighbors),
        "local_epsilon_noise_std": float(noise_std),
        "local_epsilon_edit_mean": float(np.mean(edit_values)),
        "local_epsilon_edit_median": float(np.median(edit_values)),
        "local_epsilon_identical_fraction": float(np.mean(identical_flags)),
        "local_epsilon_global_unique_neighbor_fraction": float(len(set(all_neighbors)) / max(len(all_neighbors), 1)),
    }


@torch.no_grad()
def random_shell_generation_metrics(
    vae: GRUVAE,
    diffusion: LatentDiffusion,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    n_samples: int,
    ddim_steps: int,
    temperature: float,
    device: torch.device,
) -> Dict[str, float]:
    vae.eval()
    diffusion.eval()
    n_samples = int(n_samples)
    if n_samples <= 0:
        return {"shell_unique_fraction": float("nan"), "shell_dominant_fraction": float("nan")}
    radius = math.sqrt(vae.cfg.latent_dim)
    eps = torch.randn(n_samples, vae.cfg.latent_dim, device=device)
    eps = eps / eps.norm(dim=-1, keepdim=True).clamp(min=1e-8) * radius
    h = diffusion.ddim_sample(eps, inference_steps=ddim_steps)
    z = unstandardize_latent(h, latent_mean, latent_std)
    _, peptides = autoregressive_decode(vae, z, temperature=temperature)
    counts: Dict[str, int] = {}
    for pep in peptides:
        counts[pep] = counts.get(pep, 0) + 1
    return {
        "shell_n_samples": float(n_samples),
        "shell_unique_fraction": float(len(set(peptides)) / max(n_samples, 1)),
        "shell_dominant_fraction": float(max(counts.values()) / max(n_samples, 1) if counts else float("nan")),
        "shell_radius_mean": float(eps.norm(dim=-1).mean().cpu()),
    }


@torch.no_grad()
def evaluate(
    vae: GRUVAE,
    diffusion: LatentDiffusion,
    score_head: ScoreHead,
    loader: DataLoader,
    all_peptides: List[str],
    x_all: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    vae.eval()
    diffusion.eval()
    score_head.eval()
    sums = {"loss": 0.0, "diffusion_epsilon_mse": 0.0, "score_mse": 0.0, "recon": 0.0, "token_acc": 0.0}
    preds_all: List[torch.Tensor] = []
    y_all_list: List[torch.Tensor] = []
    n_batches = 0
    val_indices: List[int] = []
    for x, y, idx in loader:
        x = x.to(device)
        y = y.to(device)
        mu, _, _ = vae.enc(x)
        h0 = standardize_mu(mu, latent_mean, latent_std)
        diffusion_loss = diffusion.training_loss(h0)
        pred_score = score_head(h0)
        score_mse = F.mse_loss(pred_score, y)
        logits, _ = vae.dec(mu, x)
        recon, rec_metrics = vae_reconstruction_loss(x, logits)
        loss = args.diffusion_loss_weight * diffusion_loss + args.score_loss_weight * score_mse + args.recon_loss_weight * recon
        sums["loss"] += float(loss.cpu())
        sums["diffusion_epsilon_mse"] += float(diffusion_loss.cpu())
        sums["score_mse"] += float(score_mse.cpu())
        sums["recon"] += rec_metrics["recon"]
        sums["token_acc"] += rec_metrics["token_acc"]
        preds_all.append(pred_score.detach().cpu())
        y_all_list.append(y.detach().cpu())
        val_indices.extend([int(i) for i in idx.tolist()])
        n_batches += 1
    metrics = {k: v / max(1, n_batches) for k, v in sums.items()}
    metrics["score_pearson"] = pearson_corr_safe(torch.cat(preds_all), torch.cat(y_all_list)) if preds_all else float("nan")
    x_val = x_all[val_indices].to(device)
    val_peptides = [all_peptides[i] for i in val_indices]
    metrics.update(
        diffusion_inversion_metrics(
            vae=vae,
            diffusion=diffusion,
            x_onehot=x_val,
            source_peptides=val_peptides,
            latent_mean=latent_mean,
            latent_std=latent_std,
            ddim_steps=args.ddim_steps,
            temperature=args.decode_temperature,
            project_epsilon_to_sphere=args.project_inverted_epsilon_to_sphere,
        )
    )
    metrics.update(
        local_epsilon_smoothness_metrics(
            vae=vae,
            diffusion=diffusion,
            x_onehot=x_val,
            latent_mean=latent_mean,
            latent_std=latent_std,
            ddim_steps=args.ddim_steps,
            temperature=args.decode_temperature,
            n_subset=args.local_smoothness_subset,
            n_neighbors=args.local_neighbors,
            noise_std=args.local_noise_std,
            seed=args.seed + 777,
            renormalize_to_sphere=args.local_renormalize_to_sphere,
        )
    )
    metrics.update(
        random_shell_generation_metrics(
            vae=vae,
            diffusion=diffusion,
            latent_mean=latent_mean,
            latent_std=latent_std,
            n_samples=args.shell_samples,
            ddim_steps=args.ddim_steps,
            temperature=args.decode_temperature,
            device=device,
        )
    )
    return metrics


def build_optimizer(vae: GRUVAE, diffusion: LatentDiffusion, score_head: ScoreHead, args: argparse.Namespace) -> torch.optim.Optimizer:
    params = [
        {"params": diffusion.parameters(), "lr": args.diffusion_lr, "name": "latent_diffusion"},
        {"params": score_head.parameters(), "lr": args.score_head_lr, "name": "score_head"},
    ]
    if args.finetune_vae:
        params.append({"params": vae.parameters(), "lr": args.vae_lr, "name": "gru_vae"})
    return torch.optim.AdamW(params, weight_decay=args.weight_decay)


def save_checkpoint(
    path: str,
    vae: GRUVAE,
    diffusion: LatentDiffusion,
    score_head: ScoreHead,
    cfg: ModelConfig,
    diff_cfg: DiffusionConfig,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    metrics: Dict[str, float],
    selection: str,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "checkpoint_type": "cu_gru_vae_h64_z64_latent_conditioned_latent_diffusion",
        "vae_state_dict": vae.state_dict(),
        "diffusion_state_dict": diffusion.state_dict(),
        "score_head_state_dict": score_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": asdict(cfg),
        "diffusion_config": asdict(diff_cfg),
        "latent_mean": latent_mean.detach().cpu(),
        "latent_std": latent_std.detach().cpu(),
        "args": vars(args),
        "epoch": int(epoch),
        "metrics": metrics,
        "checkpoint_selection": selection,
        "aa": AA,
        "seq_len": SEQ_LEN,
        "decoder_latent_conditioning": cfg.decoder_conditioning,
        "flow_type": "none_replaced_by_latent_diffusion",
        "diffusion_target": "standardized_encoder_mu",
        "diffusion_prediction": "epsilon",
        "diffusion_sampler": "deterministic_ddim_eta0",
        "bo_coordinate_space": "diffusion_base_noise_spherical_epsilon",
        "recommended_bo_radius": float(math.sqrt(cfg.latent_dim)),
        "vae_finetuned": bool(args.finetune_vae),
        "score_col": args.score_col,
        "cu_csv": args.cu_csv,
    }
    torch.save(payload, path)

    # Keep the .pt checkpoint fully tensor-based, but make the sidecar .json human-readable
    # and JSON-safe. The previous version tried to dump latent_mean/latent_std tensors
    # directly, causing: TypeError: Object of type Tensor is not JSON serializable.
    json_payload = {
        k: v
        for k, v in payload.items()
        if k not in {
            "vae_state_dict",
            "diffusion_state_dict",
            "score_head_state_dict",
            "optimizer_state_dict",
        }
    }
    with open(path + ".json", "w", encoding="utf-8") as handle:
        json.dump(json_safe(json_payload), handle, indent=2)


@torch.no_grad()
def export_inversion_coordinates(
    path: str,
    vae: GRUVAE,
    diffusion: LatentDiffusion,
    peptides: List[str],
    x_all: torch.Tensor,
    y_all: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    vae.eval()
    diffusion.eval()
    rows: List[Dict[str, object]] = []
    for start in range(0, x_all.size(0), args.batch_size):
        end = min(start + args.batch_size, x_all.size(0))
        x = x_all[start:end].to(device)
        y = y_all[start:end]
        pep_batch = peptides[start:end]
        mu, _, _ = vae.enc(x)
        h0 = standardize_mu(mu, latent_mean, latent_std)
        eps = diffusion.ddim_invert(h0, inference_steps=args.ddim_steps)
        if args.project_inverted_epsilon_to_sphere:
            radius = math.sqrt(vae.cfg.latent_dim)
            eps = eps / eps.norm(dim=-1, keepdim=True).clamp(min=1e-8) * radius
        h_back = diffusion.ddim_sample(eps, inference_steps=args.ddim_steps)
        mu_back = unstandardize_latent(h_back, latent_mean, latent_std)
        _, decoded = autoregressive_decode(vae, mu_back, temperature=args.decode_temperature)
        h_l2 = torch.linalg.norm(h0 - h_back, dim=-1)
        h_cos = F.cosine_similarity(h0, h_back, dim=-1, eps=1e-8)
        eps_norm = torch.linalg.norm(eps, dim=-1)
        mu_cpu = mu.detach().cpu().numpy()
        h0_cpu = h0.detach().cpu().numpy()
        eps_cpu = eps.detach().cpu().numpy()
        for i, pep in enumerate(pep_batch):
            row: Dict[str, object] = {
                "peptide": pep,
                "peptide_len10": pep,
                "sequence": pep,
                "score": float(y[i].detach().cpu()),
                args.score_col: float(y[i].detach().cpu()),
                "decoded_from_inverted_epsilon": decoded[i],
                "source_to_decoded_edit": levenshtein_edit_distance(pep, decoded[i]),
                "epsilon_norm": float(eps_norm[i].detach().cpu()),
                "ddim_h_l2": float(h_l2[i].detach().cpu()),
                "ddim_h_cosine": float(h_cos[i].detach().cpu()),
            }
            for j in range(eps_cpu.shape[1]):
                row[f"epsilon_{j:02d}"] = float(eps_cpu[i, j])
                row[f"mu_{j:02d}"] = float(mu_cpu[i, j])
                row[f"h0_standardized_{j:02d}"] = float(h0_cpu[i, j])
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved DDIM inversion coordinates for BO: {path}")


def train(args: argparse.Namespace) -> str:
    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    peptides, x_all, y_all = load_cu_dataset(args.cu_csv, args.peptide_col, args.score_col)
    n = x_all.size(0)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
    n_val = max(1, int(round(args.val_frac * n)))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train_loader = make_loader(x_all, y_all, train_idx, args.batch_size, shuffle=True)
    val_loader = make_loader(x_all, y_all, val_idx, args.batch_size, shuffle=False)

    cfg = ModelConfig(args.hidden_size, args.latent_dim, args.n_layers, args.dropout)
    diff_cfg = DiffusionConfig(args.diffusion_hidden_dim, args.diffusion_time_dim, args.diffusion_blocks, args.diffusion_train_steps, args.beta_start, args.beta_end)
    vae = GRUVAE(cfg).to(device)
    load_pretrained_vae_weights(vae, args.init_checkpoint, device)
    diffusion = LatentDiffusion(cfg.latent_dim, diff_cfg).to(device)
    maybe_load_diffusion_weights(diffusion, args.init_diffusion_checkpoint, device)
    score_head = ScoreHead(cfg.latent_dim, args.score_head_hidden_dim).to(device)

    if not args.finetune_vae:
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        print("VAE is frozen. Training latent diffusion + score head only.")
    else:
        print("VAE fine-tuning enabled. Latent statistics are computed before fine-tuning and kept fixed.")

    latent_mean, latent_std = compute_latent_stats(vae, x_all, args.batch_size, device)
    print(f"Latent stats computed from {n} peptides: mean_shape={tuple(latent_mean.shape)}, std_shape={tuple(latent_std.shape)}")
    print(f"Recommended epsilon shell radius: sqrt({cfg.latent_dim}) = {math.sqrt(cfg.latent_dim):.6f}")

    optimizer = build_optimizer(vae, diffusion, score_head, args)
    history_path = os.path.join(args.out_dir, "training_history_cu_gru_vae_latent_diffusion.csv")
    summary_path = os.path.join(args.out_dir, "checkpoint_summary.csv")
    if os.path.exists(history_path) and not args.append_history:
        os.remove(history_path)
    if os.path.exists(summary_path) and not args.append_history:
        os.remove(summary_path)

    best_loss_path = os.path.join(args.out_dir, f"best_val_loss_h{cfg.hidden_size}_z{cfg.latent_dim}_latent_diffusion.pt")
    best_diffusion_path = os.path.join(args.out_dir, f"best_val_diffusion_mse_h{cfg.hidden_size}_z{cfg.latent_dim}_latent_diffusion.pt")
    best_inversion_path = os.path.join(args.out_dir, f"best_ddim_inversion_l2_h{cfg.hidden_size}_z{cfg.latent_dim}_latent_diffusion.pt")
    best_score_path = os.path.join(args.out_dir, f"best_score_mse_h{cfg.hidden_size}_z{cfg.latent_dim}_latent_diffusion.pt")
    last_path = os.path.join(args.out_dir, f"last_epoch_h{cfg.hidden_size}_z{cfg.latent_dim}_latent_diffusion.pt")

    best_loss = best_diff_mse = best_inv_l2 = best_score_mse = float("inf")
    best_epochs = {"loss": 0, "diff": 0, "inv": 0, "score": 0}

    print(f"Cu rows: train={len(train_idx)} val={len(val_idx)} total={n}")
    print(f"Training target: standardized encoder mu from H{cfg.hidden_size}/Z{cfg.latent_dim} latent-conditioned GRU-VAE")
    print("Architecture: GRU-VAE + latent diffusion; RealNVP removed.")

    for epoch in range(1, args.epochs + 1):
        vae.train(args.finetune_vae)
        diffusion.train()
        score_head.train()
        sums = {"loss": 0.0, "diffusion_epsilon_mse": 0.0, "score_mse": 0.0, "recon": 0.0, "token_acc": 0.0}
        n_batches = 0
        for x, y, _idx in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            if args.finetune_vae:
                mu, _, _ = vae.enc(x)
            else:
                with torch.no_grad():
                    mu, _, _ = vae.enc(x)
            h0 = standardize_mu(mu, latent_mean, latent_std)
            h0_for_aux = h0 if args.finetune_vae else h0.detach()
            diffusion_loss = diffusion.training_loss(h0_for_aux)
            pred_score = score_head(h0_for_aux)
            score_mse = F.mse_loss(pred_score, y)
            if args.recon_loss_weight > 0.0:
                logits = decode_scheduled_sampling(vae, mu, x, args.teacher_forcing_ratio)
                recon, rec_metrics = vae_reconstruction_loss(x, logits)
            else:
                recon = torch.zeros((), device=device)
                rec_metrics = {"recon": 0.0, "token_acc": 0.0}
            loss = args.diffusion_loss_weight * diffusion_loss + args.score_loss_weight * score_mse + args.recon_loss_weight * recon
            loss.backward()
            params_for_clip = list(diffusion.parameters()) + list(score_head.parameters()) + (list(vae.parameters()) if args.finetune_vae else [])
            nn.utils.clip_grad_norm_(params_for_clip, args.grad_clip)
            optimizer.step()
            sums["loss"] += float(loss.detach().cpu())
            sums["diffusion_epsilon_mse"] += float(diffusion_loss.detach().cpu())
            sums["score_mse"] += float(score_mse.detach().cpu())
            sums["recon"] += rec_metrics["recon"]
            sums["token_acc"] += rec_metrics["token_acc"]
            n_batches += 1
        train_metrics = {k: v / max(1, n_batches) for k, v in sums.items()}
        val_metrics = evaluate(vae, diffusion, score_head, val_loader, peptides, x_all, latent_mean, latent_std, args, device)
        row: Dict[str, object] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "epoch": epoch,
            "model_type": "gru_vae_latent_diffusion",
            "vae_finetuned": bool(args.finetune_vae),
            "hidden_size": cfg.hidden_size,
            "latent_dim": cfg.latent_dim,
            "diffusion_hidden_dim": diff_cfg.hidden_dim,
            "diffusion_blocks": diff_cfg.n_blocks,
            "diffusion_train_steps": diff_cfg.train_steps,
            "diffusion_lr": args.diffusion_lr,
            "score_head_lr": args.score_head_lr,
            "vae_lr": args.vae_lr,
            "diffusion_loss_weight": args.diffusion_loss_weight,
            "score_loss_weight": args.score_loss_weight,
            "recon_loss_weight": args.recon_loss_weight,
        }
        for k, v in train_metrics.items():
            row[f"train_{k}"] = v
        for k, v in val_metrics.items():
            row[f"val_{k}"] = v
        append_history_row(history_path, row)
        print(f"epoch={epoch} train={train_metrics} val={val_metrics}")

        if val_metrics["loss"] <= best_loss:
            best_loss = val_metrics["loss"]
            best_epochs["loss"] = epoch
            save_checkpoint(best_loss_path, vae, diffusion, score_head, cfg, diff_cfg, latent_mean, latent_std, optimizer, args, epoch, val_metrics, "minimum_validation_total_loss")
        if val_metrics["diffusion_epsilon_mse"] <= best_diff_mse:
            best_diff_mse = val_metrics["diffusion_epsilon_mse"]
            best_epochs["diff"] = epoch
            save_checkpoint(best_diffusion_path, vae, diffusion, score_head, cfg, diff_cfg, latent_mean, latent_std, optimizer, args, epoch, val_metrics, "minimum_validation_diffusion_epsilon_mse")
        if val_metrics["ddim_inversion_h_l2_mean"] <= best_inv_l2:
            best_inv_l2 = val_metrics["ddim_inversion_h_l2_mean"]
            best_epochs["inv"] = epoch
            save_checkpoint(best_inversion_path, vae, diffusion, score_head, cfg, diff_cfg, latent_mean, latent_std, optimizer, args, epoch, val_metrics, "minimum_validation_ddim_inversion_h_l2")
        if val_metrics["score_mse"] <= best_score_mse:
            best_score_mse = val_metrics["score_mse"]
            best_epochs["score"] = epoch
            save_checkpoint(best_score_path, vae, diffusion, score_head, cfg, diff_cfg, latent_mean, latent_std, optimizer, args, epoch, val_metrics, "minimum_validation_score_mse")
        save_checkpoint(last_path, vae, diffusion, score_head, cfg, diff_cfg, latent_mean, latent_std, optimizer, args, epoch, val_metrics, "last_completed_epoch")

    rows = []
    for selection, path in [
        ("best_val_loss", best_loss_path),
        ("best_diffusion_mse", best_diffusion_path),
        ("best_ddim_inversion_l2", best_inversion_path),
        ("best_score_mse", best_score_path),
        ("last_epoch", last_path),
    ]:
        ckpt = torch.load(path, map_location="cpu")
        m = ckpt.get("metrics", {})
        rows.append({
            "selection": selection,
            "checkpoint_path": path,
            "epoch": int(ckpt.get("epoch", -1)),
            "val_loss": m.get("loss", float("nan")),
            "val_diffusion_epsilon_mse": m.get("diffusion_epsilon_mse", float("nan")),
            "val_score_mse": m.get("score_mse", float("nan")),
            "val_score_pearson": m.get("score_pearson", float("nan")),
            "val_ddim_inversion_h_l2_mean": m.get("ddim_inversion_h_l2_mean", float("nan")),
            "val_ddim_inversion_h_cosine_mean": m.get("ddim_inversion_h_cosine_mean", float("nan")),
            "val_ddim_decode_unique_fraction": m.get("ddim_decode_unique_fraction", float("nan")),
            "val_ddim_source_to_decode_edit_mean": m.get("ddim_source_to_decode_edit_mean", float("nan")),
            "val_local_epsilon_edit_mean": m.get("local_epsilon_edit_mean", float("nan")),
            "val_local_epsilon_identical_fraction": m.get("local_epsilon_identical_fraction", float("nan")),
            "val_shell_unique_fraction": m.get("shell_unique_fraction", float("nan")),
            "val_shell_dominant_fraction": m.get("shell_dominant_fraction", float("nan")),
        })
    pd.DataFrame(rows).to_csv(summary_path, index=False)

    selected_export_ckpt = best_inversion_path if args.export_checkpoint_selection == "best_inversion" else last_path
    if args.export_inversion_coordinates:
        if args.export_checkpoint_selection != "last":
            ckpt = torch.load(selected_export_ckpt, map_location=device)
            vae.load_state_dict(ckpt["vae_state_dict"], strict=True)
            diffusion.load_state_dict(ckpt["diffusion_state_dict"], strict=True)
            score_head.load_state_dict(ckpt["score_head_state_dict"], strict=True)
            latent_mean = ckpt["latent_mean"].to(device).float()
            latent_std = ckpt["latent_std"].to(device).float().clamp(min=1e-6)
        export_path = os.path.join(args.out_dir, "cu_ddim_inversion_coordinates_for_bo.csv")
        export_inversion_coordinates(export_path, vae, diffusion, peptides, x_all, y_all, latent_mean, latent_std, args, device)

    print(f"Saved checkpoint summary: {summary_path}")
    print(f"Best val loss: epoch={best_epochs['loss']}, value={best_loss:.6f}")
    print(f"Best diffusion MSE: epoch={best_epochs['diff']}, value={best_diff_mse:.6f}")
    print(f"Best DDIM inversion L2: epoch={best_epochs['inv']}, value={best_inv_l2:.6f}")
    print(f"Best score MSE: epoch={best_epochs['score']}, value={best_score_mse:.6f}")
    return best_inversion_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune H64/Z64 latent-conditioned GRU-VAE with latent diffusion instead of RealNVP. "
            "The diffusion model is trained on standardized encoder-mu latents and exports DDIM epsilon coordinates for BO."
        )
    )
    parser.add_argument("--init-checkpoint", default="transfer_gru_vae_checkpoints_h64_z64_latent_conditioned_high_confidence_dataset/pretrained_gru_vae_latent_conditioned_h64_z64.pt")
    parser.add_argument("--init-diffusion-checkpoint", default="", help="Optional compatible latent-diffusion checkpoint. Leave empty to train diffusion from scratch.")
    parser.add_argument("--cu-csv", default="metalpdb_CU_chain_mapped_len10_high_confidence_blackbox_scored_ranked.csv")
    parser.add_argument("--peptide-col", default="peptide_len10")
    parser.add_argument("--score-col", default="final_score")
    parser.add_argument("--out-dir", default="transfer_gru_vae_latent_diffusion_checkpoints_h64_z64_high_confidence_data")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--diffusion-hidden-dim", type=int, default=128)
    parser.add_argument("--diffusion-time-dim", type=int, default=32)
    parser.add_argument("--diffusion-blocks", type=int, default=4)
    parser.add_argument("--diffusion-train-steps", type=int, default=100)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--finetune-vae", action="store_true", help="Default freezes VAE. Enable only if you want encoder/decoder to move during Cu adaptation.")
    parser.add_argument("--vae-lr", type=float, default=1e-6)
    parser.add_argument("--diffusion-lr", type=float, default=3e-5)
    parser.add_argument("--score-head-lr", type=float, default=1e-4)
    parser.add_argument("--score-head-hidden-dim", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--diffusion-loss-weight", type=float, default=1.0)
    parser.add_argument("--score-loss-weight", type=float, default=0.1)
    parser.add_argument("--recon-loss-weight", type=float, default=0.0)
    parser.add_argument("--teacher-forcing-ratio", type=float, default=0.5)
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--decode-temperature", type=float, default=1.0)
    parser.add_argument("--project-inverted-epsilon-to-sphere", action="store_true", default=True)
    parser.add_argument("--no-project-inverted-epsilon-to-sphere", dest="project_inverted_epsilon_to_sphere", action="store_false")
    parser.add_argument("--local-smoothness-subset", type=int, default=128)
    parser.add_argument("--local-neighbors", type=int, default=3)
    parser.add_argument("--local-noise-std", type=float, default=0.05)
    parser.add_argument("--local-renormalize-to-sphere", action="store_true")
    parser.add_argument("--shell-samples", type=int, default=512)
    parser.add_argument("--append-history", action="store_true")
    parser.add_argument("--export-inversion-coordinates", action="store_true", default=True)
    parser.add_argument("--no-export-inversion-coordinates", dest="export_inversion_coordinates", action="store_false")
    parser.add_argument("--export-checkpoint-selection", choices=["last", "best_inversion"], default="best_inversion")
    return parser.parse_args()


# ============================================================================
# EPSILON -> BLACK-BOX OBJECTIVE SMOOTHNESS DIAGNOSTIC
# Corrected version:
#   compares objective changes against the DECODED EPSILON CENTER peptide,
#   not against the original source peptide.
# ============================================================================

def _safe_corr(x: np.ndarray, y: np.ndarray, method: str = "pearson") -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    if method == "spearman":
        x = pd.Series(x).rank(method="average").to_numpy(float)
        y = pd.Series(y).rank(method="average").to_numpy(float)
    return float(np.corrcoef(x, y)[0, 1])


def _project_to_radius(eps: torch.Tensor, radius: float) -> torch.Tensor:
    return eps / eps.norm(dim=-1, keepdim=True).clamp(min=1e-8) * float(radius)


@torch.no_grad()
def _decode_epsilon_batch(
    vae: GRUVAE,
    diffusion: LatentDiffusion,
    epsilon: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    ddim_steps: int,
    temperature: float,
) -> Tuple[torch.Tensor, List[str]]:
    h = diffusion.ddim_sample(epsilon, inference_steps=ddim_steps)
    z = unstandardize_latent(h, latent_mean, latent_std)
    _, peptides = autoregressive_decode(vae, z, temperature=temperature)
    return h, peptides


def _resolve_objective_columns(df: pd.DataFrame, requested: List[str]) -> List[str]:
    cols = [c for c in requested if c in df.columns]
    if cols:
        return cols
    candidates = [
        "chelation_sub",
        "solubility_sub",
        "stability_sub",
        "expression_sub",
        "final_score",
    ]
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        raise ValueError(f"No objective columns found. Requested={requested}")
    return cols


def _find_peptide_col(df: pd.DataFrame, requested: str) -> str:
    for c in [requested, "peptide_len10", "sequence", "peptide"]:
        if c and c in df.columns:
            return c
    raise ValueError(
        f"Could not find peptide column. requested={requested!r}; "
        f"available={list(df.columns)}"
    )


def _make_lookup(
    df: pd.DataFrame,
    peptide_col: str,
    objective_cols: List[str],
) -> Dict[str, Dict[str, float]]:
    lookup: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        pep = clean_peptide(row.get(peptide_col))
        if pep is None:
            continue
        if any(pd.isna(row.get(c, np.nan)) for c in objective_cols):
            continue
        if pep not in lookup:
            lookup[pep] = {c: float(row[c]) for c in objective_cols}
    return lookup


def _load_scored_lookup(
    path: str,
    requested_peptide_col: str,
    objective_cols: List[str],
) -> Dict[str, Dict[str, float]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Scored peptide CSV not found: {p}")
    df = pd.read_csv(p)
    pep_col = _find_peptide_col(df, requested_peptide_col)
    return _make_lookup(df, pep_col, objective_cols)


def _merge_lookups(*lookups):
    merged = {}
    for lookup in lookups:
        merged.update(lookup)
    return merged


def _objective_delta(
    center_obj: Dict[str, float],
    neighbor_obj: Dict[str, float],
    objective_cols: List[str],
) -> Tuple[Dict[str, float], float, float]:
    deltas = {}
    ss = 0.0
    sa = 0.0
    for c in objective_cols:
        d = float(neighbor_obj[c]) - float(center_obj[c])
        deltas[c] = d
        ss += d * d
        sa += abs(d)
    return deltas, math.sqrt(ss), sa / max(len(objective_cols), 1)


@torch.no_grad()
def run_epsilon_objective_diagnostic(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device)

    mc = ckpt["model_config"]
    dc = ckpt["diffusion_config"]

    cfg = ModelConfig(
        hidden_size=int(mc["hidden_size"]),
        latent_dim=int(mc["latent_dim"]),
        n_layers=int(mc["n_layers"]),
        dropout=float(mc.get("dropout", 0.0)),
        decoder_conditioning=str(
            mc.get(
                "decoder_conditioning",
                "concat_z_at_every_decoder_step",
            )
        ),
    )

    diff_cfg = DiffusionConfig(
        hidden_dim=int(dc["hidden_dim"]),
        time_dim=int(dc["time_dim"]),
        n_blocks=int(dc["n_blocks"]),
        train_steps=int(dc["train_steps"]),
        beta_start=float(dc["beta_start"]),
        beta_end=float(dc["beta_end"]),
    )

    vae = GRUVAE(cfg).to(device)
    diffusion = LatentDiffusion(cfg.latent_dim, diff_cfg).to(device)
    vae.load_state_dict(ckpt["vae_state_dict"], strict=True)
    diffusion.load_state_dict(ckpt["diffusion_state_dict"], strict=True)
    vae.eval()
    diffusion.eval()

    latent_mean = ckpt["latent_mean"].to(device).float()
    latent_std = (
        ckpt["latent_std"].to(device).float().clamp(min=1e-6)
    )

    # ------------------------------------------------------------------
    # Original scored Cu dataset:
    # used only for selecting source peptides and as one possible source
    # of objective values for decoded center/neighbor peptides.
    # ------------------------------------------------------------------
    df = pd.read_csv(args.cu_csv)
    peptide_col = _find_peptide_col(df, args.peptide_col)
    objective_cols = _resolve_objective_columns(df, args.objective_cols)

    training_lookup = _make_lookup(
        df,
        peptide_col,
        objective_cols,
    )

    valid = []
    for idx, row in df.iterrows():
        pep = clean_peptide(row[peptide_col])
        if pep is not None and all(
            not pd.isna(row[c]) for c in objective_cols
        ):
            valid.append((idx, pep))

    if not valid:
        raise ValueError("No valid scored Cu peptides found.")

    # ------------------------------------------------------------------
    # Optional objective lookup from previously scored epsilon neighbors.
    # This is also useful for decoded centers that happened to appear among
    # the earlier generated neighbor peptides.
    # ------------------------------------------------------------------
    neighbor_lookup = _load_scored_lookup(
        args.scored_neighbor_csv,
        args.peptide_col,
        objective_cols,
    )

    # Optional dedicated scored-center CSV for the corrected second pass.
    center_scored_lookup = _load_scored_lookup(
        args.scored_center_csv,
        args.peptide_col,
        objective_cols,
    )

    all_scored_lookup = _merge_lookups(
        training_lookup,
        neighbor_lookup,
        center_scored_lookup,
    )

    # ------------------------------------------------------------------
    # Select source peptides reproducibly.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    n_centers = min(int(args.n_centers), len(valid))

    if args.center_selection == "random":
        ids = rng.choice(
            len(valid),
            size=n_centers,
            replace=False,
        )
        selected = [valid[int(i)] for i in ids]
    else:
        rank_col = (
            args.rank_objective
            if args.rank_objective in df.columns
            else objective_cols[0]
        )
        ranked = sorted(
            [
                (float(df.loc[idx, rank_col]), idx, pep)
                for idx, pep in valid
            ],
            reverse=True,
        )
        selected = [
            (idx, pep)
            for _, idx, pep in ranked[:n_centers]
        ]

    source_peptides = [pep for _, pep in selected]
    x_centers = onehot_encode_peptides(source_peptides).to(device)

    # Source objectives are retained only for round-trip auditing.
    source_values = {
        pep: {
            c: float(df.loc[idx, c])
            for c in objective_cols
        }
        for idx, pep in selected
    }

    # ------------------------------------------------------------------
    # Source peptide -> encoder mu -> standardized h0 -> DDIM epsilon
    # -> decode epsilon CENTER.
    # ------------------------------------------------------------------
    mu, _, _ = vae.enc(x_centers)
    original_h0 = standardize_mu(mu, latent_mean, latent_std)
    eps = diffusion.ddim_invert(
        original_h0,
        inference_steps=args.ddim_steps,
    )

    radius = (
        float(args.radius)
        if args.radius is not None
        else math.sqrt(cfg.latent_dim)
    )

    if args.project_centers_to_sphere:
        eps = _project_to_radius(eps, radius)

    h_center, center_decoded = _decode_epsilon_batch(
        vae,
        diffusion,
        eps,
        latent_mean,
        latent_std,
        args.ddim_steps,
        args.decode_temperature,
    )

    # ------------------------------------------------------------------
    # Export center-decoded peptides that do not yet have objective values.
    # This is the key correction to the previous diagnostic.
    # ------------------------------------------------------------------
    center_rows = []
    missing_center_peptides = []

    for i, (src, dec0) in enumerate(
        zip(source_peptides, center_decoded)
    ):
        center_obj = all_scored_lookup.get(dec0)
        row = {
            "center_index": i,
            "source_peptide": src,
            "center_decoded_peptide": dec0,
            "source_to_center_decode_edit": levenshtein_edit_distance(
                src,
                dec0,
            ),
            "center_objective_status": (
                "available"
                if center_obj is not None
                else "missing"
            ),
        }

        # Audit inversion reconstruction in standardized h-space.
        row["center_h_reconstruction_l2"] = float(
            torch.linalg.norm(
                h_center[i : i + 1]
                - original_h0[i : i + 1],
                dim=-1,
            ).item()
        )

        for c in objective_cols:
            row[f"source_{c}"] = source_values[src][c]
            row[f"center_decoded_{c}"] = (
                center_obj[c]
                if center_obj is not None
                else np.nan
            )

        center_rows.append(row)

        if center_obj is None:
            missing_center_peptides.append(dec0)

    center_df = pd.DataFrame(center_rows)

    centers_path = os.path.join(
        args.out_dir,
        "epsilon_decoded_centers.csv",
    )
    center_df.to_csv(centers_path, index=False)

    missing_centers = sorted(set(missing_center_peptides))
    missing_centers_path = os.path.join(
        args.out_dir,
        "epsilon_centers_needing_blackbox_scoring.csv",
    )
    pd.DataFrame(
        {"peptide_len10": missing_centers}
    ).to_csv(
        missing_centers_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # Generate epsilon neighbors.
    # Objective deltas are now referenced to f(decoded epsilon CENTER).
    # ------------------------------------------------------------------
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.seed + 1234)

    rows = []

    for i, src in enumerate(source_peptides):
        eps0 = eps[i : i + 1]
        hc = h_center[i : i + 1]
        dec0 = center_decoded[i]
        center_obj = all_scored_lookup.get(dec0)

        for sigma in args.sigmas:
            for j in range(args.neighbors_per_sigma):
                noise = (
                    torch.randn(
                        eps0.shape,
                        generator=gen,
                        device="cpu",
                    )
                    .to(device)
                    * float(sigma)
                )

                eps1 = eps0 + noise

                if args.renormalize_neighbors:
                    eps1 = _project_to_radius(
                        eps1,
                        radius,
                    )

                h1, decs = _decode_epsilon_batch(
                    vae,
                    diffusion,
                    eps1,
                    latent_mean,
                    latent_std,
                    args.ddim_steps,
                    args.decode_temperature,
                )

                pep1 = decs[0]
                neighbor_obj = all_scored_lookup.get(pep1)

                eps_delta = float(
                    torch.linalg.norm(
                        eps1 - eps0,
                        dim=-1,
                    ).item()
                )

                h_delta = float(
                    torch.linalg.norm(
                        h1 - hc,
                        dim=-1,
                    ).item()
                )

                edit_center = levenshtein_edit_distance(
                    dec0,
                    pep1,
                )

                row = {
                    "center_index": i,
                    "source_peptide": src,
                    "center_decoded_peptide": dec0,
                    "neighbor_peptide": pep1,
                    "sigma": float(sigma),
                    "neighbor_index": j,
                    "epsilon_delta_l2": eps_delta,
                    "epsilon_cosine": float(
                        F.cosine_similarity(
                            eps0,
                            eps1,
                            dim=-1,
                        ).item()
                    ),
                    "h_delta_l2": h_delta,
                    "edit_from_center_decode": edit_center,
                    "edit_from_source": levenshtein_edit_distance(
                        src,
                        pep1,
                    ),
                    "neighbor_equals_center_decode": int(
                        pep1 == dec0
                    ),
                    "center_objective_available": int(
                        center_obj is not None
                    ),
                    "neighbor_objective_available": int(
                        neighbor_obj is not None
                    ),
                }

                # Source objectives remain audit-only.
                for c in objective_cols:
                    row[f"source_{c}"] = source_values[src][c]

                # Correct reference = decoded center peptide.
                if center_obj is not None:
                    for c in objective_cols:
                        row[f"center_{c}"] = center_obj[c]
                else:
                    for c in objective_cols:
                        row[f"center_{c}"] = np.nan

                if neighbor_obj is not None:
                    for c in objective_cols:
                        row[f"neighbor_{c}"] = neighbor_obj[c]
                else:
                    for c in objective_cols:
                        row[f"neighbor_{c}"] = np.nan

                if (
                    center_obj is not None
                    and neighbor_obj is not None
                ):
                    deltas, delta_l2, delta_ma = _objective_delta(
                        center_obj,
                        neighbor_obj,
                        objective_cols,
                    )

                    for c in objective_cols:
                        d = deltas[c]
                        row[f"delta_{c}"] = d
                        row[f"abs_delta_{c}"] = abs(d)

                    row["objective_delta_l2"] = delta_l2
                    row["objective_delta_mean_abs"] = delta_ma
                    row["local_objective_per_epsilon"] = (
                        delta_l2 / max(eps_delta, 1e-12)
                    )
                    row["objective_score_status"] = "complete"
                else:
                    for c in objective_cols:
                        row[f"delta_{c}"] = np.nan
                        row[f"abs_delta_{c}"] = np.nan

                    row["objective_delta_l2"] = np.nan
                    row["objective_delta_mean_abs"] = np.nan
                    row["local_objective_per_epsilon"] = np.nan

                    if center_obj is None and neighbor_obj is None:
                        row["objective_score_status"] = (
                            "missing_center_and_neighbor"
                        )
                    elif center_obj is None:
                        row["objective_score_status"] = "missing_center"
                    else:
                        row["objective_score_status"] = "missing_neighbor"

                rows.append(row)

    details = pd.DataFrame(rows)

    details_path = os.path.join(
        args.out_dir,
        "epsilon_objective_neighbor_details_corrected.csv",
    )
    details.to_csv(details_path, index=False)

    # Export any missing neighbors as well, although the previous scored
    # neighbor CSV will normally cover all of these.
    missing_neighbors = sorted(
        set(
            details.loc[
                details["neighbor_objective_available"] == 0,
                "neighbor_peptide",
            ].astype(str)
        )
    )

    missing_neighbors_path = os.path.join(
        args.out_dir,
        "epsilon_neighbors_needing_blackbox_scoring_corrected.csv",
    )

    pd.DataFrame(
        {"peptide_len10": missing_neighbors}
    ).to_csv(
        missing_neighbors_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # Per-sigma summary.
    # ------------------------------------------------------------------
    summary_rows = []

    for sigma, g in details.groupby(
        "sigma",
        sort=True,
    ):
        gv = g[
            np.isfinite(
                g["objective_delta_l2"].to_numpy(float)
            )
        ].copy()

        sr = {
            "sigma": float(sigma),
            "n_neighbors": int(len(g)),
            "n_complete_pairs": int(len(gv)),
            "complete_pair_fraction": (
                len(gv) / max(len(g), 1)
            ),
            "n_unique_centers_complete": int(
                gv["center_index"].nunique()
            ),
            "epsilon_delta_l2_mean": float(
                g["epsilon_delta_l2"].mean()
            ),
            "epsilon_delta_l2_median": float(
                g["epsilon_delta_l2"].median()
            ),
            "h_delta_l2_mean": float(
                g["h_delta_l2"].mean()
            ),
            "sequence_edit_mean": float(
                g["edit_from_center_decode"].mean()
            ),
            "sequence_edit_median": float(
                g["edit_from_center_decode"].median()
            ),
            "identical_fraction": float(
                g["neighbor_equals_center_decode"].mean()
            ),
            "unique_fraction": float(
                g["neighbor_peptide"].nunique()
                / max(len(g), 1)
            ),
        }

        if len(gv) >= 3:
            xd = gv["epsilon_delta_l2"].to_numpy(float)
            yd = gv["objective_delta_l2"].to_numpy(float)
            edits = gv["edit_from_center_decode"].to_numpy(float)
            ratios = gv[
                "local_objective_per_epsilon"
            ].to_numpy(float)

            sr.update(
                {
                    "objective_delta_l2_mean": float(
                        np.mean(yd)
                    ),
                    "objective_delta_l2_median": float(
                        np.median(yd)
                    ),
                    "objective_delta_l2_p90": float(
                        np.quantile(yd, 0.90)
                    ),
                    "objective_delta_mean_abs_mean": float(
                        gv["objective_delta_mean_abs"].mean()
                    ),
                    "pearson_epsilon_vs_objective": _safe_corr(
                        xd,
                        yd,
                        "pearson",
                    ),
                    "spearman_epsilon_vs_objective": _safe_corr(
                        xd,
                        yd,
                        "spearman",
                    ),
                    "pearson_sequence_edit_vs_objective": _safe_corr(
                        edits,
                        yd,
                        "pearson",
                    ),
                    "spearman_sequence_edit_vs_objective": _safe_corr(
                        edits,
                        yd,
                        "spearman",
                    ),
                    "local_objective_per_epsilon_mean": float(
                        np.mean(ratios)
                    ),
                    "local_objective_per_epsilon_median": float(
                        np.median(ratios)
                    ),
                }
            )

            for c in objective_cols:
                sr[f"mean_abs_delta_{c}"] = float(
                    np.nanmean(gv[f"abs_delta_{c}"])
                )
                sr[f"median_abs_delta_{c}"] = float(
                    np.nanmedian(gv[f"abs_delta_{c}"])
                )

        summary_rows.append(sr)

    summary = pd.DataFrame(summary_rows)

    summary_path = os.path.join(
        args.out_dir,
        "epsilon_objective_smoothness_summary_corrected.csv",
    )
    summary.to_csv(summary_path, index=False)

    # ------------------------------------------------------------------
    # Across-sigma trend analysis.
    # This is more meaningful than within-sigma correlation because epsilon
    # distances have a narrow range inside each sigma shell.
    # ------------------------------------------------------------------
    valid_all = details[
        np.isfinite(
            details["objective_delta_l2"].to_numpy(float)
        )
    ].copy()

    report = []
    report.append(
        "CORRECTED EPSILON-SPACE OBJECTIVE SMOOTHNESS DIAGNOSTIC"
    )
    report.append("=" * 76)
    report.append(f"checkpoint={args.checkpoint}")
    report.append(f"epoch={ckpt.get('epoch')}")
    report.append(
        f"selection={ckpt.get('checkpoint_selection')}"
    )
    report.append(f"objectives={objective_cols}")
    report.append(f"n_source_centers={len(source_peptides)}")
    report.append(
        f"n_unique_decoded_centers={len(set(center_decoded))}"
    )
    report.append(
        f"decoded_centers_with_objectives="
        f"{sum(p in all_scored_lookup for p in center_decoded)}"
    )
    report.append(
        f"decoded_centers_missing_objectives={len(missing_centers)}"
    )
    report.append(f"sigmas={args.sigmas}")
    report.append(
        f"neighbors_per_sigma={args.neighbors_per_sigma}"
    )
    report.append(f"radius={radius:.6f}")
    report.append(f"total_neighbors={len(details)}")
    report.append(
        f"complete_center_neighbor_pairs={len(valid_all)}"
    )
    report.append(
        f"missing_unique_neighbors={len(missing_neighbors)}"
    )

    if len(valid_all) >= 3:
        x = valid_all[
            "epsilon_delta_l2"
        ].to_numpy(float)
        y = valid_all[
            "objective_delta_l2"
        ].to_numpy(float)
        edits = valid_all[
            "edit_from_center_decode"
        ].to_numpy(float)

        report.append(
            f"overall_pearson_epsilon_vs_objective="
            f"{_safe_corr(x, y, 'pearson'):.6f}"
        )
        report.append(
            f"overall_spearman_epsilon_vs_objective="
            f"{_safe_corr(x, y, 'spearman'):.6f}"
        )
        report.append(
            f"overall_pearson_sequence_edit_vs_objective="
            f"{_safe_corr(edits, y, 'pearson'):.6f}"
        )
        report.append(
            f"overall_spearman_sequence_edit_vs_objective="
            f"{_safe_corr(edits, y, 'spearman'):.6f}"
        )
        report.append(
            f"overall_mean_objective_delta_l2="
            f"{np.mean(y):.6f}"
        )
        report.append(
            f"overall_median_objective_delta_l2="
            f"{np.median(y):.6f}"
        )
        report.append(
            f"overall_median_local_objective_per_epsilon="
            f"{np.nanmedian(valid_all['local_objective_per_epsilon']):.6f}"
        )

        # Trend of per-sigma mean epsilon distance against per-sigma mean
        # objective change.
        trend_df = summary[
            summary["objective_delta_l2_mean"].notna()
        ].copy()

        if len(trend_df) >= 3:
            sigma_eps = trend_df[
                "epsilon_delta_l2_mean"
            ].to_numpy(float)
            sigma_obj = trend_df[
                "objective_delta_l2_mean"
            ].to_numpy(float)

            report.append(
                f"across_sigma_pearson_mean_epsilon_vs_mean_objective="
                f"{_safe_corr(sigma_eps, sigma_obj, 'pearson'):.6f}"
            )
            report.append(
                f"across_sigma_spearman_mean_epsilon_vs_mean_objective="
                f"{_safe_corr(sigma_eps, sigma_obj, 'spearman'):.6f}"
            )

        report.append("")
        report.append("Per-sigma corrected objective change:")
        for _, r in summary.iterrows():
            if pd.notna(r.get("objective_delta_l2_mean", np.nan)):
                report.append(
                    f"- sigma={r['sigma']:.6f}: "
                    f"epsilon_L2_mean={r['epsilon_delta_l2_mean']:.6f}, "
                    f"objective_delta_L2_mean={r['objective_delta_l2_mean']:.6f}, "
                    f"median={r['objective_delta_l2_median']:.6f}, "
                    f"complete_pairs={int(r['n_complete_pairs'])}"
                )

        report.append("")
        report.append("Interpretation guidance:")
        report.append(
            "- The correct local reference is the decoded epsilon-center peptide."
        )
        report.append(
            "- A monotonic increase in objective change as sigma/epsilon distance "
            "increases supports useful local BO geometry."
        )
        report.append(
            "- Small objective changes at the smallest sigma are especially important."
        )
        report.append(
            "- Sequence edits can be large without invalidating BO if objective changes "
            "remain locally smooth."
        )
        report.append(
            "- local_objective_per_epsilon is a local Lipschitz-like sensitivity "
            "diagnostic; smaller and stable values are preferable."
        )

    else:
        report.append("")
        report.append(
            "Not enough complete decoded-center/neighbor objective pairs are available."
        )

        if missing_centers:
            report.append(
                "Score epsilon_centers_needing_blackbox_scoring.csv with the SAME "
                "ESM-label + canonical Cu black-box pipeline, then rerun with "
                "--scored-center-csv."
            )

        if missing_neighbors:
            report.append(
                "Score epsilon_neighbors_needing_blackbox_scoring_corrected.csv "
                "with the SAME black-box pipeline and provide it via "
                "--scored-neighbor-csv."
            )

    report_path = os.path.join(
        args.out_dir,
        "epsilon_objective_bo_readiness_report_corrected.txt",
    )
    Path(report_path).write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print("\n".join(report))
    print()
    print(f"Saved decoded centers: {centers_path}")
    print(
        f"Saved centers needing scoring: {missing_centers_path}"
    )
    print(f"Saved corrected details: {details_path}")
    print(f"Saved corrected summary: {summary_path}")
    print(
        f"Saved missing corrected neighbors: {missing_neighbors_path}"
    )
    print(f"Saved corrected report: {report_path}")


def parse_diagnostic_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Corrected epsilon-space objective smoothness diagnostic. "
            "Objective deltas are measured relative to the decoded epsilon center."
        )
    )

    parser.add_argument(
        "--checkpoint",
        default=(
            "../../output/"
            "finetuned_gru_vae_latent_diffusion_checkpoints_h64_z64_high_confidence_dataset/"
            "best_ddim_inversion_l2_h64_z64_latent_diffusion.pt"
        ),
    )

    parser.add_argument(
        "--cu-csv",
        default=(
            "../../data/"
            "metalpdb_CU_chain_mapped_len10_high_confidence_blackbox_scored_ranked.csv"
        ),
    )

    parser.add_argument(
        "--peptide-col",
        default="peptide_len10",
    )

    parser.add_argument(
        "--objective-cols",
        nargs="+",
        default=[
            "chelation_sub",
            "solubility_sub",
            "stability_sub",
            "expression_sub",
        ],
    )

    parser.add_argument(
        "--rank-objective",
        default="final_score",
    )

    parser.add_argument(
        "--out-dir",
        default=(
            "../../output/"
            "epsilon_objective_smoothness_diagnostic_corrected"
        ),
    )

    parser.add_argument(
        "--n-centers",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--center-selection",
        choices=["random", "top"],
        default="random",
    )

    parser.add_argument(
        "--sigmas",
        nargs="+",
        type=float,
        default=[
            0.01,
            0.025,
            0.05,
            0.10,
            0.20,
        ],
    )

    parser.add_argument(
        "--neighbors-per-sigma",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--ddim-steps",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--decode-temperature",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--radius",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--project-centers-to-sphere",
        action="store_true",
        default=True,
    )

    parser.add_argument(
        "--no-project-centers-to-sphere",
        dest="project_centers_to_sphere",
        action="store_false",
    )

    parser.add_argument(
        "--renormalize-neighbors",
        action="store_true",
        default=True,
    )

    parser.add_argument(
        "--no-renormalize-neighbors",
        dest="renormalize_neighbors",
        action="store_false",
    )

    parser.add_argument(
        "--scored-neighbor-csv",
        default="",
        help=(
            "CSV with objective values for decoded neighbor peptides. "
            "Your previously generated "
            "epsilon_neighbors_blackbox_scored_with_esm_labels.csv can be used."
        ),
    )

    parser.add_argument(
        "--scored-center-csv",
        default="",
        help=(
            "Second-pass CSV with objective values for decoded epsilon-center peptides. "
            "Generate it by scoring epsilon_centers_needing_blackbox_scoring.csv "
            "with the same ESM-label + canonical Cu black-box pipeline."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_diagnostic_args()
    run_epsilon_objective_diagnostic(args)
