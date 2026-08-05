from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Configuration
# ============================================================

# Diffusion-epsilon BO Pareto set to test.
OPTIMIZED_CSV = "bo_results_cu_latent_diffusion_epsilon_bo_last_epoch/bo_pareto_dominates_train_CU_diffusion_epsilon_gp.csv"

# Optional. The scrambling analysis itself does not require a train-Pareto file.
# Set to a CSV path only when you also want generated scrambles to avoid
# sequences present in the training Pareto set.
TRAIN_PARETO_CSV = None

OUTPUT_DIR = "scrambling_control_cu_latent_diffusion_last_epoch"

N_SCRAMBLES_PER_PEPTIDE = 100
RANDOM_SEED = 42

# ============================================================
# Latent-diffusion round-trip / local smoothness configuration
# ============================================================

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_I = {a: i for i, a in enumerate(AA)}
I_TO_AA = {i: a for a, i in AA_TO_I.items()}
SEQ_LEN = 10
VOCAB = 20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LATENT_DIM = 32
HIDDEN_SIZE = 32
N_GRU_LAYERS = 2
DROPOUT = 0.0

DIFFUSION_CHECKPOINT = "cu_latent_diffusion_finetune_h32_z32/last_epoch_cu_latent_diffusion.pt"

EXPECTED_CHECKPOINT_TYPE = "cu_finetuned_gru_vae_latent_diffusion"
EXPECTED_CHECKPOINT_EPOCH = 100
EXPECTED_CHECKPOINT_SELECTION = "last_completed_epoch"
EXPECTED_DECODER_CONDITIONING = "concat_z_at_every_decoder_step"
EXPECTED_BO_SPACE = "diffusion_base_noise_epsilon"

DDIM_STEPS = 20
DECODE_TEMPERATURE = 1.0

# Local epsilon-space smoothness diagnostic.
N_LATENT_NEIGHBORS = 3
LATENT_PERTURBATION_STD = 0.05
LATENT_NEIGHBOR_SEED = 12345

# For sequence-derived inversion diagnostics, the most direct diagnostic is to
# perturb the inferred epsilon without forcing it to the shell.
# Set True if you want the local neighborhood to mimic BO's shell projection.
LOCAL_RENORMALIZE_TO_SPHERE = False

# If True, avoid generating scrambled peptides that already exist in the
# optimized set or optional training Pareto set.
REJECT_EXISTING_SEQUENCES = True

OBJECTIVE_COLS = [
    "chelation_sub",
    "solubility_sub",
    "stability_sub",
    "expression_sub",
]
FINAL_SCORE_COL = "final_score"


# ============================================================
# Objective function import
# ============================================================

try:
    from black_box_fcn_mo_CU_f import blackbox_fc
except Exception:
    from backup_code_result_folders.black_box_fcn_mo_CU_f_updated_batch_normalized import blackbox_fc


# ============================================================
# Latent-conditioned GRU-VAE + compact latent diffusion
# ============================================================

@dataclass
class ModelConfig:
    hidden_size: int = 32
    latent_dim: int = 32
    n_layers: int = 2
    dropout: float = 0.0


@dataclass
class DiffusionConfig:
    hidden_dim: int = 128
    time_dim: int = 32
    n_blocks: int = 4
    train_steps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 2e-2


class GRUEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        latent_dim: int,
        n_layers: int,
        dropout: float,
    ):
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

    def forward(
        self,
        x_onehot: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.in_proj(x_onehot)
        out_top, h_n = self.gru(x)
        h_last = h_n[-1]
        mu = self.to_mu(h_last)
        logvar = self.to_logvar(h_last).clamp(min=-8.0, max=8.0)
        return mu, logvar, out_top


class GRUDecoder(nn.Module):
    """Latent-conditioned decoder: every recurrent step sees [token_embedding ; z]."""

    def __init__(
        self,
        hidden_size: int,
        latent_dim: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
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

    def initial_hidden(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        return self.z_to_h(z).view(
            self.gru.num_layers,
            batch_size,
            self.gru.hidden_size,
        )

    def decoder_step(
        self,
        z: torch.Tensor,
        current_token: torch.Tensor,
        h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.token_embed(current_token)
        z_step = z.unsqueeze(1)
        decoder_input = torch.cat([emb, z_step], dim=-1)
        out_step, h = self.gru(decoder_input, h)
        logits = self.to_logits(out_step[:, -1, :])
        return logits, h


class GRUVAE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.enc = GRUEncoder(
            cfg.hidden_size,
            cfg.latent_dim,
            cfg.n_layers,
            cfg.dropout,
        )
        self.dec = GRUDecoder(
            cfg.hidden_size,
            cfg.latent_dim,
            cfg.n_layers,
            cfg.dropout,
        )


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time_dim must be even")
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        exponent = -math.log(10000.0) * torch.arange(
            half,
            device=t.device,
            dtype=torch.float32,
        ) / max(half - 1, 1)
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
    def __init__(
        self,
        latent_dim: int,
        cfg: DiffusionConfig,
    ):
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(cfg.time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.time_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.in_proj = nn.Linear(latent_dim, cfg.hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualDenoiserBlock(cfg.hidden_dim) for _ in range(cfg.n_blocks)]
        )
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
    def __init__(
        self,
        latent_dim: int,
        cfg: DiffusionConfig,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.cfg = cfg
        self.denoiser = LatentDenoiser(latent_dim, cfg)

        betas = torch.linspace(
            cfg.beta_start,
            cfg.beta_end,
            cfg.train_steps,
            dtype=torch.float32,
        )
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    @torch.no_grad()
    def ddim_sample(
        self,
        base_noise: torch.Tensor,
        inference_steps: int = 20,
    ) -> torch.Tensor:
        """Deterministic DDIM sampler; base_noise is epsilon."""
        self.eval()
        n_train = self.cfg.train_steps
        inference_steps = min(max(int(inference_steps), 1), n_train)

        timesteps = torch.linspace(
            n_train - 1,
            0,
            inference_steps,
            device=base_noise.device,
        ).round().long()
        timesteps = torch.unique_consecutive(timesteps)

        h = base_noise

        for step_index, t_scalar in enumerate(timesteps):
            t = torch.full(
                (h.size(0),),
                int(t_scalar.item()),
                device=h.device,
                dtype=torch.long,
            )
            pred_noise = self.denoiser(h, t)
            alpha_bar_t = self.alpha_bars[t_scalar]

            pred_h0 = (
                h - torch.sqrt(1.0 - alpha_bar_t) * pred_noise
            ) / torch.sqrt(alpha_bar_t)

            if step_index == len(timesteps) - 1:
                h = pred_h0
                break

            t_prev = timesteps[step_index + 1]
            alpha_bar_prev = self.alpha_bars[t_prev]

            h = (
                torch.sqrt(alpha_bar_prev) * pred_h0
                + torch.sqrt(1.0 - alpha_bar_prev) * pred_noise
            )

        return h

    @torch.no_grad()
    def ddim_invert(
        self,
        h0: torch.Tensor,
        inference_steps: int = 20,
    ) -> torch.Tensor:
        """Approximate deterministic DDIM inversion from clean standardized h0 to epsilon."""
        self.eval()
        n_train = self.cfg.train_steps
        inference_steps = min(max(int(inference_steps), 1), n_train)

        timesteps = torch.linspace(
            0,
            n_train - 1,
            inference_steps,
            device=h0.device,
        ).round().long()
        timesteps = torch.unique_consecutive(timesteps)

        h = h0

        for step_index, t_scalar in enumerate(timesteps[:-1]):
            t = torch.full(
                (h.size(0),),
                int(t_scalar.item()),
                device=h.device,
                dtype=torch.long,
            )
            pred_noise = self.denoiser(h, t)
            alpha_bar_t = self.alpha_bars[t_scalar]

            pred_h0 = (
                h - torch.sqrt(1.0 - alpha_bar_t) * pred_noise
            ) / torch.sqrt(alpha_bar_t)

            t_next = timesteps[step_index + 1]
            alpha_bar_next = self.alpha_bars[t_next]

            h = (
                torch.sqrt(alpha_bar_next) * pred_h0
                + torch.sqrt(1.0 - alpha_bar_next) * pred_noise
            )

        return h


# ============================================================
# Basic peptide utilities
# ============================================================

def onehot_encode_peptides(peptides: List[str]) -> torch.Tensor:
    x = torch.zeros((len(peptides), SEQ_LEN, VOCAB), dtype=torch.float32)
    for n, peptide in enumerate(peptides):
        peptide = str(peptide).strip().upper()
        if len(peptide) != SEQ_LEN:
            raise ValueError(
                f"Expected peptide length {SEQ_LEN}, got {len(peptide)} for {peptide!r}"
            )
        for t, aa in enumerate(peptide):
            if aa not in AA_TO_I:
                raise ValueError(f"Unsupported amino acid {aa!r} in {peptide!r}")
            x[n, t, AA_TO_I[aa]] = 1.0
    return x


def levenshtein_edit_distance(a: str, b: str) -> int:
    a = str(a)
    b = str(b)
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + int(char_a != char_b)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return int(previous[-1])


def project_to_shell(epsilon: torch.Tensor, radius: float) -> torch.Tensor:
    return epsilon / epsilon.norm(dim=-1, keepdim=True).clamp(min=1e-8) * radius


# ============================================================
# Checkpoint loading
# ============================================================

def load_diffusion_generator(checkpoint_path: str):
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            "Update DIFFUSION_CHECKPOINT in the configuration section."
        )

    ckpt = torch.load(checkpoint, map_location=DEVICE)

    if ckpt.get("checkpoint_type") != EXPECTED_CHECKPOINT_TYPE:
        raise ValueError(
            f"Expected checkpoint_type={EXPECTED_CHECKPOINT_TYPE!r}, "
            f"but found {ckpt.get('checkpoint_type')!r}."
        )

    checkpoint_epoch = int(ckpt.get("epoch", -1))
    if checkpoint_epoch != EXPECTED_CHECKPOINT_EPOCH:
        raise ValueError(
            f"Expected epoch {EXPECTED_CHECKPOINT_EPOCH}, "
            f"but checkpoint metadata reports epoch {checkpoint_epoch}."
        )

    checkpoint_selection = ckpt.get("checkpoint_selection")
    if checkpoint_selection != EXPECTED_CHECKPOINT_SELECTION:
        raise ValueError(
            f"Expected checkpoint_selection={EXPECTED_CHECKPOINT_SELECTION!r}, "
            f"but found {checkpoint_selection!r}."
        )

    decoder_conditioning = ckpt.get("decoder_latent_conditioning")
    if decoder_conditioning != EXPECTED_DECODER_CONDITIONING:
        raise ValueError(
            f"Expected decoder conditioning {EXPECTED_DECODER_CONDITIONING!r}, "
            f"but found {decoder_conditioning!r}."
        )

    bo_space = ckpt.get("bo_coordinate_space")
    if bo_space != EXPECTED_BO_SPACE:
        raise ValueError(
            f"Expected bo_coordinate_space={EXPECTED_BO_SPACE!r}, "
            f"but found {bo_space!r}."
        )

    cfg = ModelConfig(**ckpt["model_config"])
    diff_cfg = DiffusionConfig(**ckpt["diffusion_config"])

    if int(cfg.hidden_size) != HIDDEN_SIZE or int(cfg.latent_dim) != LATENT_DIM:
        raise ValueError(
            f"Checkpoint model config does not match H={HIDDEN_SIZE}, Z={LATENT_DIM}: {cfg}"
        )

    vae = GRUVAE(cfg).to(DEVICE)
    diffusion = LatentDiffusion(cfg.latent_dim, diff_cfg).to(DEVICE)

    vae.load_state_dict(ckpt["vae_state_dict"], strict=True)
    diffusion.load_state_dict(ckpt["diffusion_state_dict"], strict=True)

    vae.eval()
    diffusion.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in diffusion.parameters():
        p.requires_grad_(False)

    latent_mean = ckpt["latent_mean"].to(DEVICE).float()
    latent_std = ckpt["latent_std"].to(DEVICE).float().clamp(min=1e-6)
    radius = float(ckpt.get("recommended_bo_radius", math.sqrt(cfg.latent_dim)))

    print(f"Loaded latent-diffusion checkpoint: {checkpoint}")
    print(f"Device: {DEVICE}")
    print(f"Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"Checkpoint selection: {ckpt.get('checkpoint_selection', 'unknown')}")
    print(f"BO coordinate space: {ckpt.get('bo_coordinate_space', 'unknown')}")
    print(f"Recommended epsilon radius: {radius}")
    metrics = ckpt.get("metrics", {})
    if metrics:
        print("Checkpoint val diffusion_epsilon_mse:", metrics.get("diffusion_epsilon_mse", "unknown"))
        print("Checkpoint val score_pearson:", metrics.get("score_pearson", "unknown"))
        print("Checkpoint val ddim_inversion_h_l2_mean:", metrics.get("ddim_inversion_h_l2_mean", "unknown"))
        print("Checkpoint val ddim_decode_unique_fraction:", metrics.get("ddim_decode_unique_fraction", "unknown"))
        print("Checkpoint val local_epsilon_edit_mean:", metrics.get("local_epsilon_edit_mean", "unknown"))

    return vae, diffusion, latent_mean, latent_std, radius


# ============================================================
# Latent-diffusion diagnostics
# ============================================================

@torch.no_grad()
def encode_peptide_to_standardized_mu(
    vae: GRUVAE,
    peptide: str,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
) -> torch.Tensor:
    x = onehot_encode_peptides([peptide]).to(DEVICE)
    mu, _, _ = vae.enc(x)
    h0 = (mu - latent_mean.to(mu.device)) / latent_std.to(mu.device).clamp(min=1e-6)
    return h0


@torch.no_grad()
def encode_peptide_to_epsilon(
    vae: GRUVAE,
    diffusion: LatentDiffusion,
    peptide: str,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
) -> torch.Tensor:
    h0 = encode_peptide_to_standardized_mu(
        vae=vae,
        peptide=peptide,
        latent_mean=latent_mean,
        latent_std=latent_std,
    )
    epsilon = diffusion.ddim_invert(h0, inference_steps=DDIM_STEPS)
    return epsilon


@torch.no_grad()
def epsilon_to_decoder_latent(
    diffusion: LatentDiffusion,
    epsilon: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
) -> torch.Tensor:
    h = diffusion.ddim_sample(epsilon, inference_steps=DDIM_STEPS)
    z = h * latent_std.to(h.device).clamp(min=1e-6) + latent_mean.to(h.device)
    return z


@torch.no_grad()
def decode_from_decoder_latent(
    vae: GRUVAE,
    z: torch.Tensor,
    temperature: float = DECODE_TEMPERATURE,
) -> str:
    z = z.reshape(1, -1).to(DEVICE)
    batch_size = z.size(0)
    h = vae.dec.initial_hidden(z)

    current_token = torch.zeros(
        batch_size,
        1,
        VOCAB,
        dtype=z.dtype,
        device=z.device,
    )

    token_indices = []
    tau = max(float(temperature), 1e-6)

    for _ in range(SEQ_LEN):
        logits, h = vae.dec.decoder_step(
            z=z,
            current_token=current_token,
            h=h,
        )
        logits = logits / tau
        token_idx = logits.argmax(dim=-1)
        token_indices.append(int(token_idx.item()))
        current_token = F.one_hot(
            token_idx,
            num_classes=VOCAB,
        ).to(dtype=z.dtype).unsqueeze(1)

    return "".join(I_TO_AA[i] for i in token_indices)


@torch.no_grad()
def decode_from_epsilon(
    vae: GRUVAE,
    diffusion: LatentDiffusion,
    epsilon: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
) -> str:
    z = epsilon_to_decoder_latent(
        diffusion=diffusion,
        epsilon=epsilon.reshape(1, -1).to(DEVICE),
        latent_mean=latent_mean,
        latent_std=latent_std,
    )
    return decode_from_decoder_latent(vae, z)


@torch.no_grad()
def compute_sequence_roundtrip_diagnostics(
    vae: GRUVAE,
    diffusion: LatentDiffusion,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    source_peptide: str,
) -> Dict[str, object]:
    """
    Deterministic diffusion path:
        source -> h0=standardized_mu(source)
        h0 -> epsilon1 = DDIM inversion(h0)
        epsilon1 -> DDIM -> z1 -> p1
        p1 -> h0_2 -> epsilon2
        epsilon2 -> DDIM -> z2 -> p2

    Peptide round trip = Levenshtein(p1, p2).
    Epsilon round trip = L2/cosine(epsilon1, epsilon2).
    Standardized latent reconstruction = L2/cosine(h0, DDIM(epsilon1)).
    """
    h0 = encode_peptide_to_standardized_mu(
        vae,
        source_peptide,
        latent_mean,
        latent_std,
    )
    epsilon1 = diffusion.ddim_invert(h0, inference_steps=DDIM_STEPS)
    h0_reconstructed = diffusion.ddim_sample(epsilon1, inference_steps=DDIM_STEPS)
    z1 = h0_reconstructed * latent_std + latent_mean
    p1 = decode_from_decoder_latent(vae, z1)

    h0_p1 = encode_peptide_to_standardized_mu(
        vae,
        p1,
        latent_mean,
        latent_std,
    )
    epsilon2 = diffusion.ddim_invert(h0_p1, inference_steps=DDIM_STEPS)
    h0_p1_reconstructed = diffusion.ddim_sample(epsilon2, inference_steps=DDIM_STEPS)
    z2 = h0_p1_reconstructed * latent_std + latent_mean
    p2 = decode_from_decoder_latent(vae, z2)

    peptide_edit_distance = levenshtein_edit_distance(p1, p2)

    epsilon_l2 = torch.linalg.norm(epsilon1 - epsilon2, dim=-1).item()
    epsilon_cosine = F.cosine_similarity(epsilon1, epsilon2, dim=-1, eps=1e-8).item()

    h_l2 = torch.linalg.norm(h0 - h0_reconstructed, dim=-1).item()
    h_cosine = F.cosine_similarity(h0, h0_reconstructed, dim=-1, eps=1e-8).item()

    source_to_p1_edit = levenshtein_edit_distance(source_peptide, p1)

    return {
        "source_peptide": source_peptide,
        "roundtrip_p1": p1,
        "roundtrip_p2": p2,
        "source_to_p1_edit_distance": int(source_to_p1_edit),
        "source_to_p1_edit_distance_normalized": float(source_to_p1_edit) / SEQ_LEN,
        "peptide_roundtrip_edit_distance": int(peptide_edit_distance),
        "peptide_roundtrip_edit_distance_normalized": (
            float(peptide_edit_distance) / max(len(p1), len(p2), 1)
        ),
        "epsilon_roundtrip_l2": float(epsilon_l2),
        "epsilon_roundtrip_cosine": float(epsilon_cosine),
        "epsilon1_norm": float(torch.linalg.norm(epsilon1, dim=-1).item()),
        "epsilon2_norm": float(torch.linalg.norm(epsilon2, dim=-1).item()),
        "standardized_h_reconstruction_l2": float(h_l2),
        "standardized_h_reconstruction_cosine": float(h_cosine),
        "h0_norm": float(torch.linalg.norm(h0, dim=-1).item()),
        "h0_reconstructed_norm": float(torch.linalg.norm(h0_reconstructed, dim=-1).item()),
    }


@torch.no_grad()
def compute_local_epsilon_smoothness(
    vae: GRUVAE,
    diffusion: LatentDiffusion,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    source_peptide: str,
    sequence_id: int,
    generator: torch.Generator,
    n_neighbors: int,
    perturbation_std: float,
    radius: float,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """
    For epsilon1 inferred from source peptide, decode center p1. Create
    n_neighbors by epsilon_neighbor = epsilon1 + noise, optionally project
    to sphere, decode each neighbor, and measure Levenshtein(neighbor, p1).
    """
    epsilon1 = encode_peptide_to_epsilon(
        vae=vae,
        diffusion=diffusion,
        peptide=source_peptide,
        latent_mean=latent_mean,
        latent_std=latent_std,
    )
    if LOCAL_RENORMALIZE_TO_SPHERE:
        epsilon1 = project_to_shell(epsilon1, radius)

    p1 = decode_from_epsilon(
        vae=vae,
        diffusion=diffusion,
        epsilon=epsilon1,
        latent_mean=latent_mean,
        latent_std=latent_std,
    )

    detail_rows = []
    edit_distances = []
    normalized_edit_distances = []

    for neighbor_index in range(n_neighbors):
        noise_cpu = torch.randn(
            epsilon1.shape,
            generator=generator,
            dtype=epsilon1.dtype,
            device="cpu",
        )
        noise = noise_cpu.to(epsilon1.device) * float(perturbation_std)
        epsilon_neighbor = epsilon1 + noise
        if LOCAL_RENORMALIZE_TO_SPHERE:
            epsilon_neighbor = project_to_shell(epsilon_neighbor, radius)

        neighbor_peptide = decode_from_epsilon(
            vae=vae,
            diffusion=diffusion,
            epsilon=epsilon_neighbor,
            latent_mean=latent_mean,
            latent_std=latent_std,
        )

        edit_distance = levenshtein_edit_distance(p1, neighbor_peptide)
        normalized_edit = float(edit_distance) / max(
            len(p1),
            len(neighbor_peptide),
            1,
        )

        edit_distances.append(edit_distance)
        normalized_edit_distances.append(normalized_edit)

        detail_rows.append(
            {
                "sequence_id": int(sequence_id),
                "source_peptide": source_peptide,
                "center_p1": p1,
                "neighbor_index": int(neighbor_index),
                "perturbation_std": float(perturbation_std),
                "perturbation_l2": float(torch.linalg.norm(noise, dim=-1).item()),
                "renormalized_to_sphere": bool(LOCAL_RENORMALIZE_TO_SPHERE),
                "epsilon_center_norm": float(torch.linalg.norm(epsilon1, dim=-1).item()),
                "epsilon_neighbor_norm": float(torch.linalg.norm(epsilon_neighbor, dim=-1).item()),
                "neighbor_peptide": neighbor_peptide,
                "neighbor_edit_distance_to_p1": int(edit_distance),
                "neighbor_edit_distance_to_p1_normalized": float(normalized_edit),
            }
        )

    summary = {
        "local_smoothness_center_p1": p1,
        "local_smoothness_n_neighbors": int(n_neighbors),
        "local_smoothness_perturbation_std": float(perturbation_std),
        "local_smoothness_renormalized_to_sphere": bool(LOCAL_RENORMALIZE_TO_SPHERE),
        "local_neighbor_edit_distance_mean": float(np.mean(edit_distances)),
        "local_neighbor_edit_distance_std": float(np.std(edit_distances, ddof=0)),
        "local_neighbor_edit_distance_min": int(np.min(edit_distances)),
        "local_neighbor_edit_distance_max": int(np.max(edit_distances)),
        "local_neighbor_normalized_edit_distance_mean": float(np.mean(normalized_edit_distances)),
        "local_neighbor_identical_fraction": float(np.mean(np.asarray(edit_distances) == 0)),
        "local_neighbor_unique_peptides": int(len({row["neighbor_peptide"] for row in detail_rows})),
    }
    return summary, detail_rows


# ============================================================
# Helper functions
# ============================================================

def get_peptide_column(df: pd.DataFrame) -> str:
    candidates = ["peptide", "peptide_len10", "sequence"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find peptide column. Available columns: {list(df.columns)}")


def scramble_sequence(seq: str, rng: random.Random) -> str:
    chars = list(seq)
    rng.shuffle(chars)
    return "".join(chars)


def generate_unique_scrambles(
    seq: str,
    n: int,
    rng: random.Random,
    forbidden: Optional[set[str]] = None,
    max_attempts: int = 10000,
) -> List[str]:
    seq = seq.strip().upper()
    forbidden = forbidden or set()

    scrambles = set()
    attempts = 0

    while len(scrambles) < n and attempts < max_attempts:
        attempts += 1
        s = scramble_sequence(seq, rng)
        if s == seq:
            continue
        if s in scrambles:
            continue
        if s in forbidden:
            continue
        scrambles.add(s)

    if len(scrambles) < n:
        print(
            f"[WARNING] Only generated {len(scrambles)} unique scrambles "
            f"for {seq}. Requested {n}."
        )

    return sorted(scrambles)


def dominates_maximize(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a >= b) and np.any(a > b))


def safe_zscore(x: float, values: np.ndarray) -> float:
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values, ddof=1))
    if not np.isfinite(std) or std == 0:
        return np.nan
    return float((x - mean) / std)


def empirical_p_value_greater_equal(
    original_value: float,
    scrambled_values: np.ndarray,
) -> float:
    scrambled_values = np.asarray(scrambled_values, dtype=float)
    n = np.sum(np.isfinite(scrambled_values))
    if n == 0:
        return np.nan
    count_ge = np.sum(scrambled_values >= original_value)
    return float((count_ge + 1) / (n + 1))


def append_overall_average_row(
    df: pd.DataFrame,
    metric_columns: List[str],
    label_column: str = "source_peptide",
) -> pd.DataFrame:
    average_row = {col: np.nan for col in df.columns}
    if label_column in average_row:
        average_row[label_column] = "__AVERAGE__"
    for col in metric_columns:
        if col in df.columns:
            average_row[col] = float(pd.to_numeric(df[col], errors="coerce").mean())
    return pd.concat([df, pd.DataFrame([average_row])], ignore_index=True)


# ============================================================
# Main analysis
# ============================================================

def main():
    rng = random.Random(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    latent_neighbor_generator = torch.Generator(device="cpu")
    latent_neighbor_generator.manual_seed(LATENT_NEIGHBOR_SEED)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    optimized_df = pd.read_csv(OPTIMIZED_CSV)

    if TRAIN_PARETO_CSV is not None:
        train_path = Path(TRAIN_PARETO_CSV)
        if not train_path.exists():
            raise FileNotFoundError(f"Configured TRAIN_PARETO_CSV does not exist: {train_path}")
        train_df = pd.read_csv(train_path)
    else:
        train_df = None

    vae, diffusion, latent_mean, latent_std, radius = load_diffusion_generator(DIFFUSION_CHECKPOINT)

    opt_pep_col = get_peptide_column(optimized_df)

    optimized_peptides = (
        optimized_df[opt_pep_col]
        .astype(str)
        .str.strip()
        .str.upper()
        .dropna()
        .unique()
        .tolist()
    )

    if train_df is not None:
        train_pep_col = get_peptide_column(train_df)
        train_peptides = (
            train_df[train_pep_col]
            .astype(str)
            .str.strip()
            .str.upper()
            .dropna()
            .unique()
            .tolist()
        )
    else:
        train_peptides = []

    forbidden = set()
    if REJECT_EXISTING_SEQUENCES:
        forbidden.update(optimized_peptides)
        forbidden.update(train_peptides)

    print(f"Optimized peptides: {len(optimized_peptides)}")
    print(f"Optional training Pareto peptides used for scramble rejection: {len(train_peptides)}")
    print(f"Optimized input CSV: {OPTIMIZED_CSV}")
    print(f"Latent-diffusion checkpoint: {DIFFUSION_CHECKPOINT}")
    print(f"Scrambles per optimized peptide: {N_SCRAMBLES_PER_PEPTIDE}")
    print(f"Local epsilon renormalized to sphere: {LOCAL_RENORMALIZE_TO_SPHERE}")

    all_scored_rows = []
    summary_rows = []
    roundtrip_rows = []
    local_smoothness_detail_rows = []

    for peptide_index, original_peptide in enumerate(optimized_peptides, start=1):
        print(f"\n[{peptide_index}/{len(optimized_peptides)}] Original: {original_peptide}")

        scrambles = generate_unique_scrambles(
            seq=original_peptide,
            n=N_SCRAMBLES_PER_PEPTIDE,
            rng=rng,
            forbidden=forbidden,
        )

        candidate_peptides = [original_peptide] + scrambles

        # ------------------------------------------------------------
        # Diffusion epsilon round-trip and local smoothness for every
        # sequence in this group.
        # ------------------------------------------------------------
        for sequence in candidate_peptides:
            sequence_id = len(roundtrip_rows)

            rt_row = compute_sequence_roundtrip_diagnostics(
                vae=vae,
                diffusion=diffusion,
                latent_mean=latent_mean,
                latent_std=latent_std,
                source_peptide=sequence,
            )
            rt_row["sequence_id"] = int(sequence_id)
            rt_row["scramble_group_id"] = int(peptide_index)
            rt_row["original_peptide"] = original_peptide
            rt_row["control_type"] = (
                "optimized_original"
                if sequence == original_peptide
                else "scrambled_control"
            )

            smooth_summary, smooth_details = compute_local_epsilon_smoothness(
                vae=vae,
                diffusion=diffusion,
                latent_mean=latent_mean,
                latent_std=latent_std,
                source_peptide=sequence,
                sequence_id=sequence_id,
                generator=latent_neighbor_generator,
                n_neighbors=N_LATENT_NEIGHBORS,
                perturbation_std=LATENT_PERTURBATION_STD,
                radius=radius,
            )
            rt_row.update(smooth_summary)

            roundtrip_rows.append(rt_row)
            local_smoothness_detail_rows.extend(smooth_details)

        # ------------------------------------------------------------
        # Re-score original + scrambles together.
        #
        # Important:
        # If blackbox_fc uses batch min-max normalization, the fairest
        # composition-preserving control is to score the original peptide
        # and its scrambled controls in the same call.
        # ------------------------------------------------------------
        scored_df = blackbox_fc(candidate_peptides)

        scored_pep_col = get_peptide_column(scored_df)
        scored_df[scored_pep_col] = scored_df[scored_pep_col].astype(str).str.upper()

        scored_df["original_peptide"] = original_peptide
        scored_df["control_type"] = np.where(
            scored_df[scored_pep_col] == original_peptide,
            "optimized_original",
            "scrambled_control",
        )
        scored_df["scramble_group_id"] = peptide_index

        all_scored_rows.append(scored_df)

        original_rows = scored_df[scored_df["control_type"] == "optimized_original"].copy()
        scrambled_rows = scored_df[scored_df["control_type"] == "scrambled_control"].copy()

        if len(original_rows) != 1:
            print(
                f"[WARNING] Expected exactly one original row for {original_peptide}, "
                f"found {len(original_rows)}."
            )
            continue

        original_row = original_rows.iloc[0]

        original_obj = original_row[OBJECTIVE_COLS].to_numpy(dtype=float)
        scrambled_obj = scrambled_rows[OBJECTIVE_COLS].to_numpy(dtype=float)
        n_scrambles = len(scrambled_rows)

        n_scrambled_dominating_original = 0
        n_scrambled_dominated_by_original = 0

        for i in range(n_scrambles):
            s_obj = scrambled_obj[i]
            if dominates_maximize(s_obj, original_obj):
                n_scrambled_dominating_original += 1
            if dominates_maximize(original_obj, s_obj):
                n_scrambled_dominated_by_original += 1

        summary = {
            "original_peptide": original_peptide,
            "n_scrambles_scored": n_scrambles,
            "original_chelation_sub": float(original_row["chelation_sub"]),
            "original_solubility_sub": float(original_row["solubility_sub"]),
            "original_stability_sub": float(original_row["stability_sub"]),
            "original_expression_sub": float(original_row["expression_sub"]),
            "original_final_score": float(original_row["final_score"]),
            "scrambled_mean_final_score": float(scrambled_rows["final_score"].mean()),
            "scrambled_std_final_score": float(scrambled_rows["final_score"].std(ddof=1)),
            "scrambled_max_final_score": float(scrambled_rows["final_score"].max()),
            "scrambled_95pct_final_score": float(scrambled_rows["final_score"].quantile(0.95)),
            "original_final_score_z_vs_scrambled": safe_zscore(
                float(original_row["final_score"]),
                scrambled_rows["final_score"].to_numpy(dtype=float),
            ),
            "empirical_p_scrambled_ge_original_final_score": empirical_p_value_greater_equal(
                float(original_row["final_score"]),
                scrambled_rows["final_score"].to_numpy(dtype=float),
            ),
            "n_scrambled_dominating_original": int(n_scrambled_dominating_original),
            "n_scrambled_dominated_by_original": int(n_scrambled_dominated_by_original),
            "fraction_scrambled_dominated_by_original": (
                n_scrambled_dominated_by_original / n_scrambles if n_scrambles > 0 else np.nan
            ),
            "fraction_scrambled_dominating_original": (
                n_scrambled_dominating_original / n_scrambles if n_scrambles > 0 else np.nan
            ),
        }

        for col in OBJECTIVE_COLS:
            original_value = float(original_row[col])
            scrambled_values = scrambled_rows[col].to_numpy(dtype=float)
            summary[f"scrambled_mean_{col}"] = float(np.nanmean(scrambled_values))
            summary[f"scrambled_std_{col}"] = float(np.nanstd(scrambled_values, ddof=1))
            summary[f"scrambled_max_{col}"] = float(np.nanmax(scrambled_values))
            summary[f"empirical_p_scrambled_ge_original_{col}"] = empirical_p_value_greater_equal(
                original_value,
                scrambled_values,
            )

        summary_rows.append(summary)

    # ============================================================
    # Save detailed and summary outputs
    # ============================================================

    if len(all_scored_rows) == 0:
        raise RuntimeError("No scored rows were generated.")

    all_scores_df = pd.concat(all_scored_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    all_scores_path = out_dir / "scrambling_control_all_scores.csv"
    summary_path = out_dir / "scrambling_control_summary.csv"

    all_scores_df.to_csv(all_scores_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    roundtrip_df = pd.DataFrame(roundtrip_rows)
    local_smoothness_detail_df = pd.DataFrame(local_smoothness_detail_rows)

    roundtrip_metric_columns = [
        "source_to_p1_edit_distance",
        "source_to_p1_edit_distance_normalized",
        "peptide_roundtrip_edit_distance",
        "peptide_roundtrip_edit_distance_normalized",
        "epsilon_roundtrip_l2",
        "epsilon_roundtrip_cosine",
        "epsilon1_norm",
        "epsilon2_norm",
        "standardized_h_reconstruction_l2",
        "standardized_h_reconstruction_cosine",
        "h0_norm",
        "h0_reconstructed_norm",
        "local_neighbor_edit_distance_mean",
        "local_neighbor_edit_distance_std",
        "local_neighbor_edit_distance_min",
        "local_neighbor_edit_distance_max",
        "local_neighbor_normalized_edit_distance_mean",
        "local_neighbor_identical_fraction",
        "local_neighbor_unique_peptides",
    ]

    roundtrip_with_average_df = append_overall_average_row(
        roundtrip_df,
        metric_columns=roundtrip_metric_columns,
        label_column="source_peptide",
    )

    roundtrip_path = out_dir / "diffusion_roundtrip_sequence_diagnostics.csv"
    local_smoothness_detail_path = out_dir / "diffusion_epsilon_local_smoothness_neighbors.csv"
    roundtrip_average_path = out_dir / "diffusion_roundtrip_and_smoothness_overall_averages.csv"

    roundtrip_with_average_df.to_csv(roundtrip_path, index=False)
    local_smoothness_detail_df.to_csv(local_smoothness_detail_path, index=False)

    overall_average_row = {
        "n_sequences_evaluated": int(len(roundtrip_df)),
        "mean_source_to_p1_edit_distance": float(roundtrip_df["source_to_p1_edit_distance"].mean()),
        "mean_peptide_roundtrip_edit_distance": float(roundtrip_df["peptide_roundtrip_edit_distance"].mean()),
        "mean_peptide_roundtrip_edit_distance_normalized": float(roundtrip_df["peptide_roundtrip_edit_distance_normalized"].mean()),
        "mean_epsilon_roundtrip_l2": float(roundtrip_df["epsilon_roundtrip_l2"].mean()),
        "mean_epsilon_roundtrip_cosine": float(roundtrip_df["epsilon_roundtrip_cosine"].mean()),
        "mean_epsilon1_norm": float(roundtrip_df["epsilon1_norm"].mean()),
        "mean_standardized_h_reconstruction_l2": float(roundtrip_df["standardized_h_reconstruction_l2"].mean()),
        "mean_standardized_h_reconstruction_cosine": float(roundtrip_df["standardized_h_reconstruction_cosine"].mean()),
        "mean_local_neighbor_edit_distance_to_p1": float(roundtrip_df["local_neighbor_edit_distance_mean"].mean()),
        "mean_local_neighbor_normalized_edit_distance_to_p1": float(roundtrip_df["local_neighbor_normalized_edit_distance_mean"].mean()),
        "mean_local_neighbor_identical_fraction": float(roundtrip_df["local_neighbor_identical_fraction"].mean()),
        "mean_local_neighbor_unique_peptides": float(roundtrip_df["local_neighbor_unique_peptides"].mean()),
        "n_latent_neighbors_per_sequence": int(N_LATENT_NEIGHBORS),
        "latent_perturbation_std": float(LATENT_PERTURBATION_STD),
        "local_renormalize_to_sphere": bool(LOCAL_RENORMALIZE_TO_SPHERE),
        "checkpoint": DIFFUSION_CHECKPOINT,
    }
    pd.DataFrame([overall_average_row]).to_csv(roundtrip_average_path, index=False)

    print(f"\nSaved detailed scores: {all_scores_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved per-sequence diffusion round-trip diagnostics: {roundtrip_path}")
    print(f"Saved local epsilon-neighbor diagnostics: {local_smoothness_detail_path}")
    print(f"Saved overall diffusion round-trip/smoothness averages: {roundtrip_average_path}")

    print("\n=== Diffusion round-trip / epsilon smoothness averages ===")
    for key, value in overall_average_row.items():
        if key != "checkpoint":
            print(f"{key}: {value}")

    # ============================================================
    # Plot 1: final score boxplot by original peptide
    # ============================================================

    plt.figure(figsize=(12, 6))
    peptides_order = (
        all_scores_df["original_peptide"]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    box_data = []
    for pep in peptides_order:
        vals = all_scores_df[
            (all_scores_df["original_peptide"] == pep)
            & (all_scores_df["control_type"] == "scrambled_control")
        ]["final_score"].to_numpy(dtype=float)
        box_data.append(vals)

    plt.boxplot(box_data, labels=peptides_order, showfliers=True)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Final score")
    plt.title("Scrambling control: final-score distribution of scrambled peptides")

    for i, pep in enumerate(peptides_order, start=1):
        orig_rows_for_plot = all_scores_df[
            (all_scores_df["original_peptide"] == pep)
            & (all_scores_df["control_type"] == "optimized_original")
        ]["final_score"]
        if orig_rows_for_plot.empty:
            print(f"[WARNING] Skipping original-score marker for {pep}: no optimized_original row was found.")
            continue
        orig_val = float(orig_rows_for_plot.iloc[0])
        plt.scatter(i, orig_val, marker="D", s=70, label="Original" if i == 1 else None)

    plt.legend()
    plt.tight_layout()
    plot_path = out_dir / "scrambling_control_final_score_boxplot.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved plot: {plot_path}")

    # ============================================================
    # Plot 2: original vs scrambled mean final score
    # ============================================================

    plt.figure(figsize=(12, 6))
    x = np.arange(len(summary_df))
    width = 0.35

    plt.bar(
        x - width / 2,
        summary_df["original_final_score"],
        width,
        label="Original optimized peptide",
    )
    plt.bar(
        x + width / 2,
        summary_df["scrambled_mean_final_score"],
        width,
        yerr=summary_df["scrambled_std_final_score"],
        capsize=3,
        label="Scrambled controls mean ± SD",
    )
    plt.xticks(x, summary_df["original_peptide"], rotation=45, ha="right")
    plt.ylabel("Final score")
    plt.title("Original optimized peptides vs scrambled controls")
    plt.legend()
    plt.tight_layout()
    plot_path = out_dir / "original_vs_scrambled_mean_final_score.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved plot: {plot_path}")

    # ============================================================
    # Plot 3: empirical p-values
    # ============================================================

    plt.figure(figsize=(12, 5))
    plt.bar(
        summary_df["original_peptide"],
        summary_df["empirical_p_scrambled_ge_original_final_score"],
    )
    plt.axhline(0.05, linestyle="--", linewidth=1, label="p = 0.05")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Empirical p-value")
    plt.title("Probability that scrambled controls score at least as high as original")
    plt.legend()
    plt.tight_layout()
    plot_path = out_dir / "empirical_p_values_final_score.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved plot: {plot_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
