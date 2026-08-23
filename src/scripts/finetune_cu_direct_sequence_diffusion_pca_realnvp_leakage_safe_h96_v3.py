# ============================================================
# Cu fine-tuning for direct peptide sequence diffusion + noise-space flow
#
# Input checkpoint expected from:
#   pretrain_peptide_direct_sequence_diffusion_chain_mapped_parts_h32.py
# or compatible direct diffusion pretraining code.
#
# This script is intentionally NOT compatible with the older GRU-VAE latent
# diffusion checkpoints. The direct diffusion checkpoint contains:
#   model_state_dict, diffusion_config, bo_coordinate_dim=200
# and no VAE encoder/decoder state.
#
# Revised matched path:
#   peptide x0 -> deterministic DDIM inversion epsilon0
#   eps0 -> RealNVP flow -> epsK
#   epsK -> deterministic DDIM direct diffusion -> peptide logits/scores
#
# Leakage control:
#   Cu peptides reuse pretraining split labels only when an exact sequence has
#   one unique split assignment. Exact sequences found in multiple pretraining
#   splits are excluded from all fine-tuning partitions. No random fallback split
#   is allowed; unmapped/conflicting peptides are dropped or error.
#
# BO export:
#   epsilon0_* = preimage coordinates before flow
#   epsilonK_* = flow-transformed Cu-aligned coordinates after flow
#   Use epsilonK_* for GP/BO after flow.
# ============================================================

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
from torch.utils.data import DataLoader, TensorDataset

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_I = {a: i for i, a in enumerate(AA)}
I_TO_AA = {i: a for i, a in enumerate(AA)}
SEQ_LEN = 10
VOCAB = 20
BO_DIM = SEQ_LEN * VOCAB


@dataclass
class SequenceDiffusionConfig:
    hidden_size: int = 96
    n_layers: int = 2
    dropout: float = 0.0
    time_dim: int = 32
    train_steps: int = 100
    beta_start: float = 1e-5
    beta_end: float = 8e-3


# ============================================================
# Basic utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_load_full(path: str, map_location):
    """
    Load full training checkpoints across PyTorch versions.

    PyTorch versions differ in the default value of weights_only. These
    checkpoints intentionally contain metadata dictionaries in addition to
    tensor state_dicts, so we explicitly request a full load when supported.
    """
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=map_location,
        )


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
        pep = clean_peptide(pep)
        if pep is None:
            raise ValueError("Invalid peptide while one-hot encoding.")
        for t, aa in enumerate(pep):
            x[n, t, AA_TO_I[aa]] = 1.0
    return x


def decode_tensor_to_peptides(x: torch.Tensor) -> List[str]:
    idx = x.argmax(dim=-1).detach().cpu().tolist()
    return ["".join(I_TO_AA[int(i)] for i in row) for row in idx]


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
            current.append(min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + int(ca != cb)))
        previous = current
    return int(previous[-1])


def pearson_corr_safe(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    if a.numel() < 2:
        return float("nan")
    if float(a.std()) < 1e-12 or float(b.std()) < 1e-12:
        return float("nan")
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1].item())


def project_to_sphere(eps: torch.Tensor, radius: float) -> torch.Tensor:
    return eps / eps.norm(dim=-1, keepdim=True).clamp(min=1e-8) * float(radius)


# ============================================================
# Direct sequence diffusion architecture from pretraining
# ============================================================

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


class DirectSequenceDenoiser(nn.Module):
    def __init__(self, cfg: SequenceDiffusionConfig):
        super().__init__()
        self.cfg = cfg
        self.x_proj = nn.Linear(VOCAB, cfg.hidden_size)
        self.time_embedding = SinusoidalTimeEmbedding(cfg.time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.time_dim, cfg.hidden_size),
            nn.SiLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
        )
        self.gru = nn.GRU(
            input_size=cfg.hidden_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.n_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.n_layers > 1 else 0.0,
        )
        self.out_norm = nn.LayerNorm(cfg.hidden_size)
        self.noise_head = nn.Linear(cfg.hidden_size, VOCAB)
        self.x0_logits_head = nn.Linear(cfg.hidden_size, VOCAB)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.x_proj(x_t)
        h = h + self.time_mlp(self.time_embedding(t)).unsqueeze(1)
        h, _ = self.gru(h)
        h = F.silu(self.out_norm(h))
        return {
            "predicted_noise": self.noise_head(h),
            "x0_logits": self.x0_logits_head(h),
        }


class DirectSequenceDiffusion(nn.Module):
    def __init__(self, cfg: SequenceDiffusionConfig):
        super().__init__()
        self.cfg = cfg
        self.denoiser = DirectSequenceDenoiser(cfg)
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.train_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    @property
    def bo_dim(self) -> int:
        return SEQ_LEN * VOCAB

    @property
    def bo_radius(self) -> float:
        return math.sqrt(self.bo_dim)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1)
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise

    def training_loss(self, x0: torch.Tensor, recon_ce_weight: float, x0_mse_weight: float) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch_size = x0.size(0)
        t = torch.randint(0, self.cfg.train_steps, (batch_size,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        out = self.denoiser(x_t, t)
        predicted_noise = out["predicted_noise"]
        x0_logits = out["x0_logits"]
        eps_mse = F.mse_loss(predicted_noise, noise)
        target = x0.argmax(dim=-1)
        recon_ce = F.cross_entropy(x0_logits.reshape(-1, VOCAB), target.reshape(-1))
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1)
        x0_from_eps = (x_t - torch.sqrt(1.0 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
        x0_mse = F.mse_loss(x0_from_eps, x0)
        loss = eps_mse + float(recon_ce_weight) * recon_ce + float(x0_mse_weight) * x0_mse
        token_acc = (x0_logits.argmax(dim=-1) == target).float().mean()
        return loss, {
            "loss": float(loss.detach().cpu()),
            "epsilon_mse": float(eps_mse.detach().cpu()),
            "recon_ce": float(recon_ce.detach().cpu()),
            "x0_mse": float(x0_mse.detach().cpu()),
            "token_acc": float(token_acc.detach().cpu()),
        }

    @torch.no_grad()
    def ddim_sample(self, base_noise: torch.Tensor, inference_steps: int = 20) -> torch.Tensor:
        return ddim_sample_with_grad(self, base_noise, inference_steps=inference_steps)


    @torch.no_grad()
    def ddim_invert(self, x0: torch.Tensor, inference_steps: int = 50) -> torch.Tensor:
        self.eval()
        if x0.ndim != 3 or x0.shape[1:] != (SEQ_LEN, VOCAB):
            raise ValueError(
                f"Expected x0 [B,{SEQ_LEN},{VOCAB}], got {tuple(x0.shape)}"
            )
        n_train_steps = self.cfg.train_steps
        inference_steps = min(max(int(inference_steps), 1), n_train_steps)
        timesteps = torch.linspace(
            0, n_train_steps - 1, inference_steps, device=x0.device
        ).round().long()
        timesteps = torch.unique_consecutive(timesteps)

        x = x0
        for step_index, t_scalar in enumerate(timesteps[:-1]):
            t = torch.full(
                (x.size(0),), int(t_scalar.item()),
                device=x.device, dtype=torch.long
            )
            out = self.denoiser(x, t)
            predicted_noise = out["predicted_noise"]
            alpha_bar_t = self.alpha_bars[t_scalar]
            x0_pred = (
                x - torch.sqrt(1.0 - alpha_bar_t) * predicted_noise
            ) / torch.sqrt(alpha_bar_t)

            t_next_scalar = timesteps[step_index + 1]
            alpha_bar_next = self.alpha_bars[t_next_scalar]
            x = (
                torch.sqrt(alpha_bar_next) * x0_pred
                + torch.sqrt(1.0 - alpha_bar_next) * predicted_noise
            )
        return x


def ddim_sample_with_grad(model: DirectSequenceDiffusion, base_noise: torch.Tensor, inference_steps: int = 20) -> torch.Tensor:
    if base_noise.ndim == 2:
        x = base_noise.view(-1, SEQ_LEN, VOCAB)
    elif base_noise.ndim == 3:
        x = base_noise
    else:
        raise ValueError(f"Expected base_noise [B,200] or [B,10,20], got {tuple(base_noise.shape)}")

    n_train_steps = model.cfg.train_steps
    inference_steps = min(max(int(inference_steps), 1), n_train_steps)
    timesteps = torch.linspace(n_train_steps - 1, 0, inference_steps, device=x.device).round().long()
    timesteps = torch.unique_consecutive(timesteps)

    for step_index, t_scalar in enumerate(timesteps):
        t = torch.full((x.size(0),), int(t_scalar.item()), device=x.device, dtype=torch.long)
        out = model.denoiser(x, t)
        predicted_noise = out["predicted_noise"]
        alpha_bar_t = model.alpha_bars[t_scalar]
        x0_pred = (x - torch.sqrt(1.0 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
        if step_index == len(timesteps) - 1:
            x = x0_pred
            break
        t_prev_scalar = timesteps[step_index + 1]
        alpha_bar_prev = model.alpha_bars[t_prev_scalar]
        x = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1.0 - alpha_bar_prev) * predicted_noise
    return x


# ============================================================
# RealNVP normalizing flow in 200-D diffusion noise space
# ============================================================

class AffineCouplingBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, mask: torch.Tensor, max_scale: float = 1.5):
        super().__init__()
        self.dim = int(dim)
        self.max_scale = float(max_scale)
        self.register_buffer("mask", mask.float())
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * dim),
        )
        # Initialize near identity.
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
    def __init__(self, dim: int = BO_DIM, n_layers: int = 6, hidden_dim: int = 256, max_scale: float = 1.5):
        super().__init__()
        masks = []
        for layer in range(n_layers):
            mask_np = np.zeros(dim, dtype=np.float32)
            mask_np[layer % 2 :: 2] = 1.0
            masks.append(torch.tensor(mask_np))
        self.blocks = nn.ModuleList([
            AffineCouplingBlock(dim, hidden_dim, mask=m, max_scale=max_scale) for m in masks
        ])

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


class ScoreHead(nn.Module):
    def __init__(self, dim: int = BO_DIM, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, eps: torch.Tensor) -> torch.Tensor:
        return self.net(eps).squeeze(-1)



# ============================================================
# Leakage-safe split reuse + train-only PCA/whitening
# ============================================================

def discover_pretraining_parts(parts_dir: str, part_glob: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(parts_dir, part_glob)))
    files = [f for f in files if "failures" not in os.path.basename(f).lower()]
    if not files:
        raise FileNotFoundError(
            f"No pretraining part files found in {parts_dir!r} using {part_glob!r}"
        )
    return files


def build_pretraining_peptide_split_map(
    files: List[str],
    peptide_col: str,
    split_col: str,
    chunksize: int,
) -> Tuple[Dict[str, str], pd.DataFrame]:
    """
    Build an exact-sequence -> split map from the pretraining parts.

    IMPORTANT
    ---------
    The attached pretraining parts can contain the SAME peptide sequence under
    more than one row/group split. A peptide-level map therefore cannot safely
    choose one of those labels without reintroducing exact-sequence leakage.

    For fine-tuning we use the conservative rule:
        * peptide occurs in exactly one split -> retain that split
        * peptide occurs in >1 split        -> EXCLUDE it from all fine-tuning
          train/validation/test partitions

    This guarantees zero exact peptide overlap across the fine-tuning splits,
    even if the original pretraining row/group split contained sequence-level
    conflicts.
    """
    peptide_splits: Dict[str, set] = {}

    for file_i, path in enumerate(files, 1):
        header = pd.read_csv(path, nrows=0).columns
        if peptide_col not in header or split_col not in header:
            raise KeyError(
                f"{path} must contain {peptide_col!r} and {split_col!r}"
            )

        for chunk in pd.read_csv(
            path,
            usecols=[peptide_col, split_col],
            chunksize=chunksize,
        ):
            for p_raw, s_raw in zip(
                chunk[peptide_col].tolist(),
                chunk[split_col].tolist(),
            ):
                p = clean_peptide(p_raw)
                if p is None:
                    continue

                s = str(s_raw).strip().lower()
                if not s or s == "nan":
                    continue

                peptide_splits.setdefault(p, set()).add(s)

        if file_i % 10 == 0 or file_i == len(files):
            n_conf = sum(len(v) > 1 for v in peptide_splits.values())
            print(
                f"[SPLIT-MAP] scanned {file_i}/{len(files)} parts; "
                f"unique peptides={len(peptide_splits)} "
                f"cross-split exact-sequence conflicts={n_conf}"
            )

    split_map: Dict[str, str] = {}
    conflict_rows = []

    for peptide, splits in peptide_splits.items():
        splits_sorted = sorted(splits)
        if len(splits_sorted) == 1:
            split_map[peptide] = splits_sorted[0]
        else:
            conflict_rows.append({
                "peptide": peptide,
                "n_splits": len(splits_sorted),
                "splits": "|".join(splits_sorted),
            })

    conflicts_df = pd.DataFrame(
        conflict_rows,
        columns=["peptide", "n_splits", "splits"],
    )

    print(
        "[SPLIT-MAP] final unique safe peptide mappings="
        f"{len(split_map)}; excluded exact-sequence conflicts={len(conflicts_df)}"
    )

    return split_map, conflicts_df


def fit_pca_whitener(x: np.ndarray, k: int, eps: float = 1e-6) -> Dict[str, np.ndarray]:
    if x.ndim != 2:
        raise ValueError(f"Expected 2-D matrix, got {x.shape}")
    n, d = x.shape
    k = min(int(k), d, max(1, n - 1))
    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    components = vt[:k].astype(np.float32)
    scores = xc @ components.T
    score_std = np.maximum(
        scores.std(axis=0, keepdims=True).astype(np.float32), eps
    )
    explained = (s ** 2) / max(1, n - 1)
    total = max(float(explained.sum()), eps)
    evr = (explained[:k] / total).astype(np.float32)
    return {
        "mean": mean.astype(np.float32),
        "components": components,
        "score_std": score_std,
        "explained_variance_ratio": evr,
    }


def pca_project(epsilon_flat: np.ndarray, pca: Dict[str, np.ndarray]) -> np.ndarray:
    return ((epsilon_flat - pca["mean"]) @ pca["components"].T) / pca["score_std"]


def pca_inverse(z_bo: np.ndarray, pca: Dict[str, np.ndarray]) -> np.ndarray:
    return (z_bo * pca["score_std"]) @ pca["components"] + pca["mean"]


def geometry_diagnostics_np(x: np.ndarray, eps: float = 1e-8) -> Dict[str, float]:
    if x.ndim != 2 or x.shape[0] < 2:
        return {}
    xc = x.astype(np.float64) - x.astype(np.float64).mean(axis=0, keepdims=True)
    cov = (xc.T @ xc) / max(1, x.shape[0] - 1)
    eig = np.linalg.eigvalsh(cov).clip(min=0.0)
    total = max(float(eig.sum()), eps)
    pr = total * total / max(float((eig ** 2).sum()), eps)
    eig_desc = eig[::-1]
    frac = eig_desc / total
    cum = np.cumsum(frac)

    z = xc / np.maximum(xc.std(axis=0, keepdims=True), eps)
    corr = (z.T @ z) / float(z.shape[0])
    off = corr - np.diag(np.diag(corr))
    denom = max(1, z.shape[1] * (z.shape[1] - 1))
    return {
        "dimension": int(x.shape[1]),
        "effective_dim_pr": float(pr),
        "pc1_variance_fraction": float(frac[0]),
        "pcs90": int(np.searchsorted(cum, 0.90) + 1),
        "pcs95": int(np.searchsorted(cum, 0.95) + 1),
        "mean_abs_offdiag_corr": float(np.abs(off).sum() / denom),
        "max_abs_offdiag_corr": float(np.abs(off).max()),
    }

# ============================================================
# Cu data: corrected chain-mapped CSV, optional proxy score
# ============================================================

def parse_binary_labels(value: object) -> List[int]:
    if pd.isna(value):
        return []
    s = str(value).replace(" ", "").strip()
    return [1 if ch == "1" else 0 for ch in s if ch in {"0", "1"}]


def hydropathy_gravy(pep: str) -> float:
    kd = {
        "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
        "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
        "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
        "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
    }
    return float(np.mean([kd[a] for a in pep]))


def compute_cu_proxy_score(pep: str, labels: List[int]) -> float:
    # Conservative fallback only when final_score is absent.
    # It rewards Cu-relevant ligating residues and chain-mapped label density,
    # while mildly penalizing very hydrophobic peptides.
    pep = clean_peptide(pep) or ""
    if not pep:
        return float("nan")
    label_density = float(sum(labels) / max(len(labels), 1)) if labels else 0.0
    h_frac = pep.count("H") / SEQ_LEN
    c_frac = pep.count("C") / SEQ_LEN
    de_frac = (pep.count("D") + pep.count("E")) / SEQ_LEN
    polar_frac = sum(pep.count(a) for a in "STNQKRDEH") / SEQ_LEN
    motif_bonus = 0.0
    for motif in ["HXH", "HXXH", "CXC", "CXXC", "HCH", "CHC"]:
        # Motifs with X wildcards.
        if len(motif) == 3:
            found = any((motif[0] == pep[i] and motif[2] == pep[i + 2]) for i in range(SEQ_LEN - 2)) if "X" in motif else motif in pep
        elif len(motif) == 4:
            found = any((motif[0] == pep[i] and motif[3] == pep[i + 3]) for i in range(SEQ_LEN - 3))
        else:
            found = motif in pep
        if found:
            motif_bonus += 0.05
    chelation = min(1.0, 0.40 * label_density + 0.25 * h_frac + 0.20 * c_frac + 0.10 * de_frac + motif_bonus)
    solubility = float(np.clip((polar_frac + max(0.0, -hydropathy_gravy(pep)) / 4.5) / 2.0, 0.0, 1.0))
    stability = 0.75  # fixed-length 10 windows; leave neutral rather than overclaiming.
    expression = 0.70
    return float(np.clip(0.50 * chelation + 0.30 * solubility + 0.10 * stability + 0.10 * expression, 0.0, 1.0))


def load_cu_dataset(
    args: argparse.Namespace,
    split_map: Dict[str, str],
) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[str], pd.DataFrame]:
    df = pd.read_csv(args.cu_csv)

    if args.peptide_col not in df.columns:
        raise KeyError(
            f"Peptide column {args.peptide_col!r} not found in Cu CSV."
        )

    if args.score_col not in df.columns:
        if not args.auto_score_if_missing:
            raise KeyError(
                f"Score column {args.score_col!r} was not found in {args.cu_csv}."
            )
        print(
            f"[WARN] score column {args.score_col!r} absent; "
            "computing the legacy proxy score."
        )
        if args.labels_col not in df.columns:
            raise KeyError(
                f"Cannot compute proxy score because {args.labels_col!r} is missing."
            )
        df["cu_proxy_score"] = [
            compute_cu_proxy_score(str(p), parse_binary_labels(lbl))
            for p, lbl in zip(
                df[args.peptide_col].tolist(),
                df[args.labels_col].tolist(),
            )
        ]
        args.score_col = "cu_proxy_score"

    work = df.copy()
    work["_clean_peptide"] = work[args.peptide_col].map(clean_peptide)
    work["_score_used"] = pd.to_numeric(work[args.score_col], errors="coerce")
    work = work.dropna(subset=["_clean_peptide", "_score_used"]).copy()

    before = len(work)
    unique_before = work["_clean_peptide"].nunique()

    if args.duplicate_policy == "error" and before != unique_before:
        raise RuntimeError(
            f"Cu CSV contains {before - unique_before} duplicate peptide rows."
        )
    elif args.duplicate_policy == "first":
        work = work.drop_duplicates("_clean_peptide", keep="first").copy()
    elif args.duplicate_policy == "mean":
        work = (
            work.groupby("_clean_peptide", as_index=False)
            .agg({"_score_used": "mean"})
        )
    elif args.duplicate_policy not in {"error", "first", "mean"}:
        raise ValueError(args.duplicate_policy)

    work["_split_used"] = work["_clean_peptide"].map(split_map)

    unmapped = work[work["_split_used"].isna()].copy()
    if len(unmapped):
        path = os.path.join(
            args.out_dir,
            "cu_peptides_unmapped_from_pretraining_parts.csv",
        )
        unmapped.to_csv(path, index=False)
        msg = (
            f"{len(unmapped)} Cu peptides do not have a safe unique pretraining "
            "split assignment (either absent from the pretraining parts or "
            "excluded because the exact sequence appeared in multiple splits). "
            f"Saved to {path}"
        )
        if args.unmapped_policy == "error":
            raise RuntimeError(msg)
        print("[WARN]", msg, "Dropping them.")
        work = work.dropna(subset=["_split_used"]).copy()

    allowed = {
        args.train_split.lower(),
        args.validation_split.lower(),
        args.test_split.lower(),
    }
    unexpected = sorted(set(work["_split_used"]) - allowed)
    if unexpected:
        raise RuntimeError(
            f"Unexpected split labels from pretraining parts: {unexpected}"
        )

    work = work.reset_index(drop=True)

    # Explicit exact-sequence overlap verification after all exclusions.
    split_sets = {
        s: set(
            work.loc[
                work["_split_used"].astype(str).str.lower().eq(s),
                "_clean_peptide",
            ].tolist()
        )
        for s in [
            args.train_split.lower(),
            args.validation_split.lower(),
            args.test_split.lower(),
        ]
    }
    exact_train_val = len(
        split_sets[args.train_split.lower()]
        & split_sets[args.validation_split.lower()]
    )
    exact_train_test = len(
        split_sets[args.train_split.lower()]
        & split_sets[args.test_split.lower()]
    )
    exact_val_test = len(
        split_sets[args.validation_split.lower()]
        & split_sets[args.test_split.lower()]
    )
    if exact_train_val or exact_train_test or exact_val_test:
        raise RuntimeError(
            "Exact peptide overlap remains after conflict exclusion: "
            f"train-val={exact_train_val}, train-test={exact_train_test}, "
            f"validation-test={exact_val_test}"
        )

    counts = work["_split_used"].value_counts().to_dict()
    for s in allowed:
        if counts.get(s, 0) == 0:
            raise RuntimeError(f"No Cu peptides remain in required split {s!r}")

    audit = {
        "rows_before_dedup": int(before),
        "unique_before_dedup": int(unique_before),
        "duplicate_rows": int(before - unique_before),
        "duplicate_policy": args.duplicate_policy,
        "unmapped_peptides": int(len(unmapped)),
        "unmapped_policy": args.unmapped_policy,
        "final_rows": int(len(work)),
        "split_counts": {str(k): int(v) for k, v in counts.items()},
        "split_source": "pretraining part CSV split column",
        "random_resplitting_used": False,
        "exact_train_validation_overlap": int(exact_train_val),
        "exact_train_test_overlap": int(exact_train_test),
        "exact_validation_test_overlap": int(exact_val_test),
        "cross_split_conflicting_pretraining_sequences_excluded": True,
    }
    with open(
        os.path.join(args.out_dir, "cu_finetuning_data_audit.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(audit, f, indent=2)

    work.to_csv(
        os.path.join(args.out_dir, "cu_finetuning_manifest_mapped.csv"),
        index=False,
    )

    peptides = work["_clean_peptide"].tolist()
    scores = work["_score_used"].astype(float).tolist()
    splits = work["_split_used"].astype(str).tolist()
    x = onehot_encode_peptides(peptides)
    y = torch.tensor(scores, dtype=torch.float32)
    return peptides, x, y, splits, work


def make_loader(x: torch.Tensor, y: torch.Tensor, eps0: torch.Tensor, indices: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(x[indices], y[indices], eps0[indices], indices), batch_size=batch_size, shuffle=shuffle)


# ============================================================
# Checkpoint loading/saving
# ============================================================

def load_direct_diffusion_checkpoint(path: str, device: torch.device) -> Tuple[DirectSequenceDiffusion, SequenceDiffusionConfig, Dict]:
    ckpt = torch_load_full(path, map_location=device)
    expected_type = "all_metal_direct_peptide_sequence_diffusion_search_cost_sampling"
    ctype = ckpt.get("checkpoint_type", "")
    if ctype and ctype != expected_type:
        raise ValueError(
            f"Unexpected checkpoint_type={ctype!r}; expected {expected_type!r}"
        )
    if "model_state_dict" not in ckpt:
        raise KeyError(
            f"Checkpoint {path} does not contain model_state_dict. "
            "This script expects a direct sequence diffusion checkpoint, not a GRU-VAE latent diffusion checkpoint."
        )
    cfg_dict = ckpt.get("diffusion_config") or ckpt.get("config")
    if cfg_dict is None:
        raise KeyError(f"Checkpoint {path} is missing diffusion_config.")
    cfg = SequenceDiffusionConfig(**cfg_dict)
    model = DirectSequenceDiffusion(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    print(f"Loaded direct diffusion checkpoint: {path}")
    print("checkpoint_type:", ckpt.get("checkpoint_type"))
    print("epoch:", ckpt.get("epoch"))
    print("bo_coordinate_dim:", ckpt.get("bo_coordinate_dim", model.bo_dim))
    print("recommended_bo_radius:", ckpt.get("recommended_bo_radius", model.bo_radius))
    return model, cfg, ckpt


def append_history(path: str, row: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def save_checkpoint(path: str, model: DirectSequenceDiffusion, flow: RealNVPFlow, score_head: ScoreHead, optimizer: torch.optim.Optimizer, cfg: SequenceDiffusionConfig, args: argparse.Namespace, epoch: int, metrics: Dict[str, float], selection: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "checkpoint_type": "cu_direct_sequence_diffusion_noise_flow",
        "selection": selection,
        "epoch": int(epoch),
        "metrics": metrics,
        "model_state_dict": model.state_dict(),
        "flow_state_dict": flow.state_dict(),
        "score_head_state_dict": score_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "diffusion_config": asdict(cfg),
        "args": vars(args),
        "aa": AA,
        "seq_len": SEQ_LEN,
        "vocab": VOCAB,
        "bo_coordinate_space": "flow_transformed_direct_diffusion_noise_epsilonK",
        "bo_coordinate_shape": [SEQ_LEN, VOCAB],
        "bo_coordinate_dim": BO_DIM,
        "recommended_bo_radius": math.sqrt(BO_DIM),
        "decoder_path": "epsilon0_to_realnvp_flow_to_epsilonK_to_direct_ddim_to_argmax_peptide",
    }
    torch.save(payload, path)
    with open(path + ".json", "w", encoding="utf-8") as handle:
        json.dump({k: v for k, v in payload.items() if k not in {"model_state_dict", "flow_state_dict", "score_head_state_dict", "optimizer_state_dict"}}, handle, indent=2)


# ============================================================
# Epsilon preimage optimization
# ============================================================

@torch.no_grad()
def random_sphere(n: int, dim: int, radius: float, device: torch.device) -> torch.Tensor:
    eps = torch.randn(n, dim, device=device)
    return project_to_sphere(eps, radius)


def optimize_preimage_batch(model: DirectSequenceDiffusion, x: torch.Tensor, args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    radius = math.sqrt(BO_DIM)
    eps_init = random_sphere(x.size(0), BO_DIM, radius, device)
    eps = nn.Parameter(eps_init.detach().clone())
    opt = torch.optim.Adam([eps], lr=args.preimage_lr)
    target = x.argmax(dim=-1)

    # cuDNN RNN backward requires the GRU module to be in training mode whenever
    # gradients flow through it. The diffusion weights may still be frozen
    # through requires_grad=False; training mode here is only for the backward
    # graph from decoder loss to the epsilon preimage.
    was_training = model.training
    model.train()
    try:
        for _ in range(int(args.preimage_steps)):
            opt.zero_grad(set_to_none=True)
            eps_decode = project_to_sphere(eps, radius) if args.preimage_project_each_step else eps
            x0_hat = ddim_sample_with_grad(model, eps_decode, inference_steps=args.ddim_steps)
            ce = F.cross_entropy(x0_hat.reshape(-1, VOCAB), target.reshape(-1))
            prob = F.softmax(x0_hat, dim=-1)
            x0_mse = F.mse_loss(prob, x)
            anchor = F.mse_loss(eps, eps_init)
            sphere = (eps.norm(dim=-1).mean() - radius).pow(2)
            loss = args.preimage_ce_weight * ce + args.preimage_x0_mse_weight * x0_mse + args.preimage_anchor_weight * anchor + args.preimage_sphere_weight * sphere
            loss.backward()
            torch.nn.utils.clip_grad_norm_([eps], args.preimage_grad_clip)
            opt.step()
            if args.preimage_project_each_step:
                with torch.no_grad():
                    eps.copy_(project_to_sphere(eps, radius))
    finally:
        model.train(was_training)
    return project_to_sphere(eps.detach(), radius)


def precompute_or_load_epsilon_preimages(
    model: DirectSequenceDiffusion,
    x_all: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    """
    Consistent with pretraining: use deterministic DDIM inversion directly.
    No optimization of an arbitrary epsilon preimage is performed.
    """
    cache_path = args.preimage_cache
    if cache_path and os.path.exists(cache_path) and not args.recompute_preimages:
        payload = torch_load_full(cache_path, map_location="cpu")
        eps0 = payload["epsilon0"].float()
        if eps0.shape != (x_all.size(0), BO_DIM):
            raise ValueError(
                f"Cached epsilon shape {tuple(eps0.shape)} does not match "
                f"{(x_all.size(0), BO_DIM)}"
            )
        print(f"Loaded cached DDIM-inverted epsilon: {cache_path}")
        return eps0

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, x_all.size(0), args.preimage_batch_size):
            end = min(start + args.preimage_batch_size, x_all.size(0))
            x = x_all[start:end].to(device)
            eps = model.ddim_invert(
                x, inference_steps=args.ddim_steps
            ).reshape(x.size(0), -1)

            if args.project_inverted_epsilon_to_sphere:
                eps = project_to_sphere(eps, model.bo_radius)

            chunks.append(eps.cpu())
            if (start // args.preimage_batch_size + 1) % max(
                1, args.preimage_log_every
            ) == 0:
                print(f"[DDIM-INVERT] rows {end}/{x_all.size(0)}")

    eps0 = torch.cat(chunks, dim=0).float()

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        torch.save({
            "epsilon0": eps0,
            "ddim_steps": int(args.ddim_steps),
            "projected_to_sphere":
                bool(args.project_inverted_epsilon_to_sphere),
            "checkpoint": args.init_checkpoint,
        }, cache_path)
        print(f"Saved DDIM-inverted epsilon cache: {cache_path}")

    return eps0


# ============================================================
# Evaluation/export
# ============================================================

@torch.no_grad()
def decode_from_epsilon(model: DirectSequenceDiffusion, eps: torch.Tensor, args: argparse.Namespace) -> Tuple[torch.Tensor, List[str]]:
    x0_hat = model.ddim_sample(eps, inference_steps=args.ddim_steps)
    return x0_hat, decode_tensor_to_peptides(x0_hat)


@torch.no_grad()
def evaluate(model: DirectSequenceDiffusion, flow: RealNVPFlow, score_head: ScoreHead, loader: DataLoader, peptides: List[str], args: argparse.Namespace, device: torch.device) -> Dict[str, float]:
    model.eval(); flow.eval(); score_head.eval()
    sums = {"loss": 0.0, "recon_ce": 0.0, "x0_mse": 0.0, "score_mse": 0.0, "anchor_mse": 0.0, "sphere_mse": 0.0, "token_acc": 0.0}
    y_true_all: List[torch.Tensor] = []
    y_pred_all: List[torch.Tensor] = []
    edits_raw: List[int] = []
    edits_flow: List[int] = []
    decoded_flow_all: List[str] = []
    n_batches = 0
    radius = math.sqrt(BO_DIM)

    for x, y, eps0, idx in loader:
        x = x.to(device); y = y.to(device); eps0 = eps0.to(device)
        epsK_raw, _logdet = flow(eps0)
        epsK = project_to_sphere(epsK_raw, radius) if args.project_flow_output_to_sphere else epsK_raw
        x0_hat = model.ddim_sample(epsK, inference_steps=args.ddim_steps)
        target = x.argmax(dim=-1)
        recon_ce = F.cross_entropy(x0_hat.reshape(-1, VOCAB), target.reshape(-1))
        prob = F.softmax(x0_hat, dim=-1)
        x0_mse = F.mse_loss(prob, x)
        pred_score = score_head(epsK)
        score_mse = F.mse_loss(pred_score, y)
        anchor = F.mse_loss(epsK, eps0)
        sphere = (epsK.norm(dim=-1).mean() - radius).pow(2)
        loss = args.flow_recon_ce_weight * recon_ce + args.flow_x0_mse_weight * x0_mse + args.score_loss_weight * score_mse + args.flow_anchor_weight * anchor + args.sphere_loss_weight * sphere
        token_acc = (x0_hat.argmax(dim=-1) == target).float().mean()
        for k, v in [("loss", loss), ("recon_ce", recon_ce), ("x0_mse", x0_mse), ("score_mse", score_mse), ("anchor_mse", anchor), ("sphere_mse", sphere), ("token_acc", token_acc)]:
            sums[k] += float(v.detach().cpu())
        y_true_all.append(y.detach().cpu())
        y_pred_all.append(pred_score.detach().cpu())
        _, decoded_raw = decode_from_epsilon(model, project_to_sphere(eps0, radius), args)
        _, decoded_flow = decode_from_epsilon(model, epsK, args)
        idx_list = [int(i) for i in idx.detach().cpu().tolist()]
        source = [peptides[i] for i in idx_list]
        edits_raw.extend([levenshtein_edit_distance(a, b) for a, b in zip(source, decoded_raw)])
        edits_flow.extend([levenshtein_edit_distance(a, b) for a, b in zip(source, decoded_flow)])
        decoded_flow_all.extend(decoded_flow)
        n_batches += 1

    out = {k: v / max(1, n_batches) for k, v in sums.items()}
    out["score_pearson"] = pearson_corr_safe(torch.cat(y_pred_all), torch.cat(y_true_all)) if y_true_all else float("nan")
    out["raw_preimage_edit_mean"] = float(np.mean(edits_raw)) if edits_raw else float("nan")
    out["flow_decode_edit_mean"] = float(np.mean(edits_flow)) if edits_flow else float("nan")
    out["flow_decode_edit_median"] = float(np.median(edits_flow)) if edits_flow else float("nan")
    out["flow_edit_le_3_fraction"] = float(np.mean(np.array(edits_flow) <= 3)) if edits_flow else float("nan")
    out["flow_edit_le_5_fraction"] = float(np.mean(np.array(edits_flow) <= 5)) if edits_flow else float("nan")
    out["flow_decoded_unique_fraction"] = float(len(set(decoded_flow_all)) / max(len(decoded_flow_all), 1)) if decoded_flow_all else float("nan")
    out["n_batches"] = float(n_batches)
    return out


@torch.no_grad()
def generation_sanity(model: DirectSequenceDiffusion, flow: RealNVPFlow, args: argparse.Namespace, device: torch.device) -> Dict[str, object]:
    model.eval(); flow.eval()
    radius = math.sqrt(BO_DIM)
    eps0 = random_sphere(args.sanity_samples, BO_DIM, radius, device)
    epsK_raw, _ = flow(eps0)
    epsK = project_to_sphere(epsK_raw, radius) if args.project_flow_output_to_sphere else epsK_raw
    x0_hat = model.ddim_sample(epsK, inference_steps=args.ddim_steps)
    peptides = decode_tensor_to_peptides(x0_hat)
    counts: Dict[str, int] = {}
    for p in peptides:
        counts[p] = counts.get(p, 0) + 1
    return {
        "n_samples": int(args.sanity_samples),
        "unique_fraction": float(len(set(peptides)) / max(len(peptides), 1)),
        "dominant_fraction": float(max(counts.values()) / max(len(peptides), 1)) if counts else float("nan"),
        "epsilon0_radius_mean": float(eps0.norm(dim=-1).mean().cpu()),
        "epsilonK_radius_mean": float(epsK.norm(dim=-1).mean().cpu()),
        "first_20_peptides": peptides[:20],
    }


@torch.no_grad()
def export_bo_coordinates(path: str, model: DirectSequenceDiffusion, flow: RealNVPFlow, score_head: ScoreHead, peptides: List[str], x_all: torch.Tensor, y_all: torch.Tensor, splits: List[str], eps0_all: torch.Tensor, pca: Dict[str, np.ndarray], args: argparse.Namespace, device: torch.device) -> None:
    model.eval(); flow.eval(); score_head.eval()
    rows: List[Dict[str, object]] = []
    radius = math.sqrt(BO_DIM)
    for start in range(0, x_all.size(0), args.batch_size):
        end = min(start + args.batch_size, x_all.size(0))
        x = x_all[start:end].to(device)
        y = y_all[start:end].to(device)
        eps0 = eps0_all[start:end].to(device)
        epsK_raw, logdet = flow(eps0)
        epsK = project_to_sphere(epsK_raw, radius) if args.project_flow_output_to_sphere else epsK_raw
        x0_raw = model.ddim_sample(project_to_sphere(eps0, radius), inference_steps=args.ddim_steps)
        x0_flow = model.ddim_sample(epsK, inference_steps=args.ddim_steps)
        decoded_raw = decode_tensor_to_peptides(x0_raw)
        decoded_flow = decode_tensor_to_peptides(x0_flow)
        pred_score = score_head(epsK)
        eps0_np = eps0.detach().cpu().numpy()
        epsK_np = epsK.detach().cpu().numpy()
        z_bo_np = pca_project(eps0_np.astype(np.float32), pca).astype(np.float32)
        for i in range(end - start):
            pep = peptides[start + i]
            row: Dict[str, object] = {
                "peptide": pep,
                "split": splits[start + i],
                "score": float(y[i].detach().cpu()),
                "predicted_score": float(pred_score[i].detach().cpu()),
                "decoded_from_epsilon0": decoded_raw[i],
                "decoded_from_epsilonK": decoded_flow[i],
                "epsilon0_source_to_decoded_edit": levenshtein_edit_distance(pep, decoded_raw[i]),
                "epsilonK_source_to_decoded_edit": levenshtein_edit_distance(pep, decoded_flow[i]),
                "epsilon0_norm": float(eps0[i].norm().detach().cpu()),
                "epsilonK_norm": float(epsK[i].norm().detach().cpu()),
                "z_bo_norm": float(np.linalg.norm(z_bo_np[i])),
                "flow_logdet": float(logdet[i].detach().cpu()),
            }
            for j in range(z_bo_np.shape[1]):
                row[f"z_bo_{j:02d}"] = float(z_bo_np[i, j])
            for j in range(BO_DIM):
                row[f"epsilon0_{j:03d}"] = float(eps0_np[i, j])
            for j in range(BO_DIM):
                row[f"epsilonK_{j:03d}"] = float(epsK_np[i, j])
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    summary = {
        "n_rows": int(len(df)),
        "score_mean": float(df["score"].mean()),
        "predicted_score_mean": float(df["predicted_score"].mean()),
        "epsilon0_edit_mean": float(df["epsilon0_source_to_decoded_edit"].mean()),
        "epsilonK_edit_mean": float(df["epsilonK_source_to_decoded_edit"].mean()),
        "epsilonK_edit_le_5_fraction": float((df["epsilonK_source_to_decoded_edit"] <= 5).mean()),
        "epsilon0_norm_mean": float(df["epsilon0_norm"].mean()),
        "epsilonK_norm_mean": float(df["epsilonK_norm"].mean()),
        "bo_coordinate_recommendation": (
            "Compare pretraining-consistent z_bo PCA coordinates against "
            "flow-transformed epsilonK coordinates using the same GP/BO protocol."
        ),
    }
    with open(path + ".summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved BO coordinate CSV: {path}")
    print("BO coordinate summary:", summary)


# ============================================================
# Training loop
# ============================================================

def build_optimizer(model: DirectSequenceDiffusion, flow: RealNVPFlow, score_head: ScoreHead, args: argparse.Namespace) -> torch.optim.Optimizer:
    params = [
        {"params": flow.parameters(), "lr": args.flow_lr, "name": "flow"},
        {"params": score_head.parameters(), "lr": args.score_head_lr, "name": "score_head"},
    ]
    if args.finetune_diffusion:
        params.append({"params": model.parameters(), "lr": args.diffusion_lr, "name": "direct_diffusion"})
    return torch.optim.AdamW(params, weight_decay=args.weight_decay)


def train(args: argparse.Namespace) -> str:
    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    model, cfg, _ckpt = load_direct_diffusion_checkpoint(
        args.init_checkpoint, device
    )

    pretraining_files = discover_pretraining_parts(
        args.pretraining_parts_dir,
        args.pretraining_part_glob,
    )
    split_map, split_conflicts_df = build_pretraining_peptide_split_map(
        pretraining_files,
        args.pretraining_peptide_col,
        args.pretraining_split_col,
        args.split_map_chunksize,
    )

    conflict_path = os.path.join(
        args.out_dir,
        "pretraining_exact_peptide_cross_split_conflicts.csv",
    )
    split_conflicts_df.to_csv(conflict_path, index=False)

    split_audit = {
        "n_safe_unique_peptide_mappings": int(len(split_map)),
        "n_exact_peptides_present_in_multiple_pretraining_splits":
            int(len(split_conflicts_df)),
        "conflict_policy":
            "exclude_conflicting_exact_peptides_from_all_finetuning_splits",
        "conflict_csv": conflict_path,
        "fine_tuning_exact_sequence_overlap_policy":
            "zero overlap by construction after conflict exclusion",
    }
    with open(
        os.path.join(
            args.out_dir,
            "pretraining_split_sequence_conflict_audit.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(split_audit, f, indent=2)

    if len(split_conflicts_df):
        print(
            f"[LEAKAGE-SAFETY] Excluding {len(split_conflicts_df)} exact peptide "
            "sequences that occur in more than one pretraining split."
        )
        print(
            f"[LEAKAGE-SAFETY] Conflict audit saved to: {conflict_path}"
        )

    peptides, x_all, y_all, splits, kept_df = load_cu_dataset(
        args, split_map
    )

    if not args.finetune_diffusion:
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        print("Direct diffusion model is frozen; training flow + score head only.")
    else:
        print("Direct diffusion fine-tuning is enabled with small LR.")

    args.preimage_cache = args.preimage_cache or os.path.join(args.out_dir, "cu_direct_diffusion_epsilon0_preimage_cache.pt")
    eps0_all = precompute_or_load_epsilon_preimages(model, x_all, args, device)

    split_arr = np.array([s.lower() for s in splits], dtype=object)
    train_idx = torch.tensor(np.where(split_arr == args.train_split.lower())[0], dtype=torch.long)
    val_idx = torch.tensor(np.where(split_arr == args.validation_split.lower())[0], dtype=torch.long)
    test_idx = torch.tensor(np.where(split_arr == args.test_split.lower())[0], dtype=torch.long)
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError(f"Empty train or validation split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    # Pretraining-consistent BO baseline: fit PCA/whitening on Cu TRAIN
    # DDIM-inverted epsilon only. Validation/test never influence this transform.
    pca = fit_pca_whitener(
        eps0_all[train_idx].numpy().astype(np.float32),
        args.bo_latent_dim,
    )
    pca_path = os.path.join(
        args.out_dir,
        "cu_train_direct_diffusion_bo_pca_whitener.npz",
    )
    np.savez(
        pca_path,
        mean=pca["mean"],
        components=pca["components"],
        score_std=pca["score_std"],
        explained_variance_ratio=pca["explained_variance_ratio"],
    )
    train_z_bo = pca_project(
        eps0_all[train_idx].numpy().astype(np.float32),
        pca,
    )
    pca_meta = {
        "fit_split": args.train_split,
        "fit_n": int(len(train_idx)),
        "raw_epsilon_dim": int(BO_DIM),
        "bo_latent_dim": int(train_z_bo.shape[1]),
        "explained_variance_fraction":
            float(pca["explained_variance_ratio"].sum()),
        "train_raw_epsilon_geometry":
            geometry_diagnostics_np(
                eps0_all[train_idx].numpy().astype(np.float32)
            ),
        "train_z_bo_geometry":
            geometry_diagnostics_np(train_z_bo.astype(np.float32)),
        "pca_path": pca_path,
    }
    with open(
        os.path.join(args.out_dir, "cu_train_direct_diffusion_pca_metadata.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(pca_meta, f, indent=2)
    print("[PCA]", pca_meta)

    train_loader = make_loader(x_all, y_all, eps0_all, train_idx, args.batch_size, shuffle=True)
    val_loader = make_loader(x_all, y_all, eps0_all, val_idx, args.batch_size, shuffle=False)
    if len(test_idx) == 0:
        raise ValueError("Untouched Cu test split is empty; refusing to reuse validation as test.")
    test_loader = make_loader(
        x_all, y_all, eps0_all, test_idx, args.batch_size, shuffle=False
    )

    flow = RealNVPFlow(dim=BO_DIM, n_layers=args.flow_layers, hidden_dim=args.flow_hidden_dim, max_scale=args.flow_max_scale).to(device)
    score_head = ScoreHead(dim=BO_DIM, hidden_dim=args.score_head_hidden_dim).to(device)
    optimizer = build_optimizer(model, flow, score_head, args)

    history_path = os.path.join(args.out_dir, "training_history_cu_direct_diffusion_noise_flow.csv")
    summary_path = os.path.join(args.out_dir, "checkpoint_summary.csv")
    if os.path.exists(history_path) and not args.append_history:
        os.remove(history_path)
    if os.path.exists(summary_path) and not args.append_history:
        os.remove(summary_path)

    best_loss_path = os.path.join(args.out_dir, "best_val_loss_cu_direct_diffusion_noise_flow.pt")
    best_edit_path = os.path.join(args.out_dir, "best_val_edit_cu_direct_diffusion_noise_flow.pt")
    best_score_path = os.path.join(args.out_dir, "best_val_score_mse_cu_direct_diffusion_noise_flow.pt")
    last_path = os.path.join(args.out_dir, "last_epoch_cu_direct_diffusion_noise_flow.pt")
    best_loss = float("inf")
    best_edit = float("inf")
    best_score_mse = float("inf")
    best_epochs = {"loss": 0, "edit": 0, "score": 0}

    print(f"Cu corrected rows: train={len(train_idx)} validation={len(val_idx)} test={len(test_idx)} total={len(peptides)}")
    print(f"Score column used: {args.score_col}")
    print(f"BO coordinate: epsilonK in {BO_DIM} dimensions; radius target={math.sqrt(BO_DIM):.6f}")

    radius = math.sqrt(BO_DIM)
    for epoch in range(1, args.epochs + 1):
        # Even when diffusion weights are frozen, the flow loss backpropagates
        # through DDIM and the GRU denoiser to update the flow parameters.
        # cuDNN requires the GRU module to be in training mode for that backward pass.
        model.train()
        flow.train(); score_head.train()
        sums = {"loss": 0.0, "diffusion_loss": 0.0, "recon_ce": 0.0, "x0_mse": 0.0, "score_mse": 0.0, "anchor_mse": 0.0, "sphere_mse": 0.0, "logdet_reg": 0.0, "token_acc": 0.0}
        n_batches = 0

        for x, y, eps0, _idx in train_loader:
            x = x.to(device); y = y.to(device); eps0 = eps0.to(device)
            optimizer.zero_grad(set_to_none=True)
            epsK_raw, logdet = flow(eps0)
            epsK = project_to_sphere(epsK_raw, radius) if args.project_flow_output_to_sphere else epsK_raw
            x0_hat = ddim_sample_with_grad(model, epsK, inference_steps=args.ddim_steps)
            target = x.argmax(dim=-1)
            recon_ce = F.cross_entropy(x0_hat.reshape(-1, VOCAB), target.reshape(-1))
            prob = F.softmax(x0_hat, dim=-1)
            x0_mse = F.mse_loss(prob, x)
            pred_score = score_head(epsK)
            score_mse = F.mse_loss(pred_score, y)
            anchor = F.mse_loss(epsK, eps0)
            sphere = (epsK.norm(dim=-1).mean() - radius).pow(2)
            logdet_reg = -logdet.mean()
            if args.finetune_diffusion and args.diffusion_loss_weight > 0.0:
                diffusion_loss, _dm = model.training_loss(x, recon_ce_weight=args.diffusion_recon_ce_weight, x0_mse_weight=args.diffusion_x0_mse_weight)
            else:
                diffusion_loss = torch.zeros((), device=device)
            loss = (
                args.diffusion_loss_weight * diffusion_loss
                + args.flow_recon_ce_weight * recon_ce
                + args.flow_x0_mse_weight * x0_mse
                + args.score_loss_weight * score_mse
                + args.flow_anchor_weight * anchor
                + args.sphere_loss_weight * sphere
                + args.flow_logdet_weight * logdet_reg
            )
            loss.backward()
            trainable = list(flow.parameters()) + list(score_head.parameters()) + (list(model.parameters()) if args.finetune_diffusion else [])
            nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            token_acc = (x0_hat.argmax(dim=-1) == target).float().mean()
            vals = {"loss": loss, "diffusion_loss": diffusion_loss, "recon_ce": recon_ce, "x0_mse": x0_mse, "score_mse": score_mse, "anchor_mse": anchor, "sphere_mse": sphere, "logdet_reg": logdet_reg, "token_acc": token_acc}
            for k, v in vals.items():
                sums[k] += float(v.detach().cpu())
            n_batches += 1

        train_metrics = {k: v / max(1, n_batches) for k, v in sums.items()}
        val_metrics = evaluate(model, flow, score_head, val_loader, peptides, args, device)
        row: Dict[str, object] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "epoch": epoch,
            "finetune_diffusion": bool(args.finetune_diffusion),
            "flow_lr": args.flow_lr,
            "score_head_lr": args.score_head_lr,
            "diffusion_lr": args.diffusion_lr,
            "score_col": args.score_col,
        }
        for k, v in train_metrics.items():
            row[f"train_{k}"] = v
        for k, v in val_metrics.items():
            row[f"val_{k}"] = v
        append_history(history_path, row)
        print(f"epoch={epoch} train={train_metrics} val={val_metrics}")

        if val_metrics["loss"] <= best_loss:
            best_loss = val_metrics["loss"]; best_epochs["loss"] = epoch
            save_checkpoint(best_loss_path, model, flow, score_head, optimizer, cfg, args, epoch, val_metrics, "minimum_validation_total_loss")
        if val_metrics["flow_decode_edit_mean"] <= best_edit:
            best_edit = val_metrics["flow_decode_edit_mean"]; best_epochs["edit"] = epoch
            save_checkpoint(best_edit_path, model, flow, score_head, optimizer, cfg, args, epoch, val_metrics, "minimum_validation_flow_decode_edit")
        if val_metrics["score_mse"] <= best_score_mse:
            best_score_mse = val_metrics["score_mse"]; best_epochs["score"] = epoch
            save_checkpoint(best_score_path, model, flow, score_head, optimizer, cfg, args, epoch, val_metrics, "minimum_validation_score_mse")
        save_checkpoint(last_path, model, flow, score_head, optimizer, cfg, args, epoch, val_metrics, "last_completed_epoch")

    test_metrics = evaluate(model, flow, score_head, test_loader, peptides, args, device)
    with open(os.path.join(args.out_dir, "final_test_metrics_cu_direct_diffusion_noise_flow.json"), "w", encoding="utf-8") as handle:
        json.dump(test_metrics, handle, indent=2)
    gen = generation_sanity(model, flow, args, device)
    with open(os.path.join(args.out_dir, "generation_sanity_cu_direct_diffusion_noise_flow.json"), "w", encoding="utf-8") as handle:
        json.dump(gen, handle, indent=2)

    rows = []
    for selection, path in [("best_val_loss", best_loss_path), ("best_val_edit", best_edit_path), ("best_val_score_mse", best_score_path), ("last_epoch", last_path)]:
        ck = torch_load_full(path, map_location="cpu")
        m = ck.get("metrics", {})
        rows.append({"selection": selection, "checkpoint_path": path, "epoch": int(ck.get("epoch", -1)), **{f"val_{k}": v for k, v in m.items() if isinstance(v, (int, float))}})
    pd.DataFrame(rows).to_csv(summary_path, index=False)

    if args.export_bo_coordinates:
        export_bo_coordinates(
            path=os.path.join(args.out_dir, "cu_direct_diffusion_noise_flow_coordinates_for_bo.csv"),
            model=model,
            flow=flow,
            score_head=score_head,
            peptides=peptides,
            x_all=x_all,
            y_all=y_all,
            splits=splits,
            eps0_all=eps0_all,
            pca=pca,
            args=args,
            device=device,
        )

    print(f"Saved checkpoint summary: {summary_path}")
    print(f"Best val loss: epoch={best_epochs['loss']} value={best_loss:.6f}")
    print(f"Best val edit: epoch={best_epochs['edit']} value={best_edit:.6f}")
    print(f"Best val score MSE: epoch={best_epochs['score']} value={best_score_mse:.6f}")
    print("Final test metrics:", test_metrics)
    print("Generation sanity:", gen)
    return best_loss_path


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cu fine-tuning of direct peptide sequence diffusion with RealNVP noise-space normalizing flow.")
    p.add_argument("--init-checkpoint", required=True)
    p.add_argument("--cu-csv", default="metalpdb_CU_chain_mapped_len10_high_confidence.csv")
    p.add_argument("--peptide-col", default="peptide_len10")
    p.add_argument("--labels-col", default="binding_site_labels_len10")
    p.add_argument("--score-col", default="final_score")
    p.add_argument("--auto-score-if-missing", action="store_true", default=True)
    p.add_argument("--split-col", default="split")
    p.add_argument("--train-split", default="train")
    p.add_argument("--validation-split", default="validation")
    p.add_argument("--test-split", default="test")
    p.add_argument("--pretraining-parts-dir", required=True)
    p.add_argument(
        "--pretraining-part-glob",
        default="metalpdb_ALL_chain_mapped_len10_high_confidence_part_*.csv",
    )
    p.add_argument("--pretraining-peptide-col", default="peptide_len10")
    p.add_argument("--pretraining-split-col", default="split")
    p.add_argument("--split-map-chunksize", type=int, default=100000)
    p.add_argument(
        "--duplicate-policy",
        choices=["mean", "first", "error"],
        default="mean",
    )
    p.add_argument(
        "--unmapped-policy",
        choices=["drop", "error"],
        default="drop",
    )
    p.add_argument(
        "--out-dir",
        default="cu_direct_sequence_diffusion_noise_flow_leakage_safe_h96",
    )

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--append-history", action="store_true")

    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument(
        "--bo-latent-dim",
        type=int,
        default=64,
        help="PCA-whitened BO baseline dimension, fit using Cu TRAIN epsilon only.",
    )
    p.add_argument(
        "--project-inverted-epsilon-to-sphere",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Match pretraining default: do not force DDIM-inverted epsilon to the sphere.",
    )
    p.add_argument(
        "--project-flow-output-to-sphere",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optional ablation; pretraining does not require sphere projection.",
    )

    p.add_argument("--flow-layers", type=int, default=4)
    p.add_argument("--flow-hidden-dim", type=int, default=128)
    p.add_argument("--flow-max-scale", type=float, default=0.5)
    p.add_argument("--flow-lr", type=float, default=3e-5)
    p.add_argument("--score-head-lr", type=float, default=1e-4)
    p.add_argument("--score-head-hidden-dim", type=int, default=128)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=5.0)

    p.add_argument("--finetune-diffusion", action="store_true")
    p.add_argument("--diffusion-lr", type=float, default=1e-5)
    p.add_argument("--diffusion-loss-weight", type=float, default=0.0)
    p.add_argument("--diffusion-recon-ce-weight", type=float, default=0.5)
    p.add_argument("--diffusion-x0-mse-weight", type=float, default=0.1)

    p.add_argument("--flow-recon-ce-weight", type=float, default=1.0)
    p.add_argument("--flow-x0-mse-weight", type=float, default=0.2)
    p.add_argument("--score-loss-weight", type=float, default=0.2)
    p.add_argument("--flow-anchor-weight", type=float, default=0.01)
    p.add_argument("--sphere-loss-weight", type=float, default=0.001)
    p.add_argument("--flow-logdet-weight", type=float, default=0.0)

    p.add_argument("--preimage-cache", default=None)
    p.add_argument("--recompute-preimages", action="store_true")
    p.add_argument("--preimage-batch-size", type=int, default=64)
    p.add_argument("--preimage-steps", type=int, default=0, help="Deprecated: DDIM inversion is used directly.")
    p.add_argument("--preimage-lr", type=float, default=0.0, help="Deprecated.")
    p.add_argument("--preimage-ce-weight", type=float, default=1.0)
    p.add_argument("--preimage-x0-mse-weight", type=float, default=0.2)
    p.add_argument("--preimage-anchor-weight", type=float, default=0.01)
    p.add_argument("--preimage-sphere-weight", type=float, default=0.001)
    p.add_argument("--preimage-grad-clip", type=float, default=5.0)
    p.add_argument("--preimage-project-each-step", action="store_true", default=True)
    p.add_argument("--preimage-log-every", type=int, default=5)

    p.add_argument("--sanity-samples", type=int, default=512)
    p.add_argument("--export-bo-coordinates", action="store_true", default=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train(args)
