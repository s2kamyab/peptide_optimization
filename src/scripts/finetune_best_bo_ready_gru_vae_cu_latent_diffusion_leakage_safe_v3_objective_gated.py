from __future__ import annotations

"""
Cu-specific latent-diffusion fine-tuning on the BEST BO-ready H64/Z64 GRU-VAE.

Designed for checkpoints produced by:
  pretrain_gru_vae_bo_ready_h64_z64_with_validation.py

Key design choices
------------------
1. Load the revised BO-ready pretrained GRU-VAE checkpoint directly and verify
   its BO-aware selection metadata and leakage-safe pretraining split.
2. Freeze the GRU-VAE by default to preserve its validated latent geometry and
   autoregressive decoder.
3. Reuse pretraining_leakage_safe_split_manifest.csv so Cu fine-tuning preserves
   the same PDB/exact-peptide/Hamming-safe train/validation/test partition.
4. Drop manifest-excluded near-neighbors and explicitly audit Cu peptides that
   are absent from the pretraining manifest.
5. Compute latent mean/std from TRAINING peptides only.
6. Train latent diffusion on standardized encoder means:
       h0 = (mu - latent_mean) / latent_std
7. Train a small multi-objective auxiliary head on the same h0 coordinates for:
       chelation_sub, solubility_sub, stability_sub, expression_sub
8. Use deterministic DDIM inversion/sample diagnostics.
9. Monitor global geometry for mu, h0, raw epsilon, and the BO epsilon coordinates.
10. Preserve an untouched Cu test set and evaluate selected checkpoints once after
    validation-based model selection.
11. Export epsilon, mu, h0, split labels, and Cu objective values for downstream BO.
12. Select BO-ready checkpoints with objective-quality gates so early checkpoints
    with good inversion but poor objective learnability cannot win.

BO coordinate path
------------------
peptide -> encoder mu -> standardize -> DDIM invert -> epsilon
epsilon -> DDIM sample -> h0 -> unstandardize -> decoder mu -> peptide

The diffusion base-noise coordinate can optionally be projected to the
sqrt(latent_dim) sphere. For Z64 this radius is 8.
"""

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

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

DEFAULT_OBJECTIVE_COLS = [
    "chelation_sub",
    "solubility_sub",
    "stability_sub",
    "expression_sub",
]


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
    beta_start: float = 1e-5
    beta_end: float = 8e-3


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_load_full(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def clean_peptide(x: object) -> Optional[str]:
    pep = str(x).strip().upper()
    if len(pep) != SEQ_LEN:
        return None
    if any(ch not in AA_TO_I for ch in pep):
        return None
    return pep


def onehot_encode_peptides(peptides: Sequence[str]) -> torch.Tensor:
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


# ---------------------------------------------------------------------
# GRU-VAE architecture: kept checkpoint-compatible with the pretrainer
# ---------------------------------------------------------------------

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

    def forward(self, x_onehot: torch.Tensor):
        h = self.in_proj(x_onehot)
        out_top, h_n = self.gru(h)
        h_last = h_n[-1]
        mu = self.to_mu(h_last)
        logvar = self.to_logvar(h_last).clamp(min=-8.0, max=8.0)
        return mu, logvar, out_top


class LatentConditionedGRUDecoder(nn.Module):
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

    def forward(self, z: torch.Tensor, x_onehot: torch.Tensor):
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

    def step(self, z: torch.Tensor, current_token: torch.Tensor, h: torch.Tensor):
        emb = self.token_embed(current_token)
        dec_input = torch.cat([emb, z.unsqueeze(1)], dim=-1)
        out_step, h = self.gru(dec_input, h)
        logits = self.to_logits(out_step[:, -1, :])
        return logits, h


class GRUVAE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.enc = GRUEncoder(cfg.hidden_size, cfg.latent_dim, cfg.n_layers, cfg.dropout)
        self.dec = LatentConditionedGRUDecoder(
            cfg.hidden_size, cfg.latent_dim, cfg.n_layers, cfg.dropout
        )


# ---------------------------------------------------------------------
# Latent diffusion
# ---------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time_dim must be even")
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        exponent = (
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
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
        return x + self.fc2(F.silu(self.fc1(self.norm(x))))


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
        self.blocks = nn.ModuleList(
            [ResidualDenoiserBlock(cfg.hidden_dim) for _ in range(cfg.n_blocks)]
        )
        self.out_norm = nn.LayerNorm(cfg.hidden_dim)
        self.out_proj = nn.Linear(cfg.hidden_dim, latent_dim)

    def forward(self, h_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(h_t) + self.time_mlp(self.time_embedding(t))
        for block in self.blocks:
            h = block(h)
        return self.out_proj(F.silu(self.out_norm(h)))


class LatentDiffusion(nn.Module):
    def __init__(self, latent_dim: int, cfg: DiffusionConfig):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.cfg = cfg
        self.denoiser = LatentDenoiser(latent_dim, cfg)

        betas = torch.linspace(
            cfg.beta_start, cfg.beta_end, cfg.train_steps, dtype=torch.float32
        )
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", torch.cumprod(alphas, dim=0))

    def q_sample(self, h0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        abar = self.alpha_bars[t].unsqueeze(-1)
        return torch.sqrt(abar) * h0 + torch.sqrt(1.0 - abar) * noise

    def training_loss(self, h0: torch.Tensor):
        b = h0.size(0)
        t = torch.randint(0, self.cfg.train_steps, (b,), device=h0.device)
        noise = torch.randn_like(h0)
        h_t = self.q_sample(h0, t, noise)
        pred_noise = self.denoiser(h_t, t)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def ddim_sample(self, base_noise: torch.Tensor, inference_steps: int = 20):
        self.eval()
        n_train = self.cfg.train_steps
        inference_steps = min(max(int(inference_steps), 1), n_train)
        timesteps = torch.linspace(
            n_train - 1, 0, inference_steps, device=base_noise.device
        ).round().long()
        timesteps = torch.unique_consecutive(timesteps)

        h = base_noise
        for j, t_scalar in enumerate(timesteps):
            t = torch.full(
                (h.size(0),), int(t_scalar.item()), device=h.device, dtype=torch.long
            )
            pred_noise = self.denoiser(h, t)
            abar_t = self.alpha_bars[t_scalar]
            pred_h0 = (
                h - torch.sqrt(1.0 - abar_t) * pred_noise
            ) / torch.sqrt(abar_t)

            if j == len(timesteps) - 1:
                h = pred_h0
                break

            t_prev = timesteps[j + 1]
            abar_prev = self.alpha_bars[t_prev]
            h = (
                torch.sqrt(abar_prev) * pred_h0
                + torch.sqrt(1.0 - abar_prev) * pred_noise
            )
        return h

    @torch.no_grad()
    def ddim_invert(self, h0: torch.Tensor, inference_steps: int = 20):
        self.eval()
        n_train = self.cfg.train_steps
        inference_steps = min(max(int(inference_steps), 1), n_train)
        timesteps = torch.linspace(
            0, n_train - 1, inference_steps, device=h0.device
        ).round().long()
        timesteps = torch.unique_consecutive(timesteps)

        h = h0
        for j, t_scalar in enumerate(timesteps[:-1]):
            t = torch.full(
                (h.size(0),), int(t_scalar.item()), device=h.device, dtype=torch.long
            )
            pred_noise = self.denoiser(h, t)
            abar_t = self.alpha_bars[t_scalar]
            pred_h0 = (
                h - torch.sqrt(1.0 - abar_t) * pred_noise
            ) / torch.sqrt(abar_t)

            t_next = timesteps[j + 1]
            abar_next = self.alpha_bars[t_next]
            h = (
                torch.sqrt(abar_next) * pred_h0
                + torch.sqrt(1.0 - abar_next) * pred_noise
            )
        return h


class MultiObjectiveHead(nn.Module):
    def __init__(self, latent_dim: int, n_objectives: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_objectives),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


# ---------------------------------------------------------------------
# Data and checkpoint handling
# ---------------------------------------------------------------------


def load_manifest(path: str) -> pd.DataFrame:
    m = pd.read_csv(path)
    if "peptide" not in m.columns or "split" not in m.columns:
        raise KeyError(
            "Pretraining split manifest must contain columns ['peptide', 'split']."
        )
    m = m[["peptide", "split"]].copy()
    m["peptide"] = m["peptide"].map(clean_peptide)
    m = m.dropna(subset=["peptide", "split"])
    m["split"] = m["split"].astype(str)

    dup = m[m.duplicated("peptide", keep=False)]
    if len(dup):
        conflicts = dup.groupby("peptide")["split"].nunique()
        bad = conflicts[conflicts > 1]
        if len(bad):
            raise RuntimeError(
                "Manifest assigns some peptides to multiple splits; "
                f"examples={bad.head().to_dict()}"
            )
        m = m.drop_duplicates("peptide", keep="first")
    return m


def _resolve_peptide_column(df: pd.DataFrame, requested: str) -> str:
    if requested in df.columns:
        return requested
    for alt in ["peptide_len10", "sequence", "peptide"]:
        if alt in df.columns:
            return alt
    raise KeyError(
        f"Could not find peptide column {requested!r}. "
        f"Available columns include: {list(df.columns)[:60]}"
    )


def load_cu_dataset_with_manifest(
    csv_path: str,
    peptide_col: str,
    objective_cols: Sequence[str],
    manifest_path: str,
    duplicate_policy: str,
    unmapped_policy: str,
    out_dir: str,
):
    raw = pd.read_csv(csv_path)
    peptide_col = _resolve_peptide_column(raw, peptide_col)

    missing = [c for c in objective_cols if c not in raw.columns]
    if missing:
        raise ValueError(
            f"Missing objective columns in {csv_path}: {missing}. "
            f"Available columns: {list(raw.columns)}"
        )

    keep_cols = [peptide_col] + list(objective_cols)
    if "final_score" in raw.columns:
        keep_cols.append("final_score")

    df = raw[keep_cols].copy()
    df["peptide"] = df[peptide_col].map(clean_peptide)

    for c in objective_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "final_score" in df.columns:
        df["final_score"] = pd.to_numeric(df["final_score"], errors="coerce")

    df = df.dropna(subset=["peptide"] + list(objective_cols)).reset_index(drop=True)

    rows_before_dedup = int(len(df))
    unique_before = int(df["peptide"].nunique())
    duplicate_rows = rows_before_dedup - unique_before

    if duplicate_policy == "error" and duplicate_rows:
        raise RuntimeError(
            f"Cu dataset contains {duplicate_rows} duplicate peptide rows."
        )
    elif duplicate_policy == "first":
        df = df.drop_duplicates("peptide", keep="first").copy()
    elif duplicate_policy == "mean":
        agg = {c: "mean" for c in objective_cols}
        if "final_score" in df.columns:
            agg["final_score"] = "mean"
        df = df.groupby("peptide", as_index=False).agg(agg)
    elif duplicate_policy != "error":
        raise ValueError(f"Unknown duplicate_policy={duplicate_policy!r}")

    if "final_score" not in df.columns:
        df["final_score"] = np.nan

    manifest = load_manifest(manifest_path)
    split_map = dict(zip(manifest["peptide"], manifest["split"]))
    df["split"] = df["peptide"].map(split_map)

    n_unmapped = int(df["split"].isna().sum())
    if n_unmapped:
        unmapped_df = df.loc[df["split"].isna()].copy()
        os.makedirs(out_dir, exist_ok=True)
        unmapped_path = os.path.join(
            out_dir, "cu_peptides_unmapped_from_pretraining_manifest.csv"
        )
        unmapped_df.to_csv(unmapped_path, index=False)
        examples = unmapped_df["peptide"].head(10).tolist()
        msg = (
            f"{n_unmapped} Cu peptides were not found in the pretraining split "
            f"manifest; examples={examples}. Full list saved to {unmapped_path}"
        )
        if unmapped_policy == "error":
            raise RuntimeError(
                msg + ". Use --unmapped-policy drop for the strict matched experiment."
            )
        print("WARNING:", msg)
        print(
            "Dropping unmapped Cu peptides so no new split bypasses the pretraining "
            "PDB/Hamming leakage controls."
        )
        df = df.dropna(subset=["split"]).copy()

    excluded_labels = {"excluded_near_train", "excluded_near_development"}
    n_excluded = int(df["split"].isin(excluded_labels).sum())
    if n_excluded:
        print(
            f"Excluding {n_excluded} Cu peptides marked as strict-holdout "
            "near-neighbors in the pretraining manifest."
        )
        df = df[~df["split"].isin(excluded_labels)].copy()

    allowed = {"train", "val", "test"}
    unexpected = sorted(set(df["split"].unique()) - allowed)
    if unexpected:
        raise RuntimeError(f"Unexpected manifest split labels: {unexpected}")

    df = df.reset_index(drop=True)
    split_counts = (
        df["split"].value_counts()
        .reindex(["train", "val", "test"])
        .fillna(0).astype(int).to_dict()
    )
    for split in ["train", "val", "test"]:
        if split_counts[split] == 0:
            raise RuntimeError(f"No Cu {split} peptides remain after manifest mapping.")

    audit = {
        "cu_csv": csv_path,
        "peptide_column": peptide_col,
        "rows_before_dedup": rows_before_dedup,
        "unique_peptides_before_dedup": unique_before,
        "duplicate_rows": int(duplicate_rows),
        "duplicate_policy": duplicate_policy,
        "rows_after_dedup_and_manifest_filtering": int(len(df)),
        "unmapped_peptides": n_unmapped,
        "unmapped_policy": unmapped_policy,
        "unmapped_peptides_excluded": int(n_unmapped if unmapped_policy == "drop" else 0),
        "excluded_manifest_near_neighbors": n_excluded,
        "split_counts": split_counts,
        "split_source": manifest_path,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(
        os.path.join(out_dir, "cu_latent_diffusion_data_audit.json"),
        "w", encoding="utf-8",
    ) as f:
        json.dump(audit, f, indent=2)

    df.to_csv(
        os.path.join(out_dir, "cu_latent_diffusion_manifest_mapped.csv"),
        index=False,
    )

    peptides = df["peptide"].astype(str).tolist()
    x = onehot_encode_peptides(peptides)
    y = torch.tensor(df[list(objective_cols)].to_numpy(dtype=np.float32))
    final_score = torch.tensor(df["final_score"].to_numpy(dtype=np.float32))

    split_to_idx = {
        split: torch.tensor(
            np.flatnonzero(df["split"].to_numpy() == split), dtype=torch.long
        )
        for split in ["train", "val", "test"]
    }

    print(
        "Cu leakage-safe split: "
        f"total={len(df)} train={len(split_to_idx['train'])} "
        f"val={len(split_to_idx['val'])} test={len(split_to_idx['test'])}"
    )

    return df, peptides, x, y, final_score, split_to_idx, peptide_col, audit


def validate_pretrained_checkpoint(
    vae: GRUVAE,
    checkpoint_path: str,
    device: torch.device,
    require_bo_ready_selection: bool = True,
):
    ckpt = torch_load_full(checkpoint_path, map_location=device)

    ctype = ckpt.get("checkpoint_type", "")
    expected_type = "pretrained_gru_vae_latent_conditioned_bo_ready_no_flow"
    if ctype and ctype != expected_type:
        raise ValueError(
            f"Unexpected checkpoint_type={ctype!r}; expected {expected_type!r}"
        )

    saved_cfg = ckpt.get("model_config", {})
    for key, expected in [
        ("hidden_size", vae.cfg.hidden_size),
        ("latent_dim", vae.cfg.latent_dim),
        ("n_layers", vae.cfg.n_layers),
        ("dropout", vae.cfg.dropout),
        ("decoder_conditioning", vae.cfg.decoder_conditioning),
    ]:
        if key in saved_cfg and saved_cfg[key] != expected:
            raise ValueError(
                f"Checkpoint architecture mismatch for {key}: "
                f"saved={saved_cfg[key]!r}, requested={expected!r}"
            )

    state = ckpt.get("model_state_dict")
    if state is None:
        raise KeyError("Pretrained checkpoint does not contain model_state_dict.")
    vae.load_state_dict(state, strict=True)

    metrics = ckpt.get("metrics", {})
    selection_metric = metrics.get("selection_metric", "")
    if require_bo_ready_selection and selection_metric != "bo_ready_score":
        raise RuntimeError(
            "Expected the revised best_bo_ready pretraining checkpoint, but "
            f"selection_metric={selection_metric!r}."
        )

    validation = ckpt.get("validation", {})
    if validation.get("split_method") != (
        "connected_component_hash_of_peptide_PDB_bipartite_graph"
    ):
        raise RuntimeError(
            "Checkpoint does not advertise the revised leakage-safe split method."
        )

    print(
        f"Loaded BO-ready pretrained GRU-VAE epoch={ckpt.get('epoch')} "
        f"selection={selection_metric} "
        f"score={metrics.get('selection_value')}"
    )
    print(
        "Pretrained latent geometry: "
        f"effective_dim_std="
        f"{metrics.get('selection_effective_dim_pr_standardized', float('nan'))}, "
        f"mean_abs_corr="
        f"{metrics.get('selection_mean_abs_offdiag_corr', float('nan'))}, "
        f"val_free_recon="
        f"{metrics.get('selection_val_free_recon', float('nan'))}"
    )
    return ckpt


@torch.no_grad()
def compute_latent_stats_from_indices(
    vae: GRUVAE,
    x_all: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    device: torch.device,
):
    vae.eval()
    mus = []
    for start in range(0, len(indices), batch_size):
        idx = indices[start:start + batch_size]
        x = x_all[idx].to(device)
        mu, _, _ = vae.enc(x)
        mus.append(mu.detach())
    mu_all = torch.cat(mus, dim=0)
    return (
        mu_all.mean(dim=0),
        mu_all.std(dim=0, unbiased=False).clamp(min=1e-6),
    )


def standardize_mu(mu, latent_mean, latent_std):
    return (mu - latent_mean.to(mu.device)) / latent_std.to(mu.device).clamp(min=1e-6)


def unstandardize_h(h, latent_mean, latent_std):
    return h * latent_std.to(h.device).clamp(min=1e-6) + latent_mean.to(h.device)


def latent_geometry_diagnostics(x: torch.Tensor, eps: float = 1e-8) -> Dict[str, float]:
    if x.ndim != 2 or x.size(0) < 2:
        return {
            "effective_dim_pr": float("nan"),
            "effective_dim_pr_standardized": float("nan"),
            "pc1_variance_fraction": float("nan"),
            "pcs90": float("nan"),
            "pcs95": float("nan"),
            "mean_abs_offdiag_corr": float("nan"),
            "max_abs_offdiag_corr": float("nan"),
        }

    def spectrum_stats(v: torch.Tensor):
        v = v.double()
        v = v - v.mean(dim=0, keepdim=True)
        cov = (v.T @ v) / max(1, v.size(0) - 1)
        eig = torch.linalg.eigvalsh(cov).clamp_min(0.0)
        total = eig.sum().clamp_min(eps)
        pr = (total * total / eig.pow(2).sum().clamp_min(eps)).item()
        eig_desc = torch.flip(eig, dims=[0])
        frac = eig_desc / total
        cum = torch.cumsum(frac, dim=0)
        pc1 = float(frac[0].item())
        pcs90 = int(
            torch.searchsorted(
                cum, torch.tensor(0.90, dtype=cum.dtype, device=cum.device)
            ).item() + 1
        )
        pcs95 = int(
            torch.searchsorted(
                cum, torch.tensor(0.95, dtype=cum.dtype, device=cum.device)
            ).item() + 1
        )
        return pr, pc1, pcs90, pcs95

    raw_pr, pc1, pcs90, pcs95 = spectrum_stats(x)
    z = x.double() - x.double().mean(dim=0, keepdim=True)
    z = z / z.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    std_pr, _, _, _ = spectrum_stats(z)
    corr = (z.T @ z) / float(z.size(0))
    off = corr - torch.diag(torch.diagonal(corr))
    denom = max(1, z.size(1) * (z.size(1) - 1))

    return {
        "effective_dim_pr": float(raw_pr),
        "effective_dim_pr_standardized": float(std_pr),
        "pc1_variance_fraction": float(pc1),
        "pcs90": float(pcs90),
        "pcs95": float(pcs95),
        "mean_abs_offdiag_corr": float(off.abs().sum().item() / denom),
        "max_abs_offdiag_corr": float(off.abs().max().item()),
    }


def prefixed_geometry(prefix: str, x: torch.Tensor) -> Dict[str, float]:
    return {
        f"{prefix}_{k}": v
        for k, v in latent_geometry_diagnostics(x).items()
    }


def project_to_radius(x: torch.Tensor, radius: float) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8) * float(radius)



def make_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    shuffle: bool,
):
    return DataLoader(
        TensorDataset(x[indices], y[indices], indices),
        batch_size=batch_size,
        shuffle=shuffle,
    )


# ---------------------------------------------------------------------
# Decoding and metrics
# ---------------------------------------------------------------------

@torch.no_grad()
def autoregressive_decode(
    vae: GRUVAE, z: torch.Tensor, temperature: float = 1.0
):
    vae.eval()
    b = z.size(0)
    h = vae.dec.initial_hidden(z)
    current = torch.zeros(b, 1, VOCAB, device=z.device, dtype=z.dtype)
    token_steps = []

    tau = max(float(temperature), 1e-6)
    for _ in range(SEQ_LEN):
        logits, h = vae.dec.step(z, current, h)
        idx = (logits / tau).argmax(dim=-1)
        token_steps.append(idx.unsqueeze(1))
        current = F.one_hot(idx, num_classes=VOCAB).to(z.dtype).unsqueeze(1)

    token_idx = torch.cat(token_steps, dim=1)
    peptides = [
        "".join(I_TO_AA[int(i)] for i in row)
        for row in token_idx.detach().cpu().tolist()
    ]
    return peptides


def pearson_safe(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().cpu().reshape(-1)
    b = b.detach().float().cpu().reshape(-1)
    if a.numel() < 2 or float(a.std()) < 1e-12 or float(b.std()) < 1e-12:
        return float("nan")
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1].item())


def objective_locality_loss(
    h: torch.Tensor,
    y: torch.Tensor,
    tau_latent: float = 4.0,
    tau_objective: float = 0.35,
) -> torch.Tensor:
    """Encourage local latent neighborhoods to respect objective similarity.

    This is intentionally lightweight and optional. It is useful for BO because
    GP surrogates behave better when nearby coordinates tend to have nearby
    objective vectors. The diagonal is excluded so the loss is not dominated by
    trivial self-similarity.
    """
    if h.size(0) < 3:
        return h.new_tensor(0.0)

    h_dist = torch.cdist(h.float(), h.float(), p=2)
    y_dist = torch.cdist(y.float(), y.float(), p=2)

    h_sim = torch.exp(-h_dist / max(float(tau_latent), 1e-8))
    y_sim = torch.exp(-y_dist / max(float(tau_objective), 1e-8))

    mask = ~torch.eye(h.size(0), dtype=torch.bool, device=h.device)
    return F.mse_loss(h_sim[mask], y_sim[mask])


@torch.no_grad()
def inversion_metrics(
    vae: GRUVAE,
    diffusion: LatentDiffusion,
    x_val: torch.Tensor,
    source_peptides: Sequence[str],
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    ddim_steps: int,
    project_to_sphere: bool,
    temperature: float,
):
    mu, _, _ = vae.enc(x_val)
    h0 = standardize_mu(mu, latent_mean, latent_std)

    eps_raw = diffusion.ddim_invert(h0, inference_steps=ddim_steps)
    eps_bo = eps_raw
    if project_to_sphere:
        eps_bo = project_to_radius(eps_raw, math.sqrt(vae.cfg.latent_dim))

    h_back = diffusion.ddim_sample(eps_bo, inference_steps=ddim_steps)
    mu_back = unstandardize_h(h_back, latent_mean, latent_std)
    decoded = autoregressive_decode(vae, mu_back, temperature=temperature)

    h_l2 = torch.linalg.norm(h0 - h_back, dim=-1)
    h_cos = F.cosine_similarity(h0, h_back, dim=-1, eps=1e-8)
    edits = [
        levenshtein_edit_distance(a, b)
        for a, b in zip(source_peptides, decoded)
    ]

    metrics = {
        "ddim_inversion_h_l2_mean": float(h_l2.mean().cpu()),
        "ddim_inversion_h_l2_median": float(h_l2.median().cpu()),
        "ddim_inversion_h_cosine_mean": float(h_cos.mean().cpu()),
        "ddim_source_to_decode_edit_mean": float(np.mean(edits)),
        "ddim_source_to_decode_edit_median": float(np.median(edits)),
        "ddim_decode_unique_fraction": float(len(set(decoded)) / max(1, len(decoded))),
        "epsilon_raw_norm_mean": float(eps_raw.norm(dim=-1).mean().cpu()),
        "epsilon_raw_norm_std": float(eps_raw.norm(dim=-1).std().cpu()),
        "epsilon_bo_norm_mean": float(eps_bo.norm(dim=-1).mean().cpu()),
        "epsilon_bo_norm_std": float(eps_bo.norm(dim=-1).std().cpu()),
        "epsilon_projected_to_sphere": bool(project_to_sphere),
    }
    metrics.update(prefixed_geometry("mu", mu.cpu()))
    metrics.update(prefixed_geometry("h0", h0.cpu()))
    metrics.update(prefixed_geometry("epsilon_raw", eps_raw.cpu()))
    metrics.update(prefixed_geometry("epsilon_bo", eps_bo.cpu()))
    return metrics


@torch.no_grad()
def local_epsilon_geometry(
    vae: GRUVAE,
    diffusion: LatentDiffusion,
    x_val: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    ddim_steps: int,
    sigma: float,
    subset: int,
    neighbors: int,
    seed: int,
    renormalize: bool,
):
    n = min(int(subset), x_val.size(0))
    if n <= 0 or neighbors <= 0:
        return {}

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    x = x_val[:n]
    mu, _, _ = vae.enc(x)
    h0 = standardize_mu(mu, latent_mean, latent_std)
    eps0 = diffusion.ddim_invert(h0, inference_steps=ddim_steps)

    radius = math.sqrt(vae.cfg.latent_dim)
    if renormalize:
        eps0 = project_to_radius(eps0, radius)

    h_center = diffusion.ddim_sample(eps0, inference_steps=ddim_steps)
    mu_center = unstandardize_h(h_center, latent_mean, latent_std)
    p_center = autoregressive_decode(vae, mu_center)

    eps_dists = []
    h_dists = []
    edits = []

    for i in range(n):
        e0 = eps0[i:i+1]
        for _ in range(int(neighbors)):
            noise = torch.randn(
                e0.shape, generator=gen, dtype=torch.float32, device="cpu"
            ).to(e0.device) * float(sigma)
            en = e0 + noise
            if renormalize:
                en = project_to_radius(en, radius)

            hn = diffusion.ddim_sample(en, inference_steps=ddim_steps)
            mun = unstandardize_h(hn, latent_mean, latent_std)
            pn = autoregressive_decode(vae, mun)[0]

            eps_dists.append(float(torch.linalg.norm(en - e0).cpu()))
            h_dists.append(float(torch.linalg.norm(hn - h_center[i:i+1]).cpu()))
            edits.append(levenshtein_edit_distance(p_center[i], pn))

    return {
        "local_sigma": float(sigma),
        "local_epsilon_l2_mean": float(np.mean(eps_dists)),
        "local_h_l2_mean": float(np.mean(h_dists)),
        "local_sequence_edit_mean": float(np.mean(edits)),
        "local_sequence_edit_median": float(np.median(edits)),
        "local_identical_fraction": float(np.mean(np.asarray(edits) == 0)),
    }


@torch.no_grad()
def evaluate(
    vae: GRUVAE,
    diffusion: LatentDiffusion,
    objective_head: MultiObjectiveHead,
    loader: DataLoader,
    peptides: Sequence[str],
    x_all: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    args,
    device: torch.device,
):
    vae.eval()
    diffusion.eval()
    objective_head.eval()

    diff_sum = 0.0
    obj_sum = 0.0
    n_batches = 0
    preds = []
    targets = []
    val_indices = []

    for x, y, idx in loader:
        x = x.to(device)
        y = y.to(device)
        mu, _, _ = vae.enc(x)
        h0 = standardize_mu(mu, latent_mean, latent_std)

        diff_loss = diffusion.training_loss(h0)
        pred = objective_head(h0)
        obj_loss = F.mse_loss(pred, y)

        diff_sum += float(diff_loss.cpu())
        obj_sum += float(obj_loss.cpu())
        preds.append(pred.cpu())
        targets.append(y.cpu())
        val_indices.extend([int(i) for i in idx.tolist()])
        n_batches += 1

    pred_all = torch.cat(preds)
    y_all = torch.cat(targets)

    metrics = {
        "diffusion_epsilon_mse": diff_sum / max(1, n_batches),
        "objective_mse": obj_sum / max(1, n_batches),
    }

    for j, name in enumerate(args.objective_cols):
        metrics[f"{name}_mse"] = float(
            F.mse_loss(pred_all[:, j], y_all[:, j]).item()
        )
        metrics[f"{name}_pearson"] = pearson_safe(pred_all[:, j], y_all[:, j])

    metrics["objective_mean_pearson"] = float(
        np.nanmean([metrics[f"{c}_pearson"] for c in args.objective_cols])
    )

    x_val = x_all[val_indices].to(device)
    val_peptides = [peptides[i] for i in val_indices]

    metrics.update(
        inversion_metrics(
            vae=vae,
            diffusion=diffusion,
            x_val=x_val,
            source_peptides=val_peptides,
            latent_mean=latent_mean,
            latent_std=latent_std,
            ddim_steps=args.ddim_steps,
            project_to_sphere=args.project_inverted_epsilon_to_sphere,
            temperature=args.decode_temperature,
        )
    )

    metrics.update(
        local_epsilon_geometry(
            vae=vae,
            diffusion=diffusion,
            x_val=x_val,
            latent_mean=latent_mean,
            latent_std=latent_std,
            ddim_steps=args.ddim_steps,
            sigma=args.local_noise_std,
            subset=args.local_smoothness_subset,
            neighbors=args.local_neighbors,
            seed=args.seed + 777,
            renormalize=args.local_renormalize_to_sphere,
        )
    )
    return metrics


def json_safe(obj):
    if torch.is_tensor(obj):
        arr = obj.detach().cpu()
        return float(arr.item()) if arr.numel() == 1 else arr.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def save_checkpoint(
    path: str,
    vae: GRUVAE,
    diffusion: LatentDiffusion,
    objective_head: MultiObjectiveHead,
    cfg: ModelConfig,
    diff_cfg: DiffusionConfig,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    args,
    epoch: int,
    metrics: Dict[str, float],
    selection: str,
    pretrained_meta: Dict[str, object],
    data_audit: Dict[str, object],
):
    payload = {
        "checkpoint_type": "cu_best_bo_ready_gru_vae_latent_diffusion",
        "vae_state_dict": vae.state_dict(),
        "diffusion_state_dict": diffusion.state_dict(),
        "objective_head_state_dict": objective_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": asdict(cfg),
        "diffusion_config": asdict(diff_cfg),
        "latent_mean": latent_mean.detach().cpu(),
        "latent_std": latent_std.detach().cpu(),
        "epoch": int(epoch),
        "metrics": metrics,
        "checkpoint_selection": selection,
        "pretrained_checkpoint": args.init_checkpoint,
        "pretrained_checkpoint_epoch": int(pretrained_meta.get("epoch", -1)),
        "pretrained_checkpoint_selection": pretrained_meta.get("metrics", {}).get(
            "selection_metric", ""
        ),
        "pretraining_split_manifest": args.pretraining_split_manifest,
        "data_audit": data_audit,
        "objective_cols": list(args.objective_cols),
        "bo_coordinate_space": "diffusion_base_noise_epsilon",
        "diffusion_target": "standardized_encoder_mu",
        "diffusion_prediction": "epsilon",
        "diffusion_sampler": "deterministic_ddim_eta0",
        "recommended_bo_radius": float(math.sqrt(cfg.latent_dim)),
        "vae_frozen": not bool(args.finetune_vae),
        "cu_csv": args.cu_csv,
        "args": vars(args),
    }
    torch.save(payload, path)

    sidecar = {
        k: v for k, v in payload.items()
        if k not in {
            "vae_state_dict",
            "diffusion_state_dict",
            "objective_head_state_dict",
            "optimizer_state_dict",
        }
    }
    with open(path + ".json", "w", encoding="utf-8") as f:
        json.dump(json_safe(sidecar), f, indent=2)


@torch.no_grad()
def export_bo_coordinates(
    path: str,
    vae: GRUVAE,
    diffusion: LatentDiffusion,
    df: pd.DataFrame,
    peptides: Sequence[str],
    x_all: torch.Tensor,
    y_all: torch.Tensor,
    final_score: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    args,
    device: torch.device,
):
    vae.eval()
    diffusion.eval()
    rows = []

    for start in range(0, len(peptides), args.batch_size):
        end = min(start + args.batch_size, len(peptides))
        x = x_all[start:end].to(device)

        mu, _, _ = vae.enc(x)
        h0 = standardize_mu(mu, latent_mean, latent_std)
        eps_raw = diffusion.ddim_invert(h0, inference_steps=args.ddim_steps)
        eps = eps_raw
        if args.project_inverted_epsilon_to_sphere:
            eps = project_to_radius(eps_raw, math.sqrt(vae.cfg.latent_dim))

        h_back = diffusion.ddim_sample(eps, inference_steps=args.ddim_steps)
        mu_back = unstandardize_h(h_back, latent_mean, latent_std)
        decoded = autoregressive_decode(vae, mu_back, args.decode_temperature)

        h_l2 = torch.linalg.norm(h0 - h_back, dim=-1)
        h_cos = F.cosine_similarity(h0, h_back, dim=-1, eps=1e-8)

        mu_np = mu.cpu().numpy()
        h0_np = h0.cpu().numpy()
        eps_np = eps.cpu().numpy()
        eps_raw_np = eps_raw.cpu().numpy()

        for i in range(end - start):
            gi = start + i
            row = {
                "peptide": peptides[gi],
                "peptide_len10": peptides[gi],
                "split": str(df.iloc[gi]["split"]),
                "decoded_from_inverted_epsilon": decoded[i],
                "source_to_decoded_edit": levenshtein_edit_distance(
                    peptides[gi], decoded[i]
                ),
                "epsilon_norm": float(np.linalg.norm(eps_np[i])),
                "epsilon_raw_norm": float(np.linalg.norm(eps_raw_np[i])),
                "ddim_h_l2": float(h_l2[i].cpu()),
                "ddim_h_cosine": float(h_cos[i].cpu()),
                "final_score": (
                    float(final_score[gi])
                    if torch.isfinite(final_score[gi]) else float("nan")
                ),
            }

            for j, name in enumerate(args.objective_cols):
                row[name] = float(y_all[gi, j])

            for j in range(eps_np.shape[1]):
                row[f"epsilon_{j:02d}"] = float(eps_np[i, j])
                row[f"epsilon_raw_{j:02d}"] = float(eps_raw_np[i, j])
                row[f"mu_{j:02d}"] = float(mu_np[i, j])
                row[f"h0_standardized_{j:02d}"] = float(h0_np[i, j])

            rows.append(row)

    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved BO coordinates: {path}")


def append_history(path: str, row: Dict[str, object]):
    pd.DataFrame([row]).to_csv(
        path, mode="a", header=not os.path.exists(path), index=False
    )



@torch.no_grad()
def evaluate_selected_checkpoint_on_test(
    label: str,
    checkpoint_path: str,
    vae: GRUVAE,
    diffusion: LatentDiffusion,
    objective_head: MultiObjectiveHead,
    test_loader: DataLoader,
    peptides: Sequence[str],
    x_all: torch.Tensor,
    args,
    device: torch.device,
):
    ckpt = torch_load_full(checkpoint_path, map_location=device)
    vae.load_state_dict(ckpt["vae_state_dict"], strict=True)
    diffusion.load_state_dict(ckpt["diffusion_state_dict"], strict=True)
    objective_head.load_state_dict(ckpt["objective_head_state_dict"], strict=True)
    latent_mean = ckpt["latent_mean"].to(device).float()
    latent_std = ckpt["latent_std"].to(device).float().clamp_min(1e-6)

    tm = evaluate(
        vae=vae,
        diffusion=diffusion,
        objective_head=objective_head,
        loader=test_loader,
        peptides=peptides,
        x_all=x_all,
        latent_mean=latent_mean,
        latent_std=latent_std,
        args=args,
        device=device,
    )

    payload = {
        "selected_checkpoint_label": label,
        "checkpoint_path": checkpoint_path,
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "checkpoint_selection": ckpt.get("checkpoint_selection", ""),
        "test_metrics": tm,
    }
    out_path = os.path.join(args.out_dir, f"test_metrics_{label}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2)

    print(
        f"Test [{label}]: eps_mse={tm['diffusion_epsilon_mse']:.6f} "
        f"obj_mse={tm['objective_mse']:.6f} "
        f"inv_l2={tm['ddim_inversion_h_l2_mean']:.4f} "
        f"eps_effdim={tm['epsilon_bo_effective_dim_pr_standardized']:.2f} "
        f"decode_edit={tm['ddim_source_to_decode_edit_mean']:.4f}"
    )
    return payload


# ---------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------

def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    (
        cu_df,
        peptides,
        x_all,
        y_all,
        final_score,
        split_to_idx,
        resolved_peptide_col,
        data_audit,
    ) = load_cu_dataset_with_manifest(
        args.cu_csv,
        args.peptide_col,
        args.objective_cols,
        args.pretraining_split_manifest,
        args.duplicate_policy,
        args.unmapped_policy,
        args.out_dir,
    )
    train_idx = split_to_idx["train"]
    val_idx = split_to_idx["val"]
    test_idx = split_to_idx["test"]

    cfg = ModelConfig(
        hidden_size=args.hidden_size,
        latent_dim=args.latent_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )
    diff_cfg = DiffusionConfig(
        hidden_dim=args.diffusion_hidden_dim,
        time_dim=args.diffusion_time_dim,
        n_blocks=args.diffusion_blocks,
        train_steps=args.diffusion_train_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    )

    vae = GRUVAE(cfg).to(device)
    pretrained_meta = validate_pretrained_checkpoint(
        vae=vae,
        checkpoint_path=args.init_checkpoint,
        device=device,
        require_bo_ready_selection=not args.allow_non_bo_ready_pretraining,
    )

    diffusion = LatentDiffusion(cfg.latent_dim, diff_cfg).to(device)
    objective_head = MultiObjectiveHead(
        cfg.latent_dim,
        n_objectives=len(args.objective_cols),
        hidden_dim=args.objective_head_hidden_dim,
    ).to(device)

    if args.finetune_vae:
        print(
            "WARNING: VAE fine-tuning enabled. This can change the validated "
            "pretrained latent geometry. Use only intentionally."
        )
        for p in vae.parameters():
            p.requires_grad_(True)
    else:
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        print("GRU-VAE frozen; training diffusion + multi-objective head only.")

    # IMPORTANT: fit normalization on training data only.
    latent_mean, latent_std = compute_latent_stats_from_indices(
        vae, x_all, train_idx, args.batch_size, device
    )
    print(
        f"Training-only latent stats: mean(|mean|)="
        f"{latent_mean.abs().mean().item():.4f}, "
        f"mean(std)={latent_std.mean().item():.4f}"
    )
    print(
        f"Recommended spherical epsilon radius = sqrt({cfg.latent_dim}) "
        f"= {math.sqrt(cfg.latent_dim):.6f}"
    )

    train_loader = make_loader(
        x_all, y_all, train_idx, args.batch_size, shuffle=True
    )
    val_loader = make_loader(
        x_all, y_all, val_idx, args.batch_size, shuffle=False
    )
    test_loader = make_loader(
        x_all, y_all, test_idx, args.batch_size, shuffle=False
    )

    param_groups = [
        {
            "params": diffusion.parameters(),
            "lr": args.diffusion_lr,
            "name": "latent_diffusion",
        },
        {
            "params": objective_head.parameters(),
            "lr": args.objective_head_lr,
            "name": "objective_head",
        },
    ]
    if args.finetune_vae:
        param_groups.append(
            {"params": vae.parameters(), "lr": args.vae_lr, "name": "gru_vae"}
        )

    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=args.weight_decay
    )

    history_path = os.path.join(
        args.out_dir, "training_history_cu_best_gru_vae_latent_diffusion.csv"
    )
    if os.path.exists(history_path) and not args.append_history:
        os.remove(history_path)

    best_inv_path = os.path.join(
        args.out_dir,
        f"best_ddim_inversion_l2_h{cfg.hidden_size}_z{cfg.latent_dim}_cu_latent_diffusion.pt",
    )
    best_diff_path = os.path.join(
        args.out_dir,
        f"best_val_diffusion_mse_h{cfg.hidden_size}_z{cfg.latent_dim}_cu_latent_diffusion.pt",
    )
    best_obj_path = os.path.join(
        args.out_dir,
        f"best_val_objective_mse_h{cfg.hidden_size}_z{cfg.latent_dim}_cu_latent_diffusion.pt",
    )
    best_bo_path = os.path.join(
        args.out_dir,
        f"best_bo_ready_h{cfg.hidden_size}_z{cfg.latent_dim}_cu_latent_diffusion.pt",
    )
    last_path = os.path.join(
        args.out_dir,
        f"last_epoch_h{cfg.hidden_size}_z{cfg.latent_dim}_cu_latent_diffusion.pt",
    )

    best_inv = float("inf")
    best_diff = float("inf")
    best_obj = float("inf")
    best_bo = float("inf")

    for epoch in range(1, args.epochs + 1):
        vae.train(args.finetune_vae)
        diffusion.train()
        objective_head.train()

        train_diff = 0.0
        train_obj = 0.0
        train_locality = 0.0
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
            if not args.finetune_vae:
                h0 = h0.detach()

            diff_loss = diffusion.training_loss(h0)
            pred_obj = objective_head(h0)
            obj_loss = F.mse_loss(pred_obj, y)
            locality_loss = objective_locality_loss(
                h0,
                y,
                tau_latent=args.locality_tau_latent,
                tau_objective=args.locality_tau_objective,
            )

            loss = (
                args.diffusion_loss_weight * diff_loss
                + args.objective_loss_weight * obj_loss
                + args.locality_loss_weight * locality_loss
            )
            loss.backward()

            clip_params = list(diffusion.parameters()) + list(objective_head.parameters())
            if args.finetune_vae:
                clip_params += list(vae.parameters())
            nn.utils.clip_grad_norm_(clip_params, args.grad_clip)

            optimizer.step()

            train_diff += float(diff_loss.detach().cpu())
            train_obj += float(obj_loss.detach().cpu())
            train_locality += float(locality_loss.detach().cpu())
            n_batches += 1

        train_metrics = {
            "diffusion_epsilon_mse": train_diff / max(1, n_batches),
            "objective_mse": train_obj / max(1, n_batches),
            "objective_locality_loss": train_locality / max(1, n_batches),
        }

        val_metrics = evaluate(
            vae=vae,
            diffusion=diffusion,
            objective_head=objective_head,
            loader=val_loader,
            peptides=peptides,
            x_all=x_all,
            latent_mean=latent_mean,
            latent_std=latent_std,
            args=args,
            device=device,
        )

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "epoch": epoch,
            "pretrained_checkpoint": args.init_checkpoint,
            "pretrained_epoch": int(pretrained_meta.get("epoch", -1)),
            "pretrained_selection": pretrained_meta.get("metrics", {}).get(
                "selection_metric", ""
            ),
            "vae_finetuned": bool(args.finetune_vae),
            "train_diffusion_epsilon_mse": train_metrics["diffusion_epsilon_mse"],
            "train_objective_mse": train_metrics["objective_mse"],
            "train_objective_locality_loss": train_metrics["objective_locality_loss"],
        }
        for k, v in val_metrics.items():
            row[f"val_{k}"] = v

        append_history(history_path, row)

        print(
            f"epoch={epoch:03d} "
            f"train_eps_mse={train_metrics['diffusion_epsilon_mse']:.6f} "
            f"val_eps_mse={val_metrics['diffusion_epsilon_mse']:.6f} "
            f"val_obj_mse={val_metrics['objective_mse']:.6f} "
            f"val_obj_pearson={val_metrics['objective_mean_pearson']:.4f} "
            f"train_loc={train_metrics['objective_locality_loss']:.6f} "
            f"inv_l2={val_metrics['ddim_inversion_h_l2_mean']:.4f} "
            f"inv_cos={val_metrics['ddim_inversion_h_cosine_mean']:.4f} "
            f"h0_effdim={val_metrics['h0_effective_dim_pr_standardized']:.2f} "
            f"eps_effdim={val_metrics['epsilon_bo_effective_dim_pr_standardized']:.2f} "
            f"eps_corr={val_metrics['epsilon_bo_mean_abs_offdiag_corr']:.3f} "
            f"edit={val_metrics['ddim_source_to_decode_edit_mean']:.4f} "
            f"local_edit={val_metrics.get('local_sequence_edit_mean', float('nan')):.4f}"
        )

        if val_metrics["ddim_inversion_h_l2_mean"] < best_inv:
            best_inv = val_metrics["ddim_inversion_h_l2_mean"]
            save_checkpoint(
                best_inv_path, vae, diffusion, objective_head, cfg, diff_cfg,
                latent_mean, latent_std, optimizer, args, epoch, val_metrics,
                "minimum_validation_ddim_inversion_h_l2",
                pretrained_meta,
                data_audit,
            )

        if val_metrics["diffusion_epsilon_mse"] < best_diff:
            best_diff = val_metrics["diffusion_epsilon_mse"]
            save_checkpoint(
                best_diff_path, vae, diffusion, objective_head, cfg, diff_cfg,
                latent_mean, latent_std, optimizer, args, epoch, val_metrics,
                "minimum_validation_diffusion_epsilon_mse",
                pretrained_meta,
                data_audit,
            )

        if val_metrics["objective_mse"] < best_obj:
            best_obj = val_metrics["objective_mse"]
            save_checkpoint(
                best_obj_path, vae, diffusion, objective_head, cfg, diff_cfg,
                latent_mean, latent_std, optimizer, args, epoch, val_metrics,
                "minimum_validation_multiobjective_mse",
                pretrained_meta,
                data_audit,
            )

        # BO-ready latent-diffusion checkpoint: prioritize invertibility while
        # softly discouraging an epsilon space that is lower-dimensional or more
        # correlated than the already healthy h0 representation.
        eps_effdim = float(
            val_metrics.get("epsilon_bo_effective_dim_pr_standardized", float("nan"))
        )
        h0_effdim = float(
            val_metrics.get("h0_effective_dim_pr_standardized", float("nan"))
        )
        eps_corr = float(
            val_metrics.get("epsilon_bo_mean_abs_offdiag_corr", float("nan"))
        )
        if all(math.isfinite(v) for v in [
            val_metrics["ddim_inversion_h_l2_mean"], eps_effdim, h0_effdim, eps_corr
        ]):
            objective_mse = float(val_metrics.get("objective_mse", float("inf")))
            objective_pearson = float(
                val_metrics.get("objective_mean_pearson", float("-inf"))
            )

            eff_target = max(
                float(args.bo_selection_min_epsilon_effective_dim),
                float(args.bo_selection_effdim_fraction_of_h0) * h0_effdim,
            )
            eff_deficit = max(0.0, eff_target - eps_effdim) / max(1e-8, eff_target)

            pearson_deficit = max(
                0.0,
                float(args.bo_selection_min_objective_pearson) - objective_pearson,
            )
            mse_excess = max(
                0.0,
                objective_mse - float(args.bo_selection_max_objective_mse),
            )

            passes_objective_gate = (
                objective_pearson >= float(args.bo_selection_min_objective_pearson)
                and objective_mse <= float(args.bo_selection_max_objective_mse)
                and eps_effdim >= float(args.bo_selection_min_epsilon_effective_dim)
            )

            bo_score = (
                float(args.bo_selection_inversion_weight)
                * val_metrics["ddim_inversion_h_l2_mean"]
                + float(args.bo_selection_diffusion_weight)
                * val_metrics["diffusion_epsilon_mse"]
                + float(args.bo_selection_objective_mse_weight)
                * objective_mse
                + float(args.bo_selection_objective_pearson_penalty_weight)
                * pearson_deficit
                + float(args.bo_selection_objective_mse_penalty_weight)
                * mse_excess
                + float(args.bo_selection_effdim_penalty_weight)
                * eff_deficit
                + float(args.bo_selection_corr_penalty_weight)
                * eps_corr
                + float(args.bo_selection_local_edit_weight)
                * val_metrics.get("local_sequence_edit_mean", 0.0)
            )

            if args.bo_selection_require_objective_gate and not passes_objective_gate:
                bo_score_for_selection = float("inf")
            else:
                bo_score_for_selection = bo_score

            if bo_score_for_selection < best_bo:
                best_bo = bo_score_for_selection
                bo_metrics = dict(val_metrics)
                bo_metrics.update({
                    "bo_ready_score": bo_score,
                    "bo_ready_score_for_selection": bo_score_for_selection,
                    "bo_selection_passes_objective_gate": passes_objective_gate,
                    "bo_selection_objective_mse": objective_mse,
                    "bo_selection_objective_mean_pearson": objective_pearson,
                    "bo_selection_objective_pearson_deficit": pearson_deficit,
                    "bo_selection_objective_mse_excess": mse_excess,
                    "bo_selection_epsilon_effective_dim": eps_effdim,
                    "bo_selection_h0_effective_dim": h0_effdim,
                    "bo_selection_effective_dim_target": eff_target,
                    "bo_selection_effective_dim_deficit": eff_deficit,
                    "bo_selection_epsilon_mean_abs_corr": eps_corr,
                })
                save_checkpoint(
                    best_bo_path, vae, diffusion, objective_head, cfg, diff_cfg,
                    latent_mean, latent_std, optimizer, args, epoch, bo_metrics,
                    "minimum_objective_gated_bo_ready_score",
                    pretrained_meta,
                    data_audit,
                )

        save_checkpoint(
            last_path, vae, diffusion, objective_head, cfg, diff_cfg,
            latent_mean, latent_std, optimizer, args, epoch, val_metrics,
            "last_completed_epoch",
            pretrained_meta,
            data_audit,
        )

    # If a hard objective gate was requested and no checkpoint passed it, fall back
    # to the best validation objective checkpoint rather than exporting a poor
    # early BO-ready checkpoint.
    if args.bo_selection_require_objective_gate and not os.path.exists(best_bo_path):
        print(
            "WARNING: No checkpoint passed the objective-gated BO-ready criteria; "
            "using best validation objective-MSE checkpoint as the BO-ready fallback."
        )
        if os.path.exists(best_obj_path):
            best_bo_path = best_obj_path

    # Evaluate validation-selected checkpoints once on the untouched Cu test set.
    selected = [
        ("best_inversion", best_inv_path),
        ("best_diffusion_mse", best_diff_path),
        ("best_objective_mse", best_obj_path),
        ("best_bo_ready", best_bo_path),
    ]
    test_results = {}
    for label, path in selected:
        if os.path.exists(path):
            test_results[label] = evaluate_selected_checkpoint_on_test(
                label=label,
                checkpoint_path=path,
                vae=vae,
                diffusion=diffusion,
                objective_head=objective_head,
                test_loader=test_loader,
                peptides=peptides,
                x_all=x_all,
                args=args,
                device=device,
            )
    with open(
        os.path.join(args.out_dir, "test_metrics_selected_checkpoints.json"),
        "w", encoding="utf-8",
    ) as f:
        json.dump(json_safe(test_results), f, indent=2)

    if args.export_inversion_coordinates:
        selected_path = {
            "inversion": best_inv_path,
            "diffusion_mse": best_diff_path,
            "objective_mse": best_obj_path,
            "bo_ready": best_bo_path,
        }[args.export_checkpoint]

        ckpt = torch_load_full(selected_path, map_location=device)
        vae.load_state_dict(ckpt["vae_state_dict"], strict=True)
        diffusion.load_state_dict(ckpt["diffusion_state_dict"], strict=True)
        latent_mean = ckpt["latent_mean"].to(device).float()
        latent_std = ckpt["latent_std"].to(device).float().clamp(min=1e-6)

        export_bo_coordinates(
            path=os.path.join(
                args.out_dir, "cu_ddim_inversion_coordinates_for_bo.csv"
            ),
            vae=vae,
            diffusion=diffusion,
            df=cu_df,
            peptides=peptides,
            x_all=x_all,
            y_all=y_all,
            final_score=final_score,
            latent_mean=latent_mean,
            latent_std=latent_std,
            args=args,
            device=device,
        )
        with open(
            os.path.join(args.out_dir, "bo_coordinate_export_checkpoint.json"),
            "w", encoding="utf-8",
        ) as f:
            json.dump({
                "export_checkpoint_choice": args.export_checkpoint,
                "checkpoint_path": selected_path,
                "checkpoint_epoch": int(ckpt.get("epoch", -1)),
                "checkpoint_selection": ckpt.get("checkpoint_selection", ""),
                "project_inverted_epsilon_to_sphere":
                    bool(args.project_inverted_epsilon_to_sphere),
            }, f, indent=2)

    print(f"Best inversion checkpoint: {best_inv_path}")
    print(f"Best validation diffusion-MSE checkpoint: {best_diff_path}")
    print(f"Best validation objective-MSE checkpoint: {best_obj_path}")
    print(f"Best BO-ready checkpoint: {best_bo_path}")
    print(f"History: {history_path}")
    return best_inv_path


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Train Cu-specific latent diffusion on the best BO-ready H64/Z64 "
            "GRU-VAE encoder-mean space."
        )
    )

    p.add_argument(
        "--init-checkpoint",
        required=True,
        help=(
            "Use best_bo_ready_gru_vae_latent_conditioned_h64_z64.pt from the "
            "revised leakage-safe/decorrelated pretraining."
        ),
    )
    p.add_argument(
        "--pretraining-split-manifest",
        required=True,
        help="Path to pretraining_leakage_safe_split_manifest.csv.",
    )
    p.add_argument("--allow-non-bo-ready-pretraining", action="store_true")

    p.add_argument("--cu-csv", required=True)
    p.add_argument("--peptide-col", default="peptide_len10")
    p.add_argument(
        "--objective-cols",
        nargs="+",
        default=DEFAULT_OBJECTIVE_COLS,
    )
    p.add_argument(
        "--duplicate-policy",
        choices=["mean", "first", "error"],
        default="mean",
    )
    p.add_argument(
        "--unmapped-policy",
        choices=["drop", "error"],
        default="drop",
        help=(
            "Default drop preserves the exact leakage-safe pretraining partition."
        ),
    )

    p.add_argument(
        "--out-dir",
        default="cu_best_gru_vae_latent_diffusion_h64_z64_leakage_safe",
    )

    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--diffusion-hidden-dim", type=int, default=128)
    p.add_argument("--diffusion-time-dim", type=int, default=32)
    p.add_argument("--diffusion-blocks", type=int, default=4)
    p.add_argument("--diffusion-train-steps", type=int, default=100)
    p.add_argument("--beta-start", type=float, default=1e-5)
    p.add_argument("--beta-end", type=float, default=8e-3)

    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    p.add_argument(
        "--finetune-vae",
        action="store_true",
        help=(
            "Default: freeze the pretrained VAE. Enabling this can alter the "
            "pretrained latent geometry."
        ),
    )
    p.add_argument("--vae-lr", type=float, default=1e-6)
    p.add_argument("--diffusion-lr", type=float, default=3e-5)
    p.add_argument("--objective-head-lr", type=float, default=1e-4)
    p.add_argument("--objective-head-hidden-dim", type=int, default=64)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=5.0)

    p.add_argument("--diffusion-loss-weight", type=float, default=1.0)
    p.add_argument("--objective-loss-weight", type=float, default=0.1)
    p.add_argument(
        "--locality-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Optional objective-locality regularizer on h0. Start with 0.01 as an ablation."
        ),
    )
    p.add_argument("--locality-tau-latent", type=float, default=4.0)
    p.add_argument("--locality-tau-objective", type=float, default=0.35)

    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--decode-temperature", type=float, default=1.0)

    p.add_argument(
        "--project-inverted-epsilon-to-sphere",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Default False for the revised healthy h0 space. Sphere projection "
            "can be enabled as an explicit ablation but changes the inverted "
            "coordinate and may worsen DDIM roundtrip fidelity."
        ),
    )

    p.add_argument("--local-smoothness-subset", type=int, default=128)
    p.add_argument("--local-neighbors", type=int, default=3)
    p.add_argument("--local-noise-std", type=float, default=0.05)
    p.add_argument("--local-renormalize-to-sphere", action="store_true")

    p.add_argument(
        "--bo-selection-min-epsilon-effective-dim",
        type=float,
        default=24.0,
        help="Soft/hard floor for epsilon standardized PR effective dimension.",
    )
    p.add_argument(
        "--bo-selection-effdim-fraction-of-h0",
        type=float,
        default=0.75,
        help="Soft target: epsilon PR should retain at least this fraction of h0 PR.",
    )
    p.add_argument("--bo-selection-inversion-weight", type=float, default=1.0)
    p.add_argument("--bo-selection-diffusion-weight", type=float, default=0.05)
    p.add_argument("--bo-selection-effdim-penalty-weight", type=float, default=0.05)
    p.add_argument("--bo-selection-corr-penalty-weight", type=float, default=0.02)
    p.add_argument("--bo-selection-local-edit-weight", type=float, default=0.02)
    p.add_argument("--bo-selection-min-objective-pearson", type=float, default=0.45)
    p.add_argument("--bo-selection-max-objective-mse", type=float, default=0.008)
    p.add_argument("--bo-selection-objective-mse-weight", type=float, default=1.0)
    p.add_argument(
        "--bo-selection-objective-pearson-penalty-weight",
        type=float,
        default=1.0,
    )
    p.add_argument(
        "--bo-selection-objective-mse-penalty-weight",
        type=float,
        default=5.0,
    )
    p.add_argument(
        "--bo-selection-require-objective-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Default True: do not allow a BO-ready checkpoint unless it passes "
            "objective MSE/Pearson and epsilon effective-dimension gates."
        ),
    )

    p.add_argument("--append-history", action="store_true")
    p.add_argument(
        "--export-inversion-coordinates",
        action="store_true",
        default=True,
    )
    p.add_argument(
        "--no-export-inversion-coordinates",
        dest="export_inversion_coordinates",
        action="store_false",
    )
    p.add_argument(
        "--export-checkpoint",
        choices=["inversion", "diffusion_mse", "objective_mse", "bo_ready"],
        default="bo_ready",
        help="Checkpoint used for downstream epsilon-coordinate export.",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
