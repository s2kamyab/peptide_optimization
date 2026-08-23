from __future__ import annotations

"""
Leakage-safe Cu-specific RealNVP fine-tuning on the revised BO-ready H64/Z64 GRU-VAE.

This version is aligned with:
    pretrain_gru_vae_bo_ready_h64_z64_leakage_safe_decorrelated.py

Key changes versus the earlier RealNVP fine-tuning script
---------------------------------------------------------
1. Loads the BO-ready pretrained GRU-VAE checkpoint directly and verifies that it
   was selected with the revised BO-aware criterion (unless explicitly overridden).
2. Reuses pretraining_leakage_safe_split_manifest.csv instead of creating a new
   exact-peptide SHA256 train/validation split.
3. Preserves train / validation / test partitions and never uses the test split
   for checkpoint selection.
4. Drops peptides marked excluded_near_train / excluded_near_development.
5. Deduplicates Cu sequences before fine-tuning (default: mean aggregation if the
   same peptide has multiple scored rows).
6. Keeps the GRU-VAE frozen by default so the revised pretrained latent geometry
   is preserved and training-only latent normalization remains valid.
7. Logs latent geometry for encoder mu, standardized h0, and RealNVP zK:
      participation-ratio effective dimension
      standardized effective dimension
      PC1 variance fraction
      PCs for 90% / 95% variance
      mean / maximum absolute off-diagonal correlation
8. Saves best-flow-NLL, best-objective-MSE, best-BO-ready, and last checkpoints.
9. After model selection, evaluates each selected checkpoint once on the untouched
   Cu test split and writes JSON metrics.
10. Exports BO coordinates with each peptide's manifest split.
11. Uses a conservative 4-layer / max_scale=0.5 RealNVP by default.
12. Uses an objective-aware loss with reduced NLL dominance plus locality and
    weak identity regularization to avoid over-reshaping the revised h0 space.

Matched generative path
-----------------------
peptide
  -> frozen GRU-VAE encoder mu
  -> Cu-training-only standardization h0
  -> RealNVP
  -> zK                                      [candidate BO coordinate]

zK
  -> exact RealNVP inverse
  -> h0
  -> unstandardize
  -> decoder mu
  -> autoregressive peptide

The four Cu objectives are:
    chelation_sub
    solubility_sub
    stability_sub
    expression_sub
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


# =============================================================================
# Config
# =============================================================================

@dataclass
class ModelConfig:
    hidden_size: int = 64
    latent_dim: int = 64
    n_layers: int = 2
    dropout: float = 0.0
    decoder_conditioning: str = "concat_z_at_every_decoder_step"


@dataclass
class RealNVPConfig:
    # The revised GRU-VAE latent is already well-conditioned, so the flow is
    # deliberately conservative: fewer coupling blocks and smaller scaling.
    n_layers: int = 4
    hidden_dim: int = 128
    max_scale: float = 0.5


# =============================================================================
# General utilities
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_load_full(path: str, map_location):
    """
    Compatible with PyTorch versions where torch.load changed the weights_only
    default. These checkpoints intentionally contain metadata in addition to tensors.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def clean_peptide(x: object) -> Optional[str]:
    p = str(x).strip().upper()
    if len(p) != SEQ_LEN:
        return None
    if any(a not in AA_TO_I for a in p):
        return None
    return p


def onehot_encode_peptides(peptides: Sequence[str]) -> torch.Tensor:
    x = torch.zeros((len(peptides), SEQ_LEN, VOCAB), dtype=torch.float32)
    for n, pep in enumerate(peptides):
        p = clean_peptide(pep)
        if p is None:
            raise ValueError(f"Invalid peptide: {pep!r}")
        for t, aa in enumerate(p):
            x[n, t, AA_TO_I[aa]] = 1.0
    return x


def levenshtein_edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(
                min(
                    cur[j - 1] + 1,
                    prev[j] + 1,
                    prev[j - 1] + int(ca != cb),
                )
            )
        prev = cur
    return int(prev[-1])


def pearson_safe(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().cpu().flatten()
    b = b.detach().float().cpu().flatten()
    if a.numel() < 2:
        return float("nan")
    if float(a.std()) < 1e-12 or float(b.std()) < 1e-12:
        return float("nan")
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def json_safe(o):
    if torch.is_tensor(o):
        return float(o.item()) if o.numel() == 1 else o.detach().cpu().tolist()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    return o


def append_history(path: str, row: Dict[str, object]) -> None:
    pd.DataFrame([row]).to_csv(
        path,
        mode="a",
        header=not os.path.exists(path),
        index=False,
    )


# =============================================================================
# GRU-VAE: checkpoint-compatible with revised pretraining
# =============================================================================

class GRUEncoder(nn.Module):
    def __init__(self, hidden_size, latent_dim, n_layers, dropout):
        super().__init__()
        self.in_proj = nn.Linear(VOCAB, hidden_size)
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.to_mu = nn.Linear(hidden_size, latent_dim)
        self.to_logvar = nn.Linear(hidden_size, latent_dim)

    def forward(self, x):
        h = self.in_proj(x)
        out, h_n = self.gru(h)
        last = h_n[-1]
        return (
            self.to_mu(last),
            self.to_logvar(last).clamp(-8.0, 8.0),
            out,
        )


class LatentConditionedGRUDecoder(nn.Module):
    def __init__(self, hidden_size, latent_dim, n_layers, dropout):
        super().__init__()
        self.token_embed = nn.Linear(VOCAB, hidden_size)
        self.z_to_h = nn.Linear(latent_dim, n_layers * hidden_size)
        self.gru = nn.GRU(
            hidden_size + latent_dim,
            hidden_size,
            n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.to_logits = nn.Linear(hidden_size, VOCAB)

    def initial_hidden(self, z):
        return self.z_to_h(z).view(
            self.gru.num_layers,
            z.size(0),
            self.gru.hidden_size,
        )

    def step(self, z, current_token, h):
        emb = self.token_embed(current_token)
        dec_in = torch.cat([emb, z.unsqueeze(1)], dim=-1)
        out, h = self.gru(dec_in, h)
        return self.to_logits(out[:, -1]), h


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
        self.dec = LatentConditionedGRUDecoder(
            cfg.hidden_size,
            cfg.latent_dim,
            cfg.n_layers,
            cfg.dropout,
        )


# =============================================================================
# RealNVP
# =============================================================================

class AffineCouplingBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        mask: torch.Tensor,
        max_scale: float,
    ):
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

        # Start exactly at the identity transform.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _st(self, x_masked):
        s, t = self.net(x_masked).chunk(2, dim=-1)
        return torch.tanh(s) * self.max_scale, t

    def forward(self, x):
        xm = x * self.mask
        s, t = self._st(xm)
        im = 1.0 - self.mask
        y = xm + im * (x * torch.exp(s) + t)
        return y, (im * s).sum(dim=-1)

    def inverse(self, y):
        ym = y * self.mask
        s, t = self._st(ym)
        im = 1.0 - self.mask
        x = ym + im * ((y - t) * torch.exp(-s))
        return x, -(im * s).sum(dim=-1)


class RealNVPFlow(nn.Module):
    def __init__(self, dim: int, cfg: RealNVPConfig):
        super().__init__()
        masks = []
        for layer in range(cfg.n_layers):
            m = torch.zeros(dim)
            m[layer % 2 :: 2] = 1.0
            masks.append(m)

        self.blocks = nn.ModuleList([
            AffineCouplingBlock(
                dim,
                cfg.hidden_dim,
                mask,
                cfg.max_scale,
            )
            for mask in masks
        ])

    def forward(self, x):
        z = x
        ld = torch.zeros(
            x.size(0),
            dtype=x.dtype,
            device=x.device,
        )
        for block in self.blocks:
            z, d = block(z)
            ld = ld + d
        return z, ld

    def inverse(self, z):
        x = z
        ld = torch.zeros(
            z.size(0),
            dtype=z.dtype,
            device=z.device,
        )
        for block in reversed(self.blocks):
            x, d = block.inverse(x)
            ld = ld + d
        return x, ld


class MultiObjectiveHead(nn.Module):
    def __init__(self, latent_dim, n_objectives, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_objectives),
        )

    def forward(self, z):
        return self.net(z)


def standard_normal_log_prob(z):
    return -0.5 * (
        z.pow(2) + math.log(2.0 * math.pi)
    ).sum(dim=-1)


def flow_nll(flow, h0):
    zK, logdet = flow(h0)
    nll = -(
        standard_normal_log_prob(zK)
        + logdet
    ).mean()
    return nll, zK, logdet


def relative_distance_preservation_loss(
    h0: torch.Tensor,
    zK: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Preserve the useful neighborhood geometry already learned by the revised
    GRU-VAE while still allowing RealNVP to make a nonlinear correction.

    Pairwise distances are normalized by their batch means so the penalty
    focuses on relative geometry rather than forcing identical absolute scale.
    """
    if h0.size(0) < 2:
        return zK.new_zeros(())

    d_h = torch.pdist(h0, p=2)
    d_z = torch.pdist(zK, p=2)

    # h0 is frozen in the recommended experiment, but detach it explicitly so
    # the reference geometry is never optimized through this term.
    ref_scale = d_h.detach().mean().clamp_min(eps)
    z_scale = d_z.mean().clamp_min(eps)

    d_h_norm = d_h.detach() / ref_scale
    d_z_norm = d_z / z_scale

    return F.mse_loss(d_z_norm, d_h_norm)


def identity_preservation_loss(
    h0: torch.Tensor,
    zK: torch.Tensor,
) -> torch.Tensor:
    """
    Weak identity prior. RealNVP starts as the identity transform; this term
    discourages unnecessary distortion of an already BO-usable h0 space.
    """
    return F.mse_loss(zK, h0.detach())


# =============================================================================
# Revised pretraining checkpoint validation
# =============================================================================

def validate_pretrained_checkpoint(
    vae: GRUVAE,
    path: str,
    device: torch.device,
    expected_cfg: ModelConfig,
    require_bo_ready_selection: bool,
):
    ckpt = torch_load_full(path, map_location=device)

    ctype = ckpt.get("checkpoint_type", "")
    expected_type = "pretrained_gru_vae_latent_conditioned_bo_ready_no_flow"
    if ctype and ctype != expected_type:
        raise ValueError(
            f"Unexpected pretrained checkpoint_type={ctype!r}; "
            f"expected {expected_type!r}"
        )

    saved_cfg = ckpt.get("model_config", {})
    expected = asdict(expected_cfg)

    mismatches = {}
    for key in [
        "hidden_size",
        "latent_dim",
        "n_layers",
        "dropout",
        "decoder_conditioning",
    ]:
        if key in saved_cfg and saved_cfg.get(key) != expected.get(key):
            mismatches[key] = (
                saved_cfg.get(key),
                expected.get(key),
            )

    if mismatches:
        raise ValueError(
            f"Pretrained checkpoint architecture mismatch: {mismatches}"
        )

    state = ckpt.get("model_state_dict")
    if state is None:
        raise KeyError(
            "Revised pretraining checkpoint must contain model_state_dict."
        )

    vae.load_state_dict(state, strict=True)

    metrics = ckpt.get("metrics", {})
    selection_metric = metrics.get("selection_metric", "")
    selection_value = metrics.get("selection_value", float("nan"))

    if require_bo_ready_selection and selection_metric != "bo_ready_score":
        raise RuntimeError(
            "The supplied pretrained checkpoint was not selected using "
            f"bo_ready_score (found {selection_metric!r}). "
            "Use best_bo_ready_gru_vae_latent_conditioned_h64_z64.pt, "
            "or pass --allow-non-bo-ready-pretraining deliberately."
        )

    validation_meta = ckpt.get("validation", {})
    split_method = validation_meta.get("split_method", "")
    if split_method != "connected_component_hash_of_peptide_PDB_bipartite_graph":
        raise RuntimeError(
            "The supplied checkpoint does not advertise the revised leakage-safe "
            f"split method (found {split_method!r})."
        )

    print(
        f"Loaded revised pretrained GRU-VAE: epoch={ckpt.get('epoch')} "
        f"selection={selection_metric} value={selection_value}"
    )

    effdim = metrics.get(
        "selection_effective_dim_pr_standardized",
        metrics.get("val_effective_dim_pr_standardized", float("nan")),
    )
    corr = metrics.get(
        "selection_mean_abs_offdiag_corr",
        metrics.get("val_mean_abs_offdiag_corr", float("nan")),
    )
    val_free = metrics.get(
        "selection_val_free_recon",
        metrics.get("val_free_recon", float("nan")),
    )

    print(
        "Pretrained geometry: "
        f"val_free_recon={val_free}, "
        f"effective_dim_std={effdim}, "
        f"mean_abs_offdiag_corr={corr}"
    )

    return ckpt


# =============================================================================
# Cu data + leakage-safe manifest
# =============================================================================

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
        conflicts = (
            dup.groupby("peptide")["split"]
            .nunique()
            .sort_values(ascending=False)
        )
        bad = conflicts[conflicts > 1]
        if len(bad):
            raise RuntimeError(
                "Manifest assigns some peptides to multiple splits; "
                f"examples: {bad.head().to_dict()}"
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
        f"Could not find peptide column {requested!r} or common alternatives."
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
        raise ValueError(f"Missing objective columns: {missing}")

    keep_cols = [peptide_col] + list(objective_cols)
    if "final_score" in raw.columns:
        keep_cols.append("final_score")

    df = raw[keep_cols].copy()
    df["peptide"] = df[peptide_col].map(clean_peptide)

    for c in objective_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "final_score" in df.columns:
        df["final_score"] = pd.to_numeric(
            df["final_score"], errors="coerce"
        )

    df = df.dropna(
        subset=["peptide"] + list(objective_cols)
    ).reset_index(drop=True)

    rows_before_dedup = len(df)
    unique_before = df["peptide"].nunique()
    duplicate_rows = rows_before_dedup - unique_before

    if duplicate_policy == "error":
        if duplicate_rows > 0:
            raise RuntimeError(
                f"Cu dataset contains {duplicate_rows} duplicate peptide rows. "
                "Use --duplicate-policy mean or first."
            )

    elif duplicate_policy == "first":
        df = df.drop_duplicates("peptide", keep="first").copy()

    elif duplicate_policy == "mean":
        agg = {c: "mean" for c in objective_cols}
        if "final_score" in df.columns:
            agg["final_score"] = "mean"
        df = (
            df.groupby("peptide", as_index=False)
            .agg(agg)
        )

    else:
        raise ValueError(
            f"Unknown duplicate_policy={duplicate_policy!r}"
        )

    if "final_score" not in df.columns:
        df["final_score"] = np.nan

    manifest = load_manifest(manifest_path)
    split_map = dict(
        zip(
            manifest["peptide"],
            manifest["split"],
        )
    )
    df["split"] = df["peptide"].map(split_map)

    n_unmapped = int(df["split"].isna().sum())
    unmapped_df = df.loc[df["split"].isna()].copy()
    if n_unmapped:
        examples = (
            unmapped_df["peptide"]
            .head(10)
            .tolist()
        )

        # Save every unmapped peptide so the exclusion is explicit and auditable.
        os.makedirs(out_dir, exist_ok=True)
        unmapped_path = os.path.join(
            out_dir,
            "cu_peptides_unmapped_from_pretraining_manifest.csv",
        )
        unmapped_df.to_csv(unmapped_path, index=False)

        msg = (
            f"{n_unmapped} Cu peptides were not found in the pretraining split "
            f"manifest; examples={examples}. "
            f"Saved full list to {unmapped_path}"
        )

        if unmapped_policy == "error":
            raise RuntimeError(
                msg
                + ". For the strict leakage-safe matched experiment, the safest "
                  "choice is --unmapped-policy drop. Do NOT assign unmapped peptides "
                  "randomly to train/validation/test, because that would bypass the "
                  "PDB/Hamming leakage controls used during pretraining."
            )

        print("WARNING:", msg)
        print(
            "Dropping unmapped Cu peptides to preserve the exact leakage-safe "
            "pretraining partition."
        )
        df = df.dropna(subset=["split"]).copy()

    excluded_labels = {
        "excluded_near_train",
        "excluded_near_development",
    }
    n_excluded = int(
        df["split"].isin(excluded_labels).sum()
    )
    if n_excluded:
        print(
            f"Excluding {n_excluded} Cu peptides marked as strict-holdout "
            "near-neighbors in the pretraining manifest."
        )
        df = df[
            ~df["split"].isin(excluded_labels)
        ].copy()

    allowed = {"train", "val", "test"}
    unexpected = sorted(
        set(df["split"].unique()) - allowed
    )
    if unexpected:
        raise RuntimeError(
            f"Unexpected manifest split labels in Cu data: {unexpected}"
        )

    df = df.reset_index(drop=True)

    split_counts = (
        df["split"]
        .value_counts()
        .reindex(["train", "val", "test"])
        .fillna(0)
        .astype(int)
        .to_dict()
    )

    if split_counts["train"] == 0:
        raise RuntimeError("No Cu training peptides remain after manifest mapping.")
    if split_counts["val"] == 0:
        raise RuntimeError("No Cu validation peptides remain after manifest mapping.")
    if split_counts["test"] == 0:
        raise RuntimeError("No Cu test peptides remain after manifest mapping.")

    audit = {
        "cu_csv": csv_path,
        "peptide_column": peptide_col,
        "rows_before_dedup": int(rows_before_dedup),
        "unique_peptides_before_dedup": int(unique_before),
        "duplicate_rows": int(duplicate_rows),
        "duplicate_policy": duplicate_policy,
        "rows_after_dedup_and_manifest_filtering": int(len(df)),
        "unmapped_peptides": int(n_unmapped),
        "unmapped_policy": unmapped_policy,
        "unmapped_peptides_excluded": int(n_unmapped if unmapped_policy == "drop" else 0),
        "excluded_manifest_near_neighbors": int(n_excluded),
        "split_counts": split_counts,
        "split_source": manifest_path,
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(
        os.path.join(out_dir, "cu_finetuning_data_audit.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(audit, f, indent=2)

    df.to_csv(
        os.path.join(out_dir, "cu_finetuning_manifest_mapped.csv"),
        index=False,
    )

    peptides = df["peptide"].astype(str).tolist()
    x_all = onehot_encode_peptides(peptides)
    y_all = torch.tensor(
        df[list(objective_cols)].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    final_score = torch.tensor(
        df["final_score"].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )

    split_to_idx = {}
    for split in ["train", "val", "test"]:
        split_to_idx[split] = torch.tensor(
            np.flatnonzero(
                df["split"].to_numpy() == split
            ),
            dtype=torch.long,
        )

    print(
        "Cu leakage-safe split: "
        f"total={len(df)} "
        f"train={len(split_to_idx['train'])} "
        f"val={len(split_to_idx['val'])} "
        f"test={len(split_to_idx['test'])}"
    )

    return (
        df,
        peptides,
        x_all,
        y_all,
        final_score,
        split_to_idx,
        peptide_col,
        audit,
    )


# =============================================================================
# Latent statistics + geometry
# =============================================================================

@torch.no_grad()
def compute_latent_stats(
    vae,
    x_all,
    idx,
    batch_size,
    device,
):
    vae.eval()
    mus = []

    for s in range(0, len(idx), batch_size):
        x = x_all[
            idx[s:s + batch_size]
        ].to(device)
        mu, _, _ = vae.enc(x)
        mus.append(mu)

    mu = torch.cat(mus, dim=0)

    return (
        mu.mean(0),
        mu.std(
            0,
            unbiased=False,
        ).clamp_min(1e-6),
    )


def standardize_mu(mu, mean, std):
    return (
        mu - mean.to(mu.device)
    ) / std.to(mu.device).clamp_min(1e-6)


def unstandardize_h(h, mean, std):
    return (
        h
        * std.to(h.device).clamp_min(1e-6)
        + mean.to(h.device)
    )


def latent_geometry_diagnostics(
    x: torch.Tensor,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Covariance-spectrum diagnostics, matching the revised pretraining analysis.
    """
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

    def spectrum_stats(v):
        v = v.double()
        v = v - v.mean(
            dim=0,
            keepdim=True,
        )
        cov = (
            v.T @ v
        ) / max(
            1,
            v.size(0) - 1,
        )

        eig = torch.linalg.eigvalsh(
            cov
        ).clamp_min(0.0)

        total = eig.sum().clamp_min(
            eps
        )

        pr = (
            total * total
            / eig.pow(2).sum().clamp_min(eps)
        ).item()

        eig_desc = torch.flip(
            eig,
            dims=[0],
        )
        frac = eig_desc / total
        cum = torch.cumsum(
            frac,
            dim=0,
        )

        pc1 = float(frac[0].item())

        pcs90 = int(
            torch.searchsorted(
                cum,
                torch.tensor(
                    0.90,
                    dtype=cum.dtype,
                    device=cum.device,
                ),
            ).item()
            + 1
        )

        pcs95 = int(
            torch.searchsorted(
                cum,
                torch.tensor(
                    0.95,
                    dtype=cum.dtype,
                    device=cum.device,
                ),
            ).item()
            + 1
        )

        return pr, pc1, pcs90, pcs95

    raw_pr, pc1, pcs90, pcs95 = spectrum_stats(x)

    z = x.double()
    z = z - z.mean(
        dim=0,
        keepdim=True,
    )
    z = z / z.std(
        dim=0,
        unbiased=False,
        keepdim=True,
    ).clamp_min(eps)

    std_pr, _, _, _ = spectrum_stats(z)

    corr = (
        z.T @ z
    ) / float(z.size(0))

    off = corr - torch.diag(
        torch.diagonal(corr)
    )

    denom = max(
        1,
        z.size(1)
        * (z.size(1) - 1),
    )

    return {
        "effective_dim_pr": float(raw_pr),
        "effective_dim_pr_standardized": float(std_pr),
        "pc1_variance_fraction": float(pc1),
        "pcs90": float(pcs90),
        "pcs95": float(pcs95),
        "mean_abs_offdiag_corr": float(
            off.abs().sum().item()
            / denom
        ),
        "max_abs_offdiag_corr": float(
            off.abs().max().item()
        ),
    }


def prefixed_geometry(prefix: str, x: torch.Tensor):
    g = latent_geometry_diagnostics(x)
    return {
        f"{prefix}_{k}": v
        for k, v in g.items()
    }


# =============================================================================
# DataLoader / decoding diagnostics
# =============================================================================

def make_loader(
    x,
    y,
    idx,
    batch_size,
    shuffle,
):
    return DataLoader(
        TensorDataset(
            x[idx],
            y[idx],
            idx,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )


@torch.no_grad()
def autoregressive_decode(
    vae,
    z,
    temperature=1.0,
):
    vae.eval()

    b = z.size(0)
    h = vae.dec.initial_hidden(z)
    cur = torch.zeros(
        b,
        1,
        VOCAB,
        device=z.device,
        dtype=z.dtype,
    )

    ids = []
    tau = max(
        float(temperature),
        1e-6,
    )

    for _ in range(SEQ_LEN):
        logits, h = vae.dec.step(
            z,
            cur,
            h,
        )

        idx = (
            logits / tau
        ).argmax(-1)

        ids.append(
            idx[:, None]
        )

        cur = F.one_hot(
            idx,
            num_classes=VOCAB,
        ).to(
            z.dtype
        ).unsqueeze(1)

    tok = torch.cat(
        ids,
        dim=1,
    ).cpu().tolist()

    return [
        "".join(
            I_TO_AA[int(i)]
            for i in row
        )
        for row in tok
    ]


@torch.no_grad()
def roundtrip_metrics(
    vae,
    flow,
    x_eval,
    source_peptides,
    mean,
    std,
    temperature,
):
    mu, _, _ = vae.enc(x_eval)
    h0 = standardize_mu(
        mu,
        mean,
        std,
    )

    zK, ld1 = flow(h0)
    h_back, ld2 = flow.inverse(zK)

    mu_back = unstandardize_h(
        h_back,
        mean,
        std,
    )

    dec = autoregressive_decode(
        vae,
        mu_back,
        temperature,
    )

    l2 = torch.linalg.norm(
        h0 - h_back,
        dim=-1,
    )

    cos = F.cosine_similarity(
        h0,
        h_back,
        dim=-1,
    )

    edits = [
        levenshtein_edit_distance(a, b)
        for a, b in zip(
            source_peptides,
            dec,
        )
    ]

    return {
        "flow_roundtrip_h_l2_mean":
            float(l2.mean()),
        "flow_roundtrip_h_l2_max":
            float(l2.max()),
        "flow_roundtrip_h_cosine_mean":
            float(cos.mean()),
        "flow_logdet_cancel_abs_mean":
            float((ld1 + ld2).abs().mean()),
        "source_to_decode_edit_mean":
            float(np.mean(edits)),
        "source_to_decode_edit_median":
            float(np.median(edits)),
        "decode_unique_fraction":
            float(
                len(set(dec))
                / max(1, len(dec))
            ),
        "zK_norm_mean":
            float(zK.norm(dim=-1).mean()),
        "zK_norm_std":
            float(zK.norm(dim=-1).std()),
    }


@torch.no_grad()
def local_zk_geometry(
    vae,
    flow,
    x_eval,
    mean,
    std,
    sigma,
    subset,
    neighbors,
    seed,
):
    n = min(
        int(subset),
        x_eval.size(0),
    )

    if n <= 0 or neighbors <= 0:
        return {}

    mu, _, _ = vae.enc(
        x_eval[:n]
    )
    h0 = standardize_mu(
        mu,
        mean,
        std,
    )
    z0, _ = flow(h0)

    h_center, _ = flow.inverse(z0)
    p_center = autoregressive_decode(
        vae,
        unstandardize_h(
            h_center,
            mean,
            std,
        ),
    )

    gen = torch.Generator(
        device="cpu"
    )
    gen.manual_seed(
        seed
    )

    z_dist = []
    h_dist = []
    edits = []

    for i in range(n):
        c = z0[i:i + 1]

        for _ in range(
            int(neighbors)
        ):
            noise = torch.randn(
                c.shape,
                generator=gen,
            ).to(
                c.device
            ) * float(sigma)

            zn = c + noise
            hn, _ = flow.inverse(zn)

            pn = autoregressive_decode(
                vae,
                unstandardize_h(
                    hn,
                    mean,
                    std,
                ),
            )[0]

            z_dist.append(
                float(
                    torch.linalg.norm(
                        zn - c
                    )
                )
            )

            h_dist.append(
                float(
                    torch.linalg.norm(
                        hn - h_center[i:i + 1]
                    )
                )
            )

            edits.append(
                levenshtein_edit_distance(
                    p_center[i],
                    pn,
                )
            )

    return {
        "local_sigma":
            float(sigma),
        "local_zK_l2_mean":
            float(np.mean(z_dist)),
        "local_h0_l2_mean":
            float(np.mean(h_dist)),
        "local_sequence_edit_mean":
            float(np.mean(edits)),
        "local_sequence_edit_median":
            float(np.median(edits)),
        "local_identical_fraction":
            float(
                np.mean(
                    np.asarray(edits) == 0
                )
            ),
    }


# =============================================================================
# Evaluation: objectives + mu / h0 / zK geometry
# =============================================================================

@torch.no_grad()
def evaluate(
    vae,
    flow,
    head,
    loader,
    peptides,
    x_all,
    mean,
    std,
    args,
    device,
):
    vae.eval()
    flow.eval()
    head.eval()

    nll_sum = 0.0
    obj_sum = 0.0
    n = 0

    preds = []
    targets = []
    mus = []
    h0s = []
    zks = []
    lds = []
    indices = []

    for x, y, idx in loader:
        x = x.to(device)
        y = y.to(device)

        mu, _, _ = vae.enc(x)
        h0 = standardize_mu(
            mu,
            mean,
            std,
        )

        nll, zK, ld = flow_nll(
            flow,
            h0,
        )

        pred = head(zK)
        obj = F.mse_loss(
            pred,
            y,
        )

        b = x.size(0)

        nll_sum += float(nll) * b
        obj_sum += float(obj) * b
        n += b

        preds.append(pred.cpu())
        targets.append(y.cpu())
        mus.append(mu.cpu())
        h0s.append(h0.cpu())
        zks.append(zK.cpu())
        lds.append(ld.cpu())
        indices += idx.tolist()

    pred = torch.cat(preds)
    yy = torch.cat(targets)
    mu_all = torch.cat(mus)
    h0_all = torch.cat(h0s)
    zK_all = torch.cat(zks)
    ld_all = torch.cat(lds)

    zstd = zK_all.std(
        0,
        unbiased=False,
    )

    m = {
        "flow_nll":
            nll_sum / max(1, n),
        "flow_nll_per_dim":
            (
                nll_sum / max(1, n)
            ) / args.latent_dim,
        "objective_mse":
            obj_sum / max(1, n),
        "zK_global_abs_mean":
            float(
                zK_all.mean(0)
                .abs()
                .mean()
            ),
        "zK_dim_std_mean":
            float(zstd.mean()),
        "zK_dim_std_min":
            float(zstd.min()),
        "zK_dim_std_max":
            float(zstd.max()),
        "zK_norm_mean":
            float(
                zK_all.norm(
                    dim=-1
                ).mean()
            ),
        "zK_norm_std":
            float(
                zK_all.norm(
                    dim=-1
                ).std()
            ),
        "flow_logdet_mean":
            float(ld_all.mean()),
        "flow_logdet_std":
            float(ld_all.std()),
    }

    for j, c in enumerate(
        args.objective_cols
    ):
        m[f"{c}_mse"] = float(
            F.mse_loss(
                pred[:, j],
                yy[:, j],
            )
        )
        m[f"{c}_pearson"] = pearson_safe(
            pred[:, j],
            yy[:, j],
        )

    m["objective_mean_pearson"] = float(
        np.nanmean([
            m[f"{c}_pearson"]
            for c in args.objective_cols
        ])
    )

    # How much the flow reshapes the already healthy encoder geometry.
    m["locality_preservation_loss"] = float(
        relative_distance_preservation_loss(
            h0_all,
            zK_all,
        )
    )
    m["identity_preservation_mse"] = float(
        identity_preservation_loss(
            h0_all,
            zK_all,
        )
    )

    # Global geometry: encoder mu -> standardized h0 -> flow zK.
    m.update(
        prefixed_geometry(
            "mu",
            mu_all,
        )
    )
    m.update(
        prefixed_geometry(
            "h0",
            h0_all,
        )
    )
    m.update(
        prefixed_geometry(
            "zK",
            zK_all,
        )
    )

    xv = x_all[indices].to(device)
    eval_peptides = [
        peptides[i]
        for i in indices
    ]

    m.update(
        roundtrip_metrics(
            vae,
            flow,
            xv,
            eval_peptides,
            mean,
            std,
            args.decode_temperature,
        )
    )

    m.update(
        local_zk_geometry(
            vae,
            flow,
            xv,
            mean,
            std,
            args.local_noise_std,
            args.local_smoothness_subset,
            args.local_neighbors,
            args.seed + 777,
        )
    )

    return m


# =============================================================================
# Checkpoints
# =============================================================================

def save_checkpoint(
    path,
    vae,
    flow,
    head,
    cfg,
    flow_cfg,
    mean,
    std,
    optimizer,
    args,
    epoch,
    metrics,
    selection,
    pretrained_meta,
    data_audit,
):
    pre_metrics = pretrained_meta.get(
        "metrics",
        {},
    )

    payload = {
        "checkpoint_type":
            "cu_best_bo_ready_gru_vae_realnvp_leakage_safe",
        "vae_state_dict":
            vae.state_dict(),
        "flow_state_dict":
            flow.state_dict(),
        "objective_head_state_dict":
            head.state_dict(),
        "optimizer_state_dict":
            optimizer.state_dict(),
        "model_config":
            asdict(cfg),
        "flow_config":
            asdict(flow_cfg),
        "latent_mean":
            mean.detach().cpu(),
        "latent_std":
            std.detach().cpu(),
        "epoch":
            int(epoch),
        "metrics":
            metrics,
        "checkpoint_selection":
            selection,
        "pretrained_checkpoint":
            args.init_checkpoint,
        "pretrained_checkpoint_epoch":
            int(
                pretrained_meta.get(
                    "epoch",
                    -1,
                )
            ),
        "pretrained_checkpoint_selection":
            pre_metrics.get(
                "selection_metric",
                "",
            ),
        "pretrained_checkpoint_effective_dim":
            pre_metrics.get(
                "selection_effective_dim_pr_standardized",
                pre_metrics.get(
                    "val_effective_dim_pr_standardized",
                    float("nan"),
                ),
            ),
        "pretraining_split_manifest":
            args.pretraining_split_manifest,
        "data_audit":
            data_audit,
        "objective_cols":
            list(args.objective_cols),
        "bo_coordinate_space":
            "realnvp_base_zK",
        "flow_input":
            "standardized_encoder_mu_h0",
        "flow_base_distribution":
            "standard_normal",
        "vae_frozen":
            not bool(args.finetune_vae),
        "cu_csv":
            args.cu_csv,
        "args":
            vars(args),
    }

    torch.save(
        payload,
        path,
    )

    side = {
        k: v
        for k, v in payload.items()
        if not k.endswith("state_dict")
    }

    with open(
        path + ".json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            json_safe(side),
            f,
            indent=2,
        )


def load_finetuned_checkpoint(
    path,
    vae,
    flow,
    head,
    device,
):
    ckpt = torch_load_full(
        path,
        map_location=device,
    )

    vae.load_state_dict(
        ckpt["vae_state_dict"],
        strict=True,
    )
    flow.load_state_dict(
        ckpt["flow_state_dict"],
        strict=True,
    )
    head.load_state_dict(
        ckpt["objective_head_state_dict"],
        strict=True,
    )

    return (
        ckpt,
        ckpt["latent_mean"]
        .to(device)
        .float(),
        ckpt["latent_std"]
        .to(device)
        .float(),
    )


# =============================================================================
# BO-coordinate export
# =============================================================================

@torch.no_grad()
def export_bo_coordinates(
    path,
    vae,
    flow,
    df,
    peptides,
    x_all,
    y_all,
    final_score,
    mean,
    std,
    args,
    device,
):
    rows = []

    vae.eval()
    flow.eval()

    for s in range(
        0,
        len(peptides),
        args.batch_size,
    ):
        e = min(
            s + args.batch_size,
            len(peptides),
        )

        x = x_all[s:e].to(device)

        mu, _, _ = vae.enc(x)
        h0 = standardize_mu(
            mu,
            mean,
            std,
        )

        zK, ld = flow(h0)
        h_back, _ = flow.inverse(zK)

        dec = autoregressive_decode(
            vae,
            unstandardize_h(
                h_back,
                mean,
                std,
            ),
            args.decode_temperature,
        )

        l2 = torch.linalg.norm(
            h0 - h_back,
            dim=-1,
        )

        cos = F.cosine_similarity(
            h0,
            h_back,
            dim=-1,
        )

        mun = mu.cpu().numpy()
        h0n = h0.cpu().numpy()
        zkn = zK.cpu().numpy()

        for i in range(
            e - s
        ):
            gi = s + i

            row = {
                "peptide":
                    peptides[gi],
                "peptide_len10":
                    peptides[gi],
                "split":
                    str(df.iloc[gi]["split"]),
                "decoded_from_zK_inverse":
                    dec[i],
                "source_to_decoded_edit":
                    levenshtein_edit_distance(
                        peptides[gi],
                        dec[i],
                    ),
                "flow_h0_roundtrip_l2":
                    float(l2[i]),
                "flow_h0_roundtrip_cosine":
                    float(cos[i]),
                "flow_logdet":
                    float(ld[i]),
                "zK_norm":
                    float(
                        np.linalg.norm(
                            zkn[i]
                        )
                    ),
                "final_score":
                    (
                        float(final_score[gi])
                        if torch.isfinite(
                            final_score[gi]
                        )
                        else float("nan")
                    ),
            }

            for j, c in enumerate(
                args.objective_cols
            ):
                row[c] = float(
                    y_all[gi, j]
                )

            for j in range(
                zkn.shape[1]
            ):
                row[f"zK_{j:02d}"] = float(
                    zkn[i, j]
                )
                row[f"h0_standardized_{j:02d}"] = float(
                    h0n[i, j]
                )
                row[f"mu_{j:02d}"] = float(
                    mun[i, j]
                )

            rows.append(row)

    pd.DataFrame(
        rows
    ).to_csv(
        path,
        index=False,
    )

    print(
        f"Saved BO coordinates: {path}"
    )


# =============================================================================
# Test-set evaluation after checkpoint selection
# =============================================================================

def evaluate_selected_checkpoint_on_test(
    label,
    checkpoint_path,
    vae,
    flow,
    head,
    test_loader,
    peptides,
    x_all,
    args,
    device,
):
    ckpt, mean, std = load_finetuned_checkpoint(
        checkpoint_path,
        vae,
        flow,
        head,
        device,
    )

    tm = evaluate(
        vae,
        flow,
        head,
        test_loader,
        peptides,
        x_all,
        mean,
        std,
        args,
        device,
    )

    payload = {
        "selected_checkpoint_label":
            label,
        "checkpoint_path":
            checkpoint_path,
        "checkpoint_epoch":
            int(ckpt.get("epoch", -1)),
        "checkpoint_selection":
            ckpt.get(
                "checkpoint_selection",
                "",
            ),
        "test_metrics":
            tm,
    }

    out_path = os.path.join(
        args.out_dir,
        f"test_metrics_{label}.json",
    )

    with open(
        out_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            json_safe(payload),
            f,
            indent=2,
        )

    print(
        f"Test evaluation [{label}]: "
        f"obj_mse={tm['objective_mse']:.6f} "
        f"mean_pearson={tm['objective_mean_pearson']:.4f} "
        f"zK_effdim={tm['zK_effective_dim_pr_standardized']:.3f} "
        f"decode_edit={tm['source_to_decode_edit_mean']:.4f}"
    )

    return payload


# =============================================================================
# Training
# =============================================================================

def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)

    os.makedirs(
        args.out_dir,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Leakage-safe Cu dataset
    # -------------------------------------------------------------------------
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

    tr_idx = split_to_idx["train"]
    va_idx = split_to_idx["val"]
    te_idx = split_to_idx["test"]

    # -------------------------------------------------------------------------
    # Revised pretrained GRU-VAE
    # -------------------------------------------------------------------------
    cfg = ModelConfig(
        args.hidden_size,
        args.latent_dim,
        args.n_layers,
        args.dropout,
    )

    flow_cfg = RealNVPConfig(
        args.flow_layers,
        args.flow_hidden_dim,
        args.flow_max_scale,
    )

    vae = GRUVAE(cfg).to(device)

    pretrained_meta = validate_pretrained_checkpoint(
        vae,
        args.init_checkpoint,
        device,
        cfg,
        require_bo_ready_selection=(
            not args.allow_non_bo_ready_pretraining
        ),
    )

    # -------------------------------------------------------------------------
    # Freeze VAE by default
    # -------------------------------------------------------------------------
    if args.finetune_vae:
        print(
            "WARNING: VAE fine-tuning is enabled. This changes the revised "
            "pretrained latent geometry and can invalidate fixed latent stats."
        )
        if not args.allow_vae_finetuning:
            raise RuntimeError(
                "For this leakage-safe matched experiment the VAE should remain "
                "frozen. If you intentionally want to fine-tune it, also pass "
                "--allow-vae-finetuning."
            )

        for p in vae.parameters():
            p.requires_grad_(True)
    else:
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)

        print(
            "GRU-VAE frozen; training RealNVP + objective head only."
        )

    # -------------------------------------------------------------------------
    # Training-only standardization. NEVER use val/test to estimate these stats.
    # -------------------------------------------------------------------------
    mean, std = compute_latent_stats(
        vae,
        x_all,
        tr_idx,
        args.batch_size,
        device,
    )

    print(
        "Cu-training-only latent stats: "
        f"mean(|mean|)={mean.abs().mean().item():.4f}, "
        f"mean(std)={std.mean().item():.4f}, "
        f"min(std)={std.min().item():.4f}, "
        f"max(std)={std.max().item():.4f}"
    )

    # -------------------------------------------------------------------------
    # RealNVP + objective head
    # -------------------------------------------------------------------------
    flow = RealNVPFlow(
        cfg.latent_dim,
        flow_cfg,
    ).to(device)

    head = MultiObjectiveHead(
        cfg.latent_dim,
        len(args.objective_cols),
        args.objective_head_hidden_dim,
    ).to(device)

    tr_loader = make_loader(
        x_all,
        y_all,
        tr_idx,
        args.batch_size,
        True,
    )

    va_loader = make_loader(
        x_all,
        y_all,
        va_idx,
        args.batch_size,
        False,
    )

    te_loader = make_loader(
        x_all,
        y_all,
        te_idx,
        args.batch_size,
        False,
    )

    groups = [
        {
            "params": flow.parameters(),
            "lr": args.flow_lr,
        },
        {
            "params": head.parameters(),
            "lr": args.objective_head_lr,
        },
    ]

    if args.finetune_vae:
        groups.append({
            "params": vae.parameters(),
            "lr": args.vae_lr,
        })

    opt = torch.optim.AdamW(
        groups,
        weight_decay=args.weight_decay,
    )

    hist = os.path.join(
        args.out_dir,
        "training_history_cu_best_gru_vae_realnvp_leakage_safe.csv",
    )

    if (
        os.path.exists(hist)
        and not args.append_history
    ):
        os.remove(hist)

    best_nll_path = os.path.join(
        args.out_dir,
        f"best_val_flow_nll_h{cfg.hidden_size}_z{cfg.latent_dim}_cu_realnvp.pt",
    )

    best_obj_path = os.path.join(
        args.out_dir,
        f"best_val_objective_mse_h{cfg.hidden_size}_z{cfg.latent_dim}_cu_realnvp.pt",
    )

    best_bo_path = os.path.join(
        args.out_dir,
        f"best_bo_ready_h{cfg.hidden_size}_z{cfg.latent_dim}_cu_realnvp.pt",
    )

    last_path = os.path.join(
        args.out_dir,
        f"last_epoch_h{cfg.hidden_size}_z{cfg.latent_dim}_cu_realnvp.pt",
    )

    best_nll = float("inf")
    best_obj = float("inf")
    best_bo_score = float("inf")

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    for epoch in range(
        1,
        args.epochs + 1,
    ):
        vae.train(
            args.finetune_vae
        )
        flow.train()
        head.train()

        nll_sum = 0.0
        obj_sum = 0.0
        locality_sum = 0.0
        identity_sum = 0.0
        total_loss_sum = 0.0
        n = 0

        for x, y, _ in tr_loader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad(
                set_to_none=True
            )

            if args.finetune_vae:
                mu, _, _ = vae.enc(x)
            else:
                with torch.no_grad():
                    mu, _, _ = vae.enc(x)

            h0 = standardize_mu(
                mu,
                mean,
                std,
            )

            if not args.finetune_vae:
                h0 = h0.detach()

            nll, zK, _ = flow_nll(
                flow,
                h0,
            )

            pred = head(zK)
            obj = F.mse_loss(
                pred,
                y,
            )

            locality = relative_distance_preservation_loss(
                h0,
                zK,
            )
            identity = identity_preservation_loss(
                h0,
                zK,
            )

            # Objective-aware conservative RealNVP:
            #   * less emphasis on Gaussianization (NLL)
            #   * substantially more emphasis on Cu objective prediction
            #   * preserve neighborhoods from the revised GRU-VAE
            #   * weakly discourage unnecessary departure from identity
            loss = (
                args.flow_loss_weight
                * (
                    nll / cfg.latent_dim
                )
                + args.objective_loss_weight
                * obj
                + args.locality_loss_weight
                * locality
                + args.identity_loss_weight
                * identity
            )

            loss.backward()

            params = (
                list(flow.parameters())
                + list(head.parameters())
            )

            if args.finetune_vae:
                params += list(
                    vae.parameters()
                )

            nn.utils.clip_grad_norm_(
                params,
                args.grad_clip,
            )

            opt.step()

            b = x.size(0)

            nll_sum += (
                float(nll.detach())
                * b
            )

            obj_sum += (
                float(obj.detach())
                * b
            )
            locality_sum += (
                float(locality.detach())
                * b
            )
            identity_sum += (
                float(identity.detach())
                * b
            )
            total_loss_sum += (
                float(loss.detach())
                * b
            )

            n += b

        train_nll = (
            nll_sum / max(1, n)
        )
        train_obj = (
            obj_sum / max(1, n)
        )
        train_locality = (
            locality_sum / max(1, n)
        )
        train_identity = (
            identity_sum / max(1, n)
        )
        train_total_loss = (
            total_loss_sum / max(1, n)
        )

        vm = evaluate(
            vae,
            flow,
            head,
            va_loader,
            peptides,
            x_all,
            mean,
            std,
            args,
            device,
        )

        val_nll_per_dim = (
            vm["flow_nll"]
            / cfg.latent_dim
        )

        # BO-ready selection score. Geometry penalties are soft and only matter
        # if zK remains poorly conditioned.
        effdim = float(
            vm[
                "zK_effective_dim_pr_standardized"
            ]
        )

        corr = float(
            vm[
                "zK_mean_abs_offdiag_corr"
            ]
        )

        eff_deficit = max(
            0.0,
            args.bo_selection_min_effective_dim
            - effdim,
        ) / max(
            1e-8,
            args.bo_selection_min_effective_dim,
        )

        bo_ready_score = (
            args.bo_selection_nll_weight
            * val_nll_per_dim
            + args.bo_selection_objective_weight
            * vm["objective_mse"]
            + args.bo_selection_locality_weight
            * vm["locality_preservation_loss"]
            + args.bo_selection_identity_weight
            * vm["identity_preservation_mse"]
            + args.bo_selection_effdim_penalty_weight
            * eff_deficit
            + args.bo_selection_corr_penalty_weight
            * corr
        )

        row = {
            "timestamp":
                datetime.now()
                .isoformat(
                    timespec="seconds"
                ),
            "epoch":
                epoch,
            "pretrained_checkpoint":
                args.init_checkpoint,
            "pretrained_epoch":
                int(
                    pretrained_meta.get(
                        "epoch",
                        -1,
                    )
                ),
            "pretrained_selection":
                pretrained_meta.get(
                    "metrics",
                    {},
                ).get(
                    "selection_metric",
                    "",
                ),
            "vae_finetuned":
                bool(args.finetune_vae),
            "train_flow_nll":
                train_nll,
            "train_flow_nll_per_dim":
                train_nll
                / cfg.latent_dim,
            "train_objective_mse":
                train_obj,
            "train_locality_preservation_loss":
                train_locality,
            "train_identity_preservation_mse":
                train_identity,
            "train_total_objective_aware_loss":
                train_total_loss,
            **{
                f"val_{k}": v
                for k, v in vm.items()
            },
            "val_flow_nll_per_dim":
                val_nll_per_dim,
            "val_bo_ready_score":
                bo_ready_score,
        }

        append_history(
            hist,
            row,
        )

        print(
            f"epoch={epoch:03d} "
            f"train_nll/d={train_nll/cfg.latent_dim:.6f} "
            f"val_nll/d={val_nll_per_dim:.6f} "
            f"val_obj_mse={vm['objective_mse']:.6f} "
            f"val_obj_pearson={vm['objective_mean_pearson']:.4f} "
            f"val_locality={vm['locality_preservation_loss']:.5f} "
            f"val_identity={vm['identity_preservation_mse']:.5f} "
            f"mu_effdim={vm['mu_effective_dim_pr_standardized']:.2f} "
            f"h0_effdim={vm['h0_effective_dim_pr_standardized']:.2f} "
            f"zK_effdim={vm['zK_effective_dim_pr_standardized']:.2f} "
            f"zK_corr={vm['zK_mean_abs_offdiag_corr']:.3f} "
            f"decode_edit={vm['source_to_decode_edit_mean']:.4f} "
            f"local_edit={vm.get('local_sequence_edit_mean', float('nan')):.4f}"
        )

        # Minimum validation flow NLL
        if vm["flow_nll"] < best_nll:
            best_nll = vm["flow_nll"]

            save_checkpoint(
                best_nll_path,
                vae,
                flow,
                head,
                cfg,
                flow_cfg,
                mean,
                std,
                opt,
                args,
                epoch,
                vm,
                "minimum_validation_flow_nll",
                pretrained_meta,
                data_audit,
            )

        # Minimum validation objective MSE
        if vm["objective_mse"] < best_obj:
            best_obj = vm[
                "objective_mse"
            ]

            save_checkpoint(
                best_obj_path,
                vae,
                flow,
                head,
                cfg,
                flow_cfg,
                mean,
                std,
                opt,
                args,
                epoch,
                vm,
                "minimum_validation_multiobjective_mse",
                pretrained_meta,
                data_audit,
            )

        # BO-ready checkpoint
        if bo_ready_score < best_bo_score:
            best_bo_score = bo_ready_score

            bo_metrics = dict(vm)
            bo_metrics.update({
                "bo_ready_score":
                    bo_ready_score,
                "bo_selection_effective_dim":
                    effdim,
                "bo_selection_mean_abs_corr":
                    corr,
            })

            save_checkpoint(
                best_bo_path,
                vae,
                flow,
                head,
                cfg,
                flow_cfg,
                mean,
                std,
                opt,
                args,
                epoch,
                bo_metrics,
                "minimum_bo_ready_score",
                pretrained_meta,
                data_audit,
            )

        # Last epoch
        save_checkpoint(
            last_path,
            vae,
            flow,
            head,
            cfg,
            flow_cfg,
            mean,
            std,
            opt,
            args,
            epoch,
            vm,
            "last_completed_epoch",
            pretrained_meta,
            data_audit,
        )

    # =========================================================================
    # Untouched test-set evaluation
    # =========================================================================
    test_results = {}

    for label, path in [
        ("best_flow_nll", best_nll_path),
        ("best_objective_mse", best_obj_path),
        ("best_bo_ready", best_bo_path),
    ]:
        test_results[label] = (
            evaluate_selected_checkpoint_on_test(
                label,
                path,
                vae,
                flow,
                head,
                te_loader,
                peptides,
                x_all,
                args,
                device,
            )
        )

    with open(
        os.path.join(
            args.out_dir,
            "test_metrics_selected_checkpoints.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            json_safe(test_results),
            f,
            indent=2,
        )

    # =========================================================================
    # Export BO coordinates from selected checkpoint
    # =========================================================================
    if args.export_bo_coordinates:
        selected_path = {
            "flow_nll":
                best_nll_path,
            "objective_mse":
                best_obj_path,
            "bo_ready":
                best_bo_path,
        }[
            args.export_checkpoint
        ]

        ckpt, mean_export, std_export = (
            load_finetuned_checkpoint(
                selected_path,
                vae,
                flow,
                head,
                device,
            )
        )

        export_bo_coordinates(
            os.path.join(
                args.out_dir,
                "cu_realnvp_coordinates_for_bo.csv",
            ),
            vae,
            flow,
            cu_df,
            peptides,
            x_all,
            y_all,
            final_score,
            mean_export,
            std_export,
            args,
            device,
        )

        with open(
            os.path.join(
                args.out_dir,
                "bo_coordinate_export_checkpoint.json",
            ),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump({
                "export_checkpoint_choice":
                    args.export_checkpoint,
                "checkpoint_path":
                    selected_path,
                "checkpoint_epoch":
                    int(
                        ckpt.get(
                            "epoch",
                            -1,
                        )
                    ),
                "checkpoint_selection":
                    ckpt.get(
                        "checkpoint_selection",
                        "",
                    ),
            }, f, indent=2)

    print("\nTraining complete.")
    print(
        f"Best flow-NLL checkpoint: {best_nll_path}"
    )
    print(
        f"Best objective-MSE checkpoint: {best_obj_path}"
    )
    print(
        f"Best BO-ready checkpoint: {best_bo_path}"
    )
    print(
        f"Last checkpoint: {last_path}"
    )
    print(
        f"History: {hist}"
    )

    return best_bo_path


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Leakage-safe Cu RealNVP fine-tuning on the revised BO-ready "
            "GRU-VAE with train/validation/test manifest reuse and latent "
            "geometry monitoring."
        )
    )

    # Revised pretraining inputs
    p.add_argument(
        "--init-checkpoint",
        required=True,
        help=(
            "Use best_bo_ready_gru_vae_latent_conditioned_h64_z64.pt "
            "from the revised leakage-safe/decorrelated pretraining."
        ),
    )

    p.add_argument(
        "--pretraining-split-manifest",
        required=True,
        help=(
            "Path to pretraining_leakage_safe_split_manifest.csv generated "
            "by revised pretraining."
        ),
    )

    p.add_argument(
        "--allow-non-bo-ready-pretraining",
        action="store_true",
        help=(
            "Deliberately allow a pretrained checkpoint whose selection_metric "
            "is not bo_ready_score."
        ),
    )

    # Cu data
    p.add_argument(
        "--cu-csv",
        required=True,
    )

    p.add_argument(
        "--peptide-col",
        default="peptide_len10",
    )

    p.add_argument(
        "--objective-cols",
        nargs="+",
        default=DEFAULT_OBJECTIVE_COLS,
    )

    p.add_argument(
        "--duplicate-policy",
        choices=[
            "mean",
            "first",
            "error",
        ],
        default="mean",
        help=(
            "How to handle repeated Cu peptide sequences. 'mean' aggregates "
            "objective rows by exact peptide sequence."
        ),
    )

    p.add_argument(
        "--unmapped-policy",
        choices=[
            "error",
            "drop",
        ],
        default="drop",
        help=(
            "What to do if a Cu peptide is absent from the pretraining split "
            "manifest. Default 'drop' is leakage-safe: unmapped peptides are "
            "excluded rather than being assigned to a new split that could bypass "
            "the PDB/Hamming holdout guarantees. Use 'error' for audit-only runs."
        ),
    )

    p.add_argument(
        "--out-dir",
        default="cu_best_gru_vae_realnvp_h64_z64_objective_aware",
    )

    # Architecture must match revised pretraining checkpoint
    p.add_argument(
        "--hidden-size",
        type=int,
        default=64,
    )

    p.add_argument(
        "--latent-dim",
        type=int,
        default=64,
    )

    p.add_argument(
        "--n-layers",
        type=int,
        default=2,
    )

    p.add_argument(
        "--dropout",
        type=float,
        default=0.0,
    )

    # RealNVP
    p.add_argument(
        "--flow-layers",
        type=int,
        default=4,
        help=(
            "Conservative default for the revised GRU-VAE. Fewer coupling "
            "layers reduce unnecessary reshaping of the already healthy h0 space."
        ),
    )

    p.add_argument(
        "--flow-hidden-dim",
        type=int,
        default=128,
    )

    p.add_argument(
        "--flow-max-scale",
        type=float,
        default=0.5,
        help=(
            "Maximum affine log-scale magnitude. Reduced from 1.5 to 0.5 "
            "to prevent over-reshaping."
        ),
    )

    # Training
    p.add_argument(
        "--epochs",
        type=int,
        default=150,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    p.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    # VAE: frozen by default and strongly recommended
    p.add_argument(
        "--finetune-vae",
        action="store_true",
    )

    p.add_argument(
        "--allow-vae-finetuning",
        action="store_true",
        help=(
            "Second explicit confirmation required if --finetune-vae is used."
        ),
    )

    p.add_argument(
        "--vae-lr",
        type=float,
        default=1e-6,
    )

    # Optimizer
    p.add_argument(
        "--flow-lr",
        type=float,
        default=3e-5,
    )

    p.add_argument(
        "--objective-head-lr",
        type=float,
        default=1e-4,
    )

    p.add_argument(
        "--objective-head-hidden-dim",
        type=int,
        default=64,
    )

    p.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    p.add_argument(
        "--grad-clip",
        type=float,
        default=5.0,
    )

    p.add_argument(
        "--flow-loss-weight",
        type=float,
        default=0.25,
        help=(
            "Weight on flow NLL / latent_dim. Lowered so Gaussianization does "
            "not dominate the BO-oriented training objective."
        ),
    )

    p.add_argument(
        "--objective-loss-weight",
        type=float,
        default=5.0,
        help=(
            "Weight on four-objective prediction MSE. Increased substantially "
            "so zK is shaped for Cu objective learnability, not only N(0,I)."
        ),
    )

    p.add_argument(
        "--locality-loss-weight",
        type=float,
        default=0.10,
        help=(
            "Weight on normalized pairwise-distance preservation between h0 "
            "and zK. Helps retain useful neighborhoods learned by the revised VAE."
        ),
    )

    p.add_argument(
        "--identity-loss-weight",
        type=float,
        default=0.01,
        help=(
            "Weak MSE penalty between zK and h0. Discourages unnecessary "
            "distortion while preserving RealNVP flexibility."
        ),
    )

    # Decoder / local geometry
    p.add_argument(
        "--decode-temperature",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--local-smoothness-subset",
        type=int,
        default=128,
    )

    p.add_argument(
        "--local-neighbors",
        type=int,
        default=3,
    )

    p.add_argument(
        "--local-noise-std",
        type=float,
        default=0.05,
    )

    # BO-aware RealNVP checkpoint selection
    p.add_argument(
        "--bo-selection-min-effective-dim",
        type=float,
        default=16.0,
        help=(
            "Soft minimum standardized PR effective dimension for zK."
        ),
    )

    p.add_argument(
        "--bo-selection-nll-weight",
        type=float,
        default=0.25,
        help="Weight on validation NLL/dim in BO-ready checkpoint selection.",
    )

    p.add_argument(
        "--bo-selection-objective-weight",
        type=float,
        default=5.0,
        help="Weight on validation four-objective MSE in BO-ready selection.",
    )

    p.add_argument(
        "--bo-selection-locality-weight",
        type=float,
        default=0.10,
        help="Weight on validation h0->zK locality distortion.",
    )

    p.add_argument(
        "--bo-selection-identity-weight",
        type=float,
        default=0.01,
        help="Weak weight on validation identity distortion.",
    )

    p.add_argument(
        "--bo-selection-effdim-penalty-weight",
        type=float,
        default=0.02,
    )

    p.add_argument(
        "--bo-selection-corr-penalty-weight",
        type=float,
        default=0.02,
    )

    # Output
    p.add_argument(
        "--append-history",
        action="store_true",
    )

    p.add_argument(
        "--export-bo-coordinates",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument(
        "--export-checkpoint",
        choices=[
            "flow_nll",
            "objective_mse",
            "bo_ready",
        ],
        default="bo_ready",
        help=(
            "Which selected checkpoint should be used to export "
            "cu_realnvp_coordinates_for_bo.csv."
        ),
    )

    return p.parse_args()


if __name__ == "__main__":
    train(
        parse_args()
    )
