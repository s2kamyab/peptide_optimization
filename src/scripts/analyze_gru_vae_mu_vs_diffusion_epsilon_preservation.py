#!/usr/bin/env python
"""
Analyze whether GRU-VAE latent coordinates and diffusion epsilon/noise BO coordinates
preserve the same data patterns.

This script is designed for the model family used by:
  BO_linear_spherical_diffusion_epsilon_CU_search_cost_sampling.py

Expected model path:
  peptide -> GRU-VAE encoder mu
  mu_std = (mu - latent_mean) / latent_std
  mu_std -> DDIM inversion -> epsilon
  BO operates on spherical epsilon coordinates
  epsilon -> DDIM sampling -> mu_std_hat -> unstandardized decoder latent -> peptide

The script computes pattern-preservation criteria between:
  1) standardized GRU-VAE latent mu_std
  2) precomputed DDIM-inverted diffusion epsilon coordinates
  3) diffusion-reconstructed latent h_hat = DDIM_sample(epsilon)

Outputs include global geometry metrics, local neighborhood metrics, linear/CCA
alignment, objective predictability, and reconstruction/inversion diagnostics.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    from scipy.stats import pearsonr, spearmanr
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This script requires scipy. Install with: pip install scipy") from exc

try:
    from sklearn.cross_decomposition import CCA
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.model_selection import KFold
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This script requires scikit-learn. Install with: pip install scikit-learn") from exc

# Import the same model definitions used by the diffusion BO code.
from finetune_cu_peptide_latent_diffusion_search_cost_sampling_h32_z32 import (  # noqa: E402
    AA,
    SEQ_LEN,
    LATENT_DIM,
    ModelConfig,
    DiffusionConfig,
    GRUVAE,
    LatentDiffusion,
    onehot_encode_peptides,
)

OBJ_COLS_DEFAULT = ["chelation_sub", "solubility_sub", "stability_sub", "expression_sub"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_peptide(x: object, seq_len: int = SEQ_LEN) -> Optional[str]:
    pep = str(x).strip().upper()
    if len(pep) != int(seq_len):
        return None
    if any(ch not in AA for ch in pep):
        return None
    return pep


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def tensor_to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().float().numpy()


def standardize_np(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return (X - X.mean(axis=0, keepdims=True)) / np.maximum(X.std(axis=0, keepdims=True), eps)


def project_to_sphere_np(X: np.ndarray, radius: float) -> np.ndarray:
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norm, 1e-12) * float(radius)


def load_generator(checkpoint_path: str, device: torch.device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Diffusion checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    required = ["vae_state_dict", "diffusion_state_dict", "model_config", "diffusion_config", "latent_mean", "latent_std"]
    missing = [k for k in required if k not in ckpt]
    if missing:
        raise KeyError(f"Checkpoint missing required keys: {missing}")

    cfg = ModelConfig(**ckpt["model_config"])
    diff_cfg = DiffusionConfig(**ckpt["diffusion_config"])
    vae = GRUVAE(cfg).to(device)
    diffusion = LatentDiffusion(cfg.latent_dim, diff_cfg).to(device)
    vae.load_state_dict(ckpt["vae_state_dict"], strict=True)
    diffusion.load_state_dict(ckpt["diffusion_state_dict"], strict=True)
    vae.eval()
    diffusion.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in diffusion.parameters():
        p.requires_grad_(False)

    latent_mean = ckpt["latent_mean"].to(device).float()
    latent_std = ckpt["latent_std"].to(device).float().clamp_min(1e-6)
    return vae, diffusion, latent_mean, latent_std, ckpt


@torch.no_grad()
def encode_mu(vae: GRUVAE, peptides: List[str], device: torch.device, batch_size: int) -> torch.Tensor:
    mus: List[torch.Tensor] = []
    for start in range(0, len(peptides), batch_size):
        batch_peps = peptides[start : start + batch_size]
        x = onehot_encode_peptides(batch_peps).to(device)
        # Try common encoder APIs robustly.
        if hasattr(vae, "enc"):
            enc_out = vae.enc(x)
            mu = enc_out[0] if isinstance(enc_out, tuple) else enc_out
        elif hasattr(vae, "encoder"):
            enc_out = vae.encoder(x)
            mu = enc_out[0] if isinstance(enc_out, tuple) else enc_out
        elif hasattr(vae, "encode"):
            enc_out = vae.encode(x)
            mu = enc_out[0] if isinstance(enc_out, tuple) else enc_out
        else:
            out = vae(x)
            if isinstance(out, dict) and "mu" in out:
                mu = out["mu"]
            else:
                raise AttributeError("Could not find encoder method in GRUVAE. Expected enc, encoder, encode, or forward dict with 'mu'.")
        mus.append(mu.detach().float().cpu())
    return torch.cat(mus, dim=0)


@torch.no_grad()
def diffusion_sample_to_standardized_latent(diffusion: LatentDiffusion, eps: torch.Tensor, ddim_steps: int, batch_size: int, device: torch.device) -> torch.Tensor:
    hs: List[torch.Tensor] = []
    eps = eps.to(device).float()
    for start in range(0, eps.size(0), batch_size):
        e = eps[start : start + batch_size]
        h = diffusion.ddim_sample(e, inference_steps=ddim_steps)
        hs.append(h.detach().float().cpu())
    return torch.cat(hs, dim=0)


def load_and_align_data(data_csv: str, eps_csv: str, peptide_col: str, obj_cols: List[str]) -> pd.DataFrame:
    data = pd.read_csv(data_csv)
    coords = pd.read_csv(eps_csv)

    if peptide_col not in data.columns:
        raise KeyError(f"peptide_col={peptide_col!r} not found in data CSV columns")
    if "peptide" not in coords.columns:
        raise KeyError("epsilon coordinate CSV must contain a 'peptide' column")

    data = data.copy()
    coords = coords.copy()
    data[peptide_col] = data[peptide_col].map(clean_peptide)
    coords["peptide"] = coords["peptide"].map(clean_peptide)
    data = data.dropna(subset=[peptide_col]).drop_duplicates(subset=[peptide_col], keep="first")
    coords = coords.dropna(subset=["peptide"]).drop_duplicates(subset=["peptide"], keep="first")

    missing_obj = [c for c in obj_cols if c not in data.columns]
    if missing_obj:
        raise KeyError(f"Objective columns missing from data CSV: {missing_obj}")

    eps_cols = sorted(
        [c for c in coords.columns if str(c).startswith("epsilon_") and str(c).split("_")[-1].isdigit()],
        key=lambda c: int(str(c).split("_")[-1]),
    )
    if not eps_cols:
        raise KeyError("No epsilon coordinate columns found. Expected columns named epsilon_0, epsilon_1, ...")

    merged = coords[["peptide"] + eps_cols].merge(
        data[[peptide_col] + obj_cols], left_on="peptide", right_on=peptide_col, how="inner"
    )
    merged = merged.drop_duplicates(subset=["peptide"], keep="first").reset_index(drop=True)
    if len(merged) < 10:
        raise ValueError(f"Only {len(merged)} aligned rows found. Check peptide columns and input files.")
    return merged


def sample_pair_indices(n: int, max_pairs: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        i, j = np.triu_indices(n, k=1)
        return i, j
    i = rng.integers(0, n, size=max_pairs)
    j = rng.integers(0, n, size=max_pairs)
    mask = i != j
    return i[mask], j[mask]


def pairwise_distance_metrics(A: np.ndarray, B: np.ndarray, max_pairs: int, seed: int, label: str) -> Dict[str, float]:
    n = A.shape[0]
    i, j = sample_pair_indices(n, max_pairs=max_pairs, seed=seed)
    da = np.linalg.norm(A[i] - A[j], axis=1)
    db = np.linalg.norm(B[i] - B[j], axis=1)
    sp = spearmanr(da, db).correlation
    pr = pearsonr(da, db).statistic
    return {
        f"{label}_pairwise_distance_spearman": float(sp),
        f"{label}_pairwise_distance_pearson": float(pr),
        f"{label}_n_pairs": int(len(da)),
        f"{label}_dist_A_mean": float(np.mean(da)),
        f"{label}_dist_B_mean": float(np.mean(db)),
    }


def knn_sets(X: np.ndarray, k: int) -> np.ndarray:
    k_eff = min(k + 1, X.shape[0])
    nbrs = NearestNeighbors(n_neighbors=k_eff, metric="euclidean").fit(X)
    idx = nbrs.kneighbors(X, return_distance=False)
    return idx[:, 1:k_eff]


def knn_overlap(A: np.ndarray, B: np.ndarray, k_values: Iterable[int], label: str) -> pd.DataFrame:
    rows = []
    for k in k_values:
        k_eff = min(int(k), A.shape[0] - 1)
        if k_eff < 1:
            continue
        Ia = knn_sets(A, k_eff)
        Ib = knn_sets(B, k_eff)
        overlaps = []
        for a, b in zip(Ia, Ib):
            overlaps.append(len(set(a.tolist()).intersection(set(b.tolist()))) / k_eff)
        rows.append({"comparison": label, "k": k_eff, "knn_overlap_mean": float(np.mean(overlaps)), "knn_overlap_std": float(np.std(overlaps))})
    return pd.DataFrame(rows)


def trustworthiness_continuity(A: np.ndarray, B: np.ndarray, k_values: Iterable[int], label: str) -> pd.DataFrame:
    # Trustworthiness(A -> B): whether neighbors in B are also close in A.
    # Continuity(A -> B): trustworthiness(B -> A).
    # Implemented from rank penalties; values in [0, 1], higher is better.
    n = A.shape[0]
    DA_rank = np.argsort(np.argsort(np.linalg.norm(A[:, None, :] - A[None, :, :], axis=-1), axis=1), axis=1)
    DB_rank = np.argsort(np.argsort(np.linalg.norm(B[:, None, :] - B[None, :, :], axis=-1), axis=1), axis=1)
    rows = []
    for k in k_values:
        k = min(int(k), n - 2)
        if k < 1:
            continue
        NA = knn_sets(A, k)
        NB = knn_sets(B, k)
        penalty_t = 0.0
        penalty_c = 0.0
        for i in range(n):
            set_A = set(NA[i].tolist())
            set_B = set(NB[i].tolist())
            # Trustworthiness: points in B-neighborhood but not in A-neighborhood penalized by rank in A.
            for j in set_B.difference(set_A):
                penalty_t += DA_rank[i, j] - k
            # Continuity: points in A-neighborhood but not in B-neighborhood penalized by rank in B.
            for j in set_A.difference(set_B):
                penalty_c += DB_rank[i, j] - k
        normalizer = n * k * (2 * n - 3 * k - 1)
        trust = 1.0 - (2.0 / max(normalizer, 1.0)) * penalty_t
        cont = 1.0 - (2.0 / max(normalizer, 1.0)) * penalty_c
        rows.append({"comparison": label, "k": k, "trustworthiness_A_to_B": float(trust), "continuity_A_to_B": float(cont)})
    return pd.DataFrame(rows)


def linear_alignment(A: np.ndarray, B: np.ndarray, label: str) -> Dict[str, float]:
    # Fit B ~= A W + b, report in-sample R2/MSE. This diagnoses whether mapping is mostly linear.
    A0 = standardize_np(A)
    B0 = standardize_np(B)
    X_aug = np.concatenate([A0, np.ones((A0.shape[0], 1))], axis=1)
    W, *_ = np.linalg.lstsq(X_aug, B0, rcond=None)
    pred = X_aug @ W
    return {
        f"{label}_linear_map_r2_mean": float(np.mean([r2_score(B0[:, j], pred[:, j]) for j in range(B0.shape[1])])),
        f"{label}_linear_map_mse": float(mean_squared_error(B0, pred)),
    }


def cca_metrics(A: np.ndarray, B: np.ndarray, n_components: int, label: str) -> pd.DataFrame:
    n_components = min(int(n_components), A.shape[1], B.shape[1], A.shape[0] - 1)
    A0 = standardize_np(A)
    B0 = standardize_np(B)
    cca = CCA(n_components=n_components, max_iter=2000)
    U, V = cca.fit_transform(A0, B0)
    rows = []
    for c in range(n_components):
        corr = pearsonr(U[:, c], V[:, c]).statistic
        rows.append({"comparison": label, "component": c + 1, "cca_corr": float(corr)})
    return pd.DataFrame(rows)


def objective_predictability(X_dict: Dict[str, np.ndarray], Y: np.ndarray, obj_cols: List[str], seed: int, folds: int) -> pd.DataFrame:
    rows = []
    n = Y.shape[0]
    kfold = KFold(n_splits=min(folds, n), shuffle=True, random_state=seed)
    alphas = np.logspace(-6, 3, 20)
    for space_name, X in X_dict.items():
        X0 = standardize_np(X)
        for j, obj in enumerate(obj_cols):
            y = Y[:, j].astype(float)
            pred = np.zeros_like(y, dtype=float)
            for train_idx, test_idx in kfold.split(X0):
                scaler = StandardScaler().fit(X0[train_idx])
                Xtr = scaler.transform(X0[train_idx])
                Xte = scaler.transform(X0[test_idx])
                model = RidgeCV(alphas=alphas).fit(Xtr, y[train_idx])
                pred[test_idx] = model.predict(Xte)
            rows.append({
                "space": space_name,
                "objective": obj,
                "cv_r2": float(r2_score(y, pred)),
                "cv_rmse": float(math.sqrt(mean_squared_error(y, pred))),
                "y_std": float(np.std(y)),
            })
    return pd.DataFrame(rows)


def neighbor_objective_consistency(X_dict: Dict[str, np.ndarray], Y: np.ndarray, k_values: Iterable[int]) -> pd.DataFrame:
    rows = []
    for space_name, X in X_dict.items():
        X0 = standardize_np(X)
        for k in k_values:
            k_eff = min(int(k), X.shape[0] - 1)
            if k_eff < 1:
                continue
            idx = knn_sets(X0, k_eff)
            dists = []
            for i in range(X.shape[0]):
                yi = Y[i]
                yj = Y[idx[i]]
                d = np.linalg.norm(yj - yi[None, :], axis=1).mean()
                dists.append(d)
            rows.append({"space": space_name, "k": k_eff, "neighbor_objective_l2_mean": float(np.mean(dists)), "neighbor_objective_l2_std": float(np.std(dists))})
    return pd.DataFrame(rows)


def pca_summary(X_dict: Dict[str, np.ndarray], n_components: int) -> pd.DataFrame:
    rows = []
    for name, X in X_dict.items():
        X0 = standardize_np(X)
        pca = PCA(n_components=min(n_components, X0.shape[1], X0.shape[0] - 1))
        pca.fit(X0)
        evr = pca.explained_variance_ratio_
        rows.append({
            "space": name,
            "n_components": len(evr),
            "explained_variance_top1": float(evr[0]),
            "explained_variance_top2": float(evr[:2].sum()) if len(evr) >= 2 else float(evr[0]),
            "explained_variance_top5": float(evr[:5].sum()),
            "explained_variance_top10": float(evr[:10].sum()),
        })
    return pd.DataFrame(rows)


def make_plots(out_dir: str, metrics_summary: Dict[str, float], pred_df: pd.DataFrame, neigh_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    # Bar plot for objective predictability averaged over objectives.
    avg_pred = pred_df.groupby("space", as_index=False)["cv_r2"].mean()
    plt.figure(figsize=(7, 4))
    plt.bar(avg_pred["space"], avg_pred["cv_r2"])
    plt.ylabel("Mean CV R2 across objectives")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "objective_predictability_mean_r2.png"), dpi=200)
    plt.close()

    # kNN objective consistency: lower is better.
    plt.figure(figsize=(7, 4))
    for space, g in neigh_df.groupby("space"):
        plt.plot(g["k"], g["neighbor_objective_l2_mean"], marker="o", label=space)
    plt.xlabel("k neighbors")
    plt.ylabel("Mean neighbor objective L2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "neighbor_objective_consistency.png"), dpi=200)
    plt.close()

    # Compact summary metric plot.
    keys = [k for k in metrics_summary if k.endswith("pairwise_distance_spearman")]
    labels = [k.replace("_pairwise_distance_spearman", "") for k in keys]
    vals = [metrics_summary[k] for k in keys]
    plt.figure(figsize=(8, 4))
    plt.bar(labels, vals)
    plt.ylabel("Pairwise distance Spearman")
    plt.xticks(rotation=25, ha="right")
    plt.ylim(-1, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pairwise_distance_spearman_summary.png"), dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check pattern preservation between GRU-VAE mu latent space and diffusion epsilon BO noise space.")
    p.add_argument("--diffusion-checkpoint", default="cu_peptide_latent_diffusion_search_cost_sampling_h32_z32/last_epoch_cu_latent_diffusion.pt")
    p.add_argument("--epsilon-csv", default="cu_peptide_latent_diffusion_search_cost_sampling_h32_z32/cu_ddim_inversion_coordinates_for_bo.csv")
    p.add_argument("--data-csv", default="metalpdb_binding_windows_len10_CU_scored_ranked.csv")
    p.add_argument("--peptide-col", default="peptide_len10")
    p.add_argument("--obj-cols", nargs="+", default=OBJ_COLS_DEFAULT)
    p.add_argument("--out-dir", default="gru_mu_vs_diffusion_epsilon_pattern_preservation")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--ddim-steps", type=int, default=20)
    p.add_argument("--max-pairwise-pairs", type=int, default=200000)
    p.add_argument("--knn-values", type=int, nargs="+", default=[5, 10, 20, 50])
    p.add_argument("--cca-components", type=int, default=10)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--make-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.out_dir)
    device = torch.device(args.device)

    print("Loading aligned peptide/objective/epsilon data...")
    merged = load_and_align_data(args.data_csv, args.epsilon_csv, args.peptide_col, args.obj_cols)
    eps_cols = sorted(
        [c for c in merged.columns if str(c).startswith("epsilon_") and str(c).split("_")[-1].isdigit()],
        key=lambda c: int(str(c).split("_")[-1]),
    )
    peptides = merged["peptide"].tolist()
    eps_raw = merged[eps_cols].to_numpy(dtype=np.float64)
    Y = merged[args.obj_cols].to_numpy(dtype=np.float64)

    print(f"Aligned rows: {len(peptides)}")
    print(f"Epsilon dim: {len(eps_cols)}")

    print("Loading generator checkpoint...")
    vae, diffusion, latent_mean, latent_std, ckpt = load_generator(args.diffusion_checkpoint, device)
    latent_dim = int(len(eps_cols))
    radius = math.sqrt(latent_dim)

    print("Encoding peptides to GRU-VAE mu...")
    mu = encode_mu(vae, peptides, device=device, batch_size=args.batch_size).numpy().astype(np.float64)
    latent_mean_np = tensor_to_np(latent_mean).astype(np.float64).reshape(1, -1)
    latent_std_np = tensor_to_np(latent_std).astype(np.float64).reshape(1, -1)
    mu_std = (mu - latent_mean_np) / np.maximum(latent_std_np, 1e-6)

    eps_sphere = project_to_sphere_np(eps_raw.astype(np.float64), radius=radius)

    print("Mapping epsilon through DDIM sampling back to standardized latent h_hat...")
    eps_t = torch.tensor(eps_sphere, dtype=torch.float32)
    h_hat = diffusion_sample_to_standardized_latent(diffusion, eps_t, args.ddim_steps, args.batch_size, device).numpy().astype(np.float64)

    # Save aligned coordinate table for further manual analysis.
    coord_df = pd.DataFrame({"peptide": peptides})
    for j in range(mu.shape[1]):
        coord_df[f"mu_{j:03d}"] = mu[:, j]
        coord_df[f"mu_std_{j:03d}"] = mu_std[:, j]
    for j in range(eps_sphere.shape[1]):
        coord_df[f"epsilon_{j:03d}"] = eps_sphere[:, j]
        coord_df[f"ddim_hhat_{j:03d}"] = h_hat[:, j]
    for j, c in enumerate(args.obj_cols):
        coord_df[c] = Y[:, j]
    coord_df.to_csv(os.path.join(args.out_dir, "aligned_mu_epsilon_hhat_coordinates.csv"), index=False)

    print("Computing pattern preservation metrics...")
    summary: Dict[str, float] = {}
    comparisons = {
        "mu_std_vs_epsilon": (mu_std, eps_sphere),
        "mu_std_vs_ddim_hhat": (mu_std, h_hat),
        "epsilon_vs_ddim_hhat": (eps_sphere, h_hat),
    }
    for label, (A, B) in comparisons.items():
        summary.update(pairwise_distance_metrics(standardize_np(A), standardize_np(B), args.max_pairwise_pairs, args.seed, label))
        summary.update(linear_alignment(A, B, label))
        # Direct coordinate-level cosine is only meaningful because both are same dimensional.
        A0, B0 = standardize_np(A), standardize_np(B)
        cos = np.sum(A0 * B0, axis=1) / (np.linalg.norm(A0, axis=1) * np.linalg.norm(B0, axis=1) + 1e-12)
        summary[f"{label}_per_sample_cosine_mean"] = float(np.mean(cos))
        summary[f"{label}_per_sample_cosine_std"] = float(np.std(cos))
        summary[f"{label}_coordinate_mse_after_standardization"] = float(mean_squared_error(A0, B0))

    # Reconstruction/inversion diagnostic: h_hat should match mu_std if DDIM inversion coordinates are faithful.
    recon_err = np.linalg.norm(mu_std - h_hat, axis=1)
    summary["ddim_inversion_mu_std_to_hhat_l2_mean"] = float(np.mean(recon_err))
    summary["ddim_inversion_mu_std_to_hhat_l2_median"] = float(np.median(recon_err))
    summary["ddim_inversion_mu_std_to_hhat_l2_p90"] = float(np.quantile(recon_err, 0.90))
    summary["epsilon_norm_mean"] = float(np.linalg.norm(eps_sphere, axis=1).mean())
    summary["epsilon_norm_std"] = float(np.linalg.norm(eps_sphere, axis=1).std())
    summary["mu_std_norm_mean"] = float(np.linalg.norm(mu_std, axis=1).mean())
    summary["mu_std_norm_std"] = float(np.linalg.norm(mu_std, axis=1).std())

    knn_frames = []
    trust_frames = []
    cca_frames = []
    for label, (A, B) in comparisons.items():
        A0, B0 = standardize_np(A), standardize_np(B)
        knn_frames.append(knn_overlap(A0, B0, args.knn_values, label))
        trust_frames.append(trustworthiness_continuity(A0, B0, args.knn_values, label))
        cca_frames.append(cca_metrics(A0, B0, args.cca_components, label))
    knn_df = pd.concat(knn_frames, ignore_index=True)
    trust_df = pd.concat(trust_frames, ignore_index=True)
    cca_df = pd.concat(cca_frames, ignore_index=True)

    spaces = {
        "gru_mu": mu,
        "gru_mu_standardized": mu_std,
        "diffusion_epsilon_sphere": eps_sphere,
        "ddim_reconstructed_hhat": h_hat,
    }
    pred_df = objective_predictability(spaces, Y, args.obj_cols, seed=args.seed, folds=args.cv_folds)
    neigh_df = neighbor_objective_consistency(spaces, Y, args.knn_values)
    pca_df = pca_summary(spaces, n_components=10)

    knn_df.to_csv(os.path.join(args.out_dir, "knn_overlap.csv"), index=False)
    trust_df.to_csv(os.path.join(args.out_dir, "trustworthiness_continuity.csv"), index=False)
    cca_df.to_csv(os.path.join(args.out_dir, "cca_alignment.csv"), index=False)
    pred_df.to_csv(os.path.join(args.out_dir, "objective_predictability_cv_ridge.csv"), index=False)
    neigh_df.to_csv(os.path.join(args.out_dir, "neighbor_objective_consistency.csv"), index=False)
    pca_df.to_csv(os.path.join(args.out_dir, "pca_summary.csv"), index=False)

    # Add compact derived values to summary for quick decision-making.
    pred_avg = pred_df.groupby("space")["cv_r2"].mean().to_dict()
    neigh_avg_k10 = neigh_df[neigh_df["k"] == min(10, len(peptides) - 1)].groupby("space")["neighbor_objective_l2_mean"].mean().to_dict()
    summary.update({f"objective_cv_r2_mean_{k}": float(v) for k, v in pred_avg.items()})
    summary.update({f"neighbor_objective_l2_k10_{k}": float(v) for k, v in neigh_avg_k10.items()})
    summary["n_aligned_peptides"] = int(len(peptides))
    summary["latent_dim"] = int(latent_dim)
    summary["radius_used_for_epsilon"] = float(radius)
    summary["checkpoint_type"] = str(ckpt.get("checkpoint_type"))
    summary["checkpoint_epoch"] = int(ckpt.get("epoch", -1)) if str(ckpt.get("epoch", "")).lstrip("-").isdigit() else str(ckpt.get("epoch"))
    summary["bo_coordinate_space_from_checkpoint"] = str(ckpt.get("bo_coordinate_space"))

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Human-readable decision report.
    report_lines = []
    report_lines.append("# GRU-VAE mu vs diffusion epsilon pattern-preservation report")
    report_lines.append("")
    report_lines.append(f"Aligned peptides: {len(peptides)}")
    report_lines.append(f"Latent/noise dimension: {latent_dim}")
    report_lines.append(f"Checkpoint: {args.diffusion_checkpoint}")
    report_lines.append("")
    report_lines.append("## Key interpretation rules")
    report_lines.append("- Pairwise distance Spearman >= 0.60: strong global pattern preservation.")
    report_lines.append("- Pairwise distance Spearman 0.30-0.60: partial global preservation.")
    report_lines.append("- Pairwise distance Spearman < 0.30: global geometry is mostly reorganized.")
    report_lines.append("- kNN overlap >= 0.30 at k=10 is usually meaningful in high-dimensional latent spaces.")
    report_lines.append("- Objective CV R2 matters most for BO: if epsilon predicts objectives as well as mu, epsilon is a useful BO space even if geometry differs.")
    report_lines.append("- DDIM h_hat vs mu_std L2 checks whether the stored epsilon coordinates reconstruct the original standardized latent vectors.")
    report_lines.append("")
    report_lines.append("## Compact metrics")
    for key in sorted(summary):
        if any(token in key for token in ["pairwise_distance_spearman", "linear_map_r2_mean", "ddim_inversion", "objective_cv_r2_mean", "neighbor_objective_l2_k10"]):
            report_lines.append(f"- {key}: {summary[key]}")
    with open(os.path.join(args.out_dir, "interpretation_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    if args.make_plots:
        make_plots(args.out_dir, summary, pred_df, neigh_df)

    print("Done.")
    print(f"Output directory: {args.out_dir}")
    print("Most important files:")
    print("  summary.json")
    print("  interpretation_report.md")
    print("  objective_predictability_cv_ridge.csv")
    print("  knn_overlap.csv")
    print("  trustworthiness_continuity.csv")
    print("  aligned_mu_epsilon_hhat_coordinates.csv")


if __name__ == "__main__":
    main()
