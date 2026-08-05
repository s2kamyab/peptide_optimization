#!/usr/bin/env python
"""
Compare smoothness and optimization-easiness of peptide VAE latent spaces.

This experiment trains:
  1) a no-flow GRU-VAE
  2) a one-flow GRU-VAE

Then it compares three latent spaces on the same training data:
  A. no_flow      : mu from the no-flow VAE
  B. before_flow  : mu from the one-flow VAE
  C. after_flow   : flow(mu) from the one-flow VAE

Metrics:
  - Smoothness:
      * correlation between latent distance and objective distance
      * kNN local objective variance
      * local Lipschitz estimates
  - GP/surrogate easiness:
      * RBF-kernel-ridge surrogate validation RMSE/R2
      * top-K predicted candidates' true objective quality

Why RBF kernel ridge instead of full GP for every metric?
  Kernel ridge with an RBF kernel is a stable, fast proxy for "how easily a
  smooth GP-like surrogate can model this latent space." It avoids repeatedly
  fitting expensive exact GPs during diagnostics. You can still run your
  actual BoTorch qEHVI pipeline afterward on the most promising space.

Run:
  python compare_latent_space_smoothness_optimization.py \
      --csv metalpdb_binding_windows_len10_CU_scored_ranked.csv \
      --epochs 80 \
      --latent_dim 32 \
      --n_flows 1

Outputs:
  latent_space_comparison_results/
    - latent_space_summary.csv
    - latent_space_pairwise_smoothness.csv
    - latent_space_surrogate_cv.csv
    - latent_space_embeddings_*.csv
    - several PNG plots
"""

import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_I = {a: i for i, a in enumerate(AA)}
I_TO_AA = {i: a for i, a in enumerate(AA)}
SEQ_LEN = 10
VOCAB = 20

OBJ_COLS = ["chelation_sub", "solubility_sub", "stability_sub", "expression_sub"]
DEFAULT_TARGET_COL = "final_score"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def onehot_encode_peptides(peps: List[str], device: torch.device) -> torch.Tensor:
    x = torch.zeros((len(peps), SEQ_LEN, VOCAB), dtype=torch.float32, device=device)
    for n, s in enumerate(peps):
        s = str(s).strip().upper()
        if len(s) != SEQ_LEN:
            raise ValueError(f"Expected peptide length {SEQ_LEN}, got {len(s)} for {s}")
        for t, ch in enumerate(s):
            if ch not in AA_TO_I:
                raise ValueError(f"Unknown amino acid {ch!r} in peptide {s}")
            x[n, t, AA_TO_I[ch]] = 1.0
    return x


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


class GRUEncoder(nn.Module):
    def __init__(self, hidden_size: int, latent_dim: int, n_layers: int):
        super().__init__()
        self.in_proj = nn.Linear(VOCAB, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
        )
        self.to_mu = nn.Linear(hidden_size, latent_dim)
        self.to_logvar = nn.Linear(hidden_size, latent_dim)

    def forward(self, x_onehot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.in_proj(x_onehot)
        _, hN = self.gru(x)
        h_last = hN[-1]
        mu = self.to_mu(h_last)
        logvar = self.to_logvar(h_last)
        return mu, logvar


class GRUDecoder(nn.Module):
    def __init__(self, hidden_size: int, latent_dim: int, n_layers: int):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.token_embed = nn.Linear(VOCAB, hidden_size)
        self.z_to_h = nn.Linear(latent_dim, n_layers * hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
        )
        self.to_logits = nn.Linear(hidden_size, VOCAB)

    def forward(self, z: torch.Tensor, x_onehot: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        x_shift = torch.zeros_like(x_onehot)
        x_shift[:, 1:, :] = x_onehot[:, :-1, :]
        emb = self.token_embed(x_shift)
        h0 = self.z_to_h(z).view(self.n_layers, B, self.hidden_size)
        out, _ = self.gru(emb, h0)
        return self.to_logits(out)


class FlowGRUVAE(nn.Module):
    def __init__(self, hidden_size: int, latent_dim: int, n_layers: int, n_flows: int):
        super().__init__()
        self.enc = GRUEncoder(hidden_size, latent_dim, n_layers)
        self.dec = GRUDecoder(hidden_size, latent_dim, n_layers)
        self.flow = FlowSequence(latent_dim, n_flows)
        self.n_flows = n_flows

    def reparam(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_onehot: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar = self.enc(x_onehot)
        z0 = self.reparam(mu, logvar)
        zK, sum_logdet = self.flow(z0)
        logits = self.dec(zK, x_onehot)
        return {
            "mu": mu,
            "logvar": logvar,
            "z0": z0,
            "zK": zK,
            "sum_logdet": sum_logdet,
            "logits": logits,
        }


def correct_flow_vae_loss(
    x_onehot: torch.Tensor,
    out: Dict[str, torch.Tensor],
    beta: float,
) -> Tuple[torch.Tensor, float, float]:
    """VAE loss with correct Monte Carlo flow KL.

    KL = E[log q0(z0|x) - log|det dzK/dz0| - log p(zK)]
    For n_flows=0, zK=z0 and sum_logdet=0, reducing to the standard VAE KL.
    """
    target = x_onehot.argmax(dim=-1)
    recon = F.cross_entropy(
        out["logits"].reshape(-1, VOCAB),
        target.reshape(-1),
        reduction="mean",
    )

    mu = out["mu"]
    logvar = out["logvar"]
    z0 = out["z0"]
    zK = out["zK"]
    sum_logdet = out["sum_logdet"]

    log2pi = math.log(2.0 * math.pi)
    var = torch.exp(logvar)

    log_q0 = -0.5 * (((z0 - mu) ** 2) / var + logvar + log2pi).sum(dim=-1)
    log_p_zK = -0.5 * (zK.pow(2) + log2pi).sum(dim=-1)

    kl = (log_q0 - sum_logdet - log_p_zK).mean()
    kl = torch.clamp(kl, min=0.0)

    loss = recon + beta * kl
    return loss, float(recon.detach().cpu()), float(kl.detach().cpu())


@torch.no_grad()
def reconstruction_metrics(model: FlowGRUVAE, X: torch.Tensor, batch_size: int = 256) -> Dict[str, float]:
    model.eval()
    ce_sum = 0.0
    acc_sum = 0.0
    n_batches = 0
    for start in range(0, X.size(0), batch_size):
        xb = X[start:start + batch_size]
        out = model(xb)
        target = xb.argmax(dim=-1)
        ce = F.cross_entropy(out["logits"].reshape(-1, VOCAB), target.reshape(-1), reduction="mean")
        pred = out["logits"].argmax(dim=-1)
        acc = (pred == target).float().mean()
        ce_sum += float(ce.cpu())
        acc_sum += float(acc.cpu())
        n_batches += 1
    return {"recon_ce": ce_sum / n_batches, "token_acc": acc_sum / n_batches}


def train_vae(
    X: torch.Tensor,
    n_flows: int,
    hidden_size: int,
    latent_dim: int,
    n_layers: int,
    epochs: int,
    batch_size: int,
    lr: float,
    beta: float,
    seed: int,
    label: str,
) -> FlowGRUVAE:
    set_seed(seed)
    model = FlowGRUVAE(hidden_size, latent_dim, n_layers, n_flows).to(X.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n = X.size(0)
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=X.device)
        losses, recons, kls = [], [], []
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb = X[idx]
            out = model(xb)
            loss, recon, kl = correct_flow_vae_loss(xb, out, beta=beta)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            recons.append(recon)
            kls.append(kl)

        if ep == 1 or ep % max(1, epochs // 10) == 0 or ep == epochs:
            print(
                f"[{label}] epoch {ep:04d}/{epochs} "
                f"loss={np.mean(losses):.4f} recon={np.mean(recons):.4f} kl={np.mean(kls):.4f}"
            )
    return model


@torch.no_grad()
def get_latents(model: FlowGRUVAE, X: torch.Tensor, batch_size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """Return before-flow mu and after-flow flow(mu)."""
    model.eval()
    all_mu, all_zK = [], []
    for start in range(0, X.size(0), batch_size):
        xb = X[start:start + batch_size]
        mu, _ = model.enc(xb)
        zK, _ = model.flow(mu)
        all_mu.append(mu.detach().cpu())
        all_zK.append(zK.detach().cpu())
    return torch.cat(all_mu).numpy(), torch.cat(all_zK).numpy()


def standardize_train_test(Z: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = Z.mean(axis=0, keepdims=True)
    std = Z.std(axis=0, keepdims=True) + eps
    return (Z - mean) / std, mean, std


def pairwise_smoothness_metrics(
    Z: np.ndarray,
    Y_multi: np.ndarray,
    y_scalar: np.ndarray,
    rng: np.random.Generator,
    n_pairs: int = 50000,
) -> Dict[str, float]:
    n = Z.shape[0]
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]

    dz = np.linalg.norm(Z[i] - Z[j], axis=1)
    dy_scalar = np.abs(y_scalar[i] - y_scalar[j])
    dy_multi = np.linalg.norm(Y_multi[i] - Y_multi[j], axis=1)

    valid = dz > 1e-12
    dz = dz[valid]
    dy_scalar = dy_scalar[valid]
    dy_multi = dy_multi[valid]

    pearson_scalar = float(np.corrcoef(dz, dy_scalar)[0, 1])
    pearson_multi = float(np.corrcoef(dz, dy_multi)[0, 1])
    spearman_scalar = float(pd.Series(dz).rank().corr(pd.Series(dy_scalar).rank()))
    spearman_multi = float(pd.Series(dz).rank().corr(pd.Series(dy_multi).rank()))

    lips_scalar = dy_scalar / (dz + 1e-12)
    lips_multi = dy_multi / (dz + 1e-12)

    return {
        "pair_distance_vs_abs_final_corr_pearson": pearson_scalar,
        "pair_distance_vs_obj_vector_corr_pearson": pearson_multi,
        "pair_distance_vs_abs_final_corr_spearman": spearman_scalar,
        "pair_distance_vs_obj_vector_corr_spearman": spearman_multi,
        "pair_lipschitz_final_median": float(np.median(lips_scalar)),
        "pair_lipschitz_final_p90": float(np.percentile(lips_scalar, 90)),
        "pair_lipschitz_obj_vector_median": float(np.median(lips_multi)),
        "pair_lipschitz_obj_vector_p90": float(np.percentile(lips_multi, 90)),
    }


def knn_indices_bruteforce(Z: np.ndarray, k: int = 10, block: int = 512) -> np.ndarray:
    """Memory-safe exact kNN by blocks. Returns [N,k] neighbor indices excluding self."""
    n = Z.shape[0]
    out = np.zeros((n, k), dtype=np.int64)
    Z2 = np.sum(Z * Z, axis=1, keepdims=True).T  # [1,N]
    for start in range(0, n, block):
        end = min(start + block, n)
        A = Z[start:end]
        A2 = np.sum(A * A, axis=1, keepdims=True)
        D2 = A2 + Z2 - 2.0 * (A @ Z.T)
        rows = np.arange(start, end) - start
        D2[rows, np.arange(start, end)] = np.inf
        out[start:end] = np.argpartition(D2, kth=k, axis=1)[:, :k]
    return out


def local_knn_smoothness(
    Z: np.ndarray,
    Y_multi: np.ndarray,
    y_scalar: np.ndarray,
    k: int = 10,
) -> Dict[str, float]:
    nn_idx = knn_indices_bruteforce(Z, k=k)
    neighbor_y = y_scalar[nn_idx]
    neighbor_Y = Y_multi[nn_idx]

    local_std_final = neighbor_y.std(axis=1)
    local_range_final = neighbor_y.max(axis=1) - neighbor_y.min(axis=1)

    center_y = y_scalar[:, None]
    center_Y = Y_multi[:, None, :]
    local_abs_diff = np.abs(neighbor_y - center_y)
    local_multi_diff = np.linalg.norm(neighbor_Y - center_Y, axis=2)

    return {
        f"knn{k}_final_std_mean": float(local_std_final.mean()),
        f"knn{k}_final_std_median": float(np.median(local_std_final)),
        f"knn{k}_final_range_mean": float(local_range_final.mean()),
        f"knn{k}_abs_final_diff_mean": float(local_abs_diff.mean()),
        f"knn{k}_obj_vector_diff_mean": float(local_multi_diff.mean()),
    }


def rbf_kernel(A: np.ndarray, B: np.ndarray, lengthscale: float) -> np.ndarray:
    A2 = np.sum(A * A, axis=1, keepdims=True)
    B2 = np.sum(B * B, axis=1, keepdims=True).T
    D2 = np.maximum(A2 + B2 - 2.0 * (A @ B.T), 0.0)
    return np.exp(-0.5 * D2 / (lengthscale ** 2 + 1e-12))


def median_lengthscale(Z: np.ndarray, rng: np.random.Generator, n_pairs: int = 20000) -> float:
    n = Z.shape[0]
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    d = np.linalg.norm(Z[i] - Z[j], axis=1)
    d = d[d > 1e-12]
    return float(np.median(d) + 1e-8)


def krr_fit_predict(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
    lengthscale: float,
    ridge: float = 1e-3,
) -> np.ndarray:
    K = rbf_kernel(Z_train, Z_train, lengthscale)
    K.flat[:: K.shape[0] + 1] += ridge
    alpha = np.linalg.solve(K, y_train)
    Ktest = rbf_kernel(Z_test, Z_train, lengthscale)
    return Ktest @ alpha


def surrogate_cv_metrics(
    Z: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_folds: int = 5,
    max_train: int = 900,
    top_k: int = 20,
) -> Dict[str, float]:
    """RBF-KRR CV as a fast GP-like surrogate-easiness test."""
    n = Z.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, n_folds)

    rows = []
    all_pred = np.zeros(n, dtype=float)
    all_seen = np.zeros(n, dtype=bool)

    y_mean = y.mean()
    y_std = y.std() + 1e-12
    yz = (y - y_mean) / y_std

    for f in range(n_folds):
        test_idx = folds[f]
        train_idx = np.setdiff1d(idx, test_idx, assume_unique=False)

        if len(train_idx) > max_train:
            train_idx = rng.choice(train_idx, size=max_train, replace=False)

        Ztr, Zte = Z[train_idx], Z[test_idx]
        ytr = yz[train_idx]

        ell = median_lengthscale(Ztr, rng)
        pred_z = krr_fit_predict(Ztr, ytr, Zte, lengthscale=ell, ridge=1e-3)
        pred = pred_z * y_std + y_mean

        all_pred[test_idx] = pred
        all_seen[test_idx] = True

        rmse = float(np.sqrt(np.mean((pred - y[test_idx]) ** 2)))
        mae = float(np.mean(np.abs(pred - y[test_idx])))
        r2 = float(1.0 - np.sum((pred - y[test_idx]) ** 2) / (np.sum((y[test_idx] - y[test_idx].mean()) ** 2) + 1e-12))
        corr = float(np.corrcoef(pred, y[test_idx])[0, 1])

        k = min(top_k, len(test_idx))
        top_pred_local = np.argsort(pred)[-k:]
        true_top_threshold = np.percentile(y[test_idx], 90)
        hit_rate_top10pct = float(np.mean(y[test_idx][top_pred_local] >= true_top_threshold))
        top_pred_true_mean = float(np.mean(y[test_idx][top_pred_local]))
        random_true_mean = float(np.mean(y[test_idx]))

        rows.append({
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "pred_true_corr": corr,
            "top_pred_true_mean": top_pred_true_mean,
            "random_true_mean": random_true_mean,
            "top_pred_lift_over_random": top_pred_true_mean - random_true_mean,
            "top_pred_hit_rate_true_top10pct": hit_rate_top10pct,
        })

    cv = pd.DataFrame(rows)
    out = {f"cv_{c}_mean": float(cv[c].mean()) for c in cv.columns}
    out.update({f"cv_{c}_std": float(cv[c].std(ddof=0)) for c in cv.columns})

    # global rank metric: if we chose the top predicted points, how good are their true values?
    valid_idx = np.where(all_seen)[0]
    k = min(top_k, len(valid_idx))
    top_global = valid_idx[np.argsort(all_pred[valid_idx])[-k:]]
    out[f"global_top{top_k}_pred_true_mean"] = float(y[top_global].mean())
    out[f"global_top{top_k}_pred_true_best"] = float(y[top_global].max())
    out[f"global_top{top_k}_pred_true_percentile_mean"] = float(
        np.mean([100.0 * (y <= yy).mean() for yy in y[top_global]])
    )
    return out


def participation_ratio(Z: np.ndarray) -> float:
    """Effective dimensionality from covariance eigenvalues."""
    C = np.cov(Z.T)
    vals = np.linalg.eigvalsh(C)
    vals = np.maximum(vals, 0.0)
    return float((vals.sum() ** 2) / (np.sum(vals ** 2) + 1e-12))


def plot_summary_bar(summary: pd.DataFrame, metric: str, out_path: str, ylabel: str) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.bar(summary["space"], summary[metric])
    plt.ylabel(ylabel)
    plt.xlabel("Latent space")
    plt.title(metric)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_latent_pca(Z: np.ndarray, y: np.ndarray, title: str, out_path: str) -> None:
    Zc = Z - Z.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    P = Zc @ Vt[:2].T
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(P[:, 0], P[:, 1], c=y, s=12, alpha=0.8)
    plt.colorbar(sc, label=DEFAULT_TARGET_COL)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="metalpdb_binding_windows_len10_CU_scored_ranked.csv")
    parser.add_argument("--out_dir", type=str, default="latent_space_comparison_results")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_flows", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kl_beta", type=float, default=0.01)
    parser.add_argument("--target_col", type=str, default=DEFAULT_TARGET_COL)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pair_samples", type=int, default=50000)
    parser.add_argument("--knn_k", type=int, default=10)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--max_krr_train", type=int, default=900)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print("DEVICE:", device)

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    required = ["peptide_len10"] + OBJ_COLS + [args.target_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {args.csv}: {missing}")

    df = df.dropna(subset=["peptide_len10"] + OBJ_COLS + [args.target_col]).copy()
    df["peptide_len10"] = df["peptide_len10"].astype(str).str.strip().str.upper()
    df = df[df["peptide_len10"].str.len() == SEQ_LEN].reset_index(drop=True)

    peptides = df["peptide_len10"].tolist()
    X = onehot_encode_peptides(peptides, device=device)

    Y_multi = df[OBJ_COLS].astype(float).to_numpy()
    y = df[args.target_col].astype(float).to_numpy()

    # Standardize target only for surrogate training internally; reported metrics use original scale.
    rng = np.random.default_rng(args.seed)

    print(f"Loaded {len(df)} valid peptides from {args.csv}")

    # 1) Train no-flow VAE
    no_flow_model = train_vae(
        X=X,
        n_flows=0,
        hidden_size=args.hidden_size,
        latent_dim=args.latent_dim,
        n_layers=args.n_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta=args.kl_beta,
        seed=args.seed,
        label="no_flow",
    )

    # 2) Train one-flow VAE
    flow_model = train_vae(
        X=X,
        n_flows=args.n_flows,
        hidden_size=args.hidden_size,
        latent_dim=args.latent_dim,
        n_layers=args.n_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta=args.kl_beta,
        seed=args.seed + 1,
        label=f"{args.n_flows}_flow",
    )

    # Reconstruction diagnostics
    recon_no = reconstruction_metrics(no_flow_model, X)
    recon_flow = reconstruction_metrics(flow_model, X)

    # Latents
    Z_no_mu, _ = get_latents(no_flow_model, X)
    Z_flow_mu, Z_flow_zK = get_latents(flow_model, X)

    spaces = {
        "no_flow": Z_no_mu,
        "before_flow": Z_flow_mu,
        "after_flow": Z_flow_zK,
    }

    summary_rows = []
    pair_rows = []
    surrogate_rows = []

    for name, Z_raw in spaces.items():
        print(f"\nAnalyzing latent space: {name}")
        Z, _, _ = standardize_train_test(Z_raw)

        # Save embeddings
        emb_df = pd.DataFrame(Z_raw, columns=[f"z{d:02d}" for d in range(Z_raw.shape[1])])
        emb_df.insert(0, "peptide", peptides)
        for col in OBJ_COLS + [args.target_col]:
            emb_df[col] = df[col].values
        emb_df.to_csv(os.path.join(args.out_dir, f"latent_space_embeddings_{name}.csv"), index=False)

        pair_metrics = pairwise_smoothness_metrics(
            Z=Z,
            Y_multi=Y_multi,
            y_scalar=y,
            rng=rng,
            n_pairs=args.pair_samples,
        )
        knn_metrics = local_knn_smoothness(
            Z=Z,
            Y_multi=Y_multi,
            y_scalar=y,
            k=args.knn_k,
        )
        surrogate_metrics = surrogate_cv_metrics(
            Z=Z,
            y=y,
            rng=rng,
            n_folds=args.cv_folds,
            max_train=args.max_krr_train,
            top_k=args.top_k,
        )

        geom_metrics = {
            "latent_participation_ratio": participation_ratio(Z),
            "latent_mean_norm": float(np.linalg.norm(Z, axis=1).mean()),
            "latent_std_norm": float(np.linalg.norm(Z, axis=1).std()),
        }

        recon_metrics = {}
        if name == "no_flow":
            recon_metrics = {f"vae_{k}": v for k, v in recon_no.items()}
        else:
            recon_metrics = {f"vae_{k}": v for k, v in recon_flow.items()}

        row = {"space": name}
        row.update(recon_metrics)
        row.update(geom_metrics)
        row.update(pair_metrics)
        row.update(knn_metrics)
        row.update(surrogate_metrics)
        summary_rows.append(row)

        pair_rows.append({"space": name, **pair_metrics, **knn_metrics})
        surrogate_rows.append({"space": name, **surrogate_metrics})

        plot_latent_pca(
            Z,
            y,
            title=f"{name}: PCA of standardized latent space",
            out_path=os.path.join(args.out_dir, f"latent_pca_{name}.png"),
        )

    summary = pd.DataFrame(summary_rows)
    pair_df = pd.DataFrame(pair_rows)
    surrogate_df = pd.DataFrame(surrogate_rows)

    summary.to_csv(os.path.join(args.out_dir, "latent_space_summary.csv"), index=False)
    pair_df.to_csv(os.path.join(args.out_dir, "latent_space_pairwise_smoothness.csv"), index=False)
    surrogate_df.to_csv(os.path.join(args.out_dir, "latent_space_surrogate_cv.csv"), index=False)

    # Simple ranking: lower smoothness roughness is better; higher surrogate/optimization metrics are better.
    rank_df = summary[[
        "space",
        "knn10_abs_final_diff_mean",
        "pair_lipschitz_final_p90",
        "cv_rmse_mean",
        "cv_r2_mean",
        f"global_top{args.top_k}_pred_true_percentile_mean",
    ]].copy()

    # Compute ordinal ranks.
    rank_df["rank_smooth_knn"] = rank_df["knn10_abs_final_diff_mean"].rank(ascending=True)
    rank_df["rank_lipschitz"] = rank_df["pair_lipschitz_final_p90"].rank(ascending=True)
    rank_df["rank_gp_rmse"] = rank_df["cv_rmse_mean"].rank(ascending=True)
    rank_df["rank_gp_r2"] = rank_df["cv_r2_mean"].rank(ascending=False)
    rank_df["rank_topk"] = rank_df[f"global_top{args.top_k}_pred_true_percentile_mean"].rank(ascending=False)
    rank_df["average_rank"] = rank_df[
        ["rank_smooth_knn", "rank_lipschitz", "rank_gp_rmse", "rank_gp_r2", "rank_topk"]
    ].mean(axis=1)
    rank_df = rank_df.sort_values("average_rank")
    rank_df.to_csv(os.path.join(args.out_dir, "latent_space_rankings.csv"), index=False)

    # Plots
    plot_summary_bar(
        summary,
        metric="knn10_abs_final_diff_mean",
        out_path=os.path.join(args.out_dir, "bar_knn_abs_final_diff_mean.png"),
        ylabel="Mean |Δ final score| among kNN; lower=smoother",
    )
    plot_summary_bar(
        summary,
        metric="pair_lipschitz_final_p90",
        out_path=os.path.join(args.out_dir, "bar_pair_lipschitz_final_p90.png"),
        ylabel="90th percentile local Lipschitz; lower=smoother",
    )
    plot_summary_bar(
        summary,
        metric="cv_rmse_mean",
        out_path=os.path.join(args.out_dir, "bar_cv_rmse_mean.png"),
        ylabel="Surrogate CV RMSE; lower=easier GP fit",
    )
    plot_summary_bar(
        summary,
        metric="cv_r2_mean",
        out_path=os.path.join(args.out_dir, "bar_cv_r2_mean.png"),
        ylabel="Surrogate CV R²; higher=easier GP fit",
    )
    plot_summary_bar(
        summary,
        metric=f"global_top{args.top_k}_pred_true_percentile_mean",
        out_path=os.path.join(args.out_dir, "bar_topk_true_percentile_mean.png"),
        ylabel=f"True percentile of top-{args.top_k} predicted; higher=easier optimize",
    )

    print("\n=== Latent-space ranking; lower average_rank is better ===")
    print(rank_df.to_string(index=False))
    print(f"\nSaved results to: {args.out_dir}")


if __name__ == "__main__":
    main()
