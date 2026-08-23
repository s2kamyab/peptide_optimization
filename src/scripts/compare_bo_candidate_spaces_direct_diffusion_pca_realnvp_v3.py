from __future__ import annotations

"""
Compare BO candidate spaces exported by the leakage-safe H96 direct sequence
Diffusion + PCA/whitening + RealNVP Cu fine-tuning framework.

Spaces compared
---------------
1) ddim_epsilon0
   Raw 200-D deterministic DDIM-inverted direct-diffusion coordinate.

2) pca_z_bo
   Train-only PCA-whitened reduced coordinate (normally 64-D). This is the
   pretraining-consistent BO baseline exported by the fine-tuner.

3) realnvp_epsilonK
   200-D Cu-aligned coordinate after the fine-tuned RealNVP transformation:
       epsilon0 -> RealNVP -> epsilonK

Leakage policy
--------------
By default, only rows whose exported `split` equals `train` are used to choose
between BO spaces. Validation and test therefore remain held out from
representation selection.

Objective support
-----------------
The attached fine-tuner optimizes a scalar score and exports it as `score`.
Therefore the default comparison is scalar BO with GP expected improvement.
If multiple objective columns are supplied, the script automatically switches
to the original multi-objective ParEGO/hypervolume benchmark.

Evaluation criteria
-------------------
A. Geometry / intrinsic dimensionality
B. Local objective smoothness
C. Held-out Gaussian-process learnability
D. Offline BO sample efficiency on the same fixed candidate pool

All objectives are assumed to be maximized.
"""

import argparse
import importlib.util
import json
import math
import os
import random
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm, spearmanr
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


DEFAULT_OBJECTIVES = ["score"]


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sorted_prefixed_columns(df: pd.DataFrame, prefix: str) -> List[str]:
    """
    Return only coordinate columns with an integer suffix.

    Examples:
      prefix="epsilon_" -> epsilon_00 ... epsilon_63

    This deliberately excludes metadata columns such as:
      epsilon_norm

    The earlier implementation used startswith(prefix), which incorrectly
    included epsilon_norm and made diffusion_epsilon 65-D instead of 64-D.
    """
    pattern = re.compile(r"^" + re.escape(prefix) + r"(\d+)$")
    matched = []
    for c in df.columns:
        m = pattern.match(c)
        if m:
            matched.append((int(m.group(1)), c))
    matched.sort(key=lambda x: x[0])
    return [c for _, c in matched]


def finite_rows(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)


def import_module_py314_safe(path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def pareto_mask_max(Y: np.ndarray) -> np.ndarray:
    """Return nondominated mask for maximization."""
    Y = np.asarray(Y, dtype=float)
    n = len(Y)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        # j dominates i if all Yj >= Yi and at least one >.
        dominates_i = np.all(Y >= Y[i], axis=1) & np.any(Y > Y[i], axis=1)
        dominates_i[i] = False
        if np.any(dominates_i):
            keep[i] = False
    return keep


def normalize_objectives_global(Y: np.ndarray):
    lo = np.nanmin(Y, axis=0)
    hi = np.nanmax(Y, axis=0)
    span = np.maximum(hi - lo, 1e-12)
    Yn = (Y - lo) / span
    return Yn, lo, hi


# ---------------------------------------------------------------------
# Load representations
# ---------------------------------------------------------------------

def load_direct_spaces(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Load the three coordinate systems exported by the direct fine-tuner."""
    spaces = {}

    eps0_cols = sorted_prefixed_columns(df, "epsilon0_")
    zbo_cols = sorted_prefixed_columns(df, "z_bo_")
    epsk_cols = sorted_prefixed_columns(df, "epsilonK_")

    print(
        "Detected coordinate columns: "
        f"epsilon0={len(eps0_cols)}, z_bo={len(zbo_cols)}, "
        f"epsilonK={len(epsk_cols)}"
    )

    if eps0_cols:
        spaces["ddim_epsilon0"] = df[eps0_cols].to_numpy(dtype=np.float64)
    if zbo_cols:
        spaces["pca_z_bo"] = df[zbo_cols].to_numpy(dtype=np.float64)
    if epsk_cols:
        spaces["realnvp_epsilonK"] = df[epsk_cols].to_numpy(dtype=np.float64)

    return spaces




# ---------------------------------------------------------------------
# Geometry / intrinsic dimensionality
# ---------------------------------------------------------------------

def geometry_metrics(X: np.ndarray) -> Dict[str, float]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA()
    pca.fit(Xs)
    eig = pca.explained_variance_
    ratio = pca.explained_variance_ratio_

    participation = float((eig.sum() ** 2) / np.maximum((eig ** 2).sum(), 1e-12))
    cumulative = np.cumsum(ratio)

    def n_for(v):
        return int(np.searchsorted(cumulative, v) + 1)

    return {
        "dimension": int(X.shape[1]),
        "effective_dimension_participation_ratio": participation,
        "pca_components_90pct": n_for(0.90),
        "pca_components_95pct": n_for(0.95),
        "pca_components_99pct": n_for(0.99),
        "pc1_explained_fraction": float(ratio[0]),
        "pc2_cumulative_fraction": float(cumulative[min(1, len(cumulative)-1)]),
        "mean_feature_std_raw": float(np.mean(np.std(X, axis=0))),
    }


# ---------------------------------------------------------------------
# Local objective smoothness
# ---------------------------------------------------------------------

def local_objective_metrics(
    X: np.ndarray,
    Y: np.ndarray,
    k: int,
    pair_sample: int,
    seed: int,
) -> Dict[str, float]:
    """
    A BO-friendly representation should tend to put similar objective vectors nearby.

    Reports:
      - kNN objective L2
      - correlation of representation distance with objective distance
    """
    Xs = StandardScaler().fit_transform(X)
    Yn, _, _ = normalize_objectives_global(Y)

    k_eff = min(k + 1, len(Xs))
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    nn.fit(Xs)
    dists, inds = nn.kneighbors(Xs)

    # Drop self-neighbor.
    dists = dists[:, 1:]
    inds = inds[:, 1:]

    obj_d = []
    for i in range(len(Xs)):
        dif = Yn[inds[i]] - Yn[i]
        obj_d.extend(np.linalg.norm(dif, axis=1).tolist())

    rng = np.random.default_rng(seed)
    m = min(pair_sample, len(Xs) * 20)
    a = rng.integers(0, len(Xs), size=m)
    b = rng.integers(0, len(Xs), size=m)
    ok = a != b
    a, b = a[ok], b[ok]

    dx = np.linalg.norm(Xs[a] - Xs[b], axis=1)
    dy = np.linalg.norm(Yn[a] - Yn[b], axis=1)

    rho = spearmanr(dx, dy, nan_policy="omit").statistic
    pearson = np.corrcoef(dx, dy)[0, 1] if len(dx) > 2 else np.nan

    return {
        "knn_k": int(k_eff - 1),
        "knn_objective_l2_mean": float(np.mean(obj_d)),
        "knn_objective_l2_median": float(np.median(obj_d)),
        "pair_distance_objective_spearman": float(rho),
        "pair_distance_objective_pearson": float(pearson),
    }


# ---------------------------------------------------------------------
# Gaussian-process held-out learnability
# ---------------------------------------------------------------------

def build_gp(random_state: int):
    # Isotropic Matern after feature standardization:
    # fair across coordinate systems and much more stable than 64+ ARD parameters.
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e2))
        * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-7, 1e0))
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-8,
        normalize_y=True,
        n_restarts_optimizer=0,
        random_state=random_state,
    )


def gp_learnability(
    X: np.ndarray,
    Y: np.ndarray,
    objective_names: Sequence[str],
    train_sizes: Sequence[int],
    test_fraction: float,
    seeds: Sequence[int],
) -> pd.DataFrame:
    rows = []
    n = len(X)

    for seed in seeds:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_test = max(1, int(round(test_fraction * n)))
        test_idx = perm[:n_test]
        pool_idx = perm[n_test:]

        for train_size in train_sizes:
            train_size = min(int(train_size), len(pool_idx))
            train_idx = pool_idx[:train_size]

            scaler = StandardScaler().fit(X[train_idx])
            Xtr = scaler.transform(X[train_idx])
            Xte = scaler.transform(X[test_idx])

            per_obj = []
            for j, obj in enumerate(objective_names):
                gp = build_gp(seed + j)
                gp.fit(Xtr, Y[train_idx, j])
                pred, std = gp.predict(Xte, return_std=True)
                std = np.maximum(std, 1e-9)

                rmse = math.sqrt(mean_squared_error(Y[test_idx, j], pred))
                mae = mean_absolute_error(Y[test_idx, j], pred)
                r2 = r2_score(Y[test_idx, j], pred)
                spear = spearmanr(Y[test_idx, j], pred, nan_policy="omit").statistic

                # Gaussian negative log predictive density.
                resid = Y[test_idx, j] - pred
                nlpd = np.mean(
                    0.5 * np.log(2.0 * np.pi * std**2)
                    + 0.5 * (resid / std)**2
                )

                coverage95 = np.mean(np.abs(resid) <= 1.96 * std)

                per_obj.append((rmse, mae, r2, spear, nlpd, coverage95))

                rows.append({
                    "seed": seed,
                    "train_size": train_size,
                    "objective": obj,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                    "spearman": spear,
                    "nlpd": nlpd,
                    "coverage95": coverage95,
                })

            arr = np.asarray(per_obj, dtype=float)
            rows.append({
                "seed": seed,
                "train_size": train_size,
                "objective": "__MEAN__",
                "rmse": float(np.nanmean(arr[:, 0])),
                "mae": float(np.nanmean(arr[:, 1])),
                "r2": float(np.nanmean(arr[:, 2])),
                "spearman": float(np.nanmean(arr[:, 3])),
                "nlpd": float(np.nanmean(arr[:, 4])),
                "coverage95": float(np.nanmean(arr[:, 5])),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Offline multi-objective BO benchmark: ParEGO
# ---------------------------------------------------------------------

def augmented_tchebycheff(Yn: np.ndarray, w: np.ndarray, rho: float = 0.05):
    """
    Maximization scalarization toward ideal point 1.
    Larger is better.
    """
    d = 1.0 - Yn
    loss = np.max(w[None, :] * d, axis=1) + rho * np.sum(w[None, :] * d, axis=1)
    return -loss


def expected_improvement_max(mu, sigma, best):
    sigma = np.maximum(sigma, 1e-12)
    z = (mu - best) / sigma
    return (mu - best) * norm.cdf(z) + sigma * norm.pdf(z)


def monte_carlo_hypervolume_max(
    Y: np.ndarray,
    ref: np.ndarray,
    upper: np.ndarray,
    samples_unit: np.ndarray,
):
    """
    Fixed-sample Monte Carlo hypervolume for maximization.
    Same MC samples are reused for all spaces/runs, making comparisons fair.
    """
    if len(Y) == 0:
        return 0.0

    P = Y[pareto_mask_max(Y)]
    box = np.maximum(upper - ref, 1e-12)
    samples = ref[None, :] + samples_unit * box[None, :]

    dominated = np.zeros(len(samples), dtype=bool)
    # chunk Pareto points to keep memory bounded
    for p in P:
        dominated |= np.all(samples <= p[None, :], axis=1)

    return float(np.prod(box) * dominated.mean())


def scalar_offline_bo(
    X: np.ndarray,
    y: np.ndarray,
    init_size: int,
    budget: int,
    seeds: Sequence[int],
) -> pd.DataFrame:
    """Discrete-candidate scalar BO benchmark using GP expected improvement."""
    y = np.asarray(y, dtype=float).reshape(-1)
    rows = []
    n = len(X)
    budget = min(int(budget), n)
    Xs = StandardScaler().fit_transform(X)

    for seed in seeds:
        rng = np.random.default_rng(seed)
        observed = rng.choice(
            n, size=min(init_size, budget), replace=False
        ).tolist()
        remaining = set(range(n)) - set(observed)

        while True:
            best = float(np.max(y[observed]))
            rows.append({
                "seed": seed,
                "n_evaluated": len(observed),
                "best_objective": best,
            })

            if len(observed) >= budget or not remaining:
                break

            gp = build_gp(seed + len(observed))
            gp.fit(Xs[observed], y[observed])
            rem = np.fromiter(remaining, dtype=int)
            mu, std = gp.predict(Xs[rem], return_std=True)
            ei = expected_improvement_max(mu, std, best)
            next_idx = int(rem[int(np.argmax(ei))])
            observed.append(next_idx)
            remaining.remove(next_idx)

    return pd.DataFrame(rows)


def parego_offline_bo(
    X: np.ndarray,
    Y: np.ndarray,
    init_size: int,
    budget: int,
    seeds: Sequence[int],
    hv_mc_samples: int,
) -> pd.DataFrame:
    """
    Discrete-candidate offline ParEGO benchmark.

    At each iteration:
      1. random objective scalarization
      2. fit a GP on observed scalarized values
      3. calculate EI over all remaining evaluated candidates
      4. reveal the objective vector of the selected candidate
      5. record hypervolume

    This tests whether a coordinate space makes BO sample-efficient on a common
    finite candidate pool.
    """
    Yn, _, _ = normalize_objectives_global(Y)
    ref = np.full(Y.shape[1], -0.05, dtype=float)
    upper = np.full(Y.shape[1], 1.05, dtype=float)

    # Common fixed MC integration points.
    common_rng = np.random.default_rng(99173)
    hv_samples = common_rng.random((hv_mc_samples, Y.shape[1]))

    rows = []
    n = len(X)
    budget = min(int(budget), n)

    for seed in seeds:
        rng = np.random.default_rng(seed)
        init = rng.choice(n, size=min(init_size, budget), replace=False).tolist()
        observed = list(init)
        remaining = set(range(n)) - set(observed)

        # Fit input scaling from full candidate geometry only.
        # This does not use objectives, so it does not leak target information.
        Xs = StandardScaler().fit_transform(X)

        for step in range(len(observed), budget + 1):
            hv = monte_carlo_hypervolume_max(
                Yn[observed], ref, upper, hv_samples
            )
            rows.append({
                "seed": seed,
                "n_evaluated": len(observed),
                "hypervolume": hv,
            })

            if len(observed) >= budget or not remaining:
                break

            w = rng.dirichlet(np.ones(Y.shape[1]))
            scalar = augmented_tchebycheff(Yn[observed], w)

            gp = build_gp(seed + step)
            gp.fit(Xs[observed], scalar)

            rem = np.fromiter(remaining, dtype=int)
            mu, std = gp.predict(Xs[rem], return_std=True)
            best_scalar = float(np.max(scalar))
            ei = expected_improvement_max(mu, std, best_scalar)

            next_idx = int(rem[int(np.argmax(ei))])
            observed.append(next_idx)
            remaining.remove(next_idx)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Aggregate ranking
# ---------------------------------------------------------------------

def aggregate_gp(gp_df: pd.DataFrame, max_train_size: int):
    d = gp_df[
        (gp_df["objective"] == "__MEAN__")
        & (gp_df["train_size"] == max_train_size)
    ]
    return {
        "gp_rmse": float(d["rmse"].mean()),
        "gp_r2": float(d["r2"].mean()),
        "gp_spearman": float(d["spearman"].mean()),
        "gp_nlpd": float(d["nlpd"].mean()),
        "gp_coverage95": float(d["coverage95"].mean()),
    }


def rank_spaces(summary: pd.DataFrame, scalar_objective: bool) -> pd.DataFrame:
    """Composite rank; lower mean rank is better."""
    d = summary.copy()

    metrics = [
        ("gp_rmse", True),
        ("gp_r2", False),
        ("gp_spearman", False),
        ("knn_objective_l2_mean", True),
        ("pair_distance_objective_spearman", False),
    ]
    if scalar_objective:
        metrics += [
            ("final_best_objective", False),
            ("best_objective_auc", False),
        ]
    else:
        metrics += [
            ("final_hypervolume", False),
            ("hypervolume_auc", False),
        ]

    rank_cols = []
    for metric, lower_better in metrics:
        if metric not in d.columns:
            continue
        col = f"rank_{metric}"
        d[col] = d[metric].rank(
            ascending=lower_better, method="average"
        )
        rank_cols.append(col)

    d["mean_rank"] = d[rank_cols].mean(axis=1, skipna=True)
    d["recommended_order"] = d["mean_rank"].rank(method="min")
    return d.sort_values(["recommended_order", "space"])


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def save_gp_plot(all_gp: pd.DataFrame, out_dir: Path):
    means = (
        all_gp[all_gp["objective"] == "__MEAN__"]
        .groupby(["space", "train_size"], as_index=False)["rmse"]
        .mean()
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    for space, g in means.groupby("space"):
        ax.plot(g["train_size"], g["rmse"], marker="o", label=space)
    ax.set_xlabel("GP training size")
    ax.set_ylabel("Mean held-out objective RMSE")
    ax.set_title("GP learnability by candidate BO space")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "gp_rmse_vs_training_size.png", dpi=300)
    plt.close(fig)


def save_bo_plot(all_bo: pd.DataFrame, out_dir: Path, scalar_objective: bool):
    metric = "best_objective" if scalar_objective else "hypervolume"
    ylabel = "Best observed objective" if scalar_objective else "Monte Carlo hypervolume"
    title = (
        "Offline GP-EI sample efficiency by candidate space"
        if scalar_objective
        else "Offline ParEGO sample efficiency by candidate space"
    )

    means = (
        all_bo.groupby(["space", "n_evaluated"], as_index=False)[metric]
        .mean()
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    for space, g in means.groupby("space"):
        ax.plot(g["n_evaluated"], g[metric], label=space)
    ax.set_xlabel("Number of objective evaluations")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "offline_bo_sample_efficiency.png", dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description=(
            "Compare direct-diffusion DDIM epsilon0, PCA-whitened z_bo, and "
            "RealNVP epsilonK spaces for Bayesian optimization."
        )
    )
    p.add_argument("--coordinate-csv", required=True)
    p.add_argument(
        "--objective-cols", nargs="+", default=DEFAULT_OBJECTIVES,
        help=(
            "Default is scalar 'score' exported by the fine-tuner. "
            "Supplying multiple columns switches to multi-objective ParEGO."
        ),
    )
    p.add_argument("--split-col", default="split")
    p.add_argument(
        "--comparison-split", default="train",
        help=(
            "Default 'train' keeps validation/test out of BO-space selection. "
            "Use 'all' only for descriptive post-hoc analysis."
        ),
    )
    p.add_argument(
        "--out-dir", default="direct_diffusion_bo_space_comparison"
    )

    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--pair-sample", type=int, default=30000)

    p.add_argument(
        "--gp-train-sizes", type=int, nargs="+",
        default=[64, 128, 256, 512, 1024],
    )
    p.add_argument("--gp-test-fraction", type=float, default=0.20)
    p.add_argument(
        "--gp-seeds", type=int, nargs="+", default=[11, 22, 33]
    )

    p.add_argument("--skip-offline-bo", action="store_true")
    p.add_argument("--bo-init-size", type=int, default=32)
    p.add_argument("--bo-budget", type=int, default=100)
    p.add_argument(
        "--bo-seeds", type=int, nargs="+", default=[101, 202, 303]
    )
    p.add_argument("--hv-mc-samples", type=int, default=30000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    warnings.filterwarnings(
        "ignore",
        category=ConvergenceWarning,
        message=r".*noise_level.*lower bound.*",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.coordinate_csv)
    missing_obj = [c for c in args.objective_cols if c not in df.columns]
    if missing_obj:
        raise ValueError(
            f"Missing objective columns: {missing_obj}. "
            f"Available columns include: {list(df.columns)[:80]}"
        )

    if args.comparison_split.lower() != "all":
        if args.split_col not in df.columns:
            raise ValueError(
                f"Requested comparison split {args.comparison_split!r}, but "
                f"column {args.split_col!r} is absent."
            )
        before = len(df)
        df = df[
            df[args.split_col].astype(str).str.lower()
            == args.comparison_split.lower()
        ].reset_index(drop=True)
        print(
            f"Split filter {args.comparison_split!r}: {before} -> {len(df)} rows"
        )

    Y = df[args.objective_cols].to_numpy(dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]

    spaces = load_direct_spaces(df)
    if not spaces:
        raise ValueError(
            "No direct-diffusion coordinate columns found. Expected epsilon0_*, "
            "z_bo_*, and/or epsilonK_*."
        )

    # Dimensional sanity checks matching the attached fine-tuner.
    if "ddim_epsilon0" in spaces and spaces["ddim_epsilon0"].shape[1] != 200:
        raise ValueError(
            f"ddim_epsilon0 has {spaces['ddim_epsilon0'].shape[1]} dimensions; expected 200."
        )
    if "realnvp_epsilonK" in spaces and spaces["realnvp_epsilonK"].shape[1] != 200:
        raise ValueError(
            f"realnvp_epsilonK has {spaces['realnvp_epsilonK'].shape[1]} dimensions; expected 200."
        )

    common = np.isfinite(Y).all(axis=1)
    for X in spaces.values():
        common &= np.isfinite(X).all(axis=1)
    Y = Y[common]
    df = df.loc[common].reset_index(drop=True)
    spaces = {k: v[common] for k, v in spaces.items()}

    if len(Y) < 20:
        raise RuntimeError(f"Only {len(Y)} rows remain for comparison.")

    scalar_objective = Y.shape[1] == 1
    print(f"Rows used in fair comparison: {len(Y)}")
    print(f"Objectives: {args.objective_cols}")
    print(
        "Offline BO mode: "
        + ("scalar GP-EI" if scalar_objective else "multi-objective ParEGO")
    )
    print(f"Spaces: {[(k, v.shape) for k, v in spaces.items()]}")

    all_gp = []
    all_bo = []
    summaries = []

    for space_name, X in spaces.items():
        print(f"\n=== {space_name} shape={X.shape} ===")

        geom = geometry_metrics(X)
        local = local_objective_metrics(
            X, Y, args.knn_k, args.pair_sample, args.seed
        )

        gp = gp_learnability(
            X=X,
            Y=Y,
            objective_names=args.objective_cols,
            train_sizes=args.gp_train_sizes,
            test_fraction=args.gp_test_fraction,
            seeds=args.gp_seeds,
        )
        gp.insert(0, "space", space_name)
        all_gp.append(gp)
        max_gp_size = int(gp["train_size"].max())
        gp_summary = aggregate_gp(gp, max_gp_size)

        row = {
            "space": space_name,
            "bo_eligible": True,
            **geom,
            **local,
            **gp_summary,
        }

        if not args.skip_offline_bo:
            if scalar_objective:
                bo = scalar_offline_bo(
                    X=X,
                    y=Y[:, 0],
                    init_size=args.bo_init_size,
                    budget=args.bo_budget,
                    seeds=args.bo_seeds,
                )
                bo.insert(0, "space", space_name)
                all_bo.append(bo)
                last_n = int(bo["n_evaluated"].max())
                final_best = float(
                    bo[bo["n_evaluated"] == last_n]["best_objective"].mean()
                )
                aucs = []
                for _, g in bo.groupby("seed"):
                    g = g.sort_values("n_evaluated")
                    aucs.append(
                        np.trapezoid(g["best_objective"], g["n_evaluated"])
                    )
                row["final_best_objective"] = final_best
                row["best_objective_auc"] = float(np.mean(aucs))
            else:
                bo = parego_offline_bo(
                    X=X,
                    Y=Y,
                    init_size=args.bo_init_size,
                    budget=args.bo_budget,
                    seeds=args.bo_seeds,
                    hv_mc_samples=args.hv_mc_samples,
                )
                bo.insert(0, "space", space_name)
                all_bo.append(bo)
                last_n = int(bo["n_evaluated"].max())
                final_hv = float(
                    bo[bo["n_evaluated"] == last_n]["hypervolume"].mean()
                )
                aucs = []
                for _, g in bo.groupby("seed"):
                    g = g.sort_values("n_evaluated")
                    aucs.append(
                        np.trapezoid(g["hypervolume"], g["n_evaluated"])
                    )
                row["final_hypervolume"] = final_hv
                row["hypervolume_auc"] = float(np.mean(aucs))

        summaries.append(row)

    gp_all = pd.concat(all_gp, ignore_index=True)
    gp_all.to_csv(out_dir / "gp_learnability_all_spaces.csv", index=False)
    save_gp_plot(gp_all, out_dir)

    if all_bo:
        bo_all = pd.concat(all_bo, ignore_index=True)
        bo_all.to_csv(out_dir / "offline_bo_all_spaces.csv", index=False)
        save_bo_plot(bo_all, out_dir, scalar_objective)

    summary = pd.DataFrame(summaries)
    ranked = rank_spaces(summary, scalar_objective)
    ranked.to_csv(
        out_dir / "bo_space_comparison_summary_ranked.csv", index=False
    )

    best = ranked.iloc[0]
    show_cols = [
        "space",
        "dimension",
        "effective_dimension_participation_ratio",
        "pca_components_95pct",
        "knn_objective_l2_mean",
        "pair_distance_objective_spearman",
        "gp_rmse",
        "gp_r2",
        "gp_spearman",
    ]
    if scalar_objective:
        if "final_best_objective" in ranked.columns:
            show_cols += ["final_best_objective", "best_objective_auc"]
    else:
        if "final_hypervolume" in ranked.columns:
            show_cols += ["final_hypervolume", "hypervolume_auc"]
    show_cols += ["mean_rank"]

    lines = [
        "DIRECT-DIFFUSION BO CANDIDATE-SPACE COMPARISON",
        "=" * 88,
        f"n_rows={len(Y)}",
        f"comparison_split={args.comparison_split}",
        f"objectives={args.objective_cols}",
        "",
        "RANKED SUMMARY",
        "-" * 88,
        ranked[show_cols].to_string(index=False),
        "",
        f"TOP SPACE BY COMPOSITE RANK: {best['space']}",
        "",
        "INTERPRETATION RULES",
        "-" * 88,
        "Lower GP RMSE and kNN objective distance are better.",
        "Higher GP R2, GP Spearman, distance/objective Spearman, and offline BO performance are better.",
        "Higher effective dimension is not automatically better.",
        "pca_z_bo is the pretraining-consistent reduced baseline.",
        "realnvp_epsilonK should be preferred only if it improves GP/objective locality and BO sample efficiency enough to justify the flow.",
        "ddim_epsilon0 is a valid raw 200-D generative coordinate and is retained as a baseline.",
        "By default only the fine-tuning train split is used, so validation/test remain held out from BO-space selection.",
        "Final confirmation should use matched prospective BO with the same initial observations, acquisition function, budget, and seeds.",
    ]
    report = "\n".join(lines) + "\n"
    (out_dir / "bo_space_comparison_report.txt").write_text(
        report, encoding="utf-8"
    )
    print(report)
    print(f"Saved results to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
