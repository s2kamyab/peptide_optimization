from __future__ import annotations

"""
Visualize Bayesian-optimization hypervolume history.

Expected CSV columns (the attached file contains all of these):
    bo_iter
    hypervolume
    best_hypervolume
    n_accepted
    n_labeled
    pareto_size
    improved
    acq_value
    n_dominates_train

The main publication figure plots:
    - current hypervolume at each BO iteration
    - best-so-far hypervolume
    - markers on iterations where hypervolume improved

Additional diagnostic figures plot:
    - Pareto-front size
    - accepted candidates per iteration
    - number of candidates dominating the training Pareto
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "lines.linewidth": 2.2,
})


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def require_columns(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def save_main_hypervolume_plot(df: pd.DataFrame, out_path: Path):
    x = numeric(df, "bo_iter") + 1
    hv = numeric(df, "hypervolume")
    best = numeric(df, "best_hypervolume")

    improved = (
        numeric(df, "improved").fillna(0).astype(int).astype(bool)
        if "improved" in df.columns
        else best.diff().fillna(0) > 0
    )

    fig, ax = plt.subplots(figsize=(9.5, 5.8))

    ax.plot(
        x,
        hv,
        marker="o",
        markersize=5,
        label="Current hypervolume",
    )

    ax.plot(
        x,
        best,
        linestyle="--",
        marker="s",
        markersize=4,
        label="Best-so-far hypervolume",
    )

    if improved.any():
        ax.scatter(
            x[improved],
            best[improved],
            s=70,
            marker="*",
            label="Hypervolume improvement",
            zorder=5,
        )

    ax.set_xlabel("Bayesian optimization iteration")
    ax.set_ylabel("Hypervolume")
    ax.set_title("Multi-objective Bayesian optimization hypervolume history")
    ax.grid(True, alpha=0.25)
    ax.legend()

    # Integer iteration ticks when the run is short enough.
    if len(df) <= 30:
        ax.set_xticks(x)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_best_hv_only(df: pd.DataFrame, out_path: Path):
    x = numeric(df, "bo_iter") + 1
    best = numeric(df, "best_hypervolume")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.step(
        x,
        best,
        where="post",
        linewidth=2.5,
        label="Best-so-far hypervolume",
    )
    ax.scatter(x, best, s=28)

    ax.set_xlabel("Bayesian optimization iteration")
    ax.set_ylabel("Best hypervolume")
    ax.set_title("Best-so-far hypervolume during Bayesian optimization")
    ax.grid(True, alpha=0.25)

    if len(df) <= 30:
        ax.set_xticks(x)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_pareto_size(df: pd.DataFrame, out_path: Path):
    if "pareto_size" not in df.columns:
        return

    x = numeric(df, "bo_iter") + 1
    y = numeric(df, "pareto_size")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x, y, marker="o")
    ax.set_xlabel("Bayesian optimization iteration")
    ax.set_ylabel("Pareto-front size")
    ax.set_title("Pareto-front size across Bayesian optimization")
    ax.grid(True, alpha=0.25)

    if len(df) <= 30:
        ax.set_xticks(x)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_candidate_counts(df: pd.DataFrame, out_path: Path):
    available = [
        c for c in ["n_accepted", "n_dominates_train"]
        if c in df.columns
    ]
    if not available:
        return

    x = numeric(df, "bo_iter") + 1

    fig, ax = plt.subplots(figsize=(9, 5.5))

    if "n_accepted" in df.columns:
        ax.plot(
            x,
            numeric(df, "n_accepted"),
            marker="o",
            label="Accepted candidates",
        )

    if "n_dominates_train" in df.columns:
        ax.plot(
            x,
            numeric(df, "n_dominates_train"),
            marker="s",
            label="Candidates dominating training Pareto",
        )

    ax.set_xlabel("Bayesian optimization iteration")
    ax.set_ylabel("Number of candidates")
    ax.set_title("Candidate discovery across Bayesian optimization")
    ax.grid(True, alpha=0.25)
    ax.legend()

    if len(df) <= 30:
        ax.set_xticks(x)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary(df: pd.DataFrame, out_path: Path):
    hv = numeric(df, "hypervolume")
    best = numeric(df, "best_hypervolume")
    iterations = numeric(df, "bo_iter") + 1

    initial_hv = float(hv.iloc[0])
    final_hv = float(hv.iloc[-1])
    final_best = float(best.iloc[-1])

    absolute_gain = final_best - initial_hv
    relative_gain = (
        absolute_gain / initial_hv
        if abs(initial_hv) > 1e-12
        else float("nan")
    )

    best_idx = best.idxmax()
    best_iter = int(iterations.loc[best_idx])

    improved_count = (
        int(numeric(df, "improved").fillna(0).astype(int).sum())
        if "improved" in df.columns
        else int((best.diff().fillna(0) > 0).sum())
    )

    rows = [{
        "n_bo_iterations": int(len(df)),
        "initial_hypervolume": initial_hv,
        "final_hypervolume": final_hv,
        "final_best_hypervolume": final_best,
        "absolute_best_hypervolume_gain": absolute_gain,
        "relative_best_hypervolume_gain": relative_gain,
        "best_hypervolume_iteration": best_iter,
        "number_of_improving_iterations": improved_count,
        "final_pareto_size": (
            float(numeric(df, "pareto_size").iloc[-1])
            if "pareto_size" in df.columns else np.nan
        ),
        "total_accepted_candidates": (
            float(numeric(df, "n_accepted").sum())
            if "n_accepted" in df.columns else np.nan
        ),
        "total_candidates_dominating_training_pareto": (
            float(numeric(df, "n_dominates_train").sum())
            if "n_dominates_train" in df.columns else np.nan
        ),
    }]

    pd.DataFrame(rows).to_csv(out_path, index=False)


def main():
    p = argparse.ArgumentParser(
        description="Plot multi-objective BO hypervolume history."
    )
    p.add_argument(
        "--history-csv",
        required=True,
        help="Path to bo_hypervolume_history.csv",
    )
    p.add_argument(
        "--out-dir",
        default="bo_hypervolume_plots",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.history_csv)
    require_columns(
        df,
        ["bo_iter", "hypervolume", "best_hypervolume"],
    )

    df["bo_iter"] = numeric(df, "bo_iter")
    df = df.dropna(subset=["bo_iter"]).sort_values("bo_iter").reset_index(drop=True)

    save_main_hypervolume_plot(
        df,
        out_dir / "01_hypervolume_history.png",
    )

    save_best_hv_only(
        df,
        out_dir / "02_best_hypervolume_step_plot.png",
    )

    save_pareto_size(
        df,
        out_dir / "03_pareto_front_size.png",
    )

    save_candidate_counts(
        df,
        out_dir / "04_candidate_discovery_history.png",
    )

    write_summary(
        df,
        out_dir / "hypervolume_summary.csv",
    )

    print(f"Rows loaded: {len(df)}")
    print(
        f"Initial HV: {float(numeric(df, 'hypervolume').iloc[0]):.6f}"
    )
    print(
        f"Final best HV: {float(numeric(df, 'best_hypervolume').iloc[-1]):.6f}"
    )
    print(f"Saved figures to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
