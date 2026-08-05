from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Global plot font settings
# ============================================================

FONT_FAMILY = "Arial"        # or "Times New Roman", "Calibri", "DejaVu Sans"
BASE_FONT_SIZE = 24

plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": BASE_FONT_SIZE,

    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 14,
    "axes.labelweight": "bold",

    "xtick.labelsize": 12,
    "ytick.labelsize": 12,

    "legend.fontsize": 12,
    "figure.titlesize": 16,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

DEFAULT_ROUNDTRIP_CSV = "scrambling_control_cu_latent_diffusion_last_epoch/diffusion_roundtrip_sequence_diagnostics.csv"


def choose_l2_column(df: pd.DataFrame) -> str:
    """Choose the latent-space L2 column available in the diagnostics file."""
    candidates = [
        "epsilon_roundtrip_l2",
        "latent_roundtrip_l2",
        "standardized_h_reconstruction_l2",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(
        "No latent L2 column found. Expected one of: "
        + ", ".join(candidates)
    )


def load_and_prepare(
    csv_path: str,
    control_type: str,
) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(csv_path)

    required = [
        "original_peptide",
        "control_type",
        "source_to_p1_edit_distance",
        "peptide_roundtrip_edit_distance",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns in {csv_path}: {missing}"
        )

    l2_col = choose_l2_column(df)

    # Remove optional overall average row, if present.
    if "source_peptide" in df.columns:
        df = df[df["source_peptide"].astype(str) != "__AVERAGE__"].copy()

    df["original_peptide"] = (
        df["original_peptide"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    df["control_type"] = (
        df["control_type"]
        .astype(str)
        .str.strip()
    )

    if control_type != "all":
        df = df[df["control_type"] == control_type].copy()

    if df.empty:
        raise ValueError(
            f"No rows remain after filtering control_type={control_type!r}."
        )

    numeric_cols = [
        "source_to_p1_edit_distance",
        "peptide_roundtrip_edit_distance",
        l2_col,
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols).copy()
    if df.empty:
        raise ValueError("No valid numeric rows remain after cleaning.")

    return df, l2_col


def make_summary(
    df: pd.DataFrame,
    l2_col: str,
) -> pd.DataFrame:
    summary = (
        df.groupby("original_peptide", sort=False)
        .agg(
            n_sequences=("source_to_p1_edit_distance", "size"),
            source_to_p1_edit_mean=("source_to_p1_edit_distance", "mean"),
            source_to_p1_edit_std=("source_to_p1_edit_distance", "std"),
            p1_to_p2_edit_mean=("peptide_roundtrip_edit_distance", "mean"),
            p1_to_p2_edit_std=("peptide_roundtrip_edit_distance", "std"),
            latent_l2_mean=(l2_col, "mean"),
            latent_l2_std=(l2_col, "std"),
            source_to_p1_edit_median=("source_to_p1_edit_distance", "median"),
            p1_to_p2_edit_median=("peptide_roundtrip_edit_distance", "median"),
            latent_l2_median=(l2_col, "median"),
        )
        .reset_index()
    )

    # For optimized_original there is one row per original, so std is NaN.
    for col in [
        "source_to_p1_edit_std",
        "p1_to_p2_edit_std",
        "latent_l2_std",
    ]:
        summary[col] = summary[col].fillna(0.0)

    return summary


def plot_grouped_bar(
    summary: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    labels = summary["original_peptide"].tolist()
    x = np.arange(len(labels), dtype=float)
    width = 0.25

    y1 = summary["source_to_p1_edit_mean"].to_numpy(dtype=float)
    y2 = summary["p1_to_p2_edit_mean"].to_numpy(dtype=float)
    y3 = summary["latent_l2_mean"].to_numpy(dtype=float)

    e1 = summary["source_to_p1_edit_std"].to_numpy(dtype=float)
    e2 = summary["p1_to_p2_edit_std"].to_numpy(dtype=float)
    e3 = summary["latent_l2_std"].to_numpy(dtype=float)

    plt.figure(figsize=(max(10, len(labels) * 1.25), 6))
    plt.bar(
        x - width,
        y1,
        width,
        yerr=e1,
        capsize=3,
        label="source → p1 edit distance",
    )
    plt.bar(
        x,
        y2,
        width,
        yerr=e2,
        capsize=3,
        label="p1 → p2 edit distance",
    )
    plt.bar(
        x + width,
        y3,
        width,
        yerr=e3,
        capsize=3,
        label="latent round-trip L2",
    )

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Metric value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_grouped_bar_two_axes(
    summary: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    labels = summary["original_peptide"].tolist()
    x = np.arange(len(labels), dtype=float)
    width = 0.28

    y1 = summary["source_to_p1_edit_mean"].to_numpy(dtype=float)
    y2 = summary["p1_to_p2_edit_mean"].to_numpy(dtype=float)
    y3 = summary["latent_l2_mean"].to_numpy(dtype=float)

    e1 = summary["source_to_p1_edit_std"].to_numpy(dtype=float)
    e2 = summary["p1_to_p2_edit_std"].to_numpy(dtype=float)
    e3 = summary["latent_l2_std"].to_numpy(dtype=float)

    fig, ax1 = plt.subplots(figsize=(max(10, len(labels) * 1.25), 6))
    ax2 = ax1.twinx()

    b1 = ax1.bar(
        x - width / 2,
        y1,
        width,
        yerr=e1,
        capsize=3,
        label="source → p1 edit distance",
    )
    b2 = ax1.bar(
        x + width / 2,
        y2,
        width,
        yerr=e2,
        capsize=3,
        label="p1 → p2 edit distance",
    )
    b3 = ax2.bar(
        x + 1.5 * width,
        y3,
        width,
        yerr=e3,
        capsize=3,
        label="latent round-trip L2",
        alpha=0.65,
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.set_ylabel("Edit distance")
    ax2.set_ylabel("Latent round-trip L2")
    ax1.set_title(title)

    handles = [b1, b2, b3]
    labels_legend = [h.get_label() for h in handles]
    ax1.legend(handles, labels_legend, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_metric_boxplots(
    df: pd.DataFrame,
    l2_col: str,
    out_path: Path,
    title: str,
) -> None:
    groups = df["original_peptide"].drop_duplicates().tolist()
    metrics: List[Tuple[str, str]] = [
        ("source_to_p1_edit_distance", "source → p1 edit distance"),
        ("peptide_roundtrip_edit_distance", "p1 → p2 edit distance"),
        (l2_col, "latent round-trip L2"),
    ]

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(max(10, len(groups) * 1.25), 12),
        sharex=True,
    )

    for ax, (col, ylabel) in zip(axes, metrics):
        data = [
            df.loc[df["original_peptide"] == pep, col]
            .dropna()
            .to_numpy(dtype=float)
            for pep in groups
        ]
        ax.boxplot(data, labels=groups, showfliers=True)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_title(title)
    axes[-1].tick_params(axis="x", rotation=45)
    for tick in axes[-1].get_xticklabels():
        tick.set_horizontalalignment("right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_original_marker_on_scramble_boxplots(
    df_all: pd.DataFrame,
    l2_col: str,
    out_path: Path,
    title: str,
) -> None:
    """Boxplots for scrambled controls, with optimized original overlaid.

    This is useful when the diagnostics file contains both optimized originals
    and scrambled controls.
    """
    groups = df_all["original_peptide"].drop_duplicates().tolist()
    metrics: List[Tuple[str, str]] = [
        ("source_to_p1_edit_distance", "source → p1 edit distance"),
        ("peptide_roundtrip_edit_distance", "p1 → p2 edit distance"),
        (l2_col, "latent round-trip L2"),
    ]

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(max(10, len(groups) * 1.25), 12),
        sharex=True,
    )

    scrambled = df_all[df_all["control_type"] == "scrambled_control"].copy()
    original = df_all[df_all["control_type"] == "optimized_original"].copy()

    for ax, (col, ylabel) in zip(axes, metrics):
        data = [
            scrambled.loc[scrambled["original_peptide"] == pep, col]
            .dropna()
            .to_numpy(dtype=float)
            for pep in groups
        ]

        ax.boxplot(data, labels=groups, showfliers=True)

        original_values = []
        original_x = []
        for i, pep in enumerate(groups, start=1):
            vals = original.loc[original["original_peptide"] == pep, col]
            if not vals.empty:
                original_x.append(i)
                original_values.append(float(vals.iloc[0]))

        ax.scatter(
            original_x,
            original_values,
            marker="D",
            s=60,
            label="optimized original",
        )
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper right")

    axes[0].set_title(title)
    axes[-1].tick_params(axis="x", rotation=45)
    for tick in axes[-1].get_xticklabels():
        tick.set_horizontalalignment("right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize diffusion smoothness diagnostics by dominating Pareto member."
        )
    )
    parser.add_argument(
        "--roundtrip-csv",
        default=DEFAULT_ROUNDTRIP_CSV,
        help="Path to diffusion_roundtrip_sequence_diagnostics.csv.",
    )
    parser.add_argument(
        "--out-dir",
        default="smoothness_visualizations_by_pareto_member",
    )
    parser.add_argument(
        "--control-type",
        default="optimized_original",
        choices=["optimized_original", "scrambled_control", "all"],
        help=(
            "Rows used for grouped mean bar plots. "
            "Use optimized_original for one bar group per Pareto member; "
            "use all or scrambled_control to summarize distributions."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, l2_col = load_and_prepare(
        csv_path=args.roundtrip_csv,
        control_type=args.control_type,
    )
    summary = make_summary(df, l2_col)

    summary_path = out_dir / "smoothness_summary_by_pareto_member.csv"
    summary.to_csv(summary_path, index=False)

    plot_grouped_bar(
        summary,
        out_dir / "smoothness_grouped_bar_same_axis.png",
        title=(
            "Smoothness diagnostics by dominating Pareto member "
            f"({args.control_type})"
        ),
    )

    plot_grouped_bar_two_axes(
        summary,
        out_dir / "smoothness_grouped_bar_two_axes.png",
        title=(
            "Smoothness diagnostics by dominating Pareto member "
            f"({args.control_type})"
        ),
    )

    plot_metric_boxplots(
        df,
        l2_col,
        out_dir / "smoothness_metric_boxplots.png",
        title=(
            "Smoothness metric distributions by dominating Pareto member "
            f"({args.control_type})"
        ),
    )

    # Always make a scramble-vs-original diagnostic when both row types exist.
    df_all, l2_col_all = load_and_prepare(
        csv_path=args.roundtrip_csv,
        control_type="all",
    )
    if set(df_all["control_type"].unique()) >= {
        "optimized_original",
        "scrambled_control",
    }:
        plot_original_marker_on_scramble_boxplots(
            df_all,
            l2_col_all,
            out_dir / "scrambled_boxplots_with_original_markers.png",
            title=(
                "Scrambled controls by Pareto member; optimized original shown as diamond"
            ),
        )

    print(f"Saved summary: {summary_path}")
    print(f"Saved plots in: {out_dir}")
    print("Key plots:")
    print(f" - {out_dir / 'smoothness_grouped_bar_same_axis.png'}")
    print(f" - {out_dir / 'smoothness_grouped_bar_two_axes.png'}")
    print(f" - {out_dir / 'smoothness_metric_boxplots.png'}")
    print(f" - {out_dir / 'scrambled_boxplots_with_original_markers.png'}")


if __name__ == "__main__":
    main()
