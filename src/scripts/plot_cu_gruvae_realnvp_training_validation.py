from __future__ import annotations

"""
Plot training + validation history for the Cu GRU-VAE + RealNVP fine-tuning run.

Expected history columns in the attached CSV include:
    train_flow_nll
    train_flow_nll_per_dim
    train_objective_mse
    val_flow_nll
    val_flow_nll_per_dim
    val_objective_mse

    val_<objective>_mse
    val_<objective>_pearson
    val_objective_mean_pearson

    val_zK_global_abs_mean
    val_zK_dim_std_mean
    val_zK_dim_std_min
    val_zK_dim_std_max
    val_zK_norm_mean
    val_zK_norm_std

    val_flow_logdet_mean
    val_flow_logdet_std
    val_flow_roundtrip_h_l2_mean
    val_flow_roundtrip_h_l2_max
    val_flow_roundtrip_h_cosine_mean
    val_flow_logdet_cancel_abs_mean

    val_source_to_decode_edit_mean
    val_source_to_decode_edit_median
    val_decode_unique_fraction

    val_local_sigma
    val_local_zK_l2_mean
    val_local_h0_l2_mean
    val_local_sequence_edit_mean
    val_local_sequence_edit_median
    val_local_identical_fraction

Only the flow loss and objective-head loss have both training and validation
curves. The remaining RealNVP / decoder / local-smoothness metrics are
validation diagnostics only, so this script does not fabricate training curves.
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
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
    "lines.linewidth": 2.0,
})


OBJECTIVES = [
    "chelation_sub",
    "solubility_sub",
    "stability_sub",
    "expression_sub",
]


def series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def has_finite(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and series(df, col).notna().any()


def best_epoch(df: pd.DataFrame, col: str, mode: str = "min"):
    s = series(df, col)
    valid = s.dropna()
    if valid.empty:
        return None, None
    idx = valid.idxmin() if mode == "min" else valid.idxmax()
    return int(df.loc[idx, "epoch"]), float(s.loc[idx])


def save_plot(
    df: pd.DataFrame,
    columns,
    labels,
    ylabel,
    title,
    out_path: Path,
    selected_epoch=None,
    target_lines=None,
    ylim=None,
    logy=False,
):
    valid = [
        (c, l)
        for c, l in zip(columns, labels)
        if has_finite(df, c)
    ]
    if not valid:
        print(f"Skipping {out_path.name}: no finite data.")
        return

    fig, ax = plt.subplots(figsize=(9.2, 5.6))

    for col, label in valid:
        ax.plot(df["epoch"], series(df, col), label=label)

    if selected_epoch is not None:
        ax.axvline(
            selected_epoch,
            linestyle=":",
            linewidth=1.8,
            label=f"Selected epoch {selected_epoch}",
        )

    if target_lines:
        for y, label in target_lines:
            ax.axhline(
                y,
                linestyle="--",
                linewidth=1.5,
                label=label,
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if ylim is not None:
        ax.set_ylim(*ylim)
    if logy:
        ax.set_yscale("log")

    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_objective_mse(df: pd.DataFrame, out_path: Path):
    save_plot(
        df,
        [f"val_{o}_mse" for o in OBJECTIVES],
        ["Chelation", "Solubility", "Stability", "Expression"],
        ylabel="Validation MSE",
        title="Validation MSE by Cu objective",
        out_path=out_path,
    )


def save_objective_pearson(df: pd.DataFrame, out_path: Path):
    save_plot(
        df,
        [f"val_{o}_pearson" for o in OBJECTIVES],
        ["Chelation", "Solubility", "Stability", "Expression"],
        ylabel="Pearson correlation",
        title="Validation Pearson correlation by Cu objective",
        out_path=out_path,
        ylim=(-0.15, 1.0),
    )


def write_summary(df: pd.DataFrame, out_path: Path):
    metrics = [
        ("train_flow_nll_per_dim", "min"),
        ("val_flow_nll_per_dim", "min"),
        ("train_objective_mse", "min"),
        ("val_objective_mse", "min"),
        ("val_objective_mean_pearson", "max"),
        ("val_zK_global_abs_mean", "min"),
        ("val_zK_dim_std_mean", "closest_to_one"),
        ("val_zK_norm_mean", "closest_to_eight"),
        ("val_flow_roundtrip_h_l2_mean", "min"),
        ("val_flow_roundtrip_h_cosine_mean", "max"),
        ("val_source_to_decode_edit_mean", "min"),
        ("val_decode_unique_fraction", "max"),
        ("val_local_sequence_edit_mean", "min"),
        ("val_local_identical_fraction", "max"),
    ]

    rows = []

    for col, mode in metrics:
        if not has_finite(df, col):
            continue

        s = series(df, col)
        valid = s.dropna()

        if mode == "closest_to_one":
            idx = (valid - 1.0).abs().idxmin()
            direction = "closest_to_1"
        elif mode == "closest_to_eight":
            idx = (valid - 8.0).abs().idxmin()
            direction = "closest_to_8"
        else:
            idx = valid.idxmin() if mode == "min" else valid.idxmax()
            direction = mode

        rows.append({
            "metric": col,
            "direction": direction,
            "best_epoch": int(df.loc[idx, "epoch"]),
            "best_value": float(s.loc[idx]),
            "final_value": float(s.iloc[-1]),
        })

    pd.DataFrame(rows).to_csv(out_path, index=False)


def write_report(df: pd.DataFrame, out_path: Path):
    best_nll_ep, best_nll = best_epoch(
        df, "val_flow_nll_per_dim", "min"
    )
    best_obj_ep, best_obj = best_epoch(
        df, "val_objective_mse", "min"
    )
    best_corr_ep, best_corr = best_epoch(
        df, "val_objective_mean_pearson", "max"
    )
    best_local_ep, best_local = best_epoch(
        df, "val_local_sequence_edit_mean", "min"
    )

    lines = []
    lines.append("CU GRU-VAE + REALNVP TRAINING / VALIDATION SUMMARY")
    lines.append("=" * 82)
    lines.append(f"epochs={len(df)}")
    lines.append("")
    lines.append("BEST VALIDATION EPOCHS")
    lines.append("-" * 82)

    if best_nll_ep is not None:
        lines.append(
            f"Best validation flow NLL/dim: epoch={best_nll_ep}, "
            f"value={best_nll:.8f}"
        )
    if best_obj_ep is not None:
        lines.append(
            f"Best validation multi-objective MSE: epoch={best_obj_ep}, "
            f"value={best_obj:.8f}"
        )
    if best_corr_ep is not None:
        lines.append(
            f"Best validation mean objective Pearson: epoch={best_corr_ep}, "
            f"value={best_corr:.8f}"
        )
    if best_local_ep is not None:
        lines.append(
            f"Best local zK sequence-edit metric: epoch={best_local_ep}, "
            f"value={best_local:.8f}"
        )

    lines.append("")
    lines.append("FINAL EPOCH")
    lines.append("-" * 82)

    last = df.iloc[-1]
    for col in [
        "train_flow_nll_per_dim",
        "val_flow_nll_per_dim",
        "train_objective_mse",
        "val_objective_mse",
        "val_objective_mean_pearson",
        "val_zK_global_abs_mean",
        "val_zK_dim_std_mean",
        "val_zK_norm_mean",
        "val_flow_logdet_mean",
        "val_flow_roundtrip_h_l2_mean",
        "val_flow_roundtrip_h_cosine_mean",
        "val_source_to_decode_edit_mean",
        "val_decode_unique_fraction",
        "val_local_zK_l2_mean",
        "val_local_h0_l2_mean",
        "val_local_sequence_edit_mean",
        "val_local_identical_fraction",
    ]:
        if col in df.columns:
            v = pd.to_numeric(
                pd.Series([last[col]]),
                errors="coerce",
            ).iloc[0]
            if np.isfinite(v):
                lines.append(f"{col}={v:.8f}")

    lines.append("")
    lines.append("NOTE")
    lines.append("-" * 82)
    lines.append(
        "Only flow NLL and multi-objective MSE were logged for both training "
        "and validation. RealNVP base-distribution, invertibility, decoder, and "
        "local-smoothness diagnostics are validation-only."
    )

    out_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():
    p = argparse.ArgumentParser(
        description="Plot Cu GRU-VAE + RealNVP training and validation history."
    )
    p.add_argument("--history-csv", required=True)
    p.add_argument(
        "--out-dir",
        default="cu_gruvae_realnvp_training_validation_plots",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.history_csv)

    if "epoch" not in df.columns:
        raise ValueError("History CSV must contain an 'epoch' column.")

    df["epoch"] = pd.to_numeric(
        df["epoch"], errors="coerce"
    )
    df = (
        df.dropna(subset=["epoch"])
        .sort_values("epoch")
        .reset_index(drop=True)
    )
    df["epoch"] = df["epoch"].astype(int)

    best_nll_ep, _ = best_epoch(
        df, "val_flow_nll_per_dim", "min"
    )
    best_obj_ep, _ = best_epoch(
        df, "val_objective_mse", "min"
    )

    # 1. Flow likelihood convergence.
    save_plot(
        df,
        [
            "train_flow_nll_per_dim",
            "val_flow_nll_per_dim",
        ],
        [
            "Training flow NLL / dim",
            "Validation flow NLL / dim",
        ],
        ylabel="Negative log-likelihood per latent dimension",
        title="RealNVP training vs validation likelihood",
        out_path=out_dir / "01_flow_nll_per_dim_train_vs_val.png",
        selected_epoch=best_nll_ep,
    )

    # 2. Objective-head convergence.
    save_plot(
        df,
        [
            "train_objective_mse",
            "val_objective_mse",
        ],
        [
            "Training multi-objective MSE",
            "Validation multi-objective MSE",
        ],
        ylabel="Multi-objective MSE",
        title="Cu objective-head training vs validation loss",
        out_path=out_dir / "02_objective_mse_train_vs_val.png",
        selected_epoch=best_obj_ep,
    )

    # 3-5. Objective prediction quality.
    save_objective_mse(
        df,
        out_dir / "03_validation_objective_mse_by_target.png",
    )
    save_objective_pearson(
        df,
        out_dir / "04_validation_objective_pearson_by_target.png",
    )
    save_plot(
        df,
        ["val_objective_mean_pearson"],
        ["Mean objective Pearson"],
        ylabel="Mean Pearson correlation",
        title="Mean validation objective correlation",
        out_path=out_dir / "05_validation_mean_objective_pearson.png",
        ylim=(-0.15, 1.0),
    )

    # 6. zK mean and coordinatewise variance.
    save_plot(
        df,
        ["val_zK_global_abs_mean"],
        ["Mean absolute coordinate mean"],
        ylabel="Mean |coordinate mean|",
        title="RealNVP zK centering",
        out_path=out_dir / "06_zK_centering.png",
        target_lines=[(0.0, "Target mean = 0")],
    )

    save_plot(
        df,
        [
            "val_zK_dim_std_mean",
            "val_zK_dim_std_min",
            "val_zK_dim_std_max",
        ],
        [
            "Mean coordinate std",
            "Minimum coordinate std",
            "Maximum coordinate std",
        ],
        ylabel="Coordinate standard deviation",
        title="RealNVP zK marginal scale",
        out_path=out_dir / "07_zK_coordinate_std.png",
        target_lines=[(1.0, "Standard-normal target = 1")],
    )

    # 8. Norm of zK. For 64-D N(0,I), sqrt(64)=8 is a useful reference.
    save_plot(
        df,
        ["val_zK_norm_mean"],
        ["Mean zK norm"],
        ylabel="Mean zK norm",
        title="RealNVP base-space norm",
        out_path=out_dir / "08_zK_norm.png",
        target_lines=[(8.0, "sqrt(64) = 8")],
    )

    # 9. Flow Jacobian / transformation strength.
    save_plot(
        df,
        [
            "val_flow_logdet_mean",
            "val_flow_logdet_std",
        ],
        [
            "Mean log|det J|",
            "Std log|det J|",
        ],
        ylabel="Log-determinant",
        title="RealNVP Jacobian statistics",
        out_path=out_dir / "09_flow_logdet.png",
    )

    # 10-11. Exact invertibility.
    save_plot(
        df,
        [
            "val_flow_roundtrip_h_l2_mean",
            "val_flow_roundtrip_h_l2_max",
        ],
        [
            "Mean flow roundtrip h0-L2",
            "Maximum flow roundtrip h0-L2",
        ],
        ylabel="Roundtrip L2 error",
        title="RealNVP forward-inverse numerical roundtrip",
        out_path=out_dir / "10_flow_roundtrip_l2.png",
        logy=True,
    )

    save_plot(
        df,
        ["val_flow_roundtrip_h_cosine_mean"],
        ["Mean roundtrip cosine"],
        ylabel="Cosine similarity",
        title="RealNVP forward-inverse cosine similarity",
        out_path=out_dir / "11_flow_roundtrip_cosine.png",
        target_lines=[(1.0, "Ideal cosine = 1")],
        ylim=(0.999, 1.00005),
    )

    save_plot(
        df,
        ["val_flow_logdet_cancel_abs_mean"],
        ["Mean |forward logdet + inverse logdet|"],
        ylabel="Absolute logdet cancellation error",
        title="RealNVP Jacobian inverse-consistency",
        out_path=out_dir / "12_flow_logdet_cancellation.png",
        logy=True,
    )

    # 13-14. Sequence reconstruction / uniqueness.
    save_plot(
        df,
        [
            "val_source_to_decode_edit_mean",
            "val_source_to_decode_edit_median",
        ],
        [
            "Mean source-to-decoded edit",
            "Median source-to-decoded edit",
        ],
        ylabel="Sequence edit distance",
        title="RealNVP roundtrip sequence reconstruction",
        out_path=out_dir / "13_roundtrip_sequence_edit.png",
    )

    save_plot(
        df,
        ["val_decode_unique_fraction"],
        ["Decoded unique fraction"],
        ylabel="Unique fraction",
        title="RealNVP decoded-sequence uniqueness",
        out_path=out_dir / "14_decode_unique_fraction.png",
        target_lines=[(1.0, "Ideal uniqueness = 1")],
        ylim=(0.0, 1.02),
    )

    # 15-17. Local BO-space geometry.
    save_plot(
        df,
        [
            "val_local_zK_l2_mean",
            "val_local_h0_l2_mean",
        ],
        [
            "Local zK perturbation L2",
            "Resulting inverse-flow h0 L2",
        ],
        ylabel="Mean local L2 distance",
        title="Local RealNVP zK geometry",
        out_path=out_dir / "15_local_zK_geometry_l2.png",
    )

    save_plot(
        df,
        [
            "val_local_sequence_edit_mean",
            "val_local_sequence_edit_median",
        ],
        [
            "Mean local sequence edit",
            "Median local sequence edit",
        ],
        ylabel="Sequence edit distance",
        title="Local zK perturbation sequence change",
        out_path=out_dir / "16_local_sequence_edit.png",
    )

    save_plot(
        df,
        ["val_local_identical_fraction"],
        ["Identical decoding fraction"],
        ylabel="Identical fraction",
        title="Local zK perturbation decode stability",
        out_path=out_dir / "17_local_identical_fraction.png",
        target_lines=[(1.0, "Ideal identical fraction = 1")],
        ylim=(0.0, 1.02),
    )

    # 18-19. Late-stage focused views.
    late_start = max(
        int(df["epoch"].min()),
        int(df["epoch"].max()) - 49,
    )
    late = df[df["epoch"] >= late_start].copy()

    save_plot(
        late,
        [
            "train_flow_nll_per_dim",
            "val_flow_nll_per_dim",
        ],
        [
            "Training flow NLL / dim",
            "Validation flow NLL / dim",
        ],
        ylabel="NLL per latent dimension",
        title=f"Late-stage RealNVP likelihood (epochs {late_start}-{int(df['epoch'].max())})",
        out_path=out_dir / "18_late_flow_nll_train_vs_val.png",
        selected_epoch=best_nll_ep,
    )

    save_plot(
        late,
        [
            "train_objective_mse",
            "val_objective_mse",
        ],
        [
            "Training objective MSE",
            "Validation objective MSE",
        ],
        ylabel="Multi-objective MSE",
        title=f"Late-stage objective loss (epochs {late_start}-{int(df['epoch'].max())})",
        out_path=out_dir / "19_late_objective_train_vs_val.png",
        selected_epoch=best_obj_ep,
    )

    write_summary(
        df,
        out_dir / "training_validation_metric_summary.csv",
    )
    write_report(
        df,
        out_dir / "training_validation_report.txt",
    )

    print(f"Rows loaded: {len(df)}")
    print(f"Best validation flow NLL/dim epoch: {best_nll_ep}")
    print(f"Best validation objective MSE epoch: {best_obj_ep}")
    print(f"Saved plots and summaries to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
