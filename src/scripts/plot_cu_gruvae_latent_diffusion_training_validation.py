from __future__ import annotations

"""
Plot training + validation history for the Cu GRU-VAE latent-diffusion fine-tuning run.

Expected history columns include:
    train_diffusion_epsilon_mse
    train_objective_mse
    val_diffusion_epsilon_mse
    val_objective_mse
    val_<objective>_mse
    val_<objective>_pearson
    val_objective_mean_pearson
    val_ddim_inversion_h_l2_mean
    val_ddim_inversion_h_l2_median
    val_ddim_inversion_h_cosine_mean
    val_ddim_source_to_decode_edit_mean
    val_ddim_decode_unique_fraction
    val_local_epsilon_l2_mean
    val_local_h_l2_mean
    val_local_sequence_edit_mean
    val_local_identical_fraction

The training script only logs TRAINING values for the two optimized losses:
    diffusion epsilon MSE
    multi-objective MSE

All other diagnostics are validation-only, so the plotting script does not invent
training counterparts for metrics that were never logged.
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
):
    valid = [
        (c, l)
        for c, l in zip(columns, labels)
        if has_finite(df, c)
    ]
    if not valid:
        print(f"Skipping {out_path.name}: no finite data.")
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))

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
            ax.axhline(y, linestyle="--", linewidth=1.5, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_individual_objective_mse(df: pd.DataFrame, out_path: Path):
    cols = [f"val_{o}_mse" for o in OBJECTIVES]
    labels = [
        "Chelation",
        "Solubility",
        "Stability",
        "Expression",
    ]
    save_plot(
        df,
        cols,
        labels,
        ylabel="Validation MSE",
        title="Validation MSE by Cu objective",
        out_path=out_path,
    )


def save_individual_objective_pearson(df: pd.DataFrame, out_path: Path):
    cols = [f"val_{o}_pearson" for o in OBJECTIVES]
    labels = [
        "Chelation",
        "Solubility",
        "Stability",
        "Expression",
    ]
    save_plot(
        df,
        cols,
        labels,
        ylabel="Pearson correlation",
        title="Validation Pearson correlation by Cu objective",
        out_path=out_path,
        ylim=(-0.1, 1.0),
    )


def save_inversion_metrics(df: pd.DataFrame, out_dir: Path):
    save_plot(
        df,
        [
            "val_ddim_inversion_h_l2_mean",
            "val_ddim_inversion_h_l2_median",
        ],
        [
            "Mean DDIM inversion h-L2",
            "Median DDIM inversion h-L2",
        ],
        ylabel="Latent L2 error",
        title="DDIM inversion reconstruction error",
        out_path=out_dir / "06_ddim_inversion_l2.png",
    )

    save_plot(
        df,
        ["val_ddim_inversion_h_cosine_mean"],
        ["Mean cosine similarity"],
        ylabel="Cosine similarity",
        title="DDIM inversion latent cosine similarity",
        out_path=out_dir / "07_ddim_inversion_cosine.png",
        ylim=(0.0, 1.02),
    )

    save_plot(
        df,
        [
            "val_ddim_source_to_decode_edit_mean",
            "val_ddim_source_to_decode_edit_median",
        ],
        [
            "Mean source-to-decoded edit",
            "Median source-to-decoded edit",
        ],
        ylabel="Sequence edit distance",
        title="DDIM roundtrip sequence reconstruction",
        out_path=out_dir / "08_ddim_roundtrip_sequence_edit.png",
    )

    save_plot(
        df,
        ["val_ddim_decode_unique_fraction"],
        ["Decoded unique fraction"],
        ylabel="Unique fraction",
        title="DDIM decoded-sequence uniqueness",
        out_path=out_dir / "09_ddim_unique_fraction.png",
        ylim=(0.0, 1.02),
    )


def save_local_geometry(df: pd.DataFrame, out_dir: Path):
    save_plot(
        df,
        [
            "val_local_epsilon_l2_mean",
            "val_local_h_l2_mean",
        ],
        [
            "Local epsilon L2",
            "Resulting h-space L2",
        ],
        ylabel="Mean local L2 distance",
        title="Local diffusion geometry",
        out_path=out_dir / "10_local_geometry_l2.png",
    )

    save_plot(
        df,
        ["val_local_sequence_edit_mean"],
        ["Local mean sequence edit"],
        ylabel="Mean sequence edit distance",
        title="Local epsilon perturbation sequence change",
        out_path=out_dir / "11_local_sequence_edit.png",
    )

    save_plot(
        df,
        ["val_local_identical_fraction"],
        ["Local identical-decoding fraction"],
        ylabel="Identical fraction",
        title="Local epsilon perturbation decode stability",
        out_path=out_dir / "12_local_identical_fraction.png",
        ylim=(0.0, 1.02),
    )


def write_summary(df: pd.DataFrame, out_path: Path):
    metrics = [
        ("train_diffusion_epsilon_mse", "min"),
        ("val_diffusion_epsilon_mse", "min"),
        ("train_objective_mse", "min"),
        ("val_objective_mse", "min"),
        ("val_objective_mean_pearson", "max"),
        ("val_ddim_inversion_h_l2_mean", "min"),
        ("val_ddim_inversion_h_cosine_mean", "max"),
        ("val_ddim_source_to_decode_edit_mean", "min"),
        ("val_local_sequence_edit_mean", "min"),
        ("val_local_identical_fraction", "max"),
    ]

    rows = []
    for col, mode in metrics:
        if not has_finite(df, col):
            continue
        ep, val = best_epoch(df, col, mode)
        rows.append({
            "metric": col,
            "direction": mode,
            "best_epoch": ep,
            "best_value": val,
            "final_value": float(series(df, col).iloc[-1]),
        })

    pd.DataFrame(rows).to_csv(out_path, index=False)


def write_report(df: pd.DataFrame, out_path: Path):
    best_val_diff_ep, best_val_diff = best_epoch(
        df, "val_diffusion_epsilon_mse", "min"
    )
    best_val_obj_ep, best_val_obj = best_epoch(
        df, "val_objective_mse", "min"
    )
    best_inv_ep, best_inv = best_epoch(
        df, "val_ddim_inversion_h_l2_mean", "min"
    )
    best_corr_ep, best_corr = best_epoch(
        df, "val_objective_mean_pearson", "max"
    )

    lines = []
    lines.append("CU GRU-VAE LATENT-DIFFUSION TRAINING / VALIDATION SUMMARY")
    lines.append("=" * 82)
    lines.append(f"epochs={len(df)}")
    lines.append("")
    lines.append("BEST VALIDATION CHECKPOINTS")
    lines.append("-" * 82)

    if best_val_diff_ep is not None:
        lines.append(
            f"Best val diffusion epsilon MSE: epoch={best_val_diff_ep}, "
            f"value={best_val_diff:.8f}"
        )
    if best_val_obj_ep is not None:
        lines.append(
            f"Best val multi-objective MSE: epoch={best_val_obj_ep}, "
            f"value={best_val_obj:.8f}"
        )
    if best_inv_ep is not None:
        lines.append(
            f"Best DDIM inversion h-L2: epoch={best_inv_ep}, "
            f"value={best_inv:.8f}"
        )
    if best_corr_ep is not None:
        lines.append(
            f"Best validation mean objective Pearson: epoch={best_corr_ep}, "
            f"value={best_corr:.8f}"
        )

    lines.append("")
    lines.append("FINAL EPOCH")
    lines.append("-" * 82)
    last = df.iloc[-1]
    for col in [
        "train_diffusion_epsilon_mse",
        "val_diffusion_epsilon_mse",
        "train_objective_mse",
        "val_objective_mse",
        "val_objective_mean_pearson",
        "val_ddim_inversion_h_l2_mean",
        "val_ddim_inversion_h_cosine_mean",
        "val_ddim_source_to_decode_edit_mean",
        "val_local_sequence_edit_mean",
        "val_local_identical_fraction",
    ]:
        if col in df.columns:
            v = pd.to_numeric(pd.Series([last[col]]), errors="coerce").iloc[0]
            if np.isfinite(v):
                lines.append(f"{col}={v:.8f}")

    lines.append("")
    lines.append("NOTE")
    lines.append("-" * 82)
    lines.append(
        "Only diffusion epsilon MSE and multi-objective MSE have both training "
        "and validation histories in this CSV. The remaining diagnostics were "
        "logged on validation data only, so no training curves are fabricated."
    )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(
        description="Plot Cu GRU-VAE latent-diffusion training + validation history."
    )
    p.add_argument("--history-csv", required=True)
    p.add_argument(
        "--out-dir",
        default="cu_gruvae_latent_diffusion_training_validation_plots",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.history_csv)
    if "epoch" not in df.columns:
        raise ValueError("History CSV must contain an 'epoch' column.")

    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df = df.dropna(subset=["epoch"]).copy()
    df["epoch"] = df["epoch"].astype(int)
    df = df.sort_values("epoch").reset_index(drop=True)

    best_diff_ep, _ = best_epoch(df, "val_diffusion_epsilon_mse", "min")
    best_obj_ep, _ = best_epoch(df, "val_objective_mse", "min")
    best_inv_ep, _ = best_epoch(df, "val_ddim_inversion_h_l2_mean", "min")

    # 1. Training vs validation diffusion loss
    save_plot(
        df,
        [
            "train_diffusion_epsilon_mse",
            "val_diffusion_epsilon_mse",
        ],
        [
            "Training diffusion epsilon MSE",
            "Validation diffusion epsilon MSE",
        ],
        ylabel="Epsilon prediction MSE",
        title="Latent diffusion training vs validation loss",
        out_path=out_dir / "01_diffusion_epsilon_mse_train_vs_val.png",
        selected_epoch=best_diff_ep,
    )

    # 2. Training vs validation multi-objective head loss
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

    # 3. Individual validation objective MSE
    save_individual_objective_mse(
        df,
        out_dir / "03_validation_objective_mse_by_target.png",
    )

    # 4. Individual validation objective Pearson
    save_individual_objective_pearson(
        df,
        out_dir / "04_validation_objective_pearson_by_target.png",
    )

    # 5. Mean validation objective Pearson
    save_plot(
        df,
        ["val_objective_mean_pearson"],
        ["Mean objective Pearson"],
        ylabel="Mean Pearson correlation",
        title="Mean validation objective correlation",
        out_path=out_dir / "05_validation_mean_objective_pearson.png",
        ylim=(-0.1, 1.0),
    )

    # 6–9. DDIM inversion / reconstruction metrics
    save_inversion_metrics(df, out_dir)

    # Mark selected inversion epoch on an additional focused figure
    save_plot(
        df,
        ["val_ddim_inversion_h_l2_mean"],
        ["Mean DDIM inversion h-L2"],
        ylabel="Latent L2 error",
        title="DDIM inversion checkpoint-selection metric",
        out_path=out_dir / "06b_best_ddim_inversion_metric.png",
        selected_epoch=best_inv_ep,
    )

    # 10–12. Local diffusion geometry
    save_local_geometry(df, out_dir)

    # 13. Epsilon norm
    save_plot(
        df,
        ["val_epsilon_norm_mean"],
        ["Mean epsilon norm"],
        ylabel="Epsilon norm",
        title="Validation diffusion epsilon norm",
        out_path=out_dir / "13_epsilon_norm.png",
        target_lines=[(8.0, "Radius = sqrt(64) = 8")],
    )

    # 14. Late-stage losses (last 50 epochs)
    late_start = max(int(df["epoch"].min()), int(df["epoch"].max()) - 49)
    late = df[df["epoch"] >= late_start].copy()

    save_plot(
        late,
        [
            "train_diffusion_epsilon_mse",
            "val_diffusion_epsilon_mse",
        ],
        [
            "Training diffusion MSE",
            "Validation diffusion MSE",
        ],
        ylabel="Epsilon prediction MSE",
        title=f"Late-stage diffusion loss (epochs {late_start}–{int(df['epoch'].max())})",
        out_path=out_dir / "14_late_diffusion_train_vs_val.png",
        selected_epoch=best_diff_ep,
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
        title=f"Late-stage objective-head loss (epochs {late_start}–{int(df['epoch'].max())})",
        out_path=out_dir / "15_late_objective_train_vs_val.png",
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

    print(f"Best val diffusion MSE epoch: {best_diff_ep}")
    print(f"Best val objective MSE epoch: {best_obj_ep}")
    print(f"Best DDIM inversion h-L2 epoch: {best_inv_ep}")
    print(f"Saved plots and summaries to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
