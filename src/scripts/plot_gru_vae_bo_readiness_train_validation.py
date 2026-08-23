from __future__ import annotations

"""
Plot BO-relevant GRU-VAE pretraining behavior from pretraining_history_bo_ready.csv.

This version plots TRAINING AND VALIDATION histories together when the CSV
contains train_* and val_* columns. It remains backward-compatible with older
CSV histories that only contain the legacy unprefixed training columns.

Default checkpoint-selection marker:
  1. minimum val_loss, if val_loss exists and contains finite values;
  2. otherwise minimum train_loss;
  3. otherwise minimum legacy loss.

Important:
These plots assess reconstruction quality, generalization, latent health and
training stability. They do NOT by themselves prove BO smoothness.
Objective-locality and GP/qEHVI validation are still required downstream.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FONT_SIZE = 16
LINE_WIDTH = 2.0

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "figure.titlesize": FONT_SIZE,
    "lines.linewidth": LINE_WIDTH,
})


def finite_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def has_finite(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and finite_series(df, col).notna().any()


def first_available(df: pd.DataFrame, candidates):
    for col in candidates:
        if has_finite(df, col):
            return col
    return None


def paired_metric_columns(df: pd.DataFrame, base: str):
    """
    Return available train/validation columns and display labels.

    Preferred:
      train_<base>, val_<base>
    Fallback:
      legacy <base> as training only.
    """
    cols = []
    labels = []

    train_col = f"train_{base}"
    val_col = f"val_{base}"

    if has_finite(df, train_col):
        cols.append(train_col)
        labels.append("Training")
    elif has_finite(df, base):
        cols.append(base)
        labels.append("Training")

    if has_finite(df, val_col):
        cols.append(val_col)
        labels.append("Validation")

    return cols, labels


def save_line_plot(
    df: pd.DataFrame,
    ycols,
    labels,
    ylabel: str,
    title: str,
    out_path: Path,
    target_lines=None,
    ylim=None,
    logy=False,
    best_epoch=None,
):
    valid = [
        (col, label)
        for col, label in zip(ycols, labels)
        if has_finite(df, col)
    ]
    if not valid:
        print(f"Skipping {out_path.name}: no finite requested columns.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for col, label in valid:
        ax.plot(
            df["epoch"],
            finite_series(df, col),
            label=label,
            linewidth=LINE_WIDTH,
        )

    if target_lines:
        for value, label in target_lines:
            ax.axhline(
                value,
                linestyle="--",
                linewidth=LINE_WIDTH,
                label=label,
            )

    if best_epoch is not None:
        ax.axvline(
            best_epoch,
            linestyle=":",
            linewidth=LINE_WIDTH,
            label=f"Selected epoch {best_epoch}",
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


def save_train_val_pair(
    df: pd.DataFrame,
    base: str,
    ylabel: str,
    title: str,
    out_path: Path,
    best_epoch=None,
    target_lines=None,
    ylim=None,
):
    cols, generic_labels = paired_metric_columns(df, base)
    labels = []
    pretty = {
        "loss": "total loss",
        "teacher_recon": "teacher-forced CE",
        "free_recon": "free-running CE",
        "teacher_token_acc": "teacher-forced accuracy",
        "free_token_acc": "free-running accuracy",
        "mu_std": "μ std",
        "kl_per_dim_mean": "KL / latent dim",
        "latent_spread_loss": "latent spread loss",
        "mu_mean_loss": "μ mean loss",
        "min_info_loss": "minimum-information loss",
        "logvar_floor_loss": "logvar-floor loss",
        "free_decode_edit_mean": "free-decode edit distance",
        "free_decode_unique_fraction": "decoded unique fraction",
        "free_decode_dominant_fraction": "dominant decoded fraction",
    }.get(base, base)

    for lab in generic_labels:
        labels.append(f"{lab} {pretty}")

    save_line_plot(
        df,
        cols,
        labels,
        ylabel=ylabel,
        title=title,
        out_path=out_path,
        target_lines=target_lines,
        ylim=ylim,
        best_epoch=best_epoch,
    )


def rolling_stability_plot(
    df: pd.DataFrame,
    out_path: Path,
    window: int = 10,
):
    epoch = df["epoch"]

    train_loss_col = first_available(df, ["train_loss", "loss"])
    val_loss_col = first_available(df, ["val_loss"])
    train_free_col = first_available(df, ["train_free_recon", "free_recon"])
    val_free_col = first_available(df, ["val_free_recon"])
    train_mu_col = first_available(df, ["train_mu_std", "mu_std"])
    val_mu_col = first_available(df, ["val_mu_std"])

    series = []

    def robust_norm(s):
        roll = s.rolling(window, min_periods=1).mean()
        med = np.nanmedian(np.abs(roll))
        if not np.isfinite(med) or med < 1e-12:
            med = 1.0
        return roll / med

    if train_loss_col:
        series.append((robust_norm(finite_series(df, train_loss_col)),
                       f"Train loss ({window}-epoch mean)"))
    if val_loss_col:
        series.append((robust_norm(finite_series(df, val_loss_col)),
                       f"Validation loss ({window}-epoch mean)"))
    if train_free_col:
        series.append((robust_norm(finite_series(df, train_free_col)),
                       f"Train free CE ({window}-epoch mean)"))
    if val_free_col:
        series.append((robust_norm(finite_series(df, val_free_col)),
                       f"Validation free CE ({window}-epoch mean)"))
    if train_mu_col:
        series.append((robust_norm(finite_series(df, train_mu_col)),
                       f"Train μ std ({window}-epoch mean)"))
    if val_mu_col:
        series.append((robust_norm(finite_series(df, val_mu_col)),
                       f"Validation μ std ({window}-epoch mean)"))

    if not series:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for y, label in series:
        ax.plot(epoch, y, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized rolling value")
    ax.set_title("Train/validation late-training stability")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def best_epoch_for_metric(df: pd.DataFrame, col: str, direction: str):
    s = finite_series(df, col)
    valid = s.dropna()
    if valid.empty:
        return None, None
    idx = valid.idxmin() if direction == "min" else valid.idxmax()
    return int(df.loc[idx, "epoch"]), float(s.loc[idx])


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce separate training and validation summary rows when available.
    """
    rows = []
    metrics = [
        ("loss", "min"),
        ("teacher_recon", "min"),
        ("free_recon", "min"),
        ("teacher_token_acc", "max"),
        ("free_token_acc", "max"),
        ("kl_per_dim_mean", "max"),
        ("latent_spread_loss", "min"),
        ("mu_mean_loss", "min"),
        ("free_decode_edit_mean", "min"),
    ]

    for base, direction in metrics:
        train_col = first_available(df, [f"train_{base}", base])
        val_col = first_available(df, [f"val_{base}"])

        for split, col in [("train", train_col), ("validation", val_col)]:
            if not col:
                continue
            epoch, value = best_epoch_for_metric(df, col, direction)
            if epoch is None:
                continue
            rows.append({
                "split": split,
                "metric": base,
                "source_column": col,
                "best_direction": direction,
                "best_epoch": epoch,
                "best_value": value,
                "final_value": float(finite_series(df, col).iloc[-1]),
            })

    return pd.DataFrame(rows)


def choose_selected_epoch(df: pd.DataFrame, requested_epoch=None):
    if requested_epoch is not None:
        return int(requested_epoch), "user-specified"

    candidates = [
        ("val_loss", "minimum validation loss"),
        ("train_loss", "minimum training loss"),
        ("loss", "minimum legacy/training loss"),
    ]
    for col, reason in candidates:
        if has_finite(df, col):
            s = finite_series(df, col)
            idx = s.idxmin()
            return int(df.loc[idx, "epoch"]), reason

    raise ValueError("Could not select an epoch: no usable loss column found.")


def write_text_report(
    df: pd.DataFrame,
    out_path: Path,
    target_mu_std: float,
    selected_epoch: int,
    selection_reason: str,
):
    lines = []
    lines.append("GRU-VAE TRAINING + VALIDATION BEHAVIOR SUMMARY")
    lines.append("=" * 78)
    lines.append(f"epochs={len(df)}")
    lines.append(f"selected_epoch={selected_epoch}")
    lines.append(f"selection_rule={selection_reason}")
    lines.append("")

    bases = [
        "loss",
        "teacher_recon",
        "free_recon",
        "teacher_token_acc",
        "free_token_acc",
        "mu_std",
        "kl_per_dim_mean",
        "latent_spread_loss",
        "mu_mean_loss",
    ]

    lines.append("BEST AND FINAL VALUES")
    lines.append("-" * 78)

    directions = {
        "loss": "min",
        "teacher_recon": "min",
        "free_recon": "min",
        "teacher_token_acc": "max",
        "free_token_acc": "max",
        "mu_std": "target",
        "kl_per_dim_mean": "max",
        "latent_spread_loss": "min",
        "mu_mean_loss": "min",
    }

    for base in bases:
        train_col = first_available(df, [f"train_{base}", base])
        val_col = first_available(df, [f"val_{base}"])

        for split, col in [("train", train_col), ("validation", val_col)]:
            if not col:
                continue
            s = finite_series(df, col)
            valid = s.dropna()
            if valid.empty:
                continue

            if directions[base] == "min":
                idx = valid.idxmin()
            elif directions[base] == "max":
                idx = valid.idxmax()
            else:
                idx = (valid - target_mu_std).abs().idxmin()

            lines.append(
                f"{split}.{base}: best_epoch={int(df.loc[idx, 'epoch'])}, "
                f"best/target-nearest={float(s.loc[idx]):.6f}, "
                f"final={float(s.iloc[-1]):.6f}"
            )

    lines.append("")
    lines.append("SELECTED-EPOCH TRAIN / VALIDATION VALUES")
    lines.append("-" * 78)
    selected_rows = df[df["epoch"] == selected_epoch]
    if not selected_rows.empty:
        row = selected_rows.iloc[0]
        for base in bases:
            train_col = first_available(df, [f"train_{base}", base])
            val_col = first_available(df, [f"val_{base}"])
            values = []
            if train_col:
                values.append(f"train={pd.to_numeric(pd.Series([row[train_col]]), errors='coerce').iloc[0]:.6f}")
            if val_col:
                values.append(f"val={pd.to_numeric(pd.Series([row[val_col]]), errors='coerce').iloc[0]:.6f}")
            if values:
                lines.append(f"{base}: " + ", ".join(values))

    last_n = min(20, len(df))
    late = df.tail(last_n)
    lines.append("")
    lines.append(f"LATE-{last_n}-EPOCH GENERALIZATION / STABILITY")
    lines.append("-" * 78)
    for base in ["loss", "teacher_recon", "free_recon", "teacher_token_acc",
                 "free_token_acc", "mu_std", "kl_per_dim_mean"]:
        train_col = first_available(df, [f"train_{base}", base])
        val_col = first_available(df, [f"val_{base}"])
        for split, col in [("train", train_col), ("validation", val_col)]:
            if not col:
                continue
            s = pd.to_numeric(late[col], errors="coerce")
            lines.append(
                f"{split}.{base}: mean={s.mean():.6f}, std={s.std(ddof=0):.6f}, "
                f"min={s.min():.6f}, max={s.max():.6f}"
            )

    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 78)
    lines.append(
        "Training and validation curves assess reconstruction quality, generalization, "
        "latent activity, latent scaling and late-training stability."
    )
    lines.append(
        "The default selected epoch is based on minimum validation loss when validation "
        "loss is available, rather than minimum training loss."
    )
    lines.append(
        "These diagnostics do not establish BO smoothness. After Cu-specific fine-tuning, "
        "run objective-locality and GP/qEHVI validation."
    )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(
        description="Plot GRU-VAE training and validation BO-readiness diagnostics."
    )
    p.add_argument(
        "--history-csv",
        required=True,
        help="Path to pretraining_history_bo_ready.csv",
    )
    p.add_argument(
        "--out-dir",
        default="gru_vae_training_validation_plots",
    )
    p.add_argument(
        "--target-mu-std",
        type=float,
        default=1.0,
    )
    p.add_argument(
        "--selected-epoch",
        type=int,
        default=None,
        help=(
            "Optional epoch to mark with a vertical line. If omitted, "
            "minimum val_loss is used when available."
        ),
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

    selected_epoch, selection_reason = choose_selected_epoch(
        df, requested_epoch=args.selected_epoch
    )

    # 1. Total loss: train vs validation
    save_train_val_pair(
        df, "loss",
        ylabel="Total loss",
        title="GRU-VAE total loss: training vs validation",
        out_path=out_dir / "01_total_loss_train_vs_val.png",
        best_epoch=selected_epoch,
    )

    # 2. Teacher-forced reconstruction CE: train vs validation
    save_train_val_pair(
        df, "teacher_recon",
        ylabel="Cross-entropy",
        title="Teacher-forced reconstruction: training vs validation",
        out_path=out_dir / "02_teacher_recon_train_vs_val.png",
        best_epoch=selected_epoch,
    )

    # 3. Free-running reconstruction CE: train vs validation
    save_train_val_pair(
        df, "free_recon",
        ylabel="Cross-entropy",
        title="Free-running reconstruction: training vs validation",
        out_path=out_dir / "03_free_recon_train_vs_val.png",
        best_epoch=selected_epoch,
    )

    # 4. Teacher-forced token accuracy: train vs validation
    save_train_val_pair(
        df, "teacher_token_acc",
        ylabel="Token accuracy",
        title="Teacher-forced token accuracy: training vs validation",
        out_path=out_dir / "04_teacher_accuracy_train_vs_val.png",
        best_epoch=selected_epoch,
        target_lines=[(1.0, "Perfect accuracy")],
        ylim=(0.0, 1.02),
    )

    # 5. Free-running token accuracy: train vs validation
    save_train_val_pair(
        df, "free_token_acc",
        ylabel="Token accuracy",
        title="Free-running token accuracy: training vs validation",
        out_path=out_dir / "05_free_accuracy_train_vs_val.png",
        best_epoch=selected_epoch,
        target_lines=[(1.0, "Perfect accuracy")],
        ylim=(0.0, 1.02),
    )

    # 6. μ latent standard deviation: train vs validation
    save_train_val_pair(
        df, "mu_std",
        ylabel="μ standard deviation",
        title="Latent μ scale: training vs validation",
        out_path=out_dir / "06_mu_std_train_vs_val.png",
        best_epoch=selected_epoch,
        target_lines=[(args.target_mu_std, f"Target μ std = {args.target_mu_std:g}")],
    )

    # 7. KL per latent dimension: train vs validation
    save_train_val_pair(
        df, "kl_per_dim_mean",
        ylabel="KL per latent dimension",
        title="Latent activity: training vs validation",
        out_path=out_dir / "07_kl_per_dim_train_vs_val.png",
        best_epoch=selected_epoch,
    )

    # 8. Latent spread loss
    save_train_val_pair(
        df, "latent_spread_loss",
        ylabel="Latent spread loss",
        title="Latent spread regularizer: training vs validation",
        out_path=out_dir / "08_latent_spread_train_vs_val.png",
        best_epoch=selected_epoch,
    )

    # 9. μ centering loss
    save_train_val_pair(
        df, "mu_mean_loss",
        ylabel="μ mean loss",
        title="Latent centering: training vs validation",
        out_path=out_dir / "09_mu_mean_loss_train_vs_val.png",
        best_epoch=selected_epoch,
    )

    # 10. Free-decode edit distance, if present
    save_train_val_pair(
        df, "free_decode_edit_mean",
        ylabel="Mean edit distance",
        title="Free-decode edit distance: training vs validation",
        out_path=out_dir / "10_free_decode_edit_train_vs_val.png",
        best_epoch=selected_epoch,
    )

    # 11. Unique fraction, if present
    save_train_val_pair(
        df, "free_decode_unique_fraction",
        ylabel="Unique fraction",
        title="Decoded-sequence uniqueness: training vs validation",
        out_path=out_dir / "11_unique_fraction_train_vs_val.png",
        best_epoch=selected_epoch,
        ylim=(0.0, 1.02),
    )

    # 12. Rolling train/validation stability
    rolling_stability_plot(
        df,
        out_dir / "12_train_val_rolling_stability.png",
        window=10,
    )

    # 13. Focused late-epoch train/validation total loss
    late_start = max(int(df["epoch"].min()), int(df["epoch"].max()) - 60)
    late_df = df[df["epoch"] >= late_start].copy()
    save_train_val_pair(
        late_df, "loss",
        ylabel="Total loss",
        title=f"Late-epoch total loss: training vs validation "
              f"(epochs {late_start}–{int(df['epoch'].max())})",
        out_path=out_dir / "13_late_total_loss_train_vs_val.png",
        best_epoch=selected_epoch,
    )

    # 14. Focused late-epoch free-running CE
    save_train_val_pair(
        late_df, "free_recon",
        ylabel="Free-running cross-entropy",
        title=f"Late-epoch free-running reconstruction: training vs validation "
              f"(epochs {late_start}–{int(df['epoch'].max())})",
        out_path=out_dir / "14_late_free_recon_train_vs_val.png",
        best_epoch=selected_epoch,
    )

    summary = compute_summary(df)
    summary.to_csv(
        out_dir / "gru_vae_train_validation_summary.csv",
        index=False,
    )

    write_text_report(
        df,
        out_dir / "gru_vae_train_validation_report.txt",
        target_mu_std=args.target_mu_std,
        selected_epoch=selected_epoch,
        selection_reason=selection_reason,
    )

    print(f"Selected epoch marked on plots: {selected_epoch}")
    print(f"Selection rule: {selection_reason}")
    print(f"Validation columns detected: {any(c.startswith('val_') for c in df.columns)}")
    print(f"Saved figures and summaries to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
