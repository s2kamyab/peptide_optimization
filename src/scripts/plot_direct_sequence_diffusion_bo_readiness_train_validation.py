from __future__ import annotations

"""
Plot BO-relevant training/validation behavior for direct peptide sequence diffusion.

Designed for pretraining_history_direct_sequence_diffusion.csv produced by
pretrain_peptide_direct_sequence_diffusion_bo_ready_h96.py.

Expected metrics:
  train_loss / validation_loss
  train_epsilon_mse / validation_epsilon_mse
  train_recon_ce / validation_recon_ce
  train_x0_mse / validation_x0_mse
  train_token_acc / validation_token_acc

The direct model has no VAE posterior, KL, mu-space, latent-spread loss, or
free-running autoregressive decoder metrics. Checkpoint selection follows the
pretrainer: minimum validation_loss.
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
      train_<base>, validation_<base>
    Fallback:
      legacy <base> as training only.
    """
    cols = []
    labels = []

    train_col = f"train_{base}"
    val_col = f"validation_{base}"

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
        "epsilon_mse": "epsilon MSE",
        "recon_ce": "reconstruction CE",
        "x0_mse": "x0 MSE",
        "token_acc": "token accuracy",
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


def rolling_stability_plot(df: pd.DataFrame, out_path: Path, window: int = 10):
    epoch = df["epoch"]
    candidates = [
        ("train_loss", "Train loss"),
        ("validation_loss", "Validation loss"),
        ("train_epsilon_mse", "Train epsilon MSE"),
        ("validation_epsilon_mse", "Validation epsilon MSE"),
        ("train_recon_ce", "Train reconstruction CE"),
        ("validation_recon_ce", "Validation reconstruction CE"),
    ]
    series = []
    def robust_norm(s):
        roll = s.rolling(window, min_periods=1).mean()
        med = np.nanmedian(np.abs(roll))
        if not np.isfinite(med) or med < 1e-12:
            med = 1.0
        return roll / med
    for col, label in candidates:
        if has_finite(df, col):
            series.append((robust_norm(finite_series(df, col)), label))
    if not series:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for y, label in series:
        ax.plot(epoch, y, label=f"{label} ({window}-epoch mean)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized rolling value")
    ax.set_title("Direct diffusion late-training stability")
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
    rows = []
    metrics = [
        ("loss", "min"),
        ("epsilon_mse", "min"),
        ("recon_ce", "min"),
        ("x0_mse", "min"),
        ("token_acc", "max"),
    ]
    for base, direction in metrics:
        for split, col in [
            ("train", f"train_{base}"),
            ("validation", f"validation_{base}"),
        ]:
            if not has_finite(df, col):
                continue
            epoch, value = best_epoch_for_metric(df, col, direction)
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
    for col, reason in [
        ("validation_loss", "minimum validation loss"),
        ("train_loss", "minimum training loss fallback"),
    ]:
        if has_finite(df, col):
            s = finite_series(df, col)
            idx = s.idxmin()
            return int(df.loc[idx, "epoch"]), reason
    raise ValueError("Could not select epoch: no validation_loss/train_loss found.")

def write_text_report(df: pd.DataFrame, out_path: Path, selected_epoch: int, selection_reason: str):
    lines = [
        "DIRECT SEQUENCE DIFFUSION TRAINING + VALIDATION SUMMARY",
        "=" * 78,
        f"epochs={len(df)}",
        f"selected_epoch={selected_epoch}",
        f"selection_rule={selection_reason}",
        "",
        "SELECTED-EPOCH VALUES",
        "-" * 78,
    ]
    row_df = df[df["epoch"] == selected_epoch]
    if not row_df.empty:
        row = row_df.iloc[0]
        for base in ["loss", "epsilon_mse", "recon_ce", "x0_mse", "token_acc"]:
            vals=[]
            for split, col in [("train", f"train_{base}"), ("validation", f"validation_{base}")]:
                if col in df.columns:
                    v = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
                    if np.isfinite(v): vals.append(f"{split}={v:.6f}")
            if vals: lines.append(f"{base}: " + ", ".join(vals))
    lines += ["", "BEST / FINAL VALUES", "-" * 78]
    sm = compute_summary(df)
    if len(sm): lines.append(sm.to_string(index=False))
    lines += ["", "INTERPRETATION", "-" * 78,
              "The direct diffusion model has no GRU-VAE posterior, KL loss, mu-space, latent-spread loss, or autoregressive decoder reconstruction metrics.",
              "The best checkpoint is selected by minimum validation total loss, matching the pretrainer.",
              "These plots diagnose denoising/reconstruction generalization; BO-coordinate smoothness must be audited in DDIM epsilon / PCA-whitened z_bo space."]
    out_path.write_text("\n".join(lines)+"\n", encoding="utf-8")

def main():
    p = argparse.ArgumentParser(description="Plot direct sequence diffusion training/validation diagnostics.")
    p.add_argument("--history-csv", required=True, help="pretraining_history_direct_sequence_diffusion.csv")
    p.add_argument("--out-dir", default="direct_sequence_diffusion_training_validation_plots")
    p.add_argument("--selected-epoch", type=int, default=None)
    args = p.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.history_csv)
    if "epoch" not in df.columns: raise ValueError("History CSV must contain 'epoch'.")
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df = df.dropna(subset=["epoch"]).copy(); df["epoch"] = df["epoch"].astype(int)
    df = df.sort_values("epoch").reset_index(drop=True)
    selected_epoch, reason = choose_selected_epoch(df, args.selected_epoch)

    specs = [
        ("loss", "Total loss", "Direct sequence diffusion total loss: training vs validation", "01_total_loss_train_vs_validation.png", None),
        ("epsilon_mse", "Epsilon MSE", "Noise-prediction error: training vs validation", "02_epsilon_mse_train_vs_validation.png", None),
        ("recon_ce", "Cross-entropy", "Direct peptide reconstruction CE: training vs validation", "03_reconstruction_ce_train_vs_validation.png", None),
        ("x0_mse", "x0 MSE", "Continuous x0 reconstruction MSE: training vs validation", "04_x0_mse_train_vs_validation.png", None),
        ("token_acc", "Token accuracy", "Direct peptide token accuracy: training vs validation", "05_token_accuracy_train_vs_validation.png", (0.0, 1.02)),
    ]
    for base,ylabel,title,name,ylim in specs:
        save_train_val_pair(df, base, ylabel, title, out_dir/name, best_epoch=selected_epoch, ylim=ylim,
                            target_lines=[(1.0,"Perfect accuracy")] if base=="token_acc" else None)
    rolling_stability_plot(df, out_dir/"06_rolling_stability.png", window=10)
    late_start=max(int(df["epoch"].min()), int(df["epoch"].max())-60)
    late=df[df["epoch"]>=late_start].copy()
    for base,ylabel,name in [("loss","Total loss","07_late_total_loss.png"),("epsilon_mse","Epsilon MSE","08_late_epsilon_mse.png"),("recon_ce","Cross-entropy","09_late_reconstruction_ce.png")]:
        save_train_val_pair(late, base, ylabel, f"Late-epoch {base}: training vs validation", out_dir/name, best_epoch=selected_epoch)
    compute_summary(df).to_csv(out_dir/"direct_diffusion_train_validation_summary.csv", index=False)
    write_text_report(df, out_dir/"direct_diffusion_train_validation_report.txt", selected_epoch, reason)
    print(f"Selected epoch: {selected_epoch} ({reason})")
    print(f"Saved figures and summaries to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
