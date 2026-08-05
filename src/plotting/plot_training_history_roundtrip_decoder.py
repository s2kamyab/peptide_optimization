#!/usr/bin/env python
"""
Plot GRU-VAE fine-tuning history.

Creates separate publication-style PNG figures for:
  1. decoder confidence, train/validation, if columns exist
  2. round-trip L2, train/validation
  3. round-trip cosine, train/validation
  4. reconstruction loss, train/validation
  5. token accuracy, train/validation

Default visual style:
  - font size 24
  - line width 2.4
  - individual figures, no subplots
  - no explicit colors, so matplotlib defaults are used

Example:
    python plot_training_history_roundtrip_decoder.py \
        --csv training_history_roundtrip.csv \
        --out-dir training_history_plots_roundtrip
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_CONFIDENCE_COLUMN_PAIRS: Sequence[Tuple[str, str]] = (
    ("train_decoder_confidence", "val_decoder_confidence"),
    ("train_decoder_confidence_mean", "val_decoder_confidence_mean"),
    ("train_decoder_conf_mean", "val_decoder_conf_mean"),
    ("train_confidence", "val_confidence"),
    ("train_decoder_conf", "val_decoder_conf"),
)


def find_column_pair(
    df: pd.DataFrame,
    candidate_pairs: Sequence[Tuple[str, str]],
) -> Optional[Tuple[str, str]]:
    """Return the first train/validation column pair present in the dataframe."""
    columns = set(df.columns)
    for train_col, val_col in candidate_pairs:
        if train_col in columns and val_col in columns:
            return train_col, val_col
    return None


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required column(s): "
            + ", ".join(missing)
            + "\nAvailable columns are:\n"
            + "\n".join(df.columns)
        )


def setup_matplotlib(font_size: int) -> None:
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "figure.titlesize": font_size,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_train_val(
    df: pd.DataFrame,
    x_col: str,
    train_col: str,
    val_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
    linewidth: float,
    dpi: int,
    font_size: int,
) -> None:
    """Create one train-vs-validation line plot."""
    require_columns(df, [x_col, train_col, val_col])

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(
        df[x_col],
        df[train_col],
        label="Train",
        linewidth=linewidth,
    )
    ax.plot(
        df[x_col],
        df[val_col],
        label="Validation",
        linewidth=linewidth,
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linewidth=0.8, alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="transfer_gru_vae_flow_checkpoints_h32_z32_latent_conditioned_autoreg_rt001/training_history_latent_conditioned_autoreg_rt.csv",
        help="Path to training_history_latent_conditioned_autoreg_rt.csv.",
    )
    parser.add_argument(
        "--out-dir",
        default="transfer_gru_vae_flow_checkpoints_h32_z32_latent_conditioned_autoreg_rt001/history_plots",
        help="Directory where PNG figures will be saved.",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=24,
        help="Base matplotlib font size.",
    )
    parser.add_argument(
        "--linewidth",
        type=float,
        default=2.4,
        help="Line width for all curves.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output image resolution.",
    )
    parser.add_argument(
        "--x-col",
        default="epoch",
        help="Column used for the x-axis.",
    )
    parser.add_argument(
        "--decoder-confidence-train-col",
        default=None,
        help="Optional explicit training decoder-confidence column name.",
    )
    parser.add_argument(
        "--decoder-confidence-val-col",
        default=None,
        help="Optional explicit validation decoder-confidence column name.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    require_columns(df, [args.x_col])
    setup_matplotlib(args.font_size)

    # 1. Decoder confidence, if available.
    if args.decoder_confidence_train_col and args.decoder_confidence_val_col:
        confidence_pair = (
            args.decoder_confidence_train_col,
            args.decoder_confidence_val_col,
        )
        require_columns(df, confidence_pair)
    else:
        confidence_pair = find_column_pair(df, DEFAULT_CONFIDENCE_COLUMN_PAIRS)

    if confidence_pair is not None:
        plot_train_val(
            df=df,
            x_col=args.x_col,
            train_col=confidence_pair[0],
            val_col=confidence_pair[1],
            ylabel="Decoder confidence",
            title="Decoder Confidence During Fine-Tuning",
            out_path=out_dir / "decoder_confidence_history.png",
            linewidth=args.linewidth,
            dpi=args.dpi,
            font_size=args.font_size,
        )
        print(f"Saved decoder confidence plot using columns: {confidence_pair}")
    else:
        print(
            "WARNING: No decoder-confidence columns were found, so "
            "decoder_confidence_history.png was not created.\n"
            "Expected one of these train/validation column pairs:\n"
            + "\n".join(f"  {a}, {b}" for a, b in DEFAULT_CONFIDENCE_COLUMN_PAIRS)
            + "\nYou can also pass explicit names with:\n"
            "  --decoder-confidence-train-col TRAIN_COL "
            "--decoder-confidence-val-col VAL_COL"
        )

    # 2. Round-trip L2.
    plot_train_val(
        df=df,
        x_col=args.x_col,
        train_col="train_roundtrip_l2",
        val_col="val_roundtrip_l2",
        ylabel="Round-trip L2",
        title="Latent Round-Trip L2 During Fine-Tuning",
        out_path=out_dir / "roundtrip_l2_history.png",
        linewidth=args.linewidth,
        dpi=args.dpi,
        font_size=args.font_size,
    )

    # 3. Round-trip cosine.
    plot_train_val(
        df=df,
        x_col=args.x_col,
        train_col="train_roundtrip_cosine",
        val_col="val_roundtrip_cosine",
        ylabel="Round-trip cosine",
        title="Latent Round-Trip Cosine During Fine-Tuning",
        out_path=out_dir / "roundtrip_cosine_history.png",
        linewidth=args.linewidth,
        dpi=args.dpi,
        font_size=args.font_size,
    )

    # 4. Reconstruction loss.
    plot_train_val(
        df=df,
        x_col=args.x_col,
        train_col="train_recon",
        val_col="val_recon",
        ylabel="Reconstruction loss",
        title="Reconstruction Loss During Fine-Tuning",
        out_path=out_dir / "reconstruction_loss_history.png",
        linewidth=args.linewidth,
        dpi=args.dpi,
        font_size=args.font_size,
    )

    # 5. Token accuracy.
    plot_train_val(
        df=df,
        x_col=args.x_col,
        train_col="train_token_acc",
        val_col="val_token_acc",
        ylabel="Token accuracy",
        title="Token Accuracy During Fine-Tuning",
        out_path=out_dir / "token_accuracy_history.png",
        linewidth=args.linewidth,
        dpi=args.dpi,
        font_size=args.font_size,
    )

    print(f"Saved available plots to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
