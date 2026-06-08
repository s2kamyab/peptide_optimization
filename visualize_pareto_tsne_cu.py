import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from sklearn.manifold import TSNE
except ImportError as exc:
    raise SystemExit(
        "scikit-learn is required for t-SNE. Install it with: pip install scikit-learn"
    ) from exc


AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_I = {aa: i for i, aa in enumerate(AA)}
OBJECTIVE_COLS = [
    "chelation_sub",
    "solubility_sub",
    "stability_sub",
    "expression_sub",
]


def resolve_existing_path(path: str, fallback: str | None = None) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    if fallback is not None and Path(fallback).exists():
        return Path(fallback)
    raise FileNotFoundError(f"Could not find {path}")


def one_hot_peptides(peptides: pd.Series) -> np.ndarray:
    peptides = peptides.astype(str).str.strip().str.upper()
    if peptides.empty:
        raise ValueError("No peptides found to encode.")

    seq_len = peptides.str.len().max()
    features = np.zeros((len(peptides), seq_len, len(AA)), dtype=np.float32)

    for row_i, peptide in enumerate(peptides):
        for pos_i, aa in enumerate(peptide):
            aa_i = AA_TO_I.get(aa)
            if aa_i is not None:
                features[row_i, pos_i, aa_i] = 1.0

    return features.reshape(len(peptides), -1)


def objective_features(df: pd.DataFrame) -> np.ndarray:
    missing = [col for col in OBJECTIVE_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing objective columns: {missing}")
    return df[OBJECTIVE_COLS].astype(float).to_numpy(dtype=np.float32)


def build_features(df: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "sequence":
        return one_hot_peptides(df["peptide"])
    if mode == "objectives":
        return objective_features(df)
    if mode == "combined":
        return np.hstack([one_hot_peptides(df["peptide"]), objective_features(df)])
    raise ValueError(f"Unknown feature mode: {mode}")


def load_points(train_path: Path, bo_path: Path, dominates_path: Path) -> pd.DataFrame:
    train = pd.read_csv(train_path)
    bo = pd.read_csv(bo_path)
    dominates = pd.read_csv(dominates_path)

    for name, df in [("train", train), ("bo", bo), ("dominates", dominates)]:
        if "peptide" not in df.columns:
            raise ValueError(f"{name} CSV must contain a 'peptide' column.")

    dominates_peptides = set(dominates["peptide"].astype(str).str.strip().str.upper())

    train = train.copy()
    train["source"] = "train_pareto"

    bo = bo.copy()
    bo["source"] = "bo_final_pareto"

    combined = pd.concat([train, bo], ignore_index=True, sort=False)
    combined["peptide"] = combined["peptide"].astype(str).str.strip().str.upper()
    combined["dominates_train"] = combined["peptide"].isin(dominates_peptides)
    return combined


def run_tsne(features: np.ndarray, random_state: int, perplexity: float | None) -> np.ndarray:
    n = features.shape[0]
    if n < 2:
        raise ValueError("t-SNE needs at least 2 records.")

    if perplexity is None:
        perplexity = min(30, max(2, (n - 1) // 3))
    if perplexity >= n:
        perplexity = max(1, n - 1)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    return tsne.fit_transform(features)


def plot_embedding(df: pd.DataFrame, out_png: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))

    train_mask = (df["source"] == "train_pareto") & ~df["dominates_train"]
    bo_mask = (df["source"] == "bo_final_pareto") & ~df["dominates_train"]
    red_mask = df["dominates_train"]

    ax.scatter(
        df.loc[train_mask, "tsne_1"],
        df.loc[train_mask, "tsne_2"],
        s=55,
        c="#2f6fbb",
        marker="o",
        edgecolors="white",
        linewidths=0.5,
        alpha=0.85,
        label="Train Pareto",
    )
    ax.scatter(
        df.loc[bo_mask, "tsne_1"],
        df.loc[bo_mask, "tsne_2"],
        s=70,
        c="#f2a541",
        marker="^",
        edgecolors="white",
        linewidths=0.5,
        alpha=0.9,
        label="BO Final Pareto",
    )
    ax.scatter(
        df.loc[red_mask, "tsne_1"],
        df.loc[red_mask, "tsne_2"],
        s=90,
        c="#d62728",
        marker="*",
        edgecolors="black",
        linewidths=0.4,
        alpha=0.95,
        label="BO Pareto Dominates Train",
        zorder=3,
    )

    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize train and BO Pareto peptide sets with t-SNE."
    )
    parser.add_argument(
        "--train",
        default="bo_results/train_pareto_cu_updated.csv",
        help="Train Pareto CSV.",
    )
    parser.add_argument(
        "--bo",
        default="bo_results/bo_final_pareto_CU_update.csv",
        help="BO final Pareto CSV. Falls back to bo_final_pareto_CU_updated.csv if needed.",
    )
    parser.add_argument(
        "--dominates",
        default="bo_results/bo_pareto_dominates_train_CU_updated.csv",
        help="CSV containing records to highlight in red.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["sequence", "objectives", "combined"],
        default="sequence",
        help="Features used for t-SNE. Default uses one-hot peptide sequence features.",
    )
    parser.add_argument("--perplexity", type=float, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--out-png",
        default="bo_results/pareto_tsne_CU_updated.png",
        help="Output plot path.",
    )
    parser.add_argument(
        "--out-csv",
        default="bo_results/pareto_tsne_CU_updated_coordinates.csv",
        help="Output coordinates CSV path.",
    )
    args = parser.parse_args()

    train_path = resolve_existing_path(args.train)
    bo_path = resolve_existing_path(
        args.bo,
        fallback="bo_results/bo_final_pareto_CU_updated.csv",
    )
    dominates_path = resolve_existing_path(args.dominates)

    df = load_points(train_path, bo_path, dominates_path)
    features = build_features(df, args.feature_mode)
    embedding = run_tsne(features, args.random_state, args.perplexity)

    df["tsne_1"] = embedding[:, 0]
    df["tsne_2"] = embedding[:, 1]

    out_png = Path(args.out_png)
    out_csv = Path(args.out_csv)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    title = f"CU Pareto t-SNE ({args.feature_mode} features)"
    plot_embedding(df, out_png, title)
    df.to_csv(out_csv, index=False)

    print(f"Saved plot: {out_png}")
    print(f"Saved coordinates: {out_csv}")
    print(f"Records plotted: {len(df)}")
    print(f"Red highlighted records: {int(df['dominates_train'].sum())}")


if __name__ == "__main__":
    main()
