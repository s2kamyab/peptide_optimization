from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

OPTIMIZED_CSV = "bo_results/bo_pareto_dominates_train_CU_gp_no_flow_wu_2.csv"
# "bo_results/bo_pareto_dominates_train_CU_gp_before_flow.csv"
# "bo_results/bo_pareto_dominates_train_CU_flow_space_gp.csv"#"bo_results/bo_pareto_dominates_train_no_flow_CU.csv"
TRAIN_PARETO_CSV = "bo_results/train_pareto_cu_flow_space_gp.csv"

OUTPUT_DIR = "scrambling_control_CU_gp_no_flow_wu_2"
N_SCRAMBLES_PER_PEPTIDE = 100
RANDOM_SEED = 42

# If True, avoid generating scrambled peptides that already exist in the
# optimized or training Pareto files.
REJECT_EXISTING_SEQUENCES = True

# Your objective function file must be in the same folder or importable.
# It should contain blackbox_fc.
from backup_code_result_folders.black_box_fcn_mo_CU_f_updated_batch_normalized import blackbox_fc


OBJECTIVE_COLS = [
    "chelation_sub",
    "solubility_sub",
    "stability_sub",
    "expression_sub",
]

FINAL_SCORE_COL = "final_score"


# ============================================================
# Helper functions
# ============================================================

def get_peptide_column(df: pd.DataFrame) -> str:
    """
    Detect peptide column name.
    """
    candidates = ["peptide", "peptide_len10", "sequence"]
    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        f"Could not find peptide column. Available columns: {list(df.columns)}"
    )


def scramble_sequence(seq: str, rng: random.Random) -> str:
    """
    Return one random permutation of the amino-acid sequence.
    """
    chars = list(seq)
    rng.shuffle(chars)
    return "".join(chars)


def generate_unique_scrambles(
    seq: str,
    n: int,
    rng: random.Random,
    forbidden: set[str] | None = None,
    max_attempts: int = 10000,
) -> List[str]:
    """
    Generate unique scrambled versions of a sequence.

    Preserves amino-acid composition exactly.
    Rejects:
      - the original sequence
      - duplicates
      - forbidden sequences, if provided
    """
    seq = seq.strip().upper()
    forbidden = forbidden or set()

    scrambles = set()
    attempts = 0

    while len(scrambles) < n and attempts < max_attempts:
        attempts += 1
        s = scramble_sequence(seq, rng)

        if s == seq:
            continue

        if s in scrambles:
            continue

        if s in forbidden:
            continue

        scrambles.add(s)

    if len(scrambles) < n:
        print(
            f"[WARNING] Only generated {len(scrambles)} unique scrambles "
            f"for {seq}. Requested {n}."
        )

    return sorted(scrambles)


def dominates_maximize(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Pareto dominance for maximization.
    a dominates b if:
      all a_i >= b_i and at least one a_i > b_i
    """
    return bool(np.all(a >= b) and np.any(a > b))


def safe_zscore(x: float, values: np.ndarray) -> float:
    """
    z-score of x relative to values.
    """
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values, ddof=1))

    if not np.isfinite(std) or std == 0:
        return np.nan

    return float((x - mean) / std)


def empirical_p_value_greater_equal(original_value: float, scrambled_values: np.ndarray) -> float:
    """
    One-sided empirical p-value:
    probability that scrambled score is >= original score.

    Smaller is better for showing optimized peptide is better than scrambles.
    Uses +1 smoothing.
    """
    scrambled_values = np.asarray(scrambled_values, dtype=float)
    n = np.sum(np.isfinite(scrambled_values))

    if n == 0:
        return np.nan

    count_ge = np.sum(scrambled_values >= original_value)
    return float((count_ge + 1) / (n + 1))


# ============================================================
# Main analysis
# ============================================================

def main():
    rng = random.Random(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    optimized_df = pd.read_csv(OPTIMIZED_CSV)
    train_df = pd.read_csv(TRAIN_PARETO_CSV)

    opt_pep_col = get_peptide_column(optimized_df)
    train_pep_col = get_peptide_column(train_df)

    optimized_peptides = (
        optimized_df[opt_pep_col]
        .astype(str)
        .str.strip()
        .str.upper()
        .dropna()
        .unique()
        .tolist()
    )

    train_peptides = (
        train_df[train_pep_col]
        .astype(str)
        .str.strip()
        .str.upper()
        .dropna()
        .unique()
        .tolist()
    )

    forbidden = set()
    if REJECT_EXISTING_SEQUENCES:
        forbidden.update(optimized_peptides)
        forbidden.update(train_peptides)

    print(f"Optimized peptides: {len(optimized_peptides)}")
    print(f"Training Pareto peptides: {len(train_peptides)}")
    print(f"Scrambles per optimized peptide: {N_SCRAMBLES_PER_PEPTIDE}")

    all_scored_rows = []
    summary_rows = []

    for peptide_index, original_peptide in enumerate(optimized_peptides[1:], start=1):
        print(f"\n[{peptide_index}/{len(optimized_peptides)}] Original: {original_peptide}")

        scrambles = generate_unique_scrambles(
            seq=original_peptide,
            n=N_SCRAMBLES_PER_PEPTIDE,
            rng=rng,
            forbidden=forbidden,
        )

        candidate_peptides = [original_peptide] + scrambles

        # ------------------------------------------------------------
        # Re-score original + scrambles together.
        #
        # Important:
        # Your blackbox_fc uses min-max normalization inside the batch.
        # Therefore, the fairest control is to score the original peptide
        # and its own scrambled controls in the same call.
        # ------------------------------------------------------------
        scored_df = blackbox_fc(candidate_peptides)

        scored_pep_col = get_peptide_column(scored_df)
        scored_df[scored_pep_col] = scored_df[scored_pep_col].astype(str).str.upper()

        scored_df["original_peptide"] = original_peptide
        scored_df["control_type"] = np.where(
            scored_df[scored_pep_col] == original_peptide,
            "optimized_original",
            "scrambled_control",
        )

        scored_df["scramble_group_id"] = peptide_index

        all_scored_rows.append(scored_df)

        original_rows = scored_df[scored_df["control_type"] == "optimized_original"].copy()
        scrambled_rows = scored_df[scored_df["control_type"] == "scrambled_control"].copy()

        if len(original_rows) != 1:
            print(
                f"[WARNING] Expected exactly one original row for {original_peptide}, "
                f"found {len(original_rows)}."
            )
            continue

        original_row = original_rows.iloc[0]

        original_obj = original_row[OBJECTIVE_COLS].to_numpy(dtype=float)
        scrambled_obj = scrambled_rows[OBJECTIVE_COLS].to_numpy(dtype=float)

        n_scrambles = len(scrambled_rows)

        n_scrambled_dominating_original = 0
        n_scrambled_dominated_by_original = 0

        for i in range(n_scrambles):
            s_obj = scrambled_obj[i]

            if dominates_maximize(s_obj, original_obj):
                n_scrambled_dominating_original += 1

            if dominates_maximize(original_obj, s_obj):
                n_scrambled_dominated_by_original += 1

        summary = {
            "original_peptide": original_peptide,
            "n_scrambles_scored": n_scrambles,

            "original_chelation_sub": float(original_row["chelation_sub"]),
            "original_solubility_sub": float(original_row["solubility_sub"]),
            "original_stability_sub": float(original_row["stability_sub"]),
            "original_expression_sub": float(original_row["expression_sub"]),
            "original_final_score": float(original_row["final_score"]),

            "scrambled_mean_final_score": float(scrambled_rows["final_score"].mean()),
            "scrambled_std_final_score": float(scrambled_rows["final_score"].std(ddof=1)),
            "scrambled_max_final_score": float(scrambled_rows["final_score"].max()),
            "scrambled_95pct_final_score": float(scrambled_rows["final_score"].quantile(0.95)),

            "original_final_score_z_vs_scrambled": safe_zscore(
                float(original_row["final_score"]),
                scrambled_rows["final_score"].to_numpy(dtype=float),
            ),
            "empirical_p_scrambled_ge_original_final_score": empirical_p_value_greater_equal(
                float(original_row["final_score"]),
                scrambled_rows["final_score"].to_numpy(dtype=float),
            ),

            "n_scrambled_dominating_original": int(n_scrambled_dominating_original),
            "n_scrambled_dominated_by_original": int(n_scrambled_dominated_by_original),
            "fraction_scrambled_dominated_by_original": (
                n_scrambled_dominated_by_original / n_scrambles if n_scrambles > 0 else np.nan
            ),
            "fraction_scrambled_dominating_original": (
                n_scrambled_dominating_original / n_scrambles if n_scrambles > 0 else np.nan
            ),
        }

        # Per-objective scrambled summary
        for col in OBJECTIVE_COLS:
            original_value = float(original_row[col])
            scrambled_values = scrambled_rows[col].to_numpy(dtype=float)

            summary[f"scrambled_mean_{col}"] = float(np.nanmean(scrambled_values))
            summary[f"scrambled_std_{col}"] = float(np.nanstd(scrambled_values, ddof=1))
            summary[f"scrambled_max_{col}"] = float(np.nanmax(scrambled_values))
            summary[f"empirical_p_scrambled_ge_original_{col}"] = empirical_p_value_greater_equal(
                original_value,
                scrambled_values,
            )

        summary_rows.append(summary)

    # ============================================================
    # Save detailed and summary outputs
    # ============================================================

    if len(all_scored_rows) == 0:
        raise RuntimeError("No scored rows were generated.")

    all_scores_df = pd.concat(all_scored_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    all_scores_path = out_dir / "scrambling_control_all_scores.csv"
    summary_path = out_dir / "scrambling_control_summary.csv"

    all_scores_df.to_csv(all_scores_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSaved detailed scores: {all_scores_path}")
    print(f"Saved summary: {summary_path}")

    # ============================================================
    # Plot 1: final score boxplot by original peptide
    # ============================================================

    plt.figure(figsize=(12, 6))

    peptides_order = optimized_peptides
    box_data = []

    for pep in peptides_order:
        vals = all_scores_df[
            (all_scores_df["original_peptide"] == pep)
            & (all_scores_df["control_type"] == "scrambled_control")
        ]["final_score"].to_numpy(dtype=float)

        box_data.append(vals)

    plt.boxplot(box_data, labels=peptides_order, showfliers=True)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Final score")
    plt.title("Scrambling control: final-score distribution of scrambled peptides")

    # Add original final scores as red dots
    for i, pep in enumerate(peptides_order, start=1):
        orig_val = all_scores_df[
            (all_scores_df["original_peptide"] == pep)
            & (all_scores_df["control_type"] == "optimized_original")
        ]["final_score"].iloc[0]

        plt.scatter(i, orig_val, marker="D", s=70, label="Original" if i == 1 else None)

    plt.legend()
    plt.tight_layout()

    plot_path = out_dir / "scrambling_control_final_score_boxplot.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Saved plot: {plot_path}")

    # ============================================================
    # Plot 2: original vs scrambled mean final score
    # ============================================================

    plt.figure(figsize=(12, 6))

    x = np.arange(len(summary_df))
    width = 0.35

    plt.bar(
        x - width / 2,
        summary_df["original_final_score"],
        width,
        label="Original optimized peptide",
    )

    plt.bar(
        x + width / 2,
        summary_df["scrambled_mean_final_score"],
        width,
        yerr=summary_df["scrambled_std_final_score"],
        capsize=3,
        label="Scrambled controls mean ± SD",
    )

    plt.xticks(x, summary_df["original_peptide"], rotation=45, ha="right")
    plt.ylabel("Final score")
    plt.title("Original optimized peptides vs scrambled controls")
    plt.legend()
    plt.tight_layout()

    plot_path = out_dir / "original_vs_scrambled_mean_final_score.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Saved plot: {plot_path}")

    # ============================================================
    # Plot 3: empirical p-values
    # ============================================================

    plt.figure(figsize=(12, 5))

    plt.bar(
        summary_df["original_peptide"],
        summary_df["empirical_p_scrambled_ge_original_final_score"],
    )

    plt.axhline(0.05, linestyle="--", linewidth=1, label="p = 0.05")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Empirical p-value")
    plt.title("Probability that scrambled controls score at least as high as original")
    plt.legend()
    plt.tight_layout()

    plot_path = out_dir / "empirical_p_values_final_score.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Saved plot: {plot_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()