from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# Plot style
# ============================================================
# Use 16 pt fonts consistently in all saved diagrams.
PLOT_FONT_SIZE = 20
plt.rcParams.update({
    "font.size": PLOT_FONT_SIZE,
    "axes.titlesize": PLOT_FONT_SIZE,
    "axes.labelsize": PLOT_FONT_SIZE,
    "xtick.labelsize": PLOT_FONT_SIZE,
    "ytick.labelsize": PLOT_FONT_SIZE,
    "legend.fontsize": PLOT_FONT_SIZE,
    "figure.titlesize": PLOT_FONT_SIZE,
})


# ============================================================
# Input files
# ============================================================

ALL_SCORES_CSV = "scrambling_control_CU_gp_before_flow_wu_10/scrambling_control_all_scores.csv"
# "scrambling_control_CU_no_flow_warmup_10/scrambling_control_all_scores.csv"
# "scrambling_control_CU_after_gp_warmup_10/scrambling_control_all_scores.csv"
SUMMARY_CSV = "scrambling_control_CU_gp_before_flow_wu_10/scrambling_control_summary.csv"
# "scrambling_control_CU_before_gp_warmup_10/scrambling_control_summary.csv"

OUT_DIR = Path("scrambling_control_dominance_analysis_gp_before_flow_wu_10")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Columns
# ============================================================

SEQ_COL = "peptide_len10"
ORIGINAL_COL = "original_peptide"
CONTROL_COL = "control_type"

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

def dominates_maximize(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Pareto dominance for maximization.

    a dominates b if:
      all objectives in a >= b
      and at least one objective in a > b
    """
    return bool(np.all(a >= b) and np.any(a > b))


def empirical_p_value_greater_equal(original_value: float, scrambled_values: np.ndarray) -> float:
    """
    One-sided empirical p-value:
    probability that a scrambled control scores >= original.

    Uses +1 smoothing:
      p = (count_scrambled_ge_original + 1) / (n_scrambles + 1)
    """
    scrambled_values = np.asarray(scrambled_values, dtype=float)
    scrambled_values = scrambled_values[np.isfinite(scrambled_values)]

    if len(scrambled_values) == 0:
        return np.nan

    count_ge = np.sum(scrambled_values >= original_value)
    return float((count_ge + 1) / (len(scrambled_values) + 1))


def clean_sequence_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize sequence strings.
    """
    df = df.copy()

    df[SEQ_COL] = df[SEQ_COL].astype(str).str.strip().str.upper()
    df[ORIGINAL_COL] = df[ORIGINAL_COL].astype(str).str.strip().str.upper()
    df[CONTROL_COL] = df[CONTROL_COL].astype(str).str.strip()

    for col in OBJECTIVE_COLS + [FINAL_SCORE_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ============================================================
# Main analysis
# ============================================================

def main():
    all_scores = pd.read_csv(ALL_SCORES_CSV)
    summary = pd.read_csv(SUMMARY_CSV)

    all_scores = clean_sequence_columns(all_scores)
    summary[ORIGINAL_COL] = summary[ORIGINAL_COL].astype(str).str.strip().str.upper()

    print("Loaded all scores:", all_scores.shape)
    print("Loaded summary:", summary.shape)

    detailed_rows = []
    recomputed_summary_rows = []

    original_peptides = summary[ORIGINAL_COL].tolist()

    for original_peptide in original_peptides:
        group = all_scores[all_scores[ORIGINAL_COL] == original_peptide].copy()

        original_rows = group[group[CONTROL_COL] == "optimized_original"].copy()
        scrambled_rows = group[group[CONTROL_COL] == "scrambled_control"].copy()

        if original_rows.empty:
            print(f"[WARNING] No optimized original row found for {original_peptide}")
            continue

        if scrambled_rows.empty:
            print(f"[WARNING] No scrambled rows found for {original_peptide}")
            continue

        original_row = original_rows.iloc[0]

        original_obj = original_row[OBJECTIVE_COLS].to_numpy(dtype=float)
        original_final = float(original_row[FINAL_SCORE_COL])

        n_scrambles = len(scrambled_rows)

        n_scramble_dominates_original = 0
        n_original_dominates_scramble = 0
        n_nondominated_vs_each_other = 0

        for _, scr_row in scrambled_rows.iterrows():
            scrambled_peptide = str(scr_row[SEQ_COL]).strip().upper()
            scrambled_obj = scr_row[OBJECTIVE_COLS].to_numpy(dtype=float)

            scramble_dominates_original = dominates_maximize(scrambled_obj, original_obj)
            original_dominates_scramble = dominates_maximize(original_obj, scrambled_obj)

            if scramble_dominates_original:
                n_scramble_dominates_original += 1

            if original_dominates_scramble:
                n_original_dominates_scramble += 1

            if not scramble_dominates_original and not original_dominates_scramble:
                n_nondominated_vs_each_other += 1

            row = {
                "original_peptide": original_peptide,
                "scrambled_peptide": scrambled_peptide,
                "scramble_dominates_original": int(scramble_dominates_original),
                "original_dominates_scramble": int(original_dominates_scramble),
                "nondominated_vs_each_other": int(
                    not scramble_dominates_original and not original_dominates_scramble
                ),
                "original_final_score": original_final,
                "scrambled_final_score": float(scr_row[FINAL_SCORE_COL]),
                "delta_final_score_scrambled_minus_original": float(scr_row[FINAL_SCORE_COL] - original_final),
            }

            # Add original, scrambled, and delta values for each objective
            for col in OBJECTIVE_COLS:
                row[f"original_{col}"] = float(original_row[col])
                row[f"scrambled_{col}"] = float(scr_row[col])
                row[f"delta_{col}_scrambled_minus_original"] = float(scr_row[col] - original_row[col])

            detailed_rows.append(row)

        scrambled_final_values = scrambled_rows[FINAL_SCORE_COL].to_numpy(dtype=float)

        recomputed_summary_rows.append(
            {
                "original_peptide": original_peptide,
                "n_scrambles": int(n_scrambles),
                "original_final_score": original_final,
                "scrambled_mean_final_score": float(np.nanmean(scrambled_final_values)),
                "scrambled_std_final_score": float(np.nanstd(scrambled_final_values, ddof=1)),
                "scrambled_max_final_score": float(np.nanmax(scrambled_final_values)),
                "empirical_p_scrambled_ge_original_final_score": empirical_p_value_greater_equal(
                    original_final,
                    scrambled_final_values,
                ),
                "n_scramble_dominates_original": int(n_scramble_dominates_original),
                "n_original_dominates_scramble": int(n_original_dominates_scramble),
                "n_nondominated_vs_each_other": int(n_nondominated_vs_each_other),
                "fraction_scramble_dominates_original": float(n_scramble_dominates_original / n_scrambles),
                "fraction_original_dominates_scramble": float(n_original_dominates_scramble / n_scrambles),
                "fraction_nondominated_vs_each_other": float(n_nondominated_vs_each_other / n_scrambles),
            }
        )

    detailed_df = pd.DataFrame(detailed_rows)
    recomputed_summary_df = pd.DataFrame(recomputed_summary_rows)

    # ============================================================
    # Save all detailed dominance comparisons
    # ============================================================

    detailed_path = OUT_DIR / "scramble_vs_original_detailed_dominance.csv"
    detailed_df.to_csv(detailed_path, index=False)

    # Only scrambled sequences that dominate original
    dominating_scrambles_df = detailed_df[
        detailed_df["scramble_dominates_original"] == 1
    ].copy()

    dominating_scrambles_path = OUT_DIR / "scrambles_that_dominate_original.csv"
    dominating_scrambles_df.to_csv(dominating_scrambles_path, index=False)

    # Recomputed summary
    recomputed_summary_path = OUT_DIR / "scrambling_dominance_summary_recomputed.csv"
    recomputed_summary_df.to_csv(recomputed_summary_path, index=False)

    # ============================================================
    # Create compact table:
    # each original peptide -> list of scrambled peptides that dominate it
    # ============================================================

    compact_rows = []

    for original_peptide, g in dominating_scrambles_df.groupby("original_peptide"):
        g_sorted = g.sort_values(
            by=[
                "delta_final_score_scrambled_minus_original",
                "scrambled_final_score",
            ],
            ascending=[False, False],
        )

        compact_rows.append(
            {
                "original_peptide": original_peptide,
                "n_scrambles_dominating_original": int(len(g_sorted)),
                "dominating_scrambled_peptides": "; ".join(g_sorted["scrambled_peptide"].tolist()),
                "best_dominating_scramble": g_sorted["scrambled_peptide"].iloc[0],
                "best_dominating_scramble_final_score": float(g_sorted["scrambled_final_score"].iloc[0]),
                "original_final_score": float(g_sorted["original_final_score"].iloc[0]),
                "best_delta_final_score": float(
                    g_sorted["delta_final_score_scrambled_minus_original"].iloc[0]
                ),
            }
        )

    compact_df = pd.DataFrame(compact_rows)

    compact_path = OUT_DIR / "compact_dominating_scrambles_by_original.csv"
    compact_df.to_csv(compact_path, index=False)

    # ============================================================
    # Top 10 dominating scrambles per original peptide
    # ============================================================

    top_rows = []

    for original_peptide, g in dominating_scrambles_df.groupby("original_peptide"):
        g_sorted = g.sort_values(
            by=[
                "delta_final_score_scrambled_minus_original",
                "scrambled_final_score",
            ],
            ascending=[False, False],
        ).head(10)

        top_rows.append(g_sorted)

    if len(top_rows) > 0:
        top_dominating_df = pd.concat(top_rows, ignore_index=True)
    else:
        top_dominating_df = pd.DataFrame()

    top_path = OUT_DIR / "top_dominating_scrambles_per_original.csv"
    top_dominating_df.to_csv(top_path, index=False)

    # ============================================================
    # Plot 1: final-score empirical p-values
    # ============================================================

    p_col = "empirical_p_scrambled_ge_original_final_score"

    pval_df = recomputed_summary_df.sort_values(p_col, ascending=True).copy()

    plt.figure(figsize=(12, 5))
    plt.bar(pval_df["original_peptide"], pval_df[p_col])
    plt.axhline(0.05, linestyle="--", linewidth=1.5, label="p = 0.05")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Empirical p-value")
    plt.xlabel("Original optimized peptide")
    plt.title("Scrambling control: probability scrambled controls score at least as high as original")
    plt.legend()
    plt.tight_layout()

    pval_plot_path = OUT_DIR / "p_values_final_score.png"
    plt.savefig(pval_plot_path, dpi=300)
    plt.close()

    # ============================================================
    # Plot 2: dominance counts
    # ============================================================

    count_df = recomputed_summary_df.sort_values(
        "n_scramble_dominates_original",
        ascending=False,
    ).copy()

    x = np.arange(len(count_df))
    width = 0.35

    plt.figure(figsize=(12, 5))
    plt.bar(
        x - width / 2,
        count_df["n_scramble_dominates_original"],
        width,
        label="Scrambles dominate original",
    )
    plt.bar(
        x + width / 2,
        count_df["n_original_dominates_scramble"],
        width,
        label="Original dominates scrambles",
    )
    plt.xticks(x, count_df["original_peptide"], rotation=45, ha="right")
    plt.ylabel("Count")
    plt.xlabel("Original optimized peptide")
    plt.title("Pareto dominance comparison: original vs scrambled controls")
    plt.legend()
    plt.tight_layout()

    dominance_count_plot_path = OUT_DIR / "dominance_counts_original_vs_scrambles.png"
    plt.savefig(dominance_count_plot_path, dpi=300)
    plt.close()

    # ============================================================
    # Plot 3: per-objective p-values from summary file
    # ============================================================

    objective_p_cols = [
        "empirical_p_scrambled_ge_original_chelation_sub",
        "empirical_p_scrambled_ge_original_solubility_sub",
        "empirical_p_scrambled_ge_original_stability_sub",
        "empirical_p_scrambled_ge_original_expression_sub",
    ]

    available_p_cols = [c for c in objective_p_cols if c in summary.columns]

    if available_p_cols:
        p_obj = summary[[ORIGINAL_COL] + available_p_cols].copy()
        p_obj = p_obj.sort_values(ORIGINAL_COL)

        x = np.arange(len(p_obj))
        width = 0.18

        plt.figure(figsize=(14, 6))

        for i, col in enumerate(available_p_cols):
            label = (
                col.replace("empirical_p_scrambled_ge_original_", "")
                .replace("_sub", "")
                .replace("_", " ")
            )
            plt.bar(x + (i - 1.5) * width, p_obj[col], width, label=label)

        plt.axhline(0.05, linestyle="--", linewidth=1.5, label="p = 0.05")
        plt.xticks(x, p_obj[ORIGINAL_COL], rotation=45, ha="right")
        plt.ylabel("Empirical p-value")
        plt.xlabel("Original optimized peptide")
        plt.title("Per-objective empirical p-values for scrambling controls")
        plt.legend()
        plt.tight_layout()

        obj_pval_plot_path = OUT_DIR / "p_values_by_objective.png"
        plt.savefig(obj_pval_plot_path, dpi=300)
        plt.close()

    # ============================================================
    # Print summary
    # ============================================================

    print("\nSaved outputs to:", OUT_DIR.resolve())
    print("\nMain files:")
    print("1.", detailed_path)
    print("2.", dominating_scrambles_path)
    print("3.", compact_path)
    print("4.", top_path)
    print("5.", recomputed_summary_path)
    print("6.", pval_plot_path)
    print("7.", dominance_count_plot_path)

    if available_p_cols:
        print("8.", obj_pval_plot_path)

    print("\nRecomputed dominance summary:")
    print(
        recomputed_summary_df[
            [
                "original_peptide",
                "empirical_p_scrambled_ge_original_final_score",
                "n_scramble_dominates_original",
                "n_original_dominates_scramble",
                "n_nondominated_vs_each_other",
            ]
        ].sort_values("empirical_p_scrambled_ge_original_final_score").to_string(index=False)
    )


if __name__ == "__main__":
    main()