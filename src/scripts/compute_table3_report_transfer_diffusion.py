import pandas as pd
from pathlib import Path

# ============================================================
# Input files
# ============================================================
BO_PARETO_CSV = Path("2026_06_24_Geoff_results/bo_results/bo_pareto_dominates_train_CU_after_flow_space_gp_wu_10.csv")
TRAIN_PARETO_CSV = Path("2026_06_24_Geoff_results/bo_results/train_pareto_cu_flow_space_gp.csv")

OUTPUT_CSV = Path("2026_06_24_Geoff_results/bo_results/normalization_scratch_summary_table.csv")

OBJ_COLS = [
    "chelation_sub",
    "solubility_sub",
    "stability_sub",
    "expression_sub",
]


# ============================================================
# Utilities
# ============================================================
def levenshtein_edit_distance(a: str, b: str) -> int:
    """Compute edit distance between two peptide strings."""
    a = str(a).strip().upper()
    b = str(b).strip().upper()

    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + int(char_a != char_b)
            current.append(min(insertion, deletion, substitution))
        previous = current

    return int(previous[-1])


def dominates(row_a: pd.Series, row_b: pd.Series, obj_cols=OBJ_COLS) -> bool:
    """
    Return True if row_a Pareto-dominates row_b.
    All objectives are assumed to be maximized.
    """
    a = row_a[obj_cols].astype(float)
    b = row_b[obj_cols].astype(float)

    return bool((a >= b).all() and (a > b).any())


def nearest_training_peptide(candidate_peptide: str, train_peptides) -> tuple[str, int]:
    """Find nearest training peptide by edit distance."""
    best_peptide = None
    best_distance = None

    for train_peptide in train_peptides:
        dist = levenshtein_edit_distance(candidate_peptide, train_peptide)

        if best_distance is None or dist < best_distance:
            best_distance = dist
            best_peptide = train_peptide

    return best_peptide, int(best_distance)


# ============================================================
# Main analysis
# ============================================================
def main():
    bo_df = pd.read_csv(BO_PARETO_CSV)
    train_df = pd.read_csv(TRAIN_PARETO_CSV)

    required_bo_cols = ["peptide"] + OBJ_COLS
    required_train_cols = ["peptide"] + OBJ_COLS

    missing_bo = [c for c in required_bo_cols if c not in bo_df.columns]
    missing_train = [c for c in required_train_cols if c not in train_df.columns]

    if missing_bo:
        raise ValueError(f"Missing columns in BO Pareto CSV: {missing_bo}")

    if missing_train:
        raise ValueError(f"Missing columns in training Pareto CSV: {missing_train}")

    train_peptides = train_df["peptide"].astype(str).str.strip().str.upper().tolist()

    candidate_rows = []

    for _, bo_row in bo_df.iterrows():
        bo_peptide = str(bo_row["peptide"]).strip().upper()

        dominated_count = 0
        for _, train_row in train_df.iterrows():
            if dominates(bo_row, train_row):
                dominated_count += 1

        nearest_peptide, edit_distance = nearest_training_peptide(
            bo_peptide,
            train_peptides,
        )

        candidate_rows.append(
            {
                "Candidate Method": "Diffusion - Transfer",
                "BO Peptide": bo_peptide,
                "Max #of Dominated Pareto": dominated_count,
                "Nearest Training Peptide": nearest_peptide,
                "Edit Distance": edit_distance,
            }
        )

    detailed_df = pd.DataFrame(candidate_rows)

    # Select the BO peptide that dominates the largest number of training Pareto peptides.
    # Tie-breaker: smaller edit distance, then alphabetic peptide order.
    best_row = detailed_df.sort_values(
        by=["Max #of Dominated Pareto", "Edit Distance", "BO Peptide"],
        ascending=[False, True, True],
    ).iloc[0]

    summary_df = pd.DataFrame(
        [
            {
                "Candidate Method": "Diffusion - Transfer",
                "Max #of Dominated Pareto": int(best_row["Max #of Dominated Pareto"]),
                "Nearest Training Peptide": best_row["Nearest Training Peptide"],
                "Edit Distance": int(best_row["Edit Distance"]),
                "BO Peptide": best_row["BO Peptide"],
            }
        ]
    )

    print("\nDetailed BO candidates:")
    print(detailed_df)

    print("\nSummary table:")
    print(summary_df)

    summary_df.to_csv(OUTPUT_CSV, index=False)
    detailed_df.to_csv("normalization_scratch_detailed_candidate_analysis.csv", index=False)

    print(f"\nSaved summary to: {OUTPUT_CSV}")
    print("Saved detailed candidate analysis to: normalization_scratch_detailed_candidate_analysis.csv")


if __name__ == "__main__":
    main()