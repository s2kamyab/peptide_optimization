"""
Simple sort ranking for Pareto (or any peptide list) by a weighted scalar score.

Score:
  y(x) = 0.50*f1 + 0.30*f2 + 0.10*f3 + 0.10*f4

Input CSV:
  peptide_vae_out/bo_optimized_peptides_mo_MN.csv
Expected columns:
  - peptide
  - obj_chelation, obj_solubility, obj_stability, obj_expression

Output CSV:
  peptide_vae_out/ranked_pareto_simple_sort_MN.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np

# -----------------------------
# Config
# -----------------------------
CSV_IN = Path(r"peptide_vae_out\bo_optimized_peptides_mo_MN.csv")
CSV_OUT = Path(r"peptide_vae_out\ranked_pareto_simple_sort_MN.csv")

SEQ_COL = "peptide"
OBJ_COLS = ["obj_chelation", "obj_solubility", "obj_stability", "obj_expression"]

W = np.array([0.50, 0.30, 0.10, 0.10], dtype=float)  # weights for f1..f4

# Optional tie-breaks (after score): prioritize higher chelation then solubility etc.
TIEBREAK_COLS = ["obj_chelation", "obj_solubility", "obj_stability", "obj_expression"]

# -----------------------------
# Load + validate
# -----------------------------
df = pd.read_csv(CSV_IN).copy()

missing = [c for c in [SEQ_COL] + OBJ_COLS if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}\nFound: {list(df.columns)}")

for c in OBJ_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")

bad = df[df[OBJ_COLS].isna().any(axis=1)]
if not bad.empty:
    raise ValueError(
        "Some rows have NaN objective values. Fix/drop them first. "
        f"Example bad rows:\n{bad[[SEQ_COL] + OBJ_COLS].head(10)}"
    )

# -----------------------------
# Compute scalar score + rank
# -----------------------------
F = df[OBJ_COLS].to_numpy(dtype=float)          # (N,4)
df["final_score"] = (F * W.reshape(1, -1)).sum(axis=1)

# Sort: highest final_score first; optional tiebreakers next
sort_cols = ["final_score"] + TIEBREAK_COLS
ascending = [False] * len(sort_cols)

df_ranked = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
df_ranked.insert(0, "rank", range(1, len(df_ranked) + 1))

# Save
CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
df_ranked.to_csv(CSV_OUT, index=False)

print(f"Saved ranked file to: {CSV_OUT}")
print(df_ranked[["rank", SEQ_COL, "final_score"] + OBJ_COLS].head(10))
