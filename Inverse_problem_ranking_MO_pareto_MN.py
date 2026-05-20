"""
mn_inverse_ranking.py

Inverse-problem ranking for Mn using CVXPY assignment model.

Pipeline:
  1) Read peptides from CSV
  2) Compute Mn objectives via compute_objectives_mn with *stabilized* batch normalization:
        evaluate: REF_BATCH + CAND_BATCH
        then take the tail rows corresponding to CAND_BATCH
  3) Solve assignment ranking:
        minimize ||A(P) - y_obs||^2 + lam * <C, P>     (MIQP if available)
        else minimize sum_j |A_j(P) - y_obs_j| + lam * <C, P>  (MILP)
  4) Save ranked CSV

Install:
  pip install numpy pandas cvxpy
  # For MILP solver:
  pip install highspy
  # For MIQP (optional): gurobi / cplex / mosek / scip
"""

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import cvxpy as cp


# =============================
# USER CONFIG
# =============================
CSV_IN  = "peptide_vae_out/bo_optimized_peptides_mo_MN.csv"
CSV_OUT = "ranked_peptides_inverse_mn.csv"

SEQ_COL = "peptide"  # change if your column name is different

OBJ_COLS = ["chelation_sub", "solubility_sub", "stability_sub", "expression_sub"]

# Target aggregate vector (choose your preference profile)
# Example: prioritize solubility high, then chelation, then expression, then stability
Y_OBS = np.array([1, 1, 1, 1], dtype=float)

LAM = 0.1
USE_NDCG_GAIN = True          # g(k) = 1/log2(1+k) if True, else uniform
COST_MODE = "keep_order"      # "zeros" or "keep_order"

# ---- Objective computation controls ----
RECOMPUTE_OBJECTIVES = True   # If False and OBJ_COLS exist in CSV, we will reuse them.
REF_SIZE = 2000               # size of fixed reference batch (stability anchor)
CAND_BATCH_SIZE = 256         # number of candidate peptides evaluated per call (with REF appended)

# Binding sites / DNA_sequence not used for your Mn objective (it will reverse-translate if needed)
DEFAULT_BINDING_SITES = ""
DEFAULT_DNA = ""
DEFAULT_METAL = "MN"

# Import your Mn objective function
# Adjust this import to your project structure
from black_box_fcn_mo_MN import compute_objectives_mn


AA = set("ACDEFGHIKLMNPQRSTVWY")


# =============================
# HELPERS
# =============================
def clean_peptide(s: str) -> str:
    s = (s or "").strip().upper()
    s = "".join([c for c in s if c in AA])
    return s

def pick_solver(installed: List[str]) -> Tuple[str, str]:
    """
    Return (mode, solver_name) where mode is 'miqp' or 'milp'.
    """
    miqp_candidates = ["GUROBI", "CPLEX", "MOSEK", "SCIP"]
    for s in miqp_candidates:
        if s in installed:
            return ("miqp", s)

    if "HIGHS" in installed:
        return ("milp", "HIGHS")

    milp_fallbacks = ["GLPK_MI", "CBC", "ECOS_BB"]  # ECOS_BB is very slow but sometimes available
    for s in milp_fallbacks:
        if s in installed:
            return ("milp", s)

    raise RuntimeError(
        f"No suitable MIP solver found. Installed solvers: {installed}\n"
        "Install one of: highspy (HiGHS) or gurobi/cplex/mosek/scip."
    )

def make_cost_matrix(N: int, mode: str) -> np.ndarray:
    if mode == "zeros":
        return np.zeros((N, N), dtype=float)
    if mode == "keep_order":
        ii = np.arange(N).reshape(-1, 1)
        kk = np.arange(N).reshape(1, -1)
        C = np.abs(ii - kk).astype(float)
        C = C / max(1.0, C.max())
        return C
    raise ValueError("COST_MODE must be one of: 'zeros', 'keep_order'.")

def build_gain(N: int, use_ndcg: bool) -> np.ndarray:
    k_idx = np.arange(1, N + 1)
    if use_ndcg:
        g = 1.0 / np.log2(1.0 + k_idx)
    else:
        g = np.ones(N, dtype=float)
    g = g / np.sum(g)
    return g

def stabilized_objectives(
    peptides: List[str],
    ref_peptides: List[str],
    batch_size: int,
) -> pd.DataFrame:
    """
    Compute objectives for peptides using stabilized evaluation:
      compute_objectives_mn( REF + CAND_BATCH )
      then take tail rows corresponding to CAND_BATCH, matched by sequence.
    Returns a dataframe with columns: sequence + OBJ_COLS
    """
    # Build fixed reference candidates
    ref_candidates = [
        {"sequence": s, "binding_sites": DEFAULT_BINDING_SITES, "DNA_sequence": DEFAULT_DNA, "metal": DEFAULT_METAL}
        for s in ref_peptides
    ]

    out_rows = []
    pep_unique = list(dict.fromkeys(peptides))

    for start in range(0, len(pep_unique), batch_size):
        cand = pep_unique[start : start + batch_size]
        cand_candidates = [
            {"sequence": s, "binding_sites": DEFAULT_BINDING_SITES, "DNA_sequence": DEFAULT_DNA, "metal": DEFAULT_METAL}
            for s in cand
        ]

        batch = ref_candidates + cand_candidates
        df_obj = compute_objectives_mn(batch, reverse_translate_strategy="first")

        # default: penalize missing rows
        got = {s: np.array([-1.0, -1.0, -1.0, -1.0], dtype=float) for s in cand}

        if df_obj is not None and len(df_obj) > 0:
            tail = df_obj.tail(len(cand)).copy()
            tail["sequence"] = tail["sequence"].astype(str).map(clean_peptide)

            for s in cand:
                hit = tail[tail["sequence"] == s]
                if not hit.empty:
                    r = hit.iloc[-1]
                    got[s] = np.array(
                        [float(r[c]) for c in OBJ_COLS],
                        dtype=float,
                    )

        for s in cand:
            row = {"sequence": s}
            row.update({OBJ_COLS[i]: float(got[s][i]) for i in range(len(OBJ_COLS))})
            out_rows.append(row)

    return pd.DataFrame(out_rows)


# =============================
# MAIN
# =============================
def main():
    if not os.path.isfile(CSV_IN):
        raise FileNotFoundError(f"Input file not found: {CSV_IN}")

    df = pd.read_csv(CSV_IN).copy()

    if SEQ_COL not in df.columns:
        raise ValueError(f"Missing '{SEQ_COL}' column. Found: {list(df.columns)}")

    df[SEQ_COL] = df[SEQ_COL].astype(str).map(clean_peptide)
    df = df[df[SEQ_COL].str.len() > 0].reset_index(drop=True)

    peptides = df[SEQ_COL].tolist()
    N = len(peptides)
    if N < 2:
        raise ValueError("Need at least 2 peptides to rank.")

    # 1) Objectives
    if (not RECOMPUTE_OBJECTIVES) and all(c in df.columns for c in OBJ_COLS):
        print("[info] Using existing objective columns from CSV.")
        for c in OBJ_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[OBJ_COLS].isna().any().any():
            raise ValueError("Found NaNs in existing objective columns; set RECOMPUTE_OBJECTIVES=True.")
    else:
        print("[info] Recomputing Mn objectives with stabilized batching...")

        # reference batch: sample from peptides themselves (fixed seed for reproducibility)
        ref_peps = list(dict.fromkeys(peptides))
        if len(ref_peps) > REF_SIZE:
            rng = np.random.default_rng(0)
            ref_peps = rng.choice(ref_peps, size=REF_SIZE, replace=False).tolist()

        df_obj = stabilized_objectives(peptides=peptides, ref_peptides=ref_peps, batch_size=CAND_BATCH_SIZE)

        df_obj = df_obj.rename(columns={"sequence": SEQ_COL})
        df = df.drop(columns=[c for c in OBJ_COLS if c in df.columns], errors="ignore")
        df = df.merge(df_obj, on=SEQ_COL, how="left")

        # safety check
        if df[OBJ_COLS].isna().any().any():
            bad = df[df[OBJ_COLS].isna().any(axis=1)][[SEQ_COL] + OBJ_COLS]
            raise ValueError(f"Some peptides failed objective computation:\n{bad}")

    f = df[OBJ_COLS].to_numpy(dtype=float)  # (N, m)
    m = f.shape[1]

    y_obs = np.array(Y_OBS, dtype=float).reshape(-1)
    if y_obs.shape[0] != m:
        raise ValueError(f"Y_OBS length {y_obs.shape[0]} must match m={m} objectives.")

    # 2) Gains and cost matrix
    g = build_gain(N=N, use_ndcg=USE_NDCG_GAIN)          # (N,)
    C = make_cost_matrix(N=N, mode=COST_MODE)            # (N,N)

    # 3) Decision variable P: P[i,k]=1 if item i is at position k
    P = cp.Variable((N, N), boolean=True)

    # We want row_scores[k,:] = objectives of item placed at position k
    # Using P.T @ f gives (N,m): each position k is a column in P, so transpose.
    row_scores = P.T @ f                                  # (N,m)
    A_vec = cp.sum(cp.multiply(row_scores, g[:, None]), axis=0)  # (m,)

    cost_term = cp.sum(cp.multiply(C, P))

    constraints = [
        cp.sum(P, axis=0) == 1,   # each position gets exactly one item
        cp.sum(P, axis=1) == 1,   # each item appears exactly once
    ]

    installed = cp.installed_solvers()
    mode, solver_name = pick_solver(installed)

    print("Installed solvers:", installed)
    print("Chosen mode/solver:", mode, solver_name)

    if mode == "miqp":
        fit_term = cp.sum_squares(A_vec - y_obs)
        objective = cp.Minimize(fit_term + LAM * cost_term)
    else:
        # MILP (HiGHS-compatible): L1 fit
        t = cp.Variable(m, nonneg=True)
        constraints += [t >= A_vec - y_obs, t >= -(A_vec - y_obs)]
        fit_term = cp.sum(t)
        objective = cp.Minimize(fit_term + LAM * cost_term)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=getattr(cp, solver_name) if hasattr(cp, solver_name) else solver_name, verbose=True)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Solver failed. Status={prob.status}")

    P_opt = np.rint(P.value).astype(int)

    # perm[k] = item index i at position k
    perm = np.argmax(P_opt, axis=0)

    df_ranked = df.iloc[perm].copy().reset_index(drop=True)
    df_ranked.insert(0, "rank", np.arange(1, N + 1))

    achieved = (g.reshape(-1, 1) * f[perm]).sum(axis=0)

    print("\nObjectives:", OBJ_COLS)
    print("Target y_obs:", y_obs)
    print("Achieved A(P):", achieved)

    df_ranked.to_csv(CSV_OUT, index=False)
    print(f"\nSaved ranked file to: {CSV_OUT}")
    print(df_ranked[["rank", SEQ_COL] + OBJ_COLS].head(10))


if __name__ == "__main__":
    main()
