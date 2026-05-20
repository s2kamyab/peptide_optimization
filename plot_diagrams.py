# plot_hypervolume_vs_iterations.py
# pip install numpy pandas matplotlib torch botorch gpytorch

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume

# ---------------------------------------
# 0) INPUT FILES
# ---------------------------------------
SO_PATH = "peptide_vae_out/bo_optimized_peptides_MN.csv"        # single-objective BO log
MO_PATH = "peptide_vae_out/bo_optimized_peptides_mo_MN.csv"     # multi-objective BO log

# Your TRAINING CSV (you provided this name)
TRAIN_CSV_PATH = "metalpdb_binding_windows_len10_MN.csv"  # change to an absolute path if needed

OUT_PNG = "hypervolume_vs_iterations.png"
OUT_CSV = "hypervolume_vs_iterations.csv"

# ---------------------------------------
# 1) IMPORT your objective evaluator
# ---------------------------------------
# Recommended: import from your module
from black_box_fcn_mo_MN import compute_objectives_mn

# def compute_objectives_mn(*args, **kwargs):
#     raise NotImplementedError(
#         "Import or paste your compute_objectives_mn(...) here."
#     )

OBJ_SUB_COLS = ["chelation_sub", "solubility_sub", "stability_sub", "expression_sub"]
OBJ_COLS = ["obj_chelation", "obj_solubility", "obj_stability", "obj_expression"]

# ---------------------------------------
# 2) Non-dominated (maximization)
# ---------------------------------------
def is_non_dominated_max(Y: np.ndarray) -> np.ndarray:
    N = Y.shape[0]
    nd = np.ones(N, dtype=bool)
    for i in range(N):
        if not nd[i]:
            continue
        for j in range(N):
            if i == j:
                continue
            if np.all(Y[j] >= Y[i]) and np.any(Y[j] > Y[i]):
                nd[i] = False
                break
    return nd

# ---------------------------------------
# 3) Hypervolume helpers
# ---------------------------------------
def compute_ref_point(Y_all: np.ndarray, margin: float = 0.05) -> np.ndarray:
    # For maximization, ref_point must be "worse" than all points
    return np.min(Y_all, axis=0) - margin

def hypervolume_of_nondominated(Y: np.ndarray, ref_point: np.ndarray) -> float:
    if Y.size == 0:
        return 0.0
    nd = is_non_dominated_max(Y)
    Y_nd = Y[nd]
    hv = Hypervolume(ref_point=torch.tensor(ref_point, dtype=torch.double))
    return float(hv.compute(torch.tensor(Y_nd, dtype=torch.double)))

def hypervolume_vs_evals(df: pd.DataFrame, obj_cols: list[str], ref_point: np.ndarray) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    Y = df[obj_cols].to_numpy(dtype=float)

    rows = []
    for k in range(1, len(df) + 1):
        Yk = Y[:k, :]
        hv_k = hypervolume_of_nondominated(Yk, ref_point)
        rows.append({"n_evals": k, "hv": hv_k, "n_nd": int(is_non_dominated_max(Yk).sum())})
    return pd.DataFrame(rows)

# ---------------------------------------
# 4) Stable objectives via fixed reference-batch trick
# ---------------------------------------
def stable_objectives_batch(
    peptides: list[str],
    ref_candidates: list[dict],
    binding_sites: str = "",
) -> np.ndarray:
    """
    For each peptide p: evaluate compute_objectives_mn(ref + [p]) and take candidate row.
    Returns (N,4): [chelation_sub, solubility_sub, stability_sub, expression_sub]
    """
    out = []
    for pep in peptides:
        seq = (pep or "").strip().upper()
        if not seq:
            out.append([-1.0, -1.0, -1.0, -1.0])
            continue

        batch = list(ref_candidates) + [{
            "sequence": seq,
            "binding_sites": binding_sites,
            "DNA_sequence": "",
            "metal": "MN",
        }]

        df_obj = compute_objectives_mn(batch, reverse_translate_strategy="first")
        if df_obj is None or len(df_obj) == 0:
            out.append([-1.0, -1.0, -1.0, -1.0])
            continue

        last_seq = str(df_obj["sequence"].iloc[-1]).strip().upper()
        if last_seq != seq:
            out.append([-1.0, -1.0, -1.0, -1.0])
            continue

        last = df_obj.iloc[-1]
        out.append([float(last[c]) for c in OBJ_SUB_COLS])

    return np.array(out, dtype=float)

def build_reference_candidates_from_training(
    train_csv_path: str,
    n_ref: int = 2000,
    seed: int = 0,
    seq_col: str = "peptide_len10",
) -> list[dict]:
    """
    Match your BO code:
      df = pd.read_csv(train_csv).dropna(subset=[seq_col])
      df[seq_col] = ... upper/strip
      sample n_ref with random_state=seed
    """
    df = pd.read_csv(train_csv_path).dropna(subset=[seq_col]).copy()
    df[seq_col] = df[seq_col].astype(str).str.strip().str.upper()

    if len(df) == 0:
        raise ValueError(f"No valid sequences found in {train_csv_path} using column '{seq_col}'")

    sampled = df[seq_col].sample(n=min(n_ref, len(df)), random_state=seed).tolist()
    ref = [{"sequence": s, "binding_sites": "", "DNA_sequence": "", "metal": "MN"} for s in sampled]
    return ref

# ---------------------------------------
# 5) MAIN
# ---------------------------------------
def main():
    so = pd.read_csv(SO_PATH)
    mo = pd.read_csv(MO_PATH)

    # ---- Validate MO columns ----
    required_mo = ["iter", "peptide", "obj_chelation", "obj_solubility", "obj_stability", "obj_expression"]
    for c in required_mo:
        if c not in mo.columns:
            raise ValueError(f"MO file missing column: {c}")

    # ---- Validate SO columns ----
    if "iter" not in so.columns or "peptide" not in so.columns:
        raise ValueError("Single-objective CSV must contain columns: iter, peptide")

    # ---- Build FIXED reference batch from training CSV ----
    ref_candidates = build_reference_candidates_from_training(
        TRAIN_CSV_PATH,
        n_ref=2000,
        seed=0,
        seq_col="peptide_len10",
    )
    print(f"Built reference batch: {len(ref_candidates)} sequences from {TRAIN_CSV_PATH}")

    # ---- Prepare MO df ----
    mo_df = mo[required_mo].copy().sort_values("iter").reset_index(drop=True)
    mo_df[OBJ_COLS] = mo_df[OBJ_COLS].astype(float)

    # ---- Prepare SO df + compute objectives stably ----
    so_df = so[["iter", "peptide"]].copy().sort_values("iter").reset_index(drop=True)
    so_peps = so_df["peptide"].astype(str).str.strip().str.upper().tolist()
    Y_so = stable_objectives_batch(so_peps, ref_candidates=ref_candidates, binding_sites="")

    so_df["obj_chelation"] = Y_so[:, 0]
    so_df["obj_solubility"] = Y_so[:, 1]
    so_df["obj_stability"] = Y_so[:, 2]
    so_df["obj_expression"] = Y_so[:, 3]

    # ---- Shared ref_point for FAIR HV comparison ----
    Y_all = np.vstack([
        so_df[OBJ_COLS].to_numpy(dtype=float),
        mo_df[OBJ_COLS].to_numpy(dtype=float),
    ])
    ref_point = compute_ref_point(Y_all, margin=0.05)
    print("Shared ref_point:", ref_point)

    # ---- HV curves ----
    hv_so = hypervolume_vs_evals(so_df, obj_cols=OBJ_COLS, ref_point=ref_point)
    hv_mo = hypervolume_vs_evals(mo_df, obj_cols=OBJ_COLS, ref_point=ref_point)

    hv_so["method"] = "single_objective_BO (HV of ND set)"
    hv_mo["method"] = "MOBO_qEHVI (HV of ND set)"

    hv_all = pd.concat([hv_so, hv_mo], ignore_index=True)
    hv_all.to_csv(OUT_CSV, index=False)
    print(f"[saved] {OUT_CSV}")

    # ---- plot ----
    plt.figure()
    plt.plot(hv_so["n_evals"], hv_so["hv"], label="Single-objective BO")
    plt.plot(hv_mo["n_evals"], hv_mo["hv"], label="MOBO qEHVI")
    plt.xlabel("Number of evaluations")
    plt.ylabel("Dominated hypervolume")
    plt.title("Hypervolume vs Evaluations (shared reference point)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)
    plt.close()
    print(f"[saved] {OUT_PNG}")

    print("\nFinal HV:")
    print("  single-objective:", float(hv_so["hv"].iloc[-1]))
    print("  multi-objective :", float(hv_mo["hv"].iloc[-1]))

if __name__ == "__main__":
    main()
