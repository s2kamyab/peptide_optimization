"""
Reproduce Tables 2-1, 2-2, 2-3, 2-4 (mean +/- SD across 3 independently
trained checkpoints) from the paper "Comparative Analysis of Latent
Generative Bayesian Optimization Frameworks for Multi-Objective Peptide
Design".

WHAT THIS DOES
--------------
1. Runs `audit_ddim_invertibility_gru_vae_latent_diffusion.py` once per
   checkpoint (3 runs total), producing an `out_runN/` directory each
   containing `audit_report.json`, `step_convergence_summary.csv`,
   `sequence_reconstruction_comparison.csv`, and `sphere_projection_audit.csv`.
2. Loads those outputs back in and computes mean +/- sample standard
   deviation (ddof=1, n=3) for every number that appears in Tables 2-1
   through 2-4.
3. Prints the four tables to stdout in the same layout as the paper.

REQUIREMENTS
------------
- Python 3.10+, torch (CPU is fine), pandas, numpy.
- The audit script `audit_ddim_invertibility_gru_vae_latent_diffusion.py`
  must be in the same directory as this script (or edit AUDIT_SCRIPT below).
- The 3 finetuned checkpoints, their coordinate CSVs, and the finetune
  script must be available and paths below updated to match your layout.

USAGE
-----
    python reproduce_tables_2_1_to_2_4.py

Edit the CONFIG block below to point at your own file paths before running.
Set SKIP_AUDIT_RUNS = True to skip re-running the audit script and just
recompute the tables from existing out_run1/out_run2/out_run3 directories.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# CONFIG -- edit these paths to match your environment
# ----------------------------------------------------------------------
AUDIT_SCRIPT = "audit_ddim_invertibility_gru_vae_latent_diffusion.py"
FINETUNE_SCRIPT = "finetune_best_bo_ready_gru_vae_cu_latent_diffusion_leakage_safe_v3_objective_gated.py"

# One entry per independently trained run.
RUNS = {
    1: dict(
        checkpoint="best_val_objective_mse_h64_z64_cu_latent_diffusion_run1.pt",
        coordinate_csv="cu_ddim_inversion_coordinates_for_bo_run1.csv",
        out_dir="./out_run1",
    ),
    2: dict(
        checkpoint="best_val_objective_mse_h64_z64_cu_latent_diffusion_run2.pt",
        coordinate_csv="cu_ddim_inversion_coordinates_for_bo_run2.csv",
        out_dir="./out_run2",
    ),
    3: dict(
        checkpoint="best_val_objective_mse_h64_z64_cu_latent_diffusion_run3.pt",
        coordinate_csv="cu_ddim_inversion_coordinates_for_bo_run3.csv",
        out_dir="./out_run3",
    ),
}

SPLITS = ["val", "test"]
MAX_SAMPLES = 512
STEP_COUNTS = [5, 10, 20, 30, 50, 75, 100]
DEVICE = "cpu"  # use "cuda" if you have a GPU available

SKIP_AUDIT_RUNS = False  # set True to only recompute tables from existing out_runN/


# ----------------------------------------------------------------------
# Step 1: run the audit script for each of the 3 checkpoints
# ----------------------------------------------------------------------
def run_audits():
    for run_id, cfg in RUNS.items():
        print(f"\n=== Running audit for run {run_id} ===")
        cmd = [
            sys.executable, AUDIT_SCRIPT,
            "--finetune-script", FINETUNE_SCRIPT,
            "--checkpoint", cfg["checkpoint"],
            "--coordinate-csv", cfg["coordinate_csv"],
            "--out-dir", cfg["out_dir"],
            "--splits", *SPLITS,
            "--max-samples", str(MAX_SAMPLES),
            "--step-counts", *[str(s) for s in STEP_COUNTS],
            "--device", DEVICE,
        ]
        subprocess.run(cmd, check=True)


# ----------------------------------------------------------------------
# Step 2: helpers
# ----------------------------------------------------------------------
def mean_std(values, ddof=1):
    arr = np.array(values, dtype=float)
    return arr.mean(), arr.std(ddof=ddof)


def fmt(mean, std, decimals):
    return f"{mean:.{decimals}f} \u00b1 {std:.{decimals}f}"


# ----------------------------------------------------------------------
# Step 3: Table 2-1 -- DDIM round-trip convergence across step counts
# ----------------------------------------------------------------------
def build_table_2_1():
    dfs = [pd.read_csv(Path(RUNS[i]["out_dir"]) / "step_convergence_summary.csv") for i in [1, 2, 3]]
    specs = [
        ("h_roundtrip_l2_mean", "Mean latent L2", 5),
        ("h_roundtrip_rmse_mean", "Mean per-dim RMSE", 5),
        ("h_roundtrip_cosine_mean", "Mean cosine", 6),
        ("epsilon_roundtrip_l2_mean", "Mean epsilon round-trip L2", 5),
    ]
    steps = dfs[0]["ddim_steps"].tolist()
    print("\n" + "=" * 80)
    print("Table 2-1. DDIM round-trip convergence across inference step counts.")
    print("=" * 80)
    header = ["DDIM steps"] + [label for _, label, _ in specs]
    print(" | ".join(header))
    for i, n in enumerate(steps):
        row = [str(n)]
        for col, _, dec in specs:
            vals = [df.loc[i, col] for df in dfs]
            m, s = mean_std(vals)
            row.append(fmt(m, s, dec))
        print(" | ".join(row))


# ----------------------------------------------------------------------
# Step 4: Table 2-2 -- algebraic reversibility / saved-coordinate diagnostics
# ----------------------------------------------------------------------
def build_table_2_2():
    reports = [json.load(open(Path(RUNS[i]["out_dir"]) / "audit_report.json")) for i in [1, 2, 3]]
    print("\n" + "=" * 80)
    print("Table 2-2. Algebraic reversibility and saved-coordinate consistency diagnostics.")
    print("=" * 80)

    all_true = all(r["schedule_all_exact_reverses"] for r in reports)
    print(f"Timestep grids exact reverses: {'Yes' if all_true else 'No (see per-run values)'}")

    fields = [
        ("one_step_same_epsilon_algebraic_reverse_l2_mean", "Same-epsilon one-step algebraic reverse, mean L2"),
        ("one_step_model_recomputed_reverse_l2_mean", "Model-recomputed one-step reverse, mean L2"),
        ("saved_coordinate_recompute_epsilon_l2_mean", "Saved vs recomputed epsilon, mean L2"),
        ("saved_coordinate_h_roundtrip_l2_mean", "Saved epsilon -> h0, mean L2"),
    ]
    for key, label in fields:
        vals = [r[key] for r in reports]
        m, s = mean_std(vals)
        print(f"{label}: {m:.6e} \u00b1 {s:.6e}")


# ----------------------------------------------------------------------
# Step 5: Table 2-3 -- VAE-only vs DDIM round-trip peptide reconstruction
# ----------------------------------------------------------------------
def build_table_2_3():
    dfs = [pd.read_csv(Path(RUNS[i]["out_dir"]) / "sequence_reconstruction_comparison.csv") for i in [1, 2, 3]]
    print("\n" + "=" * 80)
    print("Table 2-3. VAE-only versus DDIM round-trip peptide reconstruction comparison.")
    print("=" * 80)
    print("DDIM steps | DDIM better | Same | DDIM worse | Mean VAE edit | Mean DDIM edit")
    for s in STEP_COUNTS:
        better, same, worse, vae_edit, ddim_edit = [], [], [], [], []
        for df in dfs:
            sub = df[df["ddim_steps"] == s]
            better.append((sub["outcome_vs_vae"] == "ddim_better").sum())
            same.append((sub["outcome_vs_vae"] == "same").sum())
            worse.append((sub["outcome_vs_vae"] == "ddim_worse").sum())
            vae_edit.append(sub["vae_edit"].mean())
            ddim_edit.append(sub["ddim_edit"].mean())
        b_m, b_s = mean_std(better)
        s_m, s_s = mean_std(same)
        w_m, w_s = mean_std(worse)
        v_m, v_s = mean_std(vae_edit)
        d_m, d_s = mean_std(ddim_edit)
        print(f"{s} | {fmt(b_m,b_s,2)} | {fmt(s_m,s_s,2)} | {fmt(w_m,w_s,2)} | {fmt(v_m,v_s,5)} | {fmt(d_m,d_s,5)}")


# ----------------------------------------------------------------------
# Step 6: Table 2-4 -- effect of spherical projection on 50-step round-trip
# ----------------------------------------------------------------------
def build_table_2_4():
    reports = [json.load(open(Path(RUNS[i]["out_dir"]) / "audit_report.json")) for i in [1, 2, 3]]
    print("\n" + "=" * 80)
    print("Table 2-4. Effect of spherical projection on 50-step DDIM round-trip fidelity.")
    print("=" * 80)
    fields = [
        ("sphere_native_h_l2_mean", "Native 50-step h0 round-trip L2"),
        ("sphere_projected_h_l2_mean", "Sphere-projected 50-step h0 round-trip L2"),
        ("sphere_projection_penalty_mean", "Additional L2 error from sphere projection"),
    ]
    for key, label in fields:
        vals = [r[key] for r in reports]
        m, s = mean_std(vals)
        print(f"{label}: {m:.5f} \u00b1 {s:.5f}")


if __name__ == "__main__":
    if not SKIP_AUDIT_RUNS:
        run_audits()
    build_table_2_1()
    build_table_2_2()
    build_table_2_3()
    build_table_2_4()
