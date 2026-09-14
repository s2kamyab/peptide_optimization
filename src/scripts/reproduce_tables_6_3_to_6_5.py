"""
Reproduce Tables 6-3, 6-4, and 6-5 (native DDIM epsilon reference
distribution, candidate trustworthiness screening, and Pareto-improving
candidate retention after trust filtering -- including the per-peptide
breakdown of any rejected Pareto candidates), mean +/- SD across 3
independently trained finetuned diffusion checkpoints and their
corresponding prospective-BO runs.

WHAT THIS DOES
--------------
1. Runs `audit_latent_diffusion_bo_trustworthiness_standard_normal.py` once
   per run (3 runs total), each needing:
     - the finetune script (shared across runs)
     - that run's finetuned diffusion checkpoint (.pt)
     - that run's DDIM coordinate CSV (must contain epsilon_raw_* columns
       and a `split` column with train/val/test labels)
     - that run's BO-accepted-candidates CSV (--bo-candidates)
     - that run's BO Pareto-front CSV (--bo-pareto), optional but needed
       for Table 6-5
2. Loads each run's `reference_standard_normal_summary.csv`,
   `candidate_trust_summary.csv`, and `trustworthiness_audit_summary.json`
   and computes mean +/- sample standard deviation (ddof=1, n=3) for every
   number in Tables 6-3, 6-4, and 6-5.
3. Prints all three tables to stdout.

IMPORTANT CAVEATS (see interpretation notes printed at the end)
-----------------------------------------------------------------
- The total number of BO-accepted candidates differs across runs (this is
  itself a real result, not noise) -- Table 6-4's raw pass/fail counts
  partly reflect that varying N, so the pass-RATE columns are the fairer
  quantity to summarize with +/- SD.
- Table 6-5 is computed over a much smaller sample (roughly 19-24 Pareto
  candidates per run) than Table 6-4's full candidate pool, so its SD is
  inherently noisier.

USAGE
-----
    python reproduce_tables_6_3_to_6_5.py

Edit the CONFIG block below to match your file layout. Set
SKIP_AUDIT_RUNS = True to just recompute the tables from existing
trust_run1/2/3 directories.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# CONFIG -- edit these paths to match your environment
# ----------------------------------------------------------------------
AUDIT_SCRIPT = "audit_latent_diffusion_bo_trustworthiness_standard_normal.py"
FINETUNE_SCRIPT = "finetune_best_bo_ready_gru_vae_cu_latent_diffusion_leakage_safe_v3_objective_gated.py"

RUNS = {
    1: dict(
        checkpoint="best_val_objective_mse_h64_z64_cu_latent_diffusion_run1.pt",
        coordinate_csv="cu_ddim_inversion_coordinates_for_bo_run1.csv",
        bo_candidates="all_accepted_candidates_scored_run1.csv",
        bo_pareto="bo_pareto_dominates_train_CU_gru_vae_diffusion_epsilon_qlogehvi_leakage_safe_run1.csv",
        out_dir="./trust_run1",
    ),
    2: dict(
        checkpoint="best_val_objective_mse_h64_z64_cu_latent_diffusion_run2.pt",
        coordinate_csv="cu_ddim_inversion_coordinates_for_bo_run2.csv",
        bo_candidates="all_accepted_candidates_scored_run2.csv",
        bo_pareto="bo_pareto_dominates_train_CU_gru_vae_diffusion_epsilon_qlogehvi_leakage_safe_run2.csv",
        out_dir="./trust_run2",
    ),
    3: dict(
        checkpoint="best_val_objective_mse_h64_z64_cu_latent_diffusion_run3.pt",
        coordinate_csv="cu_ddim_inversion_coordinates_for_bo_run3.csv",
        bo_candidates="all_accepted_candidates_scored_run3.csv",
        bo_pareto="bo_pareto_dominates_train_CU_gru_vae_diffusion_epsilon_qlogehvi_leakage_safe_run3.csv",
        out_dir="./trust_run3",
    ),
}

DEVICE = "cpu"  # use "cuda" if you have a GPU available

SKIP_AUDIT_RUNS = False  # set True to only recompute tables from existing trust_runN/ dirs


# ----------------------------------------------------------------------
# Step 1: run the audit script for each of the 3 checkpoints/BO runs
# ----------------------------------------------------------------------
def run_audits():
    for run_id, cfg in RUNS.items():
        print(f"\n=== Running trustworthiness audit for run {run_id} ===")
        cmd = [
            sys.executable, AUDIT_SCRIPT,
            "--finetune-script", FINETUNE_SCRIPT,
            "--checkpoint", cfg["checkpoint"],
            "--coordinate-csv", cfg["coordinate_csv"],
            "--bo-candidates", cfg["bo_candidates"],
            "--bo-pareto", cfg["bo_pareto"],
            "--out-dir", cfg["out_dir"],
            "--device", DEVICE,
        ]
        subprocess.run(cmd, check=True)


# ----------------------------------------------------------------------
# Step 2: Table 6-3 -- native DDIM epsilon reference distribution
# ----------------------------------------------------------------------
def build_table_6_3():
    dfs = [pd.read_csv(Path(RUNS[i]["out_dir"]) / "reference_standard_normal_summary.csv").set_index("group") for i in [1, 2, 3]]

    print("\n" + "=" * 90)
    print("Table 6-3. Native DDIM epsilon distribution in the training and validation/test reference sets.")
    print("=" * 90)
    print(f"{'Reference group':16s} | {'n':14s} | {'Mean epsilon norm':20s} | {'Mean coord SD':20s} | {'Inside 0.5-99.5% shell':22s}")
    for grp, label in [("train", "Training"), ("val_test", "Validation/test")]:
        n_vals = np.array([df.loc[grp, "n"] for df in dfs], dtype=float)
        norm_vals = np.array([df.loc[grp, "epsilon_norm_mean"] for df in dfs])
        sd_vals = np.array([df.loc[grp, "coordinate_std_mean"] for df in dfs])
        shell_vals = np.array([df.loc[grp, "fraction_inside_0.5_99.5pct_radial_shell"] for df in dfs]) * 100
        print(
            f"{label:16s} | {f'{n_vals.mean():.1f} \u00b1 {n_vals.std(ddof=1):.1f}':14s} | "
            f"{f'{norm_vals.mean():.3f} \u00b1 {norm_vals.std(ddof=1):.3f}':20s} | "
            f"{f'{sd_vals.mean():.3f} \u00b1 {sd_vals.std(ddof=1):.3f}':20s} | "
            f"{f'{shell_vals.mean():.1f}% \u00b1 {shell_vals.std(ddof=1):.1f}%':22s}"
        )


# ----------------------------------------------------------------------
# Step 3: Table 6-4 -- candidate trustworthiness screening results
# ----------------------------------------------------------------------
CRITERIA = [
    ("passes_standard_normal_radial", "Standard-normal radial check"),
    ("passes_empirical_mahalanobis", "Empirical Mahalanobis check"),
    ("passes_nearest_train_distance", "Nearest-training-epsilon distance"),
    ("passes_roundtrip_fidelity", "DDIM round-trip fidelity"),
    ("passes_all_trust_filters", "All trust filters"),
]


def build_table_6_4():
    dfs = [pd.read_csv(Path(RUNS[i]["out_dir"]) / "candidate_trust_summary.csv").set_index("criterion") for i in [1, 2, 3]]
    n_cands = [int(dfs[i]["n_pass"].iloc[0] + dfs[i]["n_fail"].iloc[0]) for i in range(3)]

    print("\n" + "=" * 90)
    print("Table 6-4. Candidate trustworthiness screening results.")
    print("=" * 90)
    print(f"N candidates per run: {n_cands}  (mean={np.mean(n_cands):.1f}, sd={np.std(n_cands, ddof=1):.1f})")
    print("NOTE: N differs across runs -- pass RATE is the fairer +/- SD quantity, see caveat in module docstring.")
    print(f"{'Trust criterion':38s} | {'Pass':16s} | {'Fail':16s} | {'Pass rate':16s}")
    for key, label in CRITERIA:
        pass_vals = np.array([df.loc[key, "n_pass"] for df in dfs], dtype=float)
        fail_vals = np.array([df.loc[key, "n_fail"] for df in dfs], dtype=float)
        rate_vals = np.array([df.loc[key, "fraction_pass"] for df in dfs]) * 100
        print(
            f"{label:38s} | {f'{pass_vals.mean():.1f} \u00b1 {pass_vals.std(ddof=1):.1f}':16s} | "
            f"{f'{fail_vals.mean():.1f} \u00b1 {fail_vals.std(ddof=1):.1f}':16s} | "
            f"{f'{rate_vals.mean():.1f}% \u00b1 {rate_vals.std(ddof=1):.1f}%':16s}"
        )


# ----------------------------------------------------------------------
# Step 4: Table 6-5 -- Pareto-improving candidate retention after trust
# filtering, plus the per-peptide breakdown of any rejected Pareto
# candidates (this is the actual published Table 6-5 structure).
# ----------------------------------------------------------------------
def build_table_6_5():
    import json
    summaries = [json.load(open(Path(RUNS[i]["out_dir"]) / "trustworthiness_audit_summary.json")) for i in [1, 2, 3]]

    n_input = np.array([s["pareto_post_filter"]["n_input"] for s in summaries], dtype=float)
    n_trusted = np.array([s["pareto_post_filter"]["n_trusted"] for s in summaries], dtype=float)
    n_rejected = n_input - n_trusted
    frac = n_trusted / n_input * 100

    print("\n" + "=" * 90)
    print("Table 6-5. Pareto-improving candidate retention after trust filtering.")
    print("=" * 90)
    print("NOTE: much smaller sample (~19-24 per run) than Table 6-4's full candidate pool -- expect noisier SD.")
    print(f"{'Metric':45s} | " + " | ".join(f"Run {i}" for i in [1, 2, 3]) + " | Mean +/- SD")
    print(
        f"{'Pareto-improving candidates before filtering':45s} | "
        + " | ".join(f"{int(v):5d}" for v in n_input)
        + f" | {n_input.mean():.1f} \u00b1 {n_input.std(ddof=1):.1f}"
    )
    print(
        f"{'Pareto-improving candidates retained':45s} | "
        + " | ".join(f"{int(v):5d}" for v in n_trusted)
        + f" | {n_trusted.mean():.1f} \u00b1 {n_trusted.std(ddof=1):.1f}"
    )
    print(
        f"{'Pareto-improving candidates rejected':45s} | "
        + " | ".join(f"{int(v):5d}" for v in n_rejected)
        + f" | {n_rejected.mean():.1f} \u00b1 {n_rejected.std(ddof=1):.1f}"
    )
    print(
        f"{'Retention rate':45s} | "
        + " | ".join(f"{v:4.1f}%" for v in frac)
        + f" | {frac.mean():.1f}% \u00b1 {frac.std(ddof=1):.1f}%"
    )

    print("\nNOTE: rejections may not be evenly distributed across runs (e.g. all in one run) --")
    print("check the per-run counts above rather than reading the mean+/-SD as a smooth rate.")

    print("\nRejected Pareto peptides, per run (only shown for runs with >=1 rejection):")
    pep_cols = [
        "peptide", "dominates_train_pareto", "epsilon_norm", "standard_normal_chi2_percentile",
        "passes_standard_normal_radial", "passes_empirical_mahalanobis",
        "passes_nearest_train_distance", "passes_roundtrip_fidelity",
    ]
    for i in [1, 2, 3]:
        df = pd.read_csv(Path(RUNS[i]["out_dir"]) / "bo_pareto_with_trust_metrics.csv")
        rejected = df[df["passes_all_trust_filters"] == False]
        if len(rejected) == 0:
            print(f"  run{i}: no rejected Pareto-improving candidates.")
            continue
        print(f"  run{i}:")
        reasons = []
        for _, row in rejected.iterrows():
            failed = [c.replace("passes_", "") for c in pep_cols[4:] if not row[c]]
            reasons.append(", ".join(f"fails {f}" for f in failed) if failed else "unknown")
        out = rejected[pep_cols].copy()
        out["reason"] = reasons
        print(out.to_string(index=False))


if __name__ == "__main__":
    if not SKIP_AUDIT_RUNS:
        run_audits()
    build_table_6_3()
    build_table_6_4()
    build_table_6_5()
