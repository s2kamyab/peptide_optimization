"""
Reproduce Table 4-1 (Comparison of Bayesian-optimization candidate spaces
derived from the GRU-VAE + latent-diffusion framework), mean +/- SD across
3 independently trained finetuned diffusion checkpoints.

WHAT THIS DOES
--------------
1. Runs `compare_bo_candidate_spaces_gruvae_diffusion_fixed_v2.py` once per
   run's coordinate CSV (3 runs total). No .pt checkpoint is required for
   this table -- the script derives all three candidate spaces
   (encoder_mu, encoder_h0, diffusion_epsilon) directly from the
   mu_*, h0_standardized_*, and epsilon_* columns already present in the
   coordinate CSV. (The script does accept an optional --checkpoint to
   additionally derive decoder-side spaces, but Table 4-1 does not use
   those columns, so it is intentionally omitted here.)
2. Loads each run's `bo_space_comparison_summary_ranked.csv` and computes
   mean +/- sample standard deviation (ddof=1, n=3) for every metric in
   Table 4-1.
3. Prints the table to stdout.

RUNTIME WARNING
---------------
This is the slowest audit in the paper's reproducibility pipeline. On a
single CPU core, each run takes roughly 10-15 minutes: the GP-quality
section fits a Gaussian Process per (space x train_size x seed x objective)
combination (up to n=1024), and the offline ParEGO section runs ~68
sequential GP-fit-and-acquire iterations per (space x seed). Consider
running the three invocations in parallel background processes if you have
multiple cores, or simply be patient on a single core.

USAGE
-----
    python reproduce_table_4_1.py

Edit the CONFIG block below to match your file layout. Set
SKIP_COMPARISON_RUNS = True to just recompute the table from existing
out_run1/out_run2/out_run3 directories.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# CONFIG -- edit these paths to match your environment
# ----------------------------------------------------------------------
COMPARISON_SCRIPT = "compare_bo_candidate_spaces_gruvae_diffusion_fixed_v2.py"

RUNS = {
    1: dict(coordinate_csv="cu_ddim_inversion_coordinates_for_bo_run1.csv", out_dir="./bo_compare_run1"),
    2: dict(coordinate_csv="cu_ddim_inversion_coordinates_for_bo_run2.csv", out_dir="./bo_compare_run2"),
    3: dict(coordinate_csv="cu_ddim_inversion_coordinates_for_bo_run3.csv", out_dir="./bo_compare_run3"),
}

OBJECTIVE_COLS = ["chelation_sub", "solubility_sub", "stability_sub", "expression_sub"]
GP_TRAIN_SIZES = [64, 128, 256, 512, 1024]
GP_SEEDS = [11, 22, 33]
BO_INIT_SIZE = 32
BO_BUDGET = 100
BO_SEEDS = [101, 202, 303]
DEVICE = "cpu"  # this script doesn't actually use torch/GPU compute, cpu is fine

SKIP_COMPARISON_RUNS = False  # set True to only recompute Table 4-1 from existing out dirs


# ----------------------------------------------------------------------
# Step 1: run the comparison script for each of the 3 runs
# ----------------------------------------------------------------------
def run_comparisons():
    for run_id, cfg in RUNS.items():
        print(f"\n=== Running BO candidate-space comparison for run {run_id} ===")
        print("(this can take 10-15 minutes per run on a single CPU core)")
        cmd = [
            sys.executable, COMPARISON_SCRIPT,
            "--coordinate-csv", cfg["coordinate_csv"],
            "--objective-cols", *OBJECTIVE_COLS,
            "--gp-train-sizes", *[str(s) for s in GP_TRAIN_SIZES],
            "--gp-seeds", *[str(s) for s in GP_SEEDS],
            "--bo-init-size", str(BO_INIT_SIZE),
            "--bo-budget", str(BO_BUDGET),
            "--bo-seeds", *[str(s) for s in BO_SEEDS],
            "--out-dir", cfg["out_dir"],
            "--device", DEVICE,
        ]
        subprocess.run(cmd, check=True)


# ----------------------------------------------------------------------
# Step 2: aggregate mean +/- SD across the 3 runs into Table 4-1
# ----------------------------------------------------------------------
METRICS = [
    ("effective_dimension_participation_ratio", "Effective dimension", 2),
    ("pca_components_95pct", "PCs for 95% variance", 2),
    ("knn_objective_l2_mean", "kNN objective L2", 4),
    ("pair_distance_objective_spearman", "Pair-distance/objective Spearman", 4),
    ("gp_rmse", "GP RMSE @1024", 4),
    ("gp_r2", "GP (R^2) @1024", 4),
    ("gp_spearman", "GP Spearman @1024", 4),
    ("final_hypervolume", "Final hypervolume", 4),
    ("hypervolume_auc", "Hypervolume AUC", 2),
    ("mean_rank", "Composite rank", 2),
]

SPACES = ["diffusion_epsilon", "encoder_mu", "encoder_h0"]
SPACE_LABELS = {"diffusion_epsilon": "Diffusion (epsilon)", "encoder_mu": "Encoder (mu)", "encoder_h0": "Encoder (h_0)"}


def build_table_4_1():
    dfs = [
        pd.read_csv(Path(RUNS[i]["out_dir"]) / "bo_space_comparison_summary_ranked.csv").set_index("space")
        for i in [1, 2, 3]
    ]

    print("\n" + "=" * 100)
    print("Table 4-1. Comparison of Bayesian-optimization candidate spaces (mean +/- SD, n=3).")
    print("=" * 100)
    header = f"{'Metric':38s} | " + " | ".join(f"{SPACE_LABELS[s]:28s}" for s in SPACES)
    print(header)
    for col, label, dec in METRICS:
        cells = []
        for sp in SPACES:
            vals = np.array([df.loc[sp, col] for df in dfs], dtype=float)
            mean, std = vals.mean(), vals.std(ddof=1)
            cells.append(f"{mean:.{dec}f} \u00b1 {std:.{dec}f}")
        print(f"{label:38s} | " + " | ".join(f"{c:28s}" for c in cells))


if __name__ == "__main__":
    if not SKIP_COMPARISON_RUNS:
        run_comparisons()
    build_table_4_1()
