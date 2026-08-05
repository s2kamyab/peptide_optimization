# ============================================================
# Plotting part only: font size 24
# Uses: scrambling_control_summary.csv
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Use your generated summary CSV
# summary_path = Path("scrambling_control_epoch199_best_roundtrip_latent_conditioned_flow_space_gp\scrambling_control_summary.csv")
# summary_path = Path("scrambling_control_cu_latent_diffusion_last_epoch/scrambling_control_summary.csv")
summary_path = Path("2026_06_24_Geoff_results\scrambling_control_CU_gp_after_flow_wu_10/scrambling_control_summary.csv")
# Or, for the attached file name:
# summary_path = Path("scrambling_control_summary(1).csv")

summary_df = pd.read_csv(summary_path)

# Output directory
out_dir = Path("scrambling_control_plots_font24")
out_dir.mkdir(parents=True, exist_ok=True)

# Global font settings
plt.rcParams.update({
    "font.size": 34,
    "axes.titlesize": 34,
    "axes.labelsize": 34,
    "xtick.labelsize": 34,
    "ytick.labelsize": 34,
    "legend.fontsize": 34,
    "figure.titlesize": 34,
})


# ============================================================
# Plot 1: original vs scrambled mean final score
# ============================================================

plt.figure(figsize=(18, 9))

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
    capsize=8,
    label="Scrambled controls mean ± SD",
)

plt.xticks(
    x,
    summary_df["original_peptide"],
    rotation=45,
    ha="right",
)

plt.ylabel("Final score")
plt.title("Original optimized peptides vs scrambled controls")
plt.legend(frameon=False)
plt.tight_layout()

plot_path = out_dir / "original_vs_scrambled_mean_final_score_font24.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved plot: {plot_path}")


# ============================================================
# Plot 2: objective sub-scores, same style as your screenshot
# ============================================================

objective_cols = [
    "chelation_sub",
    "solubility_sub",
    "stability_sub",
    "expression_sub",
]

objective_labels = [
    "chelation",
    "solubility",
    "stability",
    "expression",
]

original_cols = [f"original_{c}" for c in objective_cols]
scrambled_mean_cols = [f"scrambled_mean_{c}" for c in objective_cols]
scrambled_std_cols = [f"scrambled_std_{c}" for c in objective_cols]

original_means = summary_df[original_cols].mean().to_numpy(dtype=float)
scrambled_means = summary_df[scrambled_mean_cols].mean().to_numpy(dtype=float)
scrambled_stds = summary_df[scrambled_std_cols].mean().to_numpy(dtype=float)

plt.figure(figsize=(16, 9))

x = np.arange(len(objective_cols))
width = 0.35

plt.bar(
    x - width / 2,
    original_means,
    width,
    label="Original optimized peptide",
)

plt.bar(
    x + width / 2,
    scrambled_means,
    width,
    yerr=scrambled_stds,
    capsize=8,
    label="Scrambled control, n = 100",
)

plt.xticks(
    x,
    objective_labels,
    rotation=45,
    ha="right",
)

plt.ylabel("Sub-score")
plt.title("Original optimized peptides vs scrambled controls")
plt.ylim(0.0, 1.05)
plt.legend(frameon=False)
plt.tight_layout()

plot_path = out_dir / "objective_subscores_original_vs_scrambled_font24.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved plot: {plot_path}")


# ============================================================
# Plot 3: empirical p-values
# ============================================================

plt.figure(figsize=(18, 8))

plt.bar(
    summary_df["original_peptide"],
    summary_df["empirical_p_scrambled_ge_original_final_score"],
)

plt.axhline(
    0.05,
    linestyle="--",
    linewidth=2,
    label="p = 0.05",
)

plt.xticks(rotation=45, ha="right")
plt.ylabel("Empirical p-value")
plt.title("Probability that scrambled controls score at least as high as original")
plt.legend(frameon=False)
plt.tight_layout()

plot_path = out_dir / "empirical_p_values_final_score_font24.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved plot: {plot_path}")