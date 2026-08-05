import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

INPUT_DIR = "bo_decoder_monitoring_after_flow_space_gp_transfer_h32_z32_latent_conditioned_epoch199_balanced"
OUTPUT_DIR = "bo_decoder_monitoring_after_flow_space_gp_transfer_h32_z32_latent_conditioned_epoch199_balanced_plots"

# Diagram font sizes
AXIS_FONT_SIZE = 24
TITLE_FONT_SIZE = 24
LEGEND_FONT_SIZE = 24

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Helper functions
# ============================================================

def extract_bo_iter_from_filename(path: str) -> int:
    """
    Extract BO iteration number from file names like:
    accepted_decoder_diagnostics_scored_bo_iter_019.csv
    """
    basename = os.path.basename(path)
    match = re.search(r"bo_iter_(\d+)\.csv", basename)
    if match is None:
        raise ValueError(f"Could not extract BO iteration from filename: {basename}")
    return int(match.group(1))


def mean_token_probability_string(token_prob_string) -> float:
    """
    token_probabilities column format is expected to look like:
    0.123456|0.234567|...|0.987654

    Returns the average token probability for one decoded peptide.
    """
    if pd.isna(token_prob_string):
        return np.nan

    values = []
    for item in str(token_prob_string).split("|"):
        item = item.strip()
        if item:
            try:
                values.append(float(item))
            except ValueError:
                pass

    if len(values) == 0:
        return np.nan

    return float(np.mean(values))


# ============================================================
# Load all accepted decoder diagnostic CSV files
# ============================================================

pattern = os.path.join(INPUT_DIR, "accepted_decoder_diagnostics_scored_bo_iter_*.csv")
csv_files = sorted(glob.glob(pattern), key=extract_bo_iter_from_filename)

if len(csv_files) == 0:
    raise FileNotFoundError(
        f"No files found with pattern:\n{pattern}\n"
        "Please check INPUT_DIR and file names."
    )

print(f"Found {len(csv_files)} CSV files.")

summary_rows = []

for csv_path in csv_files:
    bo_iter = extract_bo_iter_from_filename(csv_path)
    df = pd.read_csv(csv_path)

    # Keep only rows accepted for black-box if the column exists.
    # These files are already accepted files, but this makes the code safer.
    if "accepted_for_blackbox" in df.columns:
        df = df[df["accepted_for_blackbox"].astype(str).str.lower().isin(["true", "1"])].copy()

    if len(df) == 0:
        print(f"[WARNING] BO iter {bo_iter:03d}: empty accepted file.")
        summary_rows.append({
            "bo_iter": bo_iter,
            "n_samples": 0,
            "avg_decoder_confidence_mean": np.nan,
            "avg_decoder_confidence_min": np.nan,
            "avg_token_probability_mean": np.nan,
            "avg_roundtrip_cosine": np.nan,
            "avg_roundtrip_l2": np.nan,
            "mean_chelation_sub": np.nan,
            "mean_solubility_sub": np.nan,
            "mean_stability_sub": np.nan,
            "mean_expression_sub": np.nan,
        })
        continue

    # Average token probability per decoded peptide, then average across peptides in this BO iteration.
    df["avg_token_probability"] = df["token_probabilities"].apply(mean_token_probability_string)

    summary_rows.append({
        "bo_iter": bo_iter,
        "n_samples": len(df),

        # Decoder confidence summaries
        "avg_decoder_confidence_mean": df["decoder_confidence_mean"].mean(),
        "avg_decoder_confidence_min": df["decoder_confidence_min"].mean(),
        "avg_token_probability_mean": df["avg_token_probability"].mean(),

        # Round-trip summaries
        "avg_roundtrip_cosine": df["roundtrip_cosine"].mean(),
        "avg_roundtrip_l2": df["roundtrip_l2"].mean(),

        # Objective summaries
        "mean_chelation_sub": df["chelation_sub"].mean(),
        "mean_solubility_sub": df["solubility_sub"].mean(),
        "mean_stability_sub": df["stability_sub"].mean(),
        "mean_expression_sub": df["expression_sub"].mean(),
    })

summary_df = pd.DataFrame(summary_rows).sort_values("bo_iter").reset_index(drop=True)

summary_csv_path = os.path.join(OUTPUT_DIR, "decoder_bo_iteration_summary.csv")
summary_df.to_csv(summary_csv_path, index=False)

print(f"Saved summary CSV to: {summary_csv_path}")


# ============================================================
# Plot 1:
# Mean and min decoder confidence for samples accepted by decoder
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    summary_df["bo_iter"],
    summary_df["avg_decoder_confidence_mean"],
    marker="o",
    label="Average mean decoder confidence",
)

plt.plot(
    summary_df["bo_iter"],
    summary_df["avg_decoder_confidence_min"],
    marker="o",
    label="Average min decoder confidence",
)

plt.plot(
    summary_df["bo_iter"],
    summary_df["avg_token_probability_mean"],
    marker="o",
    label="Average token probability",
)

plt.xlabel("BO iteration", fontsize=AXIS_FONT_SIZE)
plt.ylabel("Decoder confidence", fontsize=AXIS_FONT_SIZE)
plt.title("Mean and min decoder confidence for samples accepted by decoder", fontsize=TITLE_FONT_SIZE)
plt.legend(fontsize=LEGEND_FONT_SIZE)
plt.grid(True, alpha=0.3)
plt.tick_params(axis="both", labelsize=AXIS_FONT_SIZE)
plt.tight_layout()

plot_path = os.path.join(OUTPUT_DIR, "01_decoder_confidence_over_iterations.png")
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"Saved: {plot_path}")


# ============================================================
# Plot 2:
# Roundtrip cosine for samples accepted by decoder
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    summary_df["bo_iter"],
    summary_df["avg_roundtrip_cosine"],
    marker="o",
    label="Average roundtrip cosine",
)

plt.xlabel("BO iteration", fontsize=AXIS_FONT_SIZE)
plt.ylabel("Cosine similarity", fontsize=AXIS_FONT_SIZE)
plt.title("Roundtrip cosine for samples accepted by decoder", fontsize=TITLE_FONT_SIZE)
plt.legend(fontsize=LEGEND_FONT_SIZE)
plt.grid(True, alpha=0.3)
plt.tick_params(axis="both", labelsize=AXIS_FONT_SIZE)
plt.tight_layout()

plot_path = os.path.join(OUTPUT_DIR, "02_roundtrip_cosine_over_iterations.png")
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"Saved: {plot_path}")


# ============================================================
# Plot 3:
# Roundtrip L2 distance for samples accepted by decoder
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    summary_df["bo_iter"],
    summary_df["avg_roundtrip_l2"],
    marker="o",
    label="Average roundtrip L2 distance",
)

plt.xlabel("BO iteration", fontsize=AXIS_FONT_SIZE)
plt.ylabel("Roundtrip L2 distance", fontsize=AXIS_FONT_SIZE)
plt.title("Roundtrip L2 distance for samples accepted by decoder", fontsize=TITLE_FONT_SIZE)
plt.legend(fontsize=LEGEND_FONT_SIZE)
plt.grid(True, alpha=0.3)
plt.tick_params(axis="both", labelsize=AXIS_FONT_SIZE)
plt.tight_layout()

plot_path = os.path.join(OUTPUT_DIR, "03_roundtrip_l2_over_iterations.png")
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"Saved: {plot_path}")


# ============================================================
# Plot 4:
# Mean objective values for samples accepted by decoder
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    summary_df["bo_iter"],
    summary_df["mean_chelation_sub"],
    marker="o",
    label="Mean chelation_sub",
)

plt.plot(
    summary_df["bo_iter"],
    summary_df["mean_solubility_sub"],
    marker="o",
    label="Mean solubility_sub",
)

plt.plot(
    summary_df["bo_iter"],
    summary_df["mean_stability_sub"],
    marker="o",
    label="Mean stability_sub",
)

plt.plot(
    summary_df["bo_iter"],
    summary_df["mean_expression_sub"],
    marker="o",
    label="Mean expression_sub",
)

plt.xlabel("BO iteration", fontsize=AXIS_FONT_SIZE)
plt.ylabel("Mean objective value", fontsize=AXIS_FONT_SIZE)
plt.title("Mean objective values for samples accepted by decoder", fontsize=TITLE_FONT_SIZE)
plt.legend(fontsize=LEGEND_FONT_SIZE)
plt.grid(True, alpha=0.3)
plt.tick_params(axis="both", labelsize=AXIS_FONT_SIZE)
plt.tight_layout()

plot_path = os.path.join(OUTPUT_DIR, "04_mean_objective_values_over_iterations.png")
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"Saved: {plot_path}")


# ============================================================
# Optional: print quick summary
# ============================================================

print("\nFirst few summary rows:")
print(summary_df.head())

print("\nDone.")