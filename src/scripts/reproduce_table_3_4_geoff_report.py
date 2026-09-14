"""
Reproduce Table 3-4 (Latent-space health metrics, mean +/- SD across 3
independently pretrained GRU-VAE checkpoints).

WHY THIS WORKS WITHOUT RE-RUNNING ANYTHING
-------------------------------------------
`pretrain_gru_vae_bo_ready_h64_z64_leakage_safe_decorrelated.py` logs a full
validation-set geometry snapshot (mu magnitude, KL/dim, participation-ratio
dimensionality, PCA variance coverage, off-diagonal correlation, etc.) into
`metrics` every time it writes a checkpoint. The *selected* checkpoint
(chosen by the script's own model-selection criterion) therefore already
contains, verbatim, every number Table 3-4 needs -- no forward passes
required.

USAGE
-----
    python reproduce_table_3_4.py \
        best_bo_ready_gru_vae_latent_conditioned_h64_z64_run1.pt \
        best_bo_ready_gru_vae_latent_conditioned_h64_z64_run2.pt \
        best_bo_ready_gru_vae_latent_conditioned_h64_z64_run3.pt

If no arguments are given, it looks for the three default filenames in the
current directory.
"""

import sys

import numpy as np
import torch

FIELDS = [
    ("val_mu_abs_mean", "Mean absolute (mu)", 4),
    ("val_mu_std", "Overall (mu) std", 4),
    ("val_kl_per_dim_mean", "KL per dimension", 4),
    ("val_effective_dim_pr", "Raw participation-ratio dimension", 2),
    ("val_effective_dim_pr_standardized", "Standardized PR dimension", 2),
    ("val_pc1_variance_fraction", "PC1 variance fraction", 4),
    ("val_pcs90", "PCs for 90% variance", 2),
    ("val_pcs95", "PCs for 95% variance", 2),
    ("val_mean_abs_offdiag_corr", "Mean absolute off-diagonal correlation", 4),
    ("val_max_abs_offdiag_corr", "Maximum absolute correlation", 4),
    ("val_geometry_n_samples", "Validation geometry samples", 0),
]

DEFAULT_CHECKPOINTS = [
    "best_bo_ready_gru_vae_latent_conditioned_h64_z64_run1.pt",
    "best_bo_ready_gru_vae_latent_conditioned_h64_z64_run2.pt",
    "best_bo_ready_gru_vae_latent_conditioned_h64_z64_run3.pt",
]


def main(checkpoint_paths):
    epochs = []
    values = {key: [] for key, _, _ in FIELDS}

    for path in checkpoint_paths:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        epochs.append(ckpt["epoch"])
        metrics = ckpt["metrics"]
        for key, _, _ in FIELDS:
            values[key].append(metrics[key])

    print(f"Selected checkpoint epochs: {epochs}")
    print()
    print("Table 3-4. Latent-space health metrics (mean +/- SD, n=%d)." % len(checkpoint_paths))
    print("-" * 70)
    for key, label, decimals in FIELDS:
        arr = np.array(values[key], dtype=float)
        mean = arr.mean()
        std = arr.std(ddof=1) if len(arr) > 1 else 0.0
        print(f"{label:45s}: {mean:.{decimals}f} \u00b1 {std:.{decimals}f}")


if __name__ == "__main__":
    paths = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_CHECKPOINTS
    main(paths)
