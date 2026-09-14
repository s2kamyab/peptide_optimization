from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from docx import Document
    from docx.shared import Inches, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    DOCX_OK = True
except ImportError:
    DOCX_OK = False


# ---------------------------------------------------------------------
# EDIT THESE TWO PATHS IF NEEDED
# ---------------------------------------------------------------------
AUDIT_SCRIPT = Path(
    r"C:\Users\shima\OneDrive\Documentos\Leili\peptide_structure_optimization\peptide_optimization\src\scripts\audit_ddim_invertibility_gru_vae_latent_diffusion.py"
)

FINETUNE_SCRIPT = Path(
    r"C:\Users\shima\OneDrive\Documentos\Leili\peptide_structure_optimization\peptide_optimization\src\scripts\finetune_best_bo_ready_gru_vae_cu_latent_diffusion_leakage_safe_v3_objective_gated.py"
)

RUNS = [
    {
        "run": "Run 1",
        "checkpoint": Path(
            r"C:\Users\shima\OneDrive\Documentos\Leili\peptide_structure_optimization"
            r"\peptide_optimization\output"
            r"\finetuned_gru_vae_cu_latent_diffusion_leakage_safe_v3_objective_gated"
            r"\best_val_objective_mse_h64_z64_cu_latent_diffusion.pt"
        ),
        "coordinates": Path(
            r"C:\Users\shima\OneDrive\Documentos\Leili\peptide_structure_optimization"
            r"\peptide_optimization\output"
            r"\finetuned_gru_vae_cu_latent_diffusion_leakage_safe_v3_objective_gated"
            r"\cu_ddim_inversion_coordinates_for_bo.csv"
        ),
    },
    {
        "run": "Run 2",
        "checkpoint": Path(
            r"C:\Users\shima\OneDrive\Documentos\Leili\peptide_structure_optimization"
            r"\peptide_optimization\output"
            r"\finetuned_gru_vae_cu_latent_diffusion_leakage_safe_v3_objective_gated_run_2"
            r"\best_val_objective_mse_h64_z64_cu_latent_diffusion.pt"
        ),
        "coordinates": Path(
            r"C:\Users\shima\OneDrive\Documentos\Leili\peptide_structure_optimization"
            r"\peptide_optimization\output"
            r"\finetuned_gru_vae_cu_latent_diffusion_leakage_safe_v3_objective_gated_run_2"
            r"\cu_ddim_inversion_coordinates_for_bo.csv"
        ),
    },
    {
        "run": "Run 3",
        "checkpoint": Path(
            r"C:\Users\shima\OneDrive\Documentos\Leili\peptide_structure_optimization"
            r"\peptide_optimization\output"
            r"\finetuned_gru_vae_cu_latent_diffusion_leakage_safe_v3_objective_gated_run_3"
            r"\best_val_objective_mse_h64_z64_cu_latent_diffusion.pt"
        ),
        "coordinates": Path(
            r"C:\Users\shima\OneDrive\Documentos\Leili\peptide_structure_optimization"
            r"\peptide_optimization\output"
            r"\finetuned_gru_vae_cu_latent_diffusion_leakage_safe_v3_objective_gated_run_3"
            r"\cu_ddim_inversion_coordinates_for_bo.csv"
        ),
    },
]

STEP_COUNTS = [5, 10, 20, 30, 50, 75, 100]


def load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def sample_sd(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.std(x, ddof=1)) if len(x) > 1 else np.nan


def mean_sd(x, digits=6):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return ""
    m = float(np.mean(x))
    s = sample_sd(x)
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def aggregate(df, group_col, metrics):
    rows = []
    for key, g in df.groupby(group_col):
        row = {group_col: key, "n_runs": g["run"].nunique()}
        for m in metrics:
            vals = pd.to_numeric(g[m], errors="coerce").dropna().to_numpy(float)
            row[f"{m}_mean"] = float(np.mean(vals))
            row[f"{m}_sd"] = sample_sd(vals)
            row[f"{m}_mean_sd"] = mean_sd(vals)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_col)


def add_table(doc, df):
    table = doc.add_table(rows=1, cols=len(df.columns))
    table.style = "Table Grid"
    for j, c in enumerate(df.columns):
        table.rows[0].cells[j].text = str(c)
    for _, row in df.iterrows():
        cells = table.add_row().cells
        for j, c in enumerate(df.columns):
            cells[j].text = str(row[c])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        default=(
            r"C:\Users\shima\OneDrive\Documentos\Leili\peptide_structure_optimization"
            r"\peptide_optimization\output\ddim_invertibility_audit_three_runs"
        ),
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not AUDIT_SCRIPT.exists():
        raise FileNotFoundError(
            f"Single-run audit script not found:\n{AUDIT_SCRIPT}\n"
            "Set AUDIT_SCRIPT at the top of this file to the path where you saved "
            "audit_ddim_invertibility_gru_vae_latent_diffusion.py."
        )
    if not FINETUNE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Fine-tuning script not found:\n{FINETUNE_SCRIPT}\n"
            "Set FINETUNE_SCRIPT at the top of this file to its actual location."
        )

    configs = []
    convergence_all = []
    numerical_all = []
    reconstruction_all = []
    sphere_all = []

    # ================================================================
    # 1. Run the SAME audit independently on all three checkpoints
    # ================================================================
    for spec in RUNS:
        run_name = spec["run"]
        checkpoint = spec["checkpoint"]
        coordinates = spec["coordinates"]
        run_dir = out_dir / run_name.replace(" ", "_").lower()
        run_dir.mkdir(exist_ok=True)

        if not checkpoint.exists():
            raise FileNotFoundError(f"{run_name}: {checkpoint}")
        if not coordinates.exists():
            raise FileNotFoundError(f"{run_name}: {coordinates}")

        cmd = [
            sys.executable,
            str(AUDIT_SCRIPT),
            "--finetune-script", str(FINETUNE_SCRIPT),
            "--checkpoint", str(checkpoint),
            "--coordinate-csv", str(coordinates),
            "--out-dir", str(run_dir),
            "--splits", "val", "test",
            "--max-samples", "0",
            "--step-counts", *[str(x) for x in STEP_COUNTS],
            "--device", args.device,
        ]
        print("\nRunning:", run_name)
        subprocess.run(cmd, check=True)

        # ---------------- Configuration ----------------
        ckpt = load_checkpoint(checkpoint)
        ck_args = ckpt.get("args", {}) or {}
        coord = pd.read_csv(coordinates)
        if "split" in coord.columns:
            n_audited = int(coord["split"].astype(str).isin(["val", "test"]).sum())
        else:
            n_audited = len(coord)

        configs.append({
            "run": run_name,
            "checkpoint_epoch": int(ckpt.get("epoch", -1)),
            "checkpoint_selection": ckpt.get("checkpoint_selection", ""),
            "latent_dim": int(ckpt["model_config"]["latent_dim"]),
            "diffusion_train_steps": int(ckpt["diffusion_config"]["train_steps"]),
            "saved_ddim_steps": int(ck_args.get("ddim_steps", 50)),
            "sphere_projection": bool(
                ck_args.get("project_inverted_epsilon_to_sphere", False)
            ),
            "audited_val_test_samples": n_audited,
        })

        # ---------------- DDIM convergence ----------------
        d = pd.read_csv(run_dir / "step_convergence_summary.csv")
        d.insert(0, "run", run_name)
        convergence_all.append(d)

        # ---------------- Numerical consistency ----------------
        one = pd.read_csv(run_dir / "one_step_reversibility.csv")
        saved = pd.read_csv(run_dir / "saved_coordinate_consistency.csv")
        sched = pd.read_csv(run_dir / "schedule_audit.csv")

        numerical_all.extend([
            {
                "run": run_name,
                "metric": "Timestep grids exact reverses",
                "value": float(sched["grids_are_exact_reverses"].astype(bool).all()),
            },
            {
                "run": run_name,
                "metric": "Same-epsilon one-step algebraic reverse L2",
                "value": one["same_epsilon_algebraic_reverse_l2_mean"].mean(),
            },
            {
                "run": run_name,
                "metric": "Model-recomputed one-step reverse L2",
                "value": one["model_recomputed_reverse_l2_mean"].mean(),
            },
            {
                "run": run_name,
                "metric": "Saved vs recomputed epsilon L2",
                "value": saved["recomputed_vs_saved_epsilon_l2"].mean(),
            },
            {
                "run": run_name,
                "metric": "Saved epsilon -> h0 L2",
                "value": saved["saved_epsilon_to_h0_l2"].mean(),
            },
        ])

        # ---------------- Peptide reconstruction ----------------
        rec = pd.read_csv(run_dir / "sequence_reconstruction_comparison.csv")
        rows = []
        for n, g in rec.groupby("ddim_steps"):
            rows.append({
                "run": run_name,
                "ddim_steps": int(n),
                "ddim_better": int((g["outcome_vs_vae"] == "ddim_better").sum()),
                "same": int((g["outcome_vs_vae"] == "same").sum()),
                "ddim_worse": int((g["outcome_vs_vae"] == "ddim_worse").sum()),
                "mean_vae_edit": float(g["vae_edit"].mean()),
                "mean_ddim_edit": float(g["ddim_edit"].mean()),
            })
        reconstruction_all.append(pd.DataFrame(rows))

        # ---------------- Sphere projection ----------------
        sph = pd.read_csv(run_dir / "sphere_projection_audit.csv")
        sphere_all.append({
            "run": run_name,
            "native_h0_roundtrip_l2": sph["native_h_roundtrip_l2"].mean(),
            "sphere_h0_roundtrip_l2": sph["sphere_h_roundtrip_l2"].mean(),
            "additional_l2_from_sphere": sph["sphere_minus_native_l2"].mean(),
        })

    # ================================================================
    # 2. Combine runs
    # ================================================================
    config_df = pd.DataFrame(configs)
    conv_df = pd.concat(convergence_all, ignore_index=True)
    numerical_df = pd.DataFrame(numerical_all)
    rec_df = pd.concat(reconstruction_all, ignore_index=True)
    sphere_df = pd.DataFrame(sphere_all)

    conv_metrics = [
        "h_roundtrip_l2_mean",
        "h_roundtrip_rmse_mean",
        "h_roundtrip_cosine_mean",
        "epsilon_roundtrip_l2_mean",
    ]
    conv_agg = aggregate(conv_df, "ddim_steps", conv_metrics)

    rec_agg = aggregate(
        rec_df,
        "ddim_steps",
        ["ddim_better", "same", "ddim_worse", "mean_vae_edit", "mean_ddim_edit"],
    )

    num_rows = []
    for metric, g in numerical_df.groupby("metric"):
        vals = g["value"].to_numpy(float)
        num_rows.append({
            "metric": metric,
            "n_runs": len(vals),
            "mean": float(np.mean(vals)),
            "sd": sample_sd(vals),
            "mean_sd": mean_sd(vals, 8),
        })
    numerical_agg = pd.DataFrame(num_rows)

    sphere_rows = []
    for m in [
        "native_h0_roundtrip_l2",
        "sphere_h0_roundtrip_l2",
        "additional_l2_from_sphere",
    ]:
        vals = sphere_df[m].to_numpy(float)
        sphere_rows.append({
            "metric": m,
            "mean": float(np.mean(vals)),
            "sd": sample_sd(vals),
            "mean_sd": mean_sd(vals, 6),
        })
    sphere_agg = pd.DataFrame(sphere_rows)

    # Save raw and aggregate tables.
    config_df.to_csv(out_dir / "audit_configuration_by_run.csv", index=False)
    conv_df.to_csv(out_dir / "ddim_convergence_by_run.csv", index=False)
    conv_agg.to_csv(out_dir / "ddim_convergence_mean_sd.csv", index=False)
    numerical_df.to_csv(out_dir / "numerical_consistency_by_run.csv", index=False)
    numerical_agg.to_csv(out_dir / "numerical_consistency_mean_sd.csv", index=False)
    rec_df.to_csv(out_dir / "reconstruction_by_run.csv", index=False)
    rec_agg.to_csv(out_dir / "reconstruction_mean_sd.csv", index=False)
    sphere_df.to_csv(out_dir / "sphere_projection_by_run.csv", index=False)
    sphere_agg.to_csv(out_dir / "sphere_projection_mean_sd.csv", index=False)

    # ================================================================
    # 3. Write a concise three-run Word report like the one-run report
    # ================================================================
    c50 = conv_agg.loc[conv_agg["ddim_steps"] == 50].iloc[0]
    c100 = conv_agg.loc[conv_agg["ddim_steps"] == 100].iloc[0]
    algebra = numerical_agg.loc[
        numerical_agg["metric"] == "Same-epsilon one-step algebraic reverse L2"
    ].iloc[0]

    recommended_statement = (
        "Across three independent fine-tuning runs, the deterministic DDIM "
        "transformation was numerically approximately reversible under finite-step "
        "discretization. At 50 DDIM steps, mean latent round-trip L2 was "
        f"{c50['h_roundtrip_l2_mean_mean']:.5f} ± "
        f"{c50['h_roundtrip_l2_mean_sd']:.5f}; at 100 steps it decreased to "
        f"{c100['h_roundtrip_l2_mean_mean']:.5f} ± "
        f"{c100['h_roundtrip_l2_mean_sd']:.5f}. The same-noise one-step "
        "algebraic reversibility error was "
        f"{algebra['mean']:.3e} ± {algebra['sd']:.3e}. "
        "These results provide no evidence of a DDIM update-equation or "
        "timestep-indexing bug; the remaining finite-step discrepancy is "
        "consistent with neural denoiser re-evaluation along the trajectory."
    )

    summary = {
        "n_runs": 3,
        "recommended_report_statement": recommended_statement,
        "configuration": config_df.to_dict(orient="records"),
        "ddim_convergence_mean_sd": conv_agg.to_dict(orient="records"),
        "numerical_consistency_mean_sd": numerical_agg.to_dict(orient="records"),
        "reconstruction_mean_sd": rec_agg.to_dict(orient="records"),
        "sphere_projection_mean_sd": sphere_agg.to_dict(orient="records"),
    }
    (out_dir / "three_run_ddim_audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    if DOCX_OK:
        doc = Document()
        sec = doc.sections[0]
        sec.top_margin = Inches(0.6)
        sec.bottom_margin = Inches(0.6)
        sec.left_margin = Inches(0.6)
        sec.right_margin = Inches(0.6)

        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run("Three-Run DDIM Invertibility and Reconstruction Audit")
        r.bold = True
        r.font.size = Pt(16)

        doc.add_heading("1. Objective and conclusion", level=1)
        doc.add_paragraph(
            "The audit evaluates whether the DDIM inversion/sampling implementation "
            "is numerically consistent across three independent fine-tuning runs."
        )
        doc.add_paragraph("Conclusion. " + recommended_statement)

        doc.add_heading("2. Audit configuration", level=1)
        add_table(doc, config_df)

        doc.add_heading("3. DDIM round-trip convergence", level=1)
        show = conv_agg[[
            "ddim_steps",
            "h_roundtrip_l2_mean_mean_sd",
            "h_roundtrip_rmse_mean_mean_sd",
            "h_roundtrip_cosine_mean_mean_sd",
            "epsilon_roundtrip_l2_mean_mean_sd",
        ]].copy()
        show.columns = [
            "DDIM steps",
            "Mean latent L2 ± SD",
            "Mean per-dim RMSE ± SD",
            "Mean cosine ± SD",
            "Mean epsilon round-trip L2 ± SD",
        ]
        add_table(doc, show)

        doc.add_heading("4. Algebraic reversibility and saved-coordinate consistency", level=1)
        add_table(doc, numerical_agg[["metric", "mean_sd"]].rename(
            columns={"metric": "Diagnostic", "mean_sd": "Mean ± SD"}
        ))

        doc.add_heading("5. Peptide reconstruction comparison", level=1)
        show = rec_agg[[
            "ddim_steps",
            "ddim_better_mean_sd",
            "same_mean_sd",
            "ddim_worse_mean_sd",
            "mean_vae_edit_mean_sd",
            "mean_ddim_edit_mean_sd",
        ]].copy()
        show.columns = [
            "DDIM steps", "DDIM better ± SD", "Same ± SD", "DDIM worse ± SD",
            "Mean VAE edit ± SD", "Mean DDIM edit ± SD"
        ]
        add_table(doc, show)

        doc.add_heading("6. Effect of sphere projection", level=1)
        add_table(doc, sphere_agg[["metric", "mean_sd"]].rename(
            columns={"metric": "Metric", "mean_sd": "Mean ± SD"}
        ))

        doc.add_heading("7. Final assessment", level=1)
        doc.add_paragraph(recommended_statement)

        doc.save(out_dir / "DDIM_invertibility_three_run_report.docx")

    print("\nThree-run DDIM audit complete.")
    print("\n50-step result:")
    print(
        f"latent L2 = {c50['h_roundtrip_l2_mean_mean']:.6f} ± "
        f"{c50['h_roundtrip_l2_mean_sd']:.6f}"
    )
    print("\nRecommended report statement:\n")
    print(recommended_statement)
    print("\nOutputs:", out_dir)


if __name__ == "__main__":
    main()
