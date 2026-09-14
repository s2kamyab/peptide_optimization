from __future__ import annotations

"""
Audit DDIM inversion/sampling consistency for the GRU-VAE + latent-diffusion framework.

This script is designed for checkpoints produced by:
    finetune_best_bo_ready_gru_vae_cu_latent_diffusion_leakage_safe_v3_objective_gated.py

It uses the saved coordinate export:
    cu_ddim_inversion_coordinates_for_bo.csv

Main questions tested
---------------------
1. Does native DDIM approximately round-trip?
       h0 -> ddim_invert -> epsilon -> ddim_sample -> h0_hat

2. Does the reverse cycle approximately round-trip?
       epsilon -> ddim_sample -> h0 -> ddim_invert -> epsilon_hat

3. Does round-trip error decrease as DDIM inference steps increase?

4. Are inversion and sampling using exactly reversed timestep grids?

5. Are the algebraic DDIM coefficients themselves reversible if the SAME
   predicted epsilon is reused?  This separates coefficient bugs from
   model/discretization error.

6. How large is the actual one-step inconsistency when the denoiser is
   re-evaluated at the reverse endpoint?

7. Does DDIM ever improve discrete peptide reconstruction relative to the
   VAE-only decoder, and if so, does that happen despite a nonzero latent
   round-trip error?

8. What happens if epsilon is projected to the sphere?  Sphere projection is
   many-to-one and therefore MUST NOT be used as evidence of DDIM bijectivity.

9. Does the saved exported epsilon_raw agree with epsilon recomputed from the
   saved h0 using the selected checkpoint?

Outputs
-------
<out-dir>/
    audit_summary.csv
    step_convergence_summary.csv
    roundtrip_per_sample.csv
    schedule_audit.csv
    one_step_reversibility.csv
    saved_coordinate_consistency.csv
    sequence_reconstruction_comparison.csv
    sphere_projection_audit.csv
    audit_report.json

Notes
-----
- This audits the *native continuous latent diffusion map*. It deliberately
  excludes BO clipping, [0,1] scaling, stochastic decoder sampling, and local
  epsilon perturbations from the reversibility test.
- Deterministic finite-step DDIM inversion with a learned denoiser is generally
  only approximately inverse to deterministic DDIM sampling. The most useful
  diagnostic is whether errors are small and converge as the timestep grid is
  refined.
"""

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


AA = "ACDEFGHIKLMNPQRSTVWY"
SEQ_LEN = 10
VOCAB = 20


def import_module_from_path(path: str):
    path = str(Path(path).expanduser().resolve())
    name = "latent_diffusion_finetune_for_ddim_audit"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import fine-tuning module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def torch_load_full(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def clean_peptide(x: object) -> str | None:
    p = str(x).strip().upper()
    if len(p) != SEQ_LEN:
        return None
    if any(a not in AA for a in p):
        return None
    return p


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                cur[j - 1] + 1,
                prev[j] + 1,
                prev[j - 1] + int(ca != cb),
            ))
        prev = cur
    return int(prev[-1])


def describe_vector_errors(err: torch.Tensor, prefix: str) -> Dict[str, float]:
    """
    err: [N,D] residual tensor
    """
    l2 = torch.linalg.norm(err, dim=-1)
    rmse = torch.sqrt(torch.mean(err.pow(2), dim=-1))
    linf = err.abs().max(dim=-1).values
    return {
        f"{prefix}_l2_mean": float(l2.mean().cpu()),
        f"{prefix}_l2_median": float(l2.median().cpu()),
        f"{prefix}_l2_p95": float(torch.quantile(l2.float(), 0.95).cpu()),
        f"{prefix}_l2_max": float(l2.max().cpu()),
        f"{prefix}_rmse_mean": float(rmse.mean().cpu()),
        f"{prefix}_rmse_p95": float(torch.quantile(rmse.float(), 0.95).cpu()),
        f"{prefix}_linf_mean": float(linf.mean().cpu()),
        f"{prefix}_linf_max": float(linf.max().cpu()),
    }


def get_ddim_timesteps(n_train: int, inference_steps: int, ascending: bool, device):
    inference_steps = min(max(int(inference_steps), 1), int(n_train))
    if ascending:
        ts = torch.linspace(0, n_train - 1, inference_steps, device=device)
    else:
        ts = torch.linspace(n_train - 1, 0, inference_steps, device=device)
    return torch.unique_consecutive(ts.round().long())


def project_to_radius(x: torch.Tensor, radius: float) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12) * float(radius)


@torch.no_grad()
def load_models(module, checkpoint_path: str, device: torch.device):
    ckpt = torch_load_full(checkpoint_path, map_location=device)
    cfg = module.ModelConfig(**ckpt["model_config"])
    diff_cfg = module.DiffusionConfig(**ckpt["diffusion_config"])

    vae = module.GRUVAE(cfg).to(device)
    diffusion = module.LatentDiffusion(cfg.latent_dim, diff_cfg).to(device)

    vae.load_state_dict(ckpt["vae_state_dict"], strict=True)
    diffusion.load_state_dict(ckpt["diffusion_state_dict"], strict=True)

    vae.eval()
    diffusion.eval()

    latent_mean = ckpt["latent_mean"].to(device).float()
    latent_std = ckpt["latent_std"].to(device).float().clamp_min(1e-6)

    return vae, diffusion, ckpt, latent_mean, latent_std


def resolve_columns(df: pd.DataFrame, prefix: str, dim: int) -> List[str]:
    cols = [f"{prefix}{i:02d}" for i in range(dim)]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing {len(missing)} columns for prefix {prefix!r}. "
            f"Examples: {missing[:5]}"
        )
    return cols


def load_saved_coordinates(path: str, latent_dim: int, splits: Sequence[str], max_samples: int, seed: int):
    df = pd.read_csv(path)

    if "peptide" not in df.columns:
        if "peptide_len10" in df.columns:
            df["peptide"] = df["peptide_len10"]
        else:
            raise KeyError("Coordinate CSV must contain 'peptide' or 'peptide_len10'.")

    df["peptide"] = df["peptide"].map(clean_peptide)
    df = df[df["peptide"].notna()].copy()

    if "split" in df.columns and splits:
        wanted = set(str(s) for s in splits)
        df = df[df["split"].astype(str).isin(wanted)].copy()

    h0_cols = resolve_columns(df, "h0_standardized_", latent_dim)
    mu_cols = resolve_columns(df, "mu_", latent_dim)
    eps_raw_cols = resolve_columns(df, "epsilon_raw_", latent_dim)

    if max_samples > 0 and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed).sort_index()

    df = df.reset_index(drop=True)

    h0 = torch.tensor(df[h0_cols].to_numpy(np.float32))
    mu = torch.tensor(df[mu_cols].to_numpy(np.float32))
    eps_raw_saved = torch.tensor(df[eps_raw_cols].to_numpy(np.float32))

    return df, h0, mu, eps_raw_saved


@torch.no_grad()
def decode_autoregressive(module, vae, z: torch.Tensor) -> List[str]:
    # Use the exact helper from the fine-tuning script if available.
    if hasattr(module, "autoregressive_decode"):
        return module.autoregressive_decode(vae, z, temperature=1.0)

    b = z.size(0)
    h = vae.dec.initial_hidden(z)
    current = torch.zeros(b, 1, VOCAB, device=z.device, dtype=z.dtype)
    ids = []
    for _ in range(SEQ_LEN):
        logits, h = vae.dec.step(z, current, h)
        idx = logits.argmax(dim=-1)
        ids.append(idx.unsqueeze(1))
        current = F.one_hot(idx, num_classes=VOCAB).to(z.dtype).unsqueeze(1)
    token_idx = torch.cat(ids, dim=1)
    return [
        "".join(AA[int(i)] for i in row)
        for row in token_idx.detach().cpu().tolist()
    ]


@torch.no_grad()
def schedule_audit(diffusion, step_counts: Sequence[int], device) -> pd.DataFrame:
    rows = []
    n_train = int(diffusion.cfg.train_steps)
    for n in step_counts:
        asc = get_ddim_timesteps(n_train, n, True, device)
        desc = get_ddim_timesteps(n_train, n, False, device)
        same_reverse = bool(torch.equal(asc, torch.flip(desc, dims=[0])))
        rows.append({
            "requested_steps": int(n),
            "actual_unique_steps": int(len(asc)),
            "ascending_first": int(asc[0]),
            "ascending_last": int(asc[-1]),
            "descending_first": int(desc[0]),
            "descending_last": int(desc[-1]),
            "grids_are_exact_reverses": same_reverse,
            "ascending_grid": ",".join(str(int(v)) for v in asc.cpu()),
            "descending_grid": ",".join(str(int(v)) for v in desc.cpu()),
        })
    return pd.DataFrame(rows)


@torch.no_grad()
def current_roundtrip_audit(
    diffusion,
    h0: torch.Tensor,
    step_counts: Sequence[int],
    device,
):
    summary_rows = []
    per_sample_parts = []

    for n in step_counts:
        eps = diffusion.ddim_invert(h0, inference_steps=n)
        h_rt = diffusion.ddim_sample(eps, inference_steps=n)
        eps_rt = diffusion.ddim_invert(h_rt, inference_steps=n)

        h_err = h_rt - h0
        eps_err = eps_rt - eps

        h_l2 = torch.linalg.norm(h_err, dim=-1)
        h_rmse = torch.sqrt(torch.mean(h_err.pow(2), dim=-1))
        h_cos = F.cosine_similarity(h0, h_rt, dim=-1, eps=1e-8)

        eps_l2 = torch.linalg.norm(eps_err, dim=-1)
        eps_rmse = torch.sqrt(torch.mean(eps_err.pow(2), dim=-1))
        eps_cos = F.cosine_similarity(eps, eps_rt, dim=-1, eps=1e-8)

        row = {
            "ddim_steps": int(n),
            "h_roundtrip_l2_mean": float(h_l2.mean().cpu()),
            "h_roundtrip_l2_median": float(h_l2.median().cpu()),
            "h_roundtrip_l2_p95": float(torch.quantile(h_l2.float(), 0.95).cpu()),
            "h_roundtrip_l2_max": float(h_l2.max().cpu()),
            "h_roundtrip_rmse_mean": float(h_rmse.mean().cpu()),
            "h_roundtrip_cosine_mean": float(h_cos.mean().cpu()),
            "h_roundtrip_cosine_min": float(h_cos.min().cpu()),
            "epsilon_roundtrip_l2_mean": float(eps_l2.mean().cpu()),
            "epsilon_roundtrip_l2_p95": float(torch.quantile(eps_l2.float(), 0.95).cpu()),
            "epsilon_roundtrip_rmse_mean": float(eps_rmse.mean().cpu()),
            "epsilon_roundtrip_cosine_mean": float(eps_cos.mean().cpu()),
            "epsilon_norm_mean": float(eps.norm(dim=-1).mean().cpu()),
            "epsilon_norm_std": float(eps.norm(dim=-1).std().cpu()),
        }
        summary_rows.append(row)

        per_sample_parts.append(pd.DataFrame({
            "sample_index": np.arange(len(h0)),
            "ddim_steps": int(n),
            "h_roundtrip_l2": h_l2.cpu().numpy(),
            "h_roundtrip_rmse": h_rmse.cpu().numpy(),
            "h_roundtrip_cosine": h_cos.cpu().numpy(),
            "epsilon_roundtrip_l2": eps_l2.cpu().numpy(),
            "epsilon_roundtrip_rmse": eps_rmse.cpu().numpy(),
            "epsilon_roundtrip_cosine": eps_cos.cpu().numpy(),
            "epsilon_norm": eps.norm(dim=-1).cpu().numpy(),
        }))

    return pd.DataFrame(summary_rows), pd.concat(per_sample_parts, ignore_index=True)


@torch.no_grad()
def saved_coordinate_consistency(
    diffusion,
    h0: torch.Tensor,
    eps_saved: torch.Tensor,
    saved_steps: int,
):
    eps_recomputed = diffusion.ddim_invert(h0, inference_steps=saved_steps)
    h_from_saved = diffusion.ddim_sample(eps_saved, inference_steps=saved_steps)

    eps_err = eps_recomputed - eps_saved
    h_err = h_from_saved - h0

    eps_l2 = torch.linalg.norm(eps_err, dim=-1)
    h_l2 = torch.linalg.norm(h_err, dim=-1)

    return pd.DataFrame({
        "sample_index": np.arange(len(h0)),
        "saved_ddim_steps": int(saved_steps),
        "recomputed_vs_saved_epsilon_l2": eps_l2.cpu().numpy(),
        "saved_epsilon_to_h0_l2": h_l2.cpu().numpy(),
        "saved_epsilon_norm": eps_saved.norm(dim=-1).cpu().numpy(),
        "recomputed_epsilon_norm": eps_recomputed.norm(dim=-1).cpu().numpy(),
    })


@torch.no_grad()
def algebraic_one_step_audit(
    diffusion,
    h0_batch: torch.Tensor,
    inference_steps: int,
    max_transitions: int = 20,
):
    """
    Two related checks:

    A) Algebraic SAME-epsilon inverse check:
       forward t -> t_next with predicted epsilon at t, then reconstruct x_t using
       the same predicted h0 and same epsilon. This should be ~floating-point zero.

    B) Model-recomputed reverse check:
       after reaching x_{t_next}, re-evaluate the denoiser at t_next and perform
       the reverse deterministic DDIM step back to t. This is not expected to be
       exact; the discrepancy measures denoiser/discretization inconsistency.
    """
    device = h0_batch.device
    ts = get_ddim_timesteps(
        int(diffusion.cfg.train_steps),
        inference_steps,
        ascending=True,
        device=device,
    )

    # Start exactly from h0. This matches the implementation under audit.
    h = h0_batch.clone()
    rows = []

    n_trans = min(len(ts) - 1, int(max_transitions))
    for j in range(n_trans):
        t_scalar = ts[j]
        t_next = ts[j + 1]

        t = torch.full((h.size(0),), int(t_scalar.item()), device=device, dtype=torch.long)
        pred_eps_t = diffusion.denoiser(h, t)
        abar_t = diffusion.alpha_bars[t_scalar]
        pred_h0_t = (h - torch.sqrt(1.0 - abar_t) * pred_eps_t) / torch.sqrt(abar_t)

        abar_next = diffusion.alpha_bars[t_next]
        h_next = (
            torch.sqrt(abar_next) * pred_h0_t
            + torch.sqrt(1.0 - abar_next) * pred_eps_t
        )

        # A) exact algebraic reverse using the SAME pred_h0_t and pred_eps_t.
        h_back_same = (
            torch.sqrt(abar_t) * pred_h0_t
            + torch.sqrt(1.0 - abar_t) * pred_eps_t
        )
        same_err = torch.linalg.norm(h_back_same - h, dim=-1)

        # B) reverse from h_next, but re-evaluate the denoiser at t_next.
        tn = torch.full((h.size(0),), int(t_next.item()), device=device, dtype=torch.long)
        pred_eps_next = diffusion.denoiser(h_next, tn)
        pred_h0_next = (
            h_next - torch.sqrt(1.0 - abar_next) * pred_eps_next
        ) / torch.sqrt(abar_next)

        h_back_model = (
            torch.sqrt(abar_t) * pred_h0_next
            + torch.sqrt(1.0 - abar_t) * pred_eps_next
        )
        model_err = torch.linalg.norm(h_back_model - h, dim=-1)

        eps_change = torch.linalg.norm(pred_eps_next - pred_eps_t, dim=-1)

        rows.append({
            "transition_index": j,
            "t": int(t_scalar.item()),
            "t_next": int(t_next.item()),
            "same_epsilon_algebraic_reverse_l2_mean": float(same_err.mean().cpu()),
            "same_epsilon_algebraic_reverse_l2_max": float(same_err.max().cpu()),
            "model_recomputed_reverse_l2_mean": float(model_err.mean().cpu()),
            "model_recomputed_reverse_l2_p95": float(torch.quantile(model_err.float(), 0.95).cpu()),
            "predicted_epsilon_change_l2_mean": float(eps_change.mean().cpu()),
        })

        h = h_next

    return pd.DataFrame(rows)


@torch.no_grad()
def sequence_reconstruction_audit(
    module,
    vae,
    diffusion,
    h0: torch.Tensor,
    mu_saved: torch.Tensor,
    peptides: Sequence[str],
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    step_counts: Sequence[int],
):
    rows = []

    # VAE-only deterministic decode from the saved encoder mu.
    vae_decoded = decode_autoregressive(module, vae, mu_saved)
    vae_edit = np.asarray([
        edit_distance(src, dec)
        for src, dec in zip(peptides, vae_decoded)
    ], dtype=np.int64)

    for n in step_counts:
        eps = diffusion.ddim_invert(h0, inference_steps=n)
        h_rt = diffusion.ddim_sample(eps, inference_steps=n)
        mu_rt = h_rt * latent_std + latent_mean
        dd_decoded = decode_autoregressive(module, vae, mu_rt)
        dd_edit = np.asarray([
            edit_distance(src, dec)
            for src, dec in zip(peptides, dd_decoded)
        ], dtype=np.int64)

        h_l2 = torch.linalg.norm(h_rt - h0, dim=-1).cpu().numpy()

        for i in range(len(peptides)):
            if dd_edit[i] < vae_edit[i]:
                outcome = "ddim_better"
            elif dd_edit[i] > vae_edit[i]:
                outcome = "ddim_worse"
            else:
                outcome = "same"

            rows.append({
                "sample_index": i,
                "peptide": peptides[i],
                "ddim_steps": int(n),
                "vae_decoded": vae_decoded[i],
                "vae_edit": int(vae_edit[i]),
                "ddim_decoded": dd_decoded[i],
                "ddim_edit": int(dd_edit[i]),
                "outcome_vs_vae": outcome,
                "h_roundtrip_l2": float(h_l2[i]),
            })

    return pd.DataFrame(rows)


@torch.no_grad()
def sphere_projection_audit(
    diffusion,
    h0: torch.Tensor,
    radius: float,
    inference_steps: int,
):
    eps = diffusion.ddim_invert(h0, inference_steps=inference_steps)
    eps_sphere = project_to_radius(eps, radius)
    h_native = diffusion.ddim_sample(eps, inference_steps=inference_steps)
    h_sphere = diffusion.ddim_sample(eps_sphere, inference_steps=inference_steps)

    native_l2 = torch.linalg.norm(h_native - h0, dim=-1)
    sphere_l2 = torch.linalg.norm(h_sphere - h0, dim=-1)

    return pd.DataFrame({
        "sample_index": np.arange(len(h0)),
        "ddim_steps": int(inference_steps),
        "epsilon_native_norm": eps.norm(dim=-1).cpu().numpy(),
        "epsilon_sphere_norm": eps_sphere.norm(dim=-1).cpu().numpy(),
        "native_h_roundtrip_l2": native_l2.cpu().numpy(),
        "sphere_h_roundtrip_l2": sphere_l2.cpu().numpy(),
        "sphere_minus_native_l2": (sphere_l2 - native_l2).cpu().numpy(),
        "epsilon_projection_l2": torch.linalg.norm(eps_sphere - eps, dim=-1).cpu().numpy(),
    })


def monotonicity_interpretation(step_df: pd.DataFrame) -> Dict[str, object]:
    d = step_df.sort_values("ddim_steps")
    vals = d["h_roundtrip_l2_mean"].to_numpy(float)
    steps = d["ddim_steps"].to_numpy(int)

    # Count adjacent improvements as steps increase.
    improvements = int(np.sum(vals[1:] <= vals[:-1]))
    comparisons = max(0, len(vals) - 1)

    best_idx = int(np.argmin(vals))
    return {
        "adjacent_step_increases_with_nonworsening_h_l2": improvements,
        "adjacent_step_comparisons": comparisons,
        "fraction_nonworsening": float(improvements / comparisons) if comparisons else float("nan"),
        "best_step_count_by_mean_h_l2": int(steps[best_idx]),
        "best_mean_h_l2": float(vals[best_idx]),
        "largest_step_count": int(steps[-1]),
        "largest_step_mean_h_l2": float(vals[-1]),
    }


def main():
    p = argparse.ArgumentParser(
        description="Audit deterministic DDIM inversion/sampling consistency using saved latent-diffusion coordinates."
    )
    p.add_argument("--finetune-script", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--coordinate-csv", required=True)
    p.add_argument("--out-dir", default="ddim_invertibility_audit")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--splits", nargs="*", default=["val", "test"])
    p.add_argument("--max-samples", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--step-counts",
        nargs="+",
        type=int,
        default=[5, 10, 20, 30, 50, 75, 100],
    )
    p.add_argument(
        "--sphere-audit-steps",
        type=int,
        default=None,
        help="Default: checkpoint/export DDIM step count.",
    )
    p.add_argument(
        "--one-step-audit-samples",
        type=int,
        default=64,
    )
    p.add_argument(
        "--one-step-audit-max-transitions",
        type=int,
        default=100,
    )
    args = p.parse_args()

    ensure_dir(args.out_dir)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    module = import_module_from_path(args.finetune_script)
    vae, diffusion, ckpt, latent_mean, latent_std = load_models(
        module, args.checkpoint, device
    )

    latent_dim = int(vae.cfg.latent_dim)
    df, h0_cpu, mu_cpu, eps_saved_cpu = load_saved_coordinates(
        args.coordinate_csv,
        latent_dim=latent_dim,
        splits=args.splits,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    h0 = h0_cpu.to(device)
    mu_saved = mu_cpu.to(device)
    eps_saved = eps_saved_cpu.to(device)
    peptides = df["peptide"].astype(str).tolist()

    checkpoint_args = ckpt.get("args", {}) or {}
    saved_steps = int(checkpoint_args.get("ddim_steps", 50))
    sphere_used_in_checkpoint = bool(
        checkpoint_args.get("project_inverted_epsilon_to_sphere", False)
    )

    print("=" * 80)
    print("DDIM INVERTIBILITY / ROUND-TRIP AUDIT")
    print("=" * 80)
    print(f"checkpoint: {args.checkpoint}")
    print(f"checkpoint epoch: {ckpt.get('epoch')}")
    print(f"checkpoint selection: {ckpt.get('checkpoint_selection')}")
    print(f"latent_dim: {latent_dim}")
    print(f"diffusion train steps: {diffusion.cfg.train_steps}")
    print(f"checkpoint DDIM steps: {saved_steps}")
    print(f"checkpoint sphere projection: {sphere_used_in_checkpoint}")
    print(f"audited samples: {len(df)}")
    print(f"splits: {sorted(df['split'].astype(str).unique().tolist()) if 'split' in df.columns else 'not recorded'}")

    # 1) Schedule audit
    schedule_df = schedule_audit(diffusion, args.step_counts, device)
    schedule_df.to_csv(Path(args.out_dir) / "schedule_audit.csv", index=False)

    # 2) Round-trip convergence
    step_df, per_sample_df = current_roundtrip_audit(
        diffusion, h0, args.step_counts, device
    )
    step_df.to_csv(Path(args.out_dir) / "step_convergence_summary.csv", index=False)

    per_sample_df = per_sample_df.merge(
        pd.DataFrame({
            "sample_index": np.arange(len(df)),
            "peptide": peptides,
            "split": df["split"].astype(str).tolist() if "split" in df.columns else [""] * len(df),
        }),
        on="sample_index",
        how="left",
    )
    per_sample_df.to_csv(Path(args.out_dir) / "roundtrip_per_sample.csv", index=False)

    # 3) Saved coordinate consistency
    saved_df = saved_coordinate_consistency(
        diffusion, h0, eps_saved, saved_steps
    )
    saved_df["peptide"] = peptides
    saved_df["split"] = df["split"].astype(str).tolist() if "split" in df.columns else ""
    saved_df.to_csv(Path(args.out_dir) / "saved_coordinate_consistency.csv", index=False)

    # 4) Algebraic one-step checks
    n_one = min(args.one_step_audit_samples, len(h0))
    one_step_df = algebraic_one_step_audit(
        diffusion,
        h0[:n_one],
        inference_steps=max(args.step_counts),
        max_transitions=args.one_step_audit_max_transitions,
    )
    one_step_df.to_csv(Path(args.out_dir) / "one_step_reversibility.csv", index=False)

    # 5) Sequence-level comparison
    seq_df = sequence_reconstruction_audit(
        module=module,
        vae=vae,
        diffusion=diffusion,
        h0=h0,
        mu_saved=mu_saved,
        peptides=peptides,
        latent_mean=latent_mean,
        latent_std=latent_std,
        step_counts=args.step_counts,
    )
    seq_df["split"] = seq_df["sample_index"].map(
        dict(enumerate(df["split"].astype(str).tolist()))
        if "split" in df.columns else {}
    ).fillna("")
    seq_df.to_csv(
        Path(args.out_dir) / "sequence_reconstruction_comparison.csv",
        index=False,
    )

    # 6) Sphere projection audit
    sphere_steps = int(args.sphere_audit_steps or saved_steps)
    sphere_df = sphere_projection_audit(
        diffusion,
        h0,
        radius=math.sqrt(latent_dim),
        inference_steps=sphere_steps,
    )
    sphere_df["peptide"] = peptides
    sphere_df["split"] = df["split"].astype(str).tolist() if "split" in df.columns else ""
    sphere_df.to_csv(Path(args.out_dir) / "sphere_projection_audit.csv", index=False)

    # 7) High-level summary
    mono = monotonicity_interpretation(step_df)

    seq_summary_rows = []
    for n, g in seq_df.groupby("ddim_steps"):
        counts = g["outcome_vs_vae"].value_counts().to_dict()
        seq_summary_rows.append({
            "ddim_steps": int(n),
            "n": int(len(g)),
            "ddim_better_than_vae": int(counts.get("ddim_better", 0)),
            "same_as_vae": int(counts.get("same", 0)),
            "ddim_worse_than_vae": int(counts.get("ddim_worse", 0)),
            "fraction_better": float(np.mean(g["outcome_vs_vae"] == "ddim_better")),
            "fraction_same": float(np.mean(g["outcome_vs_vae"] == "same")),
            "fraction_worse": float(np.mean(g["outcome_vs_vae"] == "ddim_worse")),
            "vae_edit_mean": float(g["vae_edit"].mean()),
            "ddim_edit_mean": float(g["ddim_edit"].mean()),
        })
    seq_summary_df = pd.DataFrame(seq_summary_rows)

    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "checkpoint_selection": ckpt.get("checkpoint_selection", ""),
        "latent_dim": latent_dim,
        "diffusion_train_steps": int(diffusion.cfg.train_steps),
        "saved_ddim_steps": saved_steps,
        "checkpoint_sphere_projection": sphere_used_in_checkpoint,
        "audited_samples": int(len(df)),
        "schedule_all_exact_reverses": bool(schedule_df["grids_are_exact_reverses"].all()),
        "step_convergence": mono,
        "saved_coordinate_recompute_epsilon_l2_mean": float(
            saved_df["recomputed_vs_saved_epsilon_l2"].mean()
        ),
        "saved_coordinate_h_roundtrip_l2_mean": float(
            saved_df["saved_epsilon_to_h0_l2"].mean()
        ),
        "one_step_same_epsilon_algebraic_reverse_l2_mean": float(
            one_step_df["same_epsilon_algebraic_reverse_l2_mean"].mean()
        ),
        "one_step_model_recomputed_reverse_l2_mean": float(
            one_step_df["model_recomputed_reverse_l2_mean"].mean()
        ),
        "sphere_native_h_l2_mean": float(sphere_df["native_h_roundtrip_l2"].mean()),
        "sphere_projected_h_l2_mean": float(sphere_df["sphere_h_roundtrip_l2"].mean()),
        "sphere_projection_penalty_mean": float(sphere_df["sphere_minus_native_l2"].mean()),
        "sequence_summary": seq_summary_df.to_dict(orient="records"),
    }

    # Simple interpretation flags, deliberately conservative.
    flags = []
    if not summary["schedule_all_exact_reverses"]:
        flags.append(
            "WARNING: inversion and sampling timestep grids are not exact reverses for at least one requested step count."
        )

    algebraic_mean = summary["one_step_same_epsilon_algebraic_reverse_l2_mean"]
    if algebraic_mean > 1e-5:
        flags.append(
            "WARNING: same-epsilon algebraic one-step reverse error is larger than expected; inspect DDIM coefficients/indexing."
        )
    else:
        flags.append(
            "PASS: same-epsilon one-step algebra is reversible to near floating-point precision."
        )

    if mono["fraction_nonworsening"] >= 0.5:
        flags.append(
            "PASS/LIKELY OK: round-trip error is generally stable or improves as the DDIM grid is refined."
        )
    else:
        flags.append(
            "CHECK: round-trip error often worsens as more DDIM steps are used; inspect timestep indexing and inversion/sample consistency."
        )

    if summary["saved_coordinate_recompute_epsilon_l2_mean"] < 1e-4:
        flags.append(
            "PASS: saved epsilon_raw agrees with epsilon recomputed from saved h0 using this checkpoint."
        )
    else:
        flags.append(
            "CHECK: saved epsilon_raw does not closely match epsilon recomputed from saved h0; possible checkpoint/export mismatch."
        )

    if summary["sphere_projection_penalty_mean"] > 0:
        flags.append(
            "EXPECTED: sphere projection worsens native round-trip fidelity on average; sphere projection is non-bijective and should be treated only as a BO geometry ablation."
        )

    summary["interpretation_flags"] = flags

    # Save compact audit summary CSV.
    top_rows = []
    for _, r in step_df.iterrows():
        row = {
            "section": "step_convergence",
            **r.to_dict(),
        }
        top_rows.append(row)
    for _, r in seq_summary_df.iterrows():
        row = {
            "section": "sequence_comparison",
            **r.to_dict(),
        }
        top_rows.append(row)
    pd.DataFrame(top_rows).to_csv(
        Path(args.out_dir) / "audit_summary.csv",
        index=False,
    )

    with open(Path(args.out_dir) / "audit_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("KEY RESULTS")
    print("=" * 80)
    print(step_df.to_string(index=False))
    print("\nSchedule grids exact reverses:", summary["schedule_all_exact_reverses"])
    print(
        "Same-epsilon algebraic one-step reverse mean L2:",
        f"{summary['one_step_same_epsilon_algebraic_reverse_l2_mean']:.8g}",
    )
    print(
        "Model-recomputed one-step reverse mean L2:",
        f"{summary['one_step_model_recomputed_reverse_l2_mean']:.8g}",
    )
    print(
        "Saved epsilon recompute mean L2:",
        f"{summary['saved_coordinate_recompute_epsilon_l2_mean']:.8g}",
    )
    print(
        "Native vs sphere h round-trip mean L2:",
        f"{summary['sphere_native_h_l2_mean']:.6g}",
        "vs",
        f"{summary['sphere_projected_h_l2_mean']:.6g}",
    )
    print("\nInterpretation:")
    for flag in flags:
        print(" -", flag)

    print("\nSaved audit outputs to:", Path(args.out_dir).resolve())


if __name__ == "__main__":
    main()
