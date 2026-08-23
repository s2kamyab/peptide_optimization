from __future__ import annotations

"""
Prospective leakage-safe multi-objective GP/qLogEHVI Bayesian optimization
in the objective-aware RealNVP zK space produced by
finetune_best_bo_ready_gru_vae_cu_realnvp_objective_aware_v3.py.

This version is aligned with the revised leakage-safe GRU-VAE + conservative
objective-aware RealNVP fine-tuning framework. It uses only the Cu TRAIN split
as historical BO observations and keeps validation/test peptides completely
outside BO fitting and black-box evaluation.

Path
----
peptide -> GRU-VAE encoder mu -> training-only standardization h0
        -> RealNVP forward -> zK                           [BO space]

zK candidate -> RealNVP inverse -> h0 -> unstandardize -> decoder mu
             -> autoregressive peptide -> true Cu black-box objectives

Objectives (maximized):
    chelation_sub, solubility_sub, stability_sub, expression_sub

Notes
-----
* RealNVP zK is NOT constrained to a sphere. The trained base target is N(0,I).
* For BoTorch, zK is affinely mapped from a fixed symmetric box [-B,+B]^D
  to [0,1]^D. This avoids input-scaling warnings while preserving zK geometry.
* B defaults to 4, but is expanded once before BO if the existing Cu zK data
  contain a larger coordinate.
* Hypervolume uses one FIXED reference point from the initial design, so HV
  values are comparable across BO iterations.
* Each accepted decoded peptide is re-encoded to its own zK coordinate before
  it is added to the GP.
* The preferred fine-tuned checkpoint is the minimum BO-ready-score checkpoint.
  For the attached 150-epoch history this is epoch 150, not the epoch-149
  minimum-objective-MSE checkpoint.
* Cu validation/test peptides are treated as held-out and are rejected if a
  generated peptide exactly matches them.
"""

import argparse
import gc
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import gpytorch
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
try:
    from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
except Exception:
    qLogExpectedHypervolumeImprovement = None

from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.utils.multi_objective.hypervolume import Hypervolume



AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_I = {a: i for i, a in enumerate(AA)}
I_TO_AA = {i: a for i, a in enumerate(AA)}
SEQ_LEN = 10
VOCAB = 20
OBJ_COLS = [
    "chelation_sub",
    "solubility_sub",
    "stability_sub",
    "expression_sub",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def import_module_py314_safe(path: str):
    name = "realnvp_finetune_module_for_bo"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import fine-tuning script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _add_sys_path_once(path: Path):
    path = path.resolve()
    s = str(path)
    if path.exists() and s not in sys.path:
        sys.path.insert(0, s)


def configure_project_import_paths(
    blackbox_script: str,
    finetune_script: str,
    explicit_project_roots: Sequence[str],
):
    """
    Make imports such as `import peptide_optimization...` work regardless of
    whether this BO script is launched from src/scripts, the repository root,
    or another working directory.

    A valid Python import root is the DIRECTORY CONTAINING the
    `peptide_optimization` package directory, not the package directory itself.
    """
    candidates = []

    # Explicit roots have highest priority.
    for p in explicit_project_roots or []:
        candidates.append(Path(p).expanduser())

    # Useful anchors from the current working directory and script locations.
    anchors = [
        Path.cwd(),
        Path(__file__).resolve().parent,
        Path(finetune_script).expanduser().resolve().parent,
        Path(blackbox_script).expanduser().resolve().parent,
    ]

    for a in anchors:
        candidates.append(a)
        candidates.extend(list(a.parents)[:6])

    # Add candidate roots themselves.
    for c in candidates:
        _add_sys_path_once(c)

    # Search nearby for a peptide_optimization package and add ITS PARENT.
    searched = set()
    for c in candidates:
        try:
            c = c.resolve()
        except Exception:
            continue
        if c in searched or not c.exists() or not c.is_dir():
            continue
        searched.add(c)

        # Case 1: c itself contains peptide_optimization/
        pkg = c / "peptide_optimization"
        if pkg.is_dir():
            _add_sys_path_once(c)

        # Case 2: c itself is peptide_optimization package dir.
        if c.name == "peptide_optimization":
            _add_sys_path_once(c.parent)

        # Case 3: src-layout: c/src/peptide_optimization/
        src_pkg = c / "src" / "peptide_optimization"
        if src_pkg.is_dir():
            _add_sys_path_once(c / "src")


def resolve_blackbox_script(
    requested: str,
    finetune_script: str,
) -> Path:
    p = Path(requested).expanduser()
    if p.is_file():
        return p.resolve()

    names = [
        requested,
        "black_box_fcn_mo_CU_f.py",
    ]
    roots = [
        Path.cwd(),
        Path(__file__).resolve().parent,
        Path(finetune_script).expanduser().resolve().parent,
    ]

    for root in roots:
        for name in names:
            q = (root / name)
            if q.is_file():
                return q.resolve()

    raise FileNotFoundError(
        "Could not locate the Cu black-box Python file. "
        f"Requested={requested!r}. Pass an explicit path with --blackbox-script."
    )


def load_blackbox_function(
    blackbox_script: str,
    finetune_script: str,
    project_roots: Sequence[str],
):
    path = resolve_blackbox_script(
        blackbox_script,
        finetune_script,
    )

    configure_project_import_paths(
        str(path),
        finetune_script,
        project_roots,
    )

    module_name = "cu_blackbox_module_for_bo"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for black-box file: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        sys.modules.pop(module_name, None)
        if exc.name == "peptide_optimization":
            raise RuntimeError(
                "The Cu black-box file was found, but one of its imports requires "
                "the `peptide_optimization` package and Python cannot see that "
                "package. Pass --project-root pointing to the DIRECTORY THAT "
                "CONTAINS the `peptide_optimization` package folder. "
                "For example, if the package is "
                r"C:\...\peptide_optimization\peptide_optimization, pass "
                r"C:\...\peptide_optimization as --project-root."
            ) from exc
        raise
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    fn = getattr(module, "blackbox_fc", None)
    if fn is None or not callable(fn):
        raise AttributeError(
            f"{path} does not define a callable named blackbox_fc."
        )

    print(f"Loaded Cu black-box function from: {path}")
    return fn


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_load_full(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def clean_peptide(x: object) -> Optional[str]:
    p = str(x).strip().upper()
    if len(p) != SEQ_LEN or any(a not in AA_TO_I for a in p):
        return None
    return p


def onehot_encode(peptides: Sequence[str]) -> torch.Tensor:
    x = torch.zeros(len(peptides), SEQ_LEN, VOCAB, dtype=torch.float32)
    for i, pep in enumerate(peptides):
        p = clean_peptide(pep)
        if p is None:
            raise ValueError(f"Invalid peptide: {pep!r}")
        for t, aa in enumerate(p):
            x[i, t, AA_TO_I[aa]] = 1.0
    return x


def edit_distance(a: str, b: str) -> int:
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


def pareto_mask_max(Y: torch.Tensor) -> torch.Tensor:
    if Y.numel() == 0:
        return torch.zeros(0, dtype=torch.bool, device=Y.device)
    # dominates[j,i] = j dominates i
    A = Y.unsqueeze(1)
    B = Y.unsqueeze(0)
    dominates = (A >= B).all(-1) & (A > B).any(-1)
    dominates.fill_diagonal_(False)
    dominated = dominates.any(dim=0)
    return ~dominated


def dominates(a: torch.Tensor, b: torch.Tensor) -> bool:
    return bool(((a >= b).all() & (a > b).any()).item())


def zk_to_unit(z: torch.Tensor, bound: float, clamp: bool = True):
    x = (z + bound) / (2.0 * bound)
    return x.clamp(0.0, 1.0) if clamp else x


def unit_to_zk(x: torch.Tensor, bound: float):
    return 2.0 * bound * x - bound


# ---------------------------------------------------------------------------
# Model loading / transforms
# ---------------------------------------------------------------------------

def load_model(
    checkpoint: str,
    finetune_script: str,
    device: torch.device,
    require_bo_ready: bool = True,
):
    m = import_module_py314_safe(finetune_script)
    ckpt = torch_load_full(checkpoint, map_location=device)

    expected_type = "cu_best_bo_ready_gru_vae_realnvp_leakage_safe"
    ctype = ckpt.get("checkpoint_type", "")
    if ctype and ctype != expected_type:
        raise ValueError(
            f"Unexpected checkpoint_type={ctype!r}; expected {expected_type!r}"
        )

    selection = ckpt.get("checkpoint_selection", "")
    if require_bo_ready and selection != "minimum_bo_ready_score":
        raise RuntimeError(
            "For prospective BO use the validation-selected BO-ready checkpoint. "
            f"Found checkpoint_selection={selection!r}. "
            "For the attached history the preferred checkpoint is the epoch-150 "
            "best_bo_ready_h64_z64_cu_realnvp.pt."
        )

    cfg = m.ModelConfig(**ckpt["model_config"])
    flow_cfg = m.RealNVPConfig(**ckpt["flow_config"])

    vae = m.GRUVAE(cfg).to(device)
    flow = m.RealNVPFlow(cfg.latent_dim, flow_cfg).to(device)

    vae.load_state_dict(ckpt["vae_state_dict"], strict=True)
    flow.load_state_dict(ckpt["flow_state_dict"], strict=True)

    vae.eval()
    flow.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in flow.parameters():
        p.requires_grad_(False)

    mean = ckpt["latent_mean"].to(device).float()
    std = ckpt["latent_std"].to(device).float().clamp_min(1e-6)

    ckpt_obj_cols = ckpt.get("objective_cols", [])
    print(f"Loaded checkpoint: {checkpoint}")
    print(f"epoch: {ckpt.get('epoch')}")
    print(f"selection: {selection}")
    print(f"latent_dim: {cfg.latent_dim}")
    print(f"flow_config: {ckpt.get('flow_config')}")
    print(f"checkpoint objectives: {ckpt_obj_cols}")
    print(f"bo_coordinate_space: {ckpt.get('bo_coordinate_space', 'realnvp_base_zK')}")

    return m, vae, flow, ckpt, mean, std


@torch.no_grad()
def encode_batch_to_zk(vae, flow, x, mean, std, batch_size=256):
    out = []
    for s in range(0, len(x), batch_size):
        xb = x[s:s + batch_size]
        mu, _, _ = vae.enc(xb)
        h0 = (mu - mean) / std
        zk, _ = flow(h0)
        out.append(zk.detach().cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def peptide_to_zk(vae, flow, pep, mean, std):
    x = onehot_encode([pep]).to(mean.device)
    mu, _, _ = vae.enc(x)
    h0 = (mu - mean) / std
    zk, _ = flow(h0)
    return zk


@torch.no_grad()
def zk_to_decoder_mu(flow, zk, mean, std):
    h0, _ = flow.inverse(zk)
    return h0 * std + mean


@torch.no_grad()
def decode_one(vae, decoder_mu, temperature=1.0, sample=False):
    h = vae.dec.initial_hidden(decoder_mu)
    cur = torch.zeros(
        1, 1, VOCAB,
        device=decoder_mu.device,
        dtype=decoder_mu.dtype,
    )
    ids, probs_chosen = [], []
    tau = max(float(temperature), 1e-6)

    for _ in range(SEQ_LEN):
        logits, h = vae.dec.step(decoder_mu, cur, h)
        probs = torch.softmax(logits / tau, dim=-1)
        idx = (
            torch.multinomial(probs, 1).squeeze(-1)
            if sample else probs.argmax(dim=-1)
        )
        probs_chosen.append(float(probs.gather(-1, idx[:, None]).item()))
        ids.append(int(idx.item()))
        cur = F.one_hot(idx, num_classes=VOCAB).to(decoder_mu.dtype).unsqueeze(1)

    pep = "".join(I_TO_AA[i] for i in ids)
    return pep, float(np.mean(probs_chosen)), float(np.min(probs_chosen))


@torch.no_grad()
def reencode_diagnostic(vae, flow, pep, zk_ref, mean, std):
    z2 = peptide_to_zk(vae, flow, pep, mean, std)
    l2 = float(torch.linalg.norm(z2 - zk_ref).cpu())
    cos = float(F.cosine_similarity(z2, zk_ref, dim=-1).item())
    return l2, cos


@torch.no_grad()
def decode_candidates(vae, flow, Z, bo_iter, args, mean, std):
    rows, peps = [], []

    for j in range(len(Z)):
        z = Z[j:j+1]
        mu = zk_to_decoder_mu(flow, z, mean, std)

        center, conf_mean, conf_min = decode_one(
            vae, mu, args.decoder_temperature, sample=False
        )
        rt_l2, rt_cos = reencode_diagnostic(vae, flow, center, z, mean, std)
        peps.append(center)
        rows.append({
            "bo_iter": bo_iter,
            "zK_candidate_index": j,
            "decode_type": "argmax",
            "peptide": center,
            "decoder_confidence_mean": conf_mean,
            "decoder_confidence_min": conf_min,
            "peptide_reencode_zK_l2": rt_l2,
            "peptide_reencode_zK_cosine": rt_cos,
            "zK_norm": float(z.norm().cpu()),
            "local_noise_l2": 0.0,
            "edit_distance_to_center": 0,
        })

        for _ in range(max(0, args.decoder_samples_per_zk - 1)):
            p, cm, cmin = decode_one(
                vae, mu, args.decoder_temperature, sample=True
            )
            l2, cos = reencode_diagnostic(vae, flow, p, z, mean, std)
            peps.append(p)
            rows.append({
                "bo_iter": bo_iter,
                "zK_candidate_index": j,
                "decode_type": "sample",
                "peptide": p,
                "decoder_confidence_mean": cm,
                "decoder_confidence_min": cmin,
                "peptide_reencode_zK_l2": l2,
                "peptide_reencode_zK_cosine": cos,
                "zK_norm": float(z.norm().cpu()),
                "local_noise_l2": 0.0,
                "edit_distance_to_center": edit_distance(center, p),
            })

        for sigma in args.novelty_sigmas:
            for _ in range(args.local_neighbors_per_sigma):
                zn = z + torch.randn_like(z) * float(sigma)
                zn = zn.clamp(-args.resolved_z_bound, args.resolved_z_bound)
                mun = zk_to_decoder_mu(flow, zn, mean, std)
                p, cm, cmin = decode_one(
                    vae, mun, args.decoder_temperature, sample=False
                )
                l2, cos = reencode_diagnostic(vae, flow, p, zn, mean, std)
                peps.append(p)
                rows.append({
                    "bo_iter": bo_iter,
                    "zK_candidate_index": j,
                    "decode_type": f"local_zK_sigma_{sigma:g}",
                    "peptide": p,
                    "decoder_confidence_mean": cm,
                    "decoder_confidence_min": cmin,
                    "peptide_reencode_zK_l2": l2,
                    "peptide_reencode_zK_cosine": cos,
                    "zK_norm": float(zn.norm().cpu()),
                    "local_noise_l2": float(torch.linalg.norm(zn - z).cpu()),
                    "edit_distance_to_center": edit_distance(center, p),
                })

    d = pd.DataFrame(rows)
    d["is_duplicate_raw_decode"] = d.duplicated("peptide", keep="first")
    d.to_csv(
        os.path.join(args.decoder_dir, f"decoder_diagnostics_bo_iter_{bo_iter:03d}.csv"),
        index=False,
    )
    return list(dict.fromkeys(peps)), d


# ---------------------------------------------------------------------------
# GP / qEHVI
# ---------------------------------------------------------------------------

def fit_mo_gp(X: torch.Tensor, Y: torch.Tensor, device):
    models = []
    for j in range(Y.shape[1]):
        gp = SingleTaskGP(
            X,
            Y[:, j:j+1],
            outcome_transform=Standardize(m=1),
        ).to(device)
        gp.likelihood.noise_covar.register_constraint(
            "raw_noise",
            gpytorch.constraints.GreaterThan(1e-6),
        )
        gp.likelihood.noise = 1e-4
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        models.append(gp)
    return ModelListGP(*models)


# ---------------------------------------------------------------------------
# Objectives / novelty
# ---------------------------------------------------------------------------

def evaluate_blackbox(peptides, cache, obj_cols, blackbox_fc):
    uncached = [p for p in peptides if p not in cache]

    if uncached:
        if blackbox_fc is None:
            raise RuntimeError("Cu black-box function is not loaded.")
        scored = blackbox_fc(uncached)

        if isinstance(scored, torch.Tensor):
            arr = scored.detach().cpu().numpy()
            if arr.shape != (len(uncached), len(obj_cols)):
                raise ValueError(
                    f"blackbox_fc returned {arr.shape}, "
                    f"expected {(len(uncached), len(obj_cols))}"
                )
            for p, row in zip(uncached, arr):
                cache[p] = [float(v) for v in row]

        elif isinstance(scored, pd.DataFrame):
            for _, row in scored.iterrows():
                p = clean_peptide(row.get("peptide_len10", row.get("peptide", "")))
                if p is not None:
                    cache[p] = [float(row[c]) for c in obj_cols]
        else:
            raise TypeError(f"Unsupported blackbox return type: {type(scored)}")

    missing = [p for p in peptides if p not in cache]
    if missing:
        raise RuntimeError(f"No scores returned for {missing[:5]}")

    return torch.tensor([cache[p] for p in peptides], dtype=torch.float32)


def filter_novel(
    peps,
    train_set,
    heldout_set,
    evaluated_set,
    generated_set,
    args,
):
    accepted, rejected = [], []
    for p in peps:
        reason = None
        if clean_peptide(p) is None:
            reason = "invalid"
        elif args.reject_training_peptides and p in train_set:
            reason = "in_training"
        elif args.reject_heldout_peptides and p in heldout_set:
            reason = "in_validation_or_test_holdout"
        elif args.reject_seen_peptides and p in evaluated_set:
            reason = "already_evaluated"
        elif args.reject_seen_peptides and p in generated_set:
            reason = "already_generated"

        if reason is None:
            accepted.append(p)
        else:
            rejected.append((p, reason))
    return accepted, rejected


def initial_indices(Y, n_init, seed):
    # Matches the attached diffusion BO: all training Pareto + random fill.
    pareto = torch.where(pareto_mask_max(Y))[0].cpu().tolist()
    others = [i for i in range(len(Y)) if i not in set(pareto)]
    rng = np.random.default_rng(seed)
    need = max(0, int(n_init) - len(pareto))
    extra = (
        rng.choice(others, size=min(need, len(others)), replace=False).tolist()
        if need else []
    )
    return list(dict.fromkeys(pareto + extra))


def save_pareto(peptides, labeled_idx, Y, bo_iter, out_dir, obj_cols):
    mask = pareto_mask_max(Y.detach().cpu())
    rows = []
    for k in torch.where(mask)[0].tolist():
        g = int(labeled_idx[k])
        row = {"bo_iter": bo_iter, "global_index": g, "peptide": peptides[g]}
        for j, c in enumerate(obj_cols):
            row[c] = float(Y[k, j].detach().cpu())
        rows.append(row)

    d = pd.DataFrame(rows)
    d.to_csv(
        os.path.join(out_dir, f"pareto_front_bo_iter_{bo_iter:03d}.csv"),
        index=False,
    )
    return d


# ---------------------------------------------------------------------------
# Main BO
# ---------------------------------------------------------------------------

def run(args):
    set_seed(args.seed)
    device = torch.device(args.device)

    for p in [args.out_dir, args.decoder_dir, args.pareto_dir]:
        ensure_dir(p)

    module, vae, flow, ckpt, mean, std = load_model(
        args.realnvp_checkpoint,
        args.finetune_script,
        device,
        require_bo_ready=(not args.allow_non_bo_ready_checkpoint),
    )

    checkpoint_obj_cols = list(ckpt.get("objective_cols", []))
    if checkpoint_obj_cols and checkpoint_obj_cols != list(args.obj_cols):
        raise ValueError(
            "BO objective columns do not match the fine-tuned checkpoint: "
            f"checkpoint={checkpoint_obj_cols}, requested={list(args.obj_cols)}"
        )

    blackbox_fc = load_blackbox_function(
        args.blackbox_script,
        args.finetune_script,
        args.project_root,
    )

    # Reuse exactly the same leakage-safe Cu mapping as fine-tuning.
    (
        cu_df,
        peptides_all,
        x_all_cpu,
        y_all_cpu,
        _final_score,
        split_to_idx,
        _resolved_peptide_col,
        data_audit,
    ) = module.load_cu_dataset_with_manifest(
        args.data_csv,
        args.peptide_col,
        args.obj_cols,
        args.pretraining_split_manifest,
        args.duplicate_policy,
        args.unmapped_policy,
        args.out_dir,
    )

    train_idx = split_to_idx["train"]
    val_idx = split_to_idx["val"]
    test_idx = split_to_idx["test"]

    # BO uses historical TRAIN observations only. Validation and test remain
    # unavailable to GP fitting, hypervolume initialization, and black-box cache.
    peptides = [peptides_all[i] for i in train_idx.tolist()]
    Xseq = x_all_cpu[train_idx].to(device)
    Y_full = y_all_cpu[train_idx].to(device)

    heldout_set = {
        peptides_all[i]
        for i in torch.cat([val_idx, test_idx]).tolist()
    }

    print(
        f"Leakage-safe Cu rows: total={len(peptides_all)} "
        f"BO-train={len(train_idx)} val-heldout={len(val_idx)} "
        f"test-heldout={len(test_idx)}"
    )

    Z_all = encode_batch_to_zk(
        vae, flow, Xseq, mean, std, args.encode_batch_size
    ).to(device)

    z_abs_max = float(Z_all.abs().max().cpu())
    z_bound = float(args.zk_bound)
    if args.auto_expand_zk_bound and z_abs_max >= z_bound:
        z_bound = 1.05 * z_abs_max
    args.resolved_z_bound = z_bound

    X_all_unclamped = zk_to_unit(Z_all, z_bound, clamp=False)
    if bool(((X_all_unclamped < 0.0) | (X_all_unclamped > 1.0)).any().item()):
        raise RuntimeError(
            "Resolved zK bound does not contain all Cu TRAIN zK coordinates. "
            "Increase --zk-bound or keep --auto-expand-zk-bound enabled."
        )
    X_all = X_all_unclamped

    print(
        f"zK norm mean={float(Z_all.norm(dim=-1).mean().cpu()):.6f}; "
        f"max |coordinate|={z_abs_max:.6f}"
    )
    print(f"Fixed zK BO box: [-{z_bound:.4f}, +{z_bound:.4f}]")
    print(
        f"GP cube min={float(X_all.min().cpu()):.6f}, "
        f"max={float(X_all.max().cpu()):.6f}"
    )

    # Original training Pareto
    tr_mask = pareto_mask_max(Y_full)
    tr_idx = torch.where(tr_mask)[0]
    Y_train_pareto = Y_full[tr_mask]
    train_pareto = pd.DataFrame({
        "global_index": tr_idx.cpu().numpy(),
        "peptide": [peptides[i] for i in tr_idx.cpu().tolist()],
        **{
            c: Y_train_pareto[:, j].detach().cpu().numpy()
            for j, c in enumerate(args.obj_cols)
        },
    })
    train_pareto.to_csv(
        os.path.join(args.out_dir, "train_only_pareto_CU_gru_vae_realnvp_zK.csv"),
        index=False,
    )
    print(f"Training Pareto: {len(train_pareto)} solutions")

    cache = {
        p: [float(v) for v in vals]
        for p, vals in zip(
            peptides,
            Y_full.detach().cpu().numpy().astype(np.float32).tolist(),
        )
    }

    train_set = set(peptides)
    evaluated_set, generated_set = set(), set()

    labeled_idx = initial_indices(Y_full, args.initial_labeled, args.seed + 42)
    idx = torch.tensor(labeled_idx, device=device)

    Y_lab = Y_full[idx]
    Z_lab = Z_all[idx]
    X_lab = X_all[idx]
    evaluated_set.update(peptides[i] for i in labeled_idx)

    # Fixed reference point for the entire run.
    ref = Y_lab.min(dim=0).values - float(args.ref_margin)
    initial_hv = float(Hypervolume(ref_point=ref).compute(Y_lab))
    best_hv = initial_hv

    print(
        f"Initial labeled: {len(labeled_idx)} "
        f"(includes all {int(tr_mask.sum())} train-only Pareto observations)"
    )
    print(f"Fixed HV ref point: {ref.detach().cpu().tolist()}")
    print(f"Initial HV: {initial_hv:.6f}")

    config = vars(args).copy()
    config.update({
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_selection": ckpt.get("checkpoint_selection"),
        "checkpoint_validation_bo_ready_score":
            ckpt.get("metrics", {}).get("bo_ready_score"),
        "bo_space": "realnvp_base_zK",
        "historical_observations_split": "train_only",
        "validation_test_used_for_gp_fit": False,
        "validation_test_used_for_blackbox_cache": False,
        "heldout_exact_peptide_rejection": bool(args.reject_heldout_peptides),
        "pretraining_split_manifest": args.pretraining_split_manifest,
        "blackbox_script": args.blackbox_script,
        "project_root": list(args.project_root),
        "data_audit": data_audit,
        "resolved_z_bound": z_bound,
        "initial_hypervolume": initial_hv,
        "fixed_hv_ref": ref.detach().cpu().tolist(),
        "warm_start_includes_full_training_pareto": True,
    })
    Path(args.out_dir, "bo_run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    hv_rows, all_rows = [], []

    for bo_iter in range(args.bo_iters):
        print(f"\n=== BO iter {bo_iter + 1}/{args.bo_iters} ===")

        Xgp = X_lab.double()
        Ygp = Y_lab.double()
        model = fit_mo_gp(Xgp, Ygp, device)

        # Balanced current Pareto point as trust-region center.
        nd = torch.where(pareto_mask_max(Y_lab))[0]
        Ynd = Y_lab[nd]
        lo = Ynd.min(0).values
        span = (Ynd.max(0).values - lo).clamp_min(1e-12)
        score = ((Ynd - lo) / span).mean(-1)
        center_i = int(nd[torch.argmax(score)].item())
        xc = X_lab[center_i].double()

        r_unit = float(args.tr_radius_zk) / (2.0 * z_bound)
        bounds = torch.stack([
            (xc - r_unit).clamp(0.0, 1.0),
            (xc + r_unit).clamp(0.0, 1.0),
        ])

        Ynd_gp = Ygp[pareto_mask_max(Ygp)]
        if len(Ynd_gp) > args.max_partition_points:
            yn = (
                (Ynd_gp - Ynd_gp.min(0).values)
                / (Ynd_gp.max(0).values - Ynd_gp.min(0).values).clamp_min(1e-12)
            )
            keep = torch.topk(
                yn.mean(-1), args.max_partition_points
            ).indices
            Ypart = Ynd_gp[keep]
        else:
            Ypart = Ynd_gp

        ref_d = ref.double()
        partitioning = NondominatedPartitioning(ref_point=ref_d, Y=Ypart)
        sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([args.mc_samples])
        ).to(device)

        acq_cls = (
            qLogExpectedHypervolumeImprovement
            if args.acqf == "qlogehvi"
            and qLogExpectedHypervolumeImprovement is not None
            else qExpectedHypervolumeImprovement
        )

        acq = acq_cls(
            model=model,
            ref_point=ref_d.detach().cpu().tolist(),
            partitioning=partitioning,
            sampler=sampler,
        ).to(device)

        parts, avals = [], []
        repeats = args.q_batch if args.optimize_q_one_at_a_time else 1
        q_inner = 1 if args.optimize_q_one_at_a_time else args.q_batch

        for _ in range(repeats):
            try:
                xnew, aval = optimize_acqf(
                    acq,
                    bounds=bounds,
                    q=q_inner,
                    num_restarts=args.num_restarts,
                    raw_samples=args.raw_samples,
                    sequential=True,
                )
            except torch.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                xnew, aval = optimize_acqf(
                    acq,
                    bounds=bounds,
                    q=1,
                    num_restarts=max(1, args.num_restarts // 2),
                    raw_samples=max(16, args.raw_samples // 4),
                    sequential=True,
                )
            parts.append(xnew.detach())
            avals.append(aval.detach().reshape(-1)[0])

        Xcand = torch.cat(parts).clamp(0.0, 1.0)
        Zcand = unit_to_zk(Xcand, z_bound).float()
        acq_value = float(torch.stack(avals).mean().cpu())

        print(
            f"Proposed zK mean norm={float(Zcand.norm(dim=-1).mean().cpu()):.6f}; "
            f"max |coord|={float(Zcand.abs().max().cpu()):.6f}"
        )

        raw_peps, diag = decode_candidates(
            vae, flow, Zcand, bo_iter, args, mean, std
        )
        accepted, rejected = filter_novel(
            raw_peps,
            train_set,
            heldout_set,
            evaluated_set,
            generated_set,
            args,
        )

        if args.max_blackbox_per_iter > 0:
            accepted = accepted[:args.max_blackbox_per_iter]

        if not accepted:
            print("[BO] No novel peptide survived; trying higher temperature.")
            old = args.decoder_temperature
            args.decoder_temperature = max(old, args.fallback_temperature)
            extra, diag2 = decode_candidates(
                vae, flow, Zcand, bo_iter + 100000, args, mean, std
            )
            args.decoder_temperature = old
            accepted, rejected2 = filter_novel(
                extra,
                train_set,
                heldout_set,
                evaluated_set,
                generated_set,
                args,
            )
            if args.max_blackbox_per_iter > 0:
                accepted = accepted[:args.max_blackbox_per_iter]
            rejected.extend(rejected2)
            diag = pd.concat([diag, diag2], ignore_index=True)

        rejected_map = dict(rejected)
        status = diag.copy()

        if not accepted:
            generated_set.update(raw_peps)
            status["accepted_for_blackbox"] = False
            status["rejected_reason"] = status["peptide"].map(
                lambda p: rejected_map.get(p, "not_selected")
            )
            status.to_csv(
                os.path.join(
                    args.decoder_dir,
                    f"all_decoder_diagnostics_status_bo_iter_{bo_iter:03d}.csv",
                ),
                index=False,
            )

            hv_rows.append({
                "bo_iter": bo_iter,
                "n_raw_decoded": len(raw_peps),
                "n_accepted": 0,
                "n_labeled": len(labeled_idx),
                "pareto_size": int(pareto_mask_max(Y_lab).sum()),
                "hypervolume": best_hv,
                "best_hypervolume": best_hv,
                "improved": 0,
                "acq_value": acq_value,
                "n_dominates_train": 0,
            })
            pd.DataFrame(hv_rows).to_csv(
                os.path.join(args.out_dir, "bo_hypervolume_history.csv"),
                index=False,
            )
            continue

        # True objective evaluation
        Ynew = evaluate_blackbox(accepted, cache, args.obj_cols, blackbox_fc).to(device)

        # Re-encode accepted discrete peptides into their own zK.
        Znew = torch.cat([
            peptide_to_zk(vae, flow, p, mean, std)
            for p in accepted
        ], dim=0)
        Xnew = zk_to_unit(Znew, z_bound, clamp=True)

        iter_rows = []
        for i, (p, y) in enumerate(zip(accepted, Ynew)):
            row = {
                "bo_iter": bo_iter,
                "peptide": p,
                "acq_value": acq_value,
                "zK_norm": float(Znew[i].norm().cpu()),
                "zK_max_abs_coordinate": float(Znew[i].abs().max().cpu()),
                "zK_gp_coordinate_clipped": int(
                    bool((Znew[i].abs() > z_bound).any().item())
                ),
                "dominated_by_train_pareto": int(any(
                    dominates(y0, y) for y0 in Y_train_pareto
                )),
                "dominates_any_train_pareto": int(any(
                    dominates(y, y0) for y0 in Y_train_pareto
                )),
            }
            for j, c in enumerate(args.obj_cols):
                row[c] = float(y[j].cpu())
            iter_rows.append(row)
            all_rows.append(row)

        pd.DataFrame(iter_rows).to_csv(
            os.path.join(
                args.out_dir,
                f"accepted_candidates_scored_bo_iter_{bo_iter:03d}.csv",
            ),
            index=False,
        )

        status["accepted_for_blackbox"] = status["peptide"].isin(accepted)
        status["rejected_reason"] = status["peptide"].map(
            lambda p: "" if p in accepted else rejected_map.get(p, "not_selected")
        )
        for j, c in enumerate(args.obj_cols):
            mp = {p: float(y[j].cpu()) for p, y in zip(accepted, Ynew)}
            status[c] = status["peptide"].map(lambda p: mp.get(p, np.nan))
        status.to_csv(
            os.path.join(
                args.decoder_dir,
                f"all_decoder_diagnostics_status_bo_iter_{bo_iter:03d}.csv",
            ),
            index=False,
        )

        new_global = []
        for p in accepted:
            peptides.append(p)
            new_global.append(len(peptides) - 1)

        labeled_idx.extend(new_global)
        Z_lab = torch.cat([Z_lab, Znew])
        X_lab = torch.cat([X_lab, Xnew])
        Y_lab = torch.cat([Y_lab, Ynew])

        evaluated_set.update(accepted)
        generated_set.update(raw_peps)

        hv = float(Hypervolume(ref_point=ref).compute(Y_lab))
        improved = hv > best_hv + args.hv_tol
        if improved:
            best_hv = hv

        pf = save_pareto(
            peptides, labeled_idx, Y_lab, bo_iter, args.pareto_dir, args.obj_cols
        )
        hv_row = {
            "bo_iter": bo_iter,
            "n_raw_decoded": len(raw_peps),
            "n_accepted": len(accepted),
            "n_labeled": len(labeled_idx),
            "pareto_size": len(pf),
            "hypervolume": hv,
            "best_hypervolume": best_hv,
            "improved": int(improved),
            "acq_value": acq_value,
            "n_dominates_train": int(sum(
                r["dominates_any_train_pareto"] for r in iter_rows
            )),
        }
        hv_rows.append(hv_row)

        pd.DataFrame(hv_rows).to_csv(
            os.path.join(args.out_dir, "bo_hypervolume_history.csv"),
            index=False,
        )
        pd.DataFrame(all_rows).to_csv(
            os.path.join(args.out_dir, "all_accepted_candidates_scored.csv"),
            index=False,
        )

        print(
            f"Accepted {len(accepted)} peptides; "
            f"HV={hv:.6f}; best={best_hv:.6f}; "
            f"dominates_train={hv_row['n_dominates_train']}"
        )

    # Final Pareto
    Ycpu = Y_lab.detach().cpu()
    mask = pareto_mask_max(Ycpu)
    final_rows = []

    for k in torch.where(mask)[0].tolist():
        g = labeled_idx[k]
        p = peptides[g]
        row = {
            "global_index": g,
            "peptide": p,
            "is_novel": p not in train_set,
        }
        y = Ycpu[k]
        for j, c in enumerate(args.obj_cols):
            row[c] = float(y[j])
        row["dominates_train_pareto"] = any(
            dominates(y, y0.cpu()) for y0 in Y_train_pareto
        )
        row["n_train_pareto_members_dominated"] = sum(
            dominates(y, y0.cpu()) for y0 in Y_train_pareto
        )
        final_rows.append(row)

    final_df = pd.DataFrame(final_rows)
    final_path = os.path.join(
        args.out_dir,
        "bo_final_pareto_CU_gru_vae_realnvp_zK_qlogehvi_leakage_safe.csv",
    )
    final_df.to_csv(final_path, index=False)

    dom_df = (
        final_df[
            (final_df["is_novel"] == True)
            & (final_df["dominates_train_pareto"] == True)
        ].copy()
        if len(final_df) else pd.DataFrame()
    )
    dom_path = os.path.join(
        args.out_dir,
        "bo_pareto_dominates_train_CU_gru_vae_realnvp_zK_qlogehvi_leakage_safe.csv",
    )
    dom_df.to_csv(dom_path, index=False)

    print("\nDone.")
    print(f"Final BO Pareto: {len(final_df)} -> {final_path}")
    print(
        "Novel Pareto solutions dominating training Pareto: "
        f"{len(dom_df)} -> {dom_path}"
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-objective GP/qEHVI BO in RealNVP zK space."
    )

    p.add_argument(
        "--realnvp-checkpoint",
        required=True,
        help=(
            "Recommended: best_bo_ready_h64_z64_cu_realnvp.pt. "
            "For the attached history this checkpoint is epoch 150."
        ),
    )
    p.add_argument(
        "--finetune-script",
        required=True,
        help="Path to finetune_best_bo_ready_gru_vae_cu_realnvp.py",
    )
    p.add_argument("--data-csv", required=True)
    p.add_argument("--peptide-col", default="peptide_len10")
    p.add_argument(
        "--blackbox-script",
        default="black_box_fcn_mo_CU_f.py",
        help=(
            "Path to the Cu black-box scorer Python file defining blackbox_fc. "
            "Relative paths are resolved from the working directory, BO script "
            "directory, and fine-tuning script directory."
        ),
    )
    p.add_argument(
        "--project-root",
        action="append",
        default=[],
        help=(
            "Optional Python import root. May be passed multiple times. "
            "Use the directory CONTAINING the peptide_optimization package if "
            "the black-box scorer imports peptide_optimization.*."
        ),
    )
    p.add_argument("--obj-cols", nargs=4, default=OBJ_COLS)
    p.add_argument(
        "--pretraining-split-manifest",
        required=True,
        help="Same leakage-safe manifest used by RealNVP fine-tuning.",
    )
    p.add_argument(
        "--duplicate-policy",
        choices=["mean", "first", "error"],
        default="mean",
    )
    p.add_argument(
        "--unmapped-policy",
        choices=["drop", "error"],
        default="drop",
    )
    p.add_argument(
        "--allow-non-bo-ready-checkpoint",
        action="store_true",
        help="Deliberately allow a checkpoint not selected by minimum BO-ready score.",
    )

    p.add_argument(
        "--out-dir",
        default="bo_results_CU_gru_vae_realnvp_zK_qlogehvi_leakage_safe",
    )
    p.add_argument(
        "--decoder-dir",
        default="bo_decoder_monitoring_CU_gru_vae_realnvp_zK_qlogehvi_leakage_safe",
    )
    p.add_argument(
        "--pareto-dir",
        default="pareto_front_CU_gru_vae_realnvp_zK_qlogehvi_leakage_safe",
    )

    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--encode-batch-size", type=int, default=256)

    p.add_argument("--bo-iters", type=int, default=20)
    p.add_argument("--initial-labeled", type=int, default=256)
    p.add_argument("--q-batch", type=int, default=2)
    p.add_argument(
        "--optimize-q-one-at-a-time",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--num-restarts", type=int, default=4)
    p.add_argument("--raw-samples", type=int, default=64)
    p.add_argument("--mc-samples", type=int, default=32)
    p.add_argument(
        "--acqf",
        choices=["qehvi", "qlogehvi"],
        default="qlogehvi",
    )

    # RealNVP zK geometry
    p.add_argument(
        "--zk-bound",
        type=float,
        default=4.0,
        help="Initial symmetric zK bound; auto-expanded before BO if needed.",
    )
    p.add_argument(
        "--auto-expand-zk-bound",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--tr-radius-zk",
        type=float,
        default=0.75,
        help="Trust-region half width in raw zK coordinate units.",
    )

    p.add_argument("--max-partition-points", type=int, default=20)
    p.add_argument("--ref-margin", type=float, default=0.10)
    p.add_argument("--hv-tol", type=float, default=1e-6)

    # Decoder / novelty
    p.add_argument("--decoder-samples-per-zk", type=int, default=16)
    p.add_argument("--decoder-temperature", type=float, default=0.90)
    p.add_argument("--fallback-temperature", type=float, default=1.25)
    p.add_argument(
        "--novelty-sigmas",
        nargs="*",
        type=float,
        default=[0.05, 0.10, 0.20],
    )
    p.add_argument("--local-neighbors-per-sigma", type=int, default=4)
    p.add_argument(
        "--max-blackbox-per-iter",
        type=int,
        default=8,
        help="Set <=0 for no cap.",
    )

    p.add_argument(
        "--reject-training-peptides",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--reject-heldout-peptides",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reject exact matches to Cu validation/test peptides so held-out "
            "sequences remain untouched during prospective BO."
        ),
    )
    p.add_argument(
        "--reject-seen-peptides",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
