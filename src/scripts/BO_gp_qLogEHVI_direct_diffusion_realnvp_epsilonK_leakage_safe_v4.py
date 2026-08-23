from __future__ import annotations

"""
Leakage-safe prospective multi-objective Bayesian optimization in the
RealNVP-transformed direct-sequence-diffusion noise space epsilonK.

This script is matched to:
    finetune_cu_direct_sequence_diffusion_pca_realnvp_leakage_safe_h96_v3.py

Matched fine-tuning path
------------------------
peptide x0
    -> deterministic DDIM inversion
    -> epsilon0 (200-D)
    -> trained RealNVP flow
    -> epsilonK (200-D)                         [BO space]

BO candidate epsilonK*
    -> deterministic DDIM sampling
    -> peptide
    -> leakage/novelty filters
    -> Cu black-box objectives
    -> re-encode peptide:
         peptide -> DDIM inversion -> epsilon0 -> RealNVP -> epsilonK_actual
    -> update GP / Pareto / hypervolume

Important consistency choices
-----------------------------
1. Fine-tuning defaults to:
       --no-project-inverted-epsilon-to-sphere
       --no-project-flow-output-to-sphere
   Therefore this BO script uses native, NON-SPHERICAL epsilonK by default.

2. The coordinate CSV exported by the fine-tuning run is used as the authoritative
   leakage-safe peptide/split manifest. Only rows with split=train are historical
   BO observations. Validation/test rows are held out and rejected if generated.

3. The preferred checkpoint is the minimum-validation-score-MSE checkpoint because
   the fine-tuning objective head is the Cu scalar score head and the RealNVP flow
   is optimized jointly with that score signal. For the attached history this is
   epoch 100:
       best_val_score_mse_cu_direct_diffusion_noise_flow.pt
   The exported BO coordinate CSV is also produced after the final epoch, so epoch
   100 is coordinate-consistent with that export.

4. The GP models four true black-box objectives, all maximized:
       chelation_sub, solubility_sub, stability_sub, expression_sub

5. The GP input box is fixed once from TRAIN epsilonK coordinates. It does not
   change across BO iterations.

6. The hypervolume reference point is fixed once from the initial labeled design.

7. Every accepted discrete peptide is re-encoded to its OWN epsilonK_actual before
   its black-box objective vector is appended to the GP dataset.

8. Training, held-out validation/test, already evaluated, and already generated
   peptides are rejected before expensive black-box scoring.
"""

import argparse
import gc
import importlib.util
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import gpytorch
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
)
try:
    from botorch.acquisition.multi_objective.logei import (
        qLogExpectedHypervolumeImprovement,
    )
except Exception:
    qLogExpectedHypervolumeImprovement = None

from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions import (
    NondominatedPartitioning,
)
from botorch.utils.multi_objective.hypervolume import Hypervolume


AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_I = {a: i for i, a in enumerate(AA)}
I_TO_AA = {i: a for i, a in enumerate(AA)}
SEQ_LEN = 10
VOCAB = 20
BO_DIM = SEQ_LEN * VOCAB
OBJ_COLS = [
    "chelation_sub",
    "solubility_sub",
    "stability_sub",
    "expression_sub",
]


# =============================================================================
# Utilities
# =============================================================================

def set_seed(seed: int) -> None:
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


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clean_peptide(x: object) -> Optional[str]:
    p = str(x).strip().upper()
    if len(p) != SEQ_LEN or any(a not in AA_TO_I for a in p):
        return None
    return p


def onehot_encode(peptides: Sequence[str]) -> torch.Tensor:
    x = torch.zeros((len(peptides), SEQ_LEN, VOCAB), dtype=torch.float32)
    for n, pep in enumerate(peptides):
        p = clean_peptide(pep)
        if p is None:
            raise ValueError(f"Invalid peptide: {pep!r}")
        for t, aa in enumerate(p):
            x[n, t, AA_TO_I[aa]] = 1.0
    return x


def decode_tensor_to_peptides(x: torch.Tensor) -> List[str]:
    idx = x.argmax(dim=-1).detach().cpu().tolist()
    return ["".join(I_TO_AA[int(i)] for i in row) for row in idx]


def selected_token_confidence(x: torch.Tensor):
    probs = torch.softmax(x, dim=-1)
    conf, idx = probs.max(dim=-1)
    peps = [
        "".join(I_TO_AA[int(i)] for i in row)
        for row in idx.detach().cpu().tolist()
    ]
    return (
        conf.mean(dim=-1).detach().cpu().numpy(),
        conf.min(dim=-1).values.detach().cpu().numpy(),
        peps,
    )


def levenshtein(a: str, b: str) -> int:
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
    A = Y.unsqueeze(1)
    B = Y.unsqueeze(0)
    dom = (A >= B).all(-1) & (A > B).any(-1)
    dom.fill_diagonal_(False)
    return ~dom.any(dim=0)


def dominates(a: torch.Tensor, b: torch.Tensor) -> bool:
    return bool(((a >= b).all() & (a > b).any()).item())


def eps_to_unit(eps: torch.Tensor, bound: float, clamp: bool = True):
    x = (eps + float(bound)) / (2.0 * float(bound))
    return x.clamp(0.0, 1.0) if clamp else x


def unit_to_eps(x: torch.Tensor, bound: float):
    return 2.0 * float(bound) * x - float(bound)


def _add_sys_path_once(path: Path):
    path = path.resolve()
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


def configure_import_paths(blackbox_script: str, project_roots: Sequence[str]):
    candidates = [Path.cwd(), Path(__file__).resolve().parent]
    candidates.extend(Path(p).expanduser() for p in (project_roots or []))
    candidates.append(Path(blackbox_script).expanduser().resolve().parent)
    expanded = []
    for c in candidates:
        expanded.append(c)
        expanded.extend(list(c.parents)[:6])
    for c in expanded:
        _add_sys_path_once(c)
        if (c / "src" / "peptide_optimization").is_dir():
            _add_sys_path_once(c / "src")


def load_blackbox_function(blackbox_script: str, project_roots: Sequence[str]):
    p = Path(blackbox_script).expanduser()
    if not p.is_file():
        candidates = [
            Path.cwd() / blackbox_script,
            Path(__file__).resolve().parent / blackbox_script,
        ]
        p = next((q for q in candidates if q.is_file()), p)
    if not p.is_file():
        raise FileNotFoundError(
            f"Could not find Cu black-box script: {blackbox_script}"
        )
    p = p.resolve()
    configure_import_paths(str(p), project_roots)
    name = "cu_blackbox_direct_diffusion_bo"
    spec = importlib.util.spec_from_file_location(name, str(p))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    fn = getattr(mod, "blackbox_fc", None)
    if fn is None or not callable(fn):
        raise AttributeError(f"{p} does not define callable blackbox_fc")
    print(f"Loaded Cu black-box function from: {p}")
    return fn


# =============================================================================
# Matched H96 direct diffusion + RealNVP architecture
# =============================================================================

@dataclass
class SequenceDiffusionConfig:
    hidden_size: int = 96
    n_layers: int = 2
    dropout: float = 0.0
    time_dim: int = 32
    train_steps: int = 100
    beta_start: float = 1e-5
    beta_end: float = 8e-3


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim % 2:
            raise ValueError("time_dim must be even")
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        exponent = (
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        frequencies = torch.exp(exponent)
        angles = t.float().unsqueeze(-1) * frequencies.unsqueeze(0)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class DirectSequenceDenoiser(nn.Module):
    def __init__(self, cfg: SequenceDiffusionConfig):
        super().__init__()
        self.cfg = cfg
        self.x_proj = nn.Linear(VOCAB, cfg.hidden_size)
        self.time_embedding = SinusoidalTimeEmbedding(cfg.time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.time_dim, cfg.hidden_size),
            nn.SiLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
        )
        self.gru = nn.GRU(
            input_size=cfg.hidden_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.n_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.n_layers > 1 else 0.0,
        )
        self.out_norm = nn.LayerNorm(cfg.hidden_size)
        self.noise_head = nn.Linear(cfg.hidden_size, VOCAB)
        self.x0_logits_head = nn.Linear(cfg.hidden_size, VOCAB)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor):
        h = self.x_proj(x_t)
        h = h + self.time_mlp(self.time_embedding(t)).unsqueeze(1)
        h, _ = self.gru(h)
        h = F.silu(self.out_norm(h))
        return {
            "predicted_noise": self.noise_head(h),
            "x0_logits": self.x0_logits_head(h),
        }


class DirectSequenceDiffusion(nn.Module):
    def __init__(self, cfg: SequenceDiffusionConfig):
        super().__init__()
        self.cfg = cfg
        self.denoiser = DirectSequenceDenoiser(cfg)
        betas = torch.linspace(
            cfg.beta_start, cfg.beta_end, cfg.train_steps, dtype=torch.float32
        )
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", torch.cumprod(alphas, dim=0))

    @torch.no_grad()
    def ddim_sample(self, base_noise: torch.Tensor, inference_steps: int = 50):
        if base_noise.ndim == 2:
            x = base_noise.view(-1, SEQ_LEN, VOCAB)
        else:
            x = base_noise
        n = self.cfg.train_steps
        steps = min(max(int(inference_steps), 1), n)
        ts = torch.linspace(n - 1, 0, steps, device=x.device).round().long()
        ts = torch.unique_consecutive(ts)
        for k, t_scalar in enumerate(ts):
            t = torch.full(
                (x.size(0),), int(t_scalar.item()),
                device=x.device, dtype=torch.long
            )
            out = self.denoiser(x, t)
            pred = out["predicted_noise"]
            ab = self.alpha_bars[t_scalar]
            x0 = (x - torch.sqrt(1.0 - ab) * pred) / torch.sqrt(ab)
            if k == len(ts) - 1:
                x = x0
                break
            tp = ts[k + 1]
            abp = self.alpha_bars[tp]
            x = torch.sqrt(abp) * x0 + torch.sqrt(1.0 - abp) * pred
        return x

    @torch.no_grad()
    def ddim_invert(self, x0: torch.Tensor, inference_steps: int = 50):
        n = self.cfg.train_steps
        steps = min(max(int(inference_steps), 1), n)
        ts = torch.linspace(0, n - 1, steps, device=x0.device).round().long()
        ts = torch.unique_consecutive(ts)
        x = x0
        for k, t_scalar in enumerate(ts[:-1]):
            t = torch.full(
                (x.size(0),), int(t_scalar.item()),
                device=x.device, dtype=torch.long
            )
            out = self.denoiser(x, t)
            pred = out["predicted_noise"]
            ab = self.alpha_bars[t_scalar]
            x0_pred = (x - torch.sqrt(1.0 - ab) * pred) / torch.sqrt(ab)
            tn = ts[k + 1]
            abn = self.alpha_bars[tn]
            x = torch.sqrt(abn) * x0_pred + torch.sqrt(1.0 - abn) * pred
        return x


class AffineCouplingBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        mask: torch.Tensor,
        max_scale: float = 0.5,
    ):
        super().__init__()
        self.max_scale = float(max_scale)
        self.register_buffer("mask", mask.float())
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * dim),
        )

    def forward(self, x: torch.Tensor):
        xm = x * self.mask
        s, t = self.net(xm).chunk(2, dim=-1)
        s = torch.tanh(s) * self.max_scale
        inv = 1.0 - self.mask
        y = xm + inv * (x * torch.exp(s) + t)
        return y, (inv * s).sum(-1)

    def inverse(self, y: torch.Tensor):
        ym = y * self.mask
        s, t = self.net(ym).chunk(2, dim=-1)
        s = torch.tanh(s) * self.max_scale
        inv = 1.0 - self.mask
        x = ym + inv * ((y - t) * torch.exp(-s))
        return x, -(inv * s).sum(-1)


class RealNVPFlow(nn.Module):
    def __init__(
        self,
        dim: int = BO_DIM,
        n_layers: int = 4,
        hidden_dim: int = 128,
        max_scale: float = 0.5,
    ):
        super().__init__()
        masks = []
        for layer in range(n_layers):
            m = np.zeros(dim, dtype=np.float32)
            m[layer % 2 :: 2] = 1.0
            masks.append(torch.tensor(m))
        self.blocks = nn.ModuleList([
            AffineCouplingBlock(dim, hidden_dim, m, max_scale)
            for m in masks
        ])

    def forward(self, x: torch.Tensor):
        h = x
        logdet = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
        for block in self.blocks:
            h, ld = block(h)
            logdet = logdet + ld
        return h, logdet

    def inverse(self, y: torch.Tensor):
        h = y
        logdet = torch.zeros(y.size(0), dtype=y.dtype, device=y.device)
        for block in reversed(self.blocks):
            h, ld = block.inverse(h)
            logdet = logdet + ld
        return h, logdet


# =============================================================================
# Checkpoint and safe coordinate loading
# =============================================================================

def load_checkpoint(path: str, device: torch.device):
    ckpt = torch_load_full(path, device)
    if ckpt.get("checkpoint_type") != "cu_direct_sequence_diffusion_noise_flow":
        raise ValueError(
            "Expected cu_direct_sequence_diffusion_noise_flow checkpoint, got "
            f"{ckpt.get('checkpoint_type')!r}"
        )

    cfg = SequenceDiffusionConfig(**ckpt["diffusion_config"])
    model = DirectSequenceDiffusion(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    a = ckpt.get("args", {}) or {}
    flow = RealNVPFlow(
        dim=BO_DIM,
        n_layers=int(a.get("flow_layers", 4)),
        hidden_dim=int(a.get("flow_hidden_dim", 128)),
        max_scale=float(a.get("flow_max_scale", 0.5)),
    ).to(device)
    flow.load_state_dict(ckpt["flow_state_dict"], strict=True)

    model.eval()
    flow.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in flow.parameters():
        p.requires_grad_(False)

    ddim_steps = int(a.get("ddim_steps", 50))
    projected_eps0 = bool(a.get("project_inverted_epsilon_to_sphere", False))
    projected_epsK = bool(a.get("project_flow_output_to_sphere", False))

    if projected_eps0 or projected_epsK:
        raise RuntimeError(
            "This matched BO script expects the native non-spherical fine-tuning "
            "configuration. The supplied checkpoint records sphere projection. "
            "Use a separately matched spherical BO implementation for that ablation."
        )

    print(f"Loaded checkpoint: {path}")
    print("epoch:", ckpt.get("epoch"))
    print("selection:", ckpt.get("selection"))
    print("diffusion_config:", ckpt.get("diffusion_config"))
    print(
        "flow:",
        {
            "layers": a.get("flow_layers", 4),
            "hidden_dim": a.get("flow_hidden_dim", 128),
            "max_scale": a.get("flow_max_scale", 0.5),
        },
    )
    print("DDIM steps:", ddim_steps)
    print("project_inverted_epsilon_to_sphere:", projected_eps0)
    print("project_flow_output_to_sphere:", projected_epsK)

    return model, flow, ckpt, ddim_steps


def load_leakage_safe_training_data(args):
    coord = pd.read_csv(args.coordinate_csv)
    eps_cols = [f"epsilonK_{i:03d}" for i in range(BO_DIM)]
    missing = [c for c in eps_cols if c not in coord.columns]
    if missing:
        raise KeyError(
            f"Coordinate CSV is missing epsilonK columns; examples={missing[:5]}"
        )
    if "split" not in coord.columns:
        raise KeyError(
            "Coordinate CSV must contain the leakage-safe `split` column exported "
            "by the fine-tuning framework."
        )

    pep_col_coord = "peptide" if "peptide" in coord.columns else args.peptide_col
    if pep_col_coord not in coord.columns:
        raise KeyError("Could not find peptide column in coordinate CSV.")

    coord = coord.copy()
    coord["_pep"] = coord[pep_col_coord].map(clean_peptide)
    coord["_split"] = coord["split"].astype(str).str.strip().str.lower()
    coord = coord.dropna(subset=["_pep"]).drop_duplicates("_pep", keep="first")

    # Scored CSV provides the four true BO objectives.
    scored = pd.read_csv(args.data_csv).copy()
    if args.peptide_col not in scored.columns:
        raise KeyError(f"{args.peptide_col!r} not found in scored CSV")
    for c in args.obj_cols:
        if c not in scored.columns:
            raise KeyError(f"{c!r} not found in scored CSV")
    scored["_pep"] = scored[args.peptide_col].map(clean_peptide)
    for c in args.obj_cols:
        scored[c] = pd.to_numeric(scored[c], errors="coerce")
    scored = scored.dropna(subset=["_pep"] + list(args.obj_cols))

    if scored["_pep"].duplicated().any():
        scored = (
            scored.groupby("_pep", as_index=False)[list(args.obj_cols)]
            .mean()
        )

    work = coord.merge(
        scored[["_pep"] + list(args.obj_cols)],
        on="_pep",
        how="inner",
        validate="one_to_one",
    )

    allowed = {"train", "validation", "val", "test"}
    bad = sorted(set(work["_split"]) - allowed)
    if bad:
        raise ValueError(f"Unexpected split labels in coordinate CSV: {bad}")

    # Normalize validation alias.
    work["_split"] = work["_split"].replace({"val": "validation"})

    train = work[work["_split"] == "train"].copy().reset_index(drop=True)
    val = work[work["_split"] == "validation"].copy().reset_index(drop=True)
    test = work[work["_split"] == "test"].copy().reset_index(drop=True)

    if train.empty or val.empty or test.empty:
        raise RuntimeError(
            f"Required safe splits are empty: train={len(train)}, "
            f"validation={len(val)}, test={len(test)}"
        )

    tr_set = set(train["_pep"])
    va_set = set(val["_pep"])
    te_set = set(test["_pep"])
    if tr_set & va_set or tr_set & te_set or va_set & te_set:
        raise RuntimeError("Exact peptide overlap exists between safe splits.")

    E_train = torch.tensor(
        train[eps_cols].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    Y_train = torch.tensor(
        train[list(args.obj_cols)].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    peptides_train = train["_pep"].tolist()
    heldout_set = va_set | te_set

    audit = {
        "coordinate_csv_rows": int(len(coord)),
        "joined_scored_rows": int(len(work)),
        "train_rows": int(len(train)),
        "validation_rows": int(len(val)),
        "test_rows": int(len(test)),
        "exact_train_validation_overlap": int(len(tr_set & va_set)),
        "exact_train_test_overlap": int(len(tr_set & te_set)),
        "exact_validation_test_overlap": int(len(va_set & te_set)),
        "historical_bo_observations": "train_only",
        "heldout_used_for_gp_fit": False,
        "heldout_used_for_score_cache": False,
    }
    Path(args.out_dir, "bo_leakage_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    return peptides_train, E_train, Y_train, heldout_set, audit


# =============================================================================
# Encoding / decoding
# =============================================================================

@torch.no_grad()
def peptide_to_epsilonK(
    model,
    flow,
    peptide: str,
    ddim_steps: int,
):
    x = onehot_encode([peptide]).to(next(model.parameters()).device)
    eps0 = model.ddim_invert(x, inference_steps=ddim_steps).reshape(1, -1)
    epsK, _ = flow(eps0)
    return epsK


@torch.no_grad()
def decode_from_epsilonK(model, epsK, ddim_steps):
    x0_hat = model.ddim_sample(epsK, inference_steps=ddim_steps)
    mean_conf, min_conf, peps = selected_token_confidence(x0_hat)
    return peps, mean_conf, min_conf


@torch.no_grad()
def decode_candidates(model, E, bo_iter, args, ddim_steps):
    rows = []
    peps = []

    for j in range(len(E)):
        center = E[j:j+1]
        p0, cm, cmin = decode_from_epsilonK(model, center, ddim_steps)
        center_pep = p0[0]
        peps.append(center_pep)
        rows.append({
            "bo_iter": bo_iter,
            "epsilonK_candidate_index": j,
            "decode_type": "argmax_ddim",
            "peptide": center_pep,
            "decoder_confidence_mean": float(cm[0]),
            "decoder_confidence_min": float(cmin[0]),
            "epsilonK_norm": float(center.norm().cpu()),
            "local_noise_l2": 0.0,
            "edit_distance_to_center": 0,
        })

        for sigma in args.novelty_sigmas:
            for _ in range(args.local_neighbors_per_sigma):
                en = center + torch.randn_like(center) * float(sigma)
                # Candidate is kept in the fixed BO box; no sphere projection.
                en = en.clamp(
                    -args.resolved_epsilon_bound,
                    args.resolved_epsilon_bound,
                )
                pp, mcm, mcmin = decode_from_epsilonK(model, en, ddim_steps)
                p = pp[0]
                peps.append(p)
                rows.append({
                    "bo_iter": bo_iter,
                    "epsilonK_candidate_index": j,
                    "decode_type": f"local_epsilonK_sigma_{sigma:g}",
                    "peptide": p,
                    "decoder_confidence_mean": float(mcm[0]),
                    "decoder_confidence_min": float(mcmin[0]),
                    "epsilonK_norm": float(en.norm().cpu()),
                    "local_noise_l2": float((en - center).norm().cpu()),
                    "edit_distance_to_center": levenshtein(center_pep, p),
                })

    d = pd.DataFrame(rows)
    d["is_duplicate_raw_decode"] = d.duplicated("peptide", keep="first")
    d.to_csv(
        Path(args.decoder_dir, f"decoder_diagnostics_bo_iter_{bo_iter:03d}.csv"),
        index=False,
    )
    return list(dict.fromkeys(peps)), d


# =============================================================================
# GP / black box / novelty
# =============================================================================

def fit_mo_gp(X, Y, device):
    models = []
    for j in range(Y.shape[1]):
        gp = SingleTaskGP(
            X,
            Y[:, j:j+1],
            outcome_transform=Standardize(m=1),
        ).to(device)
        gp.likelihood.noise_covar.register_constraint(
            "raw_noise", gpytorch.constraints.GreaterThan(1e-6)
        )
        gp.likelihood.noise = 1e-4
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        models.append(gp)
    return ModelListGP(*models)


def evaluate_blackbox(peptides, cache, obj_cols, blackbox_fc):
    uncached = [p for p in peptides if p not in cache]
    if uncached:
        scored = blackbox_fc(uncached)
        if isinstance(scored, torch.Tensor):
            arr = scored.detach().cpu().numpy()
            if arr.shape != (len(uncached), len(obj_cols)):
                raise ValueError(
                    f"blackbox tensor shape={arr.shape}, expected "
                    f"{(len(uncached), len(obj_cols))}"
                )
            for p, row in zip(uncached, arr):
                cache[p] = [float(v) for v in row]
        elif isinstance(scored, pd.DataFrame):
            for _, row in scored.iterrows():
                p = clean_peptide(
                    row.get("peptide_len10", row.get("peptide", ""))
                )
                if p is not None:
                    cache[p] = [float(row[c]) for c in obj_cols]
        else:
            raise TypeError(f"Unsupported blackbox return type: {type(scored)}")

    missing = [p for p in peptides if p not in cache]
    if missing:
        raise RuntimeError(f"No black-box score returned for {missing[:5]}")
    return torch.tensor([cache[p] for p in peptides], dtype=torch.float32)


def filter_novel(
    peps,
    train_set,
    heldout_set,
    evaluated_set,
    generated_set,
):
    accepted, rejected = [], []
    for p in peps:
        reason = None
        if clean_peptide(p) is None:
            reason = "invalid"
        elif p in train_set:
            reason = "in_training"
        elif p in heldout_set:
            reason = "in_validation_or_test_holdout"
        elif p in evaluated_set:
            reason = "already_evaluated"
        elif p in generated_set:
            reason = "already_generated"
        if reason is None:
            accepted.append(p)
        else:
            rejected.append((p, reason))
    return accepted, rejected


def initial_indices(Y, n_init, seed):
    pareto = torch.where(pareto_mask_max(Y))[0].cpu().tolist()
    pset = set(pareto)
    rest = [i for i in range(len(Y)) if i not in pset]
    rng = np.random.default_rng(seed)
    need = max(0, int(n_init) - len(pareto))
    extra = (
        rng.choice(rest, size=min(need, len(rest)), replace=False).tolist()
        if need else []
    )
    return list(dict.fromkeys(pareto + extra))


def save_pareto(peptides, labeled_idx, Y, bo_iter, out_dir, obj_cols):
    m = pareto_mask_max(Y.detach().cpu())
    rows = []
    for k in torch.where(m)[0].tolist():
        g = int(labeled_idx[k])
        row = {
            "bo_iter": bo_iter,
            "global_index": g,
            "peptide": peptides[g],
        }
        for j, c in enumerate(obj_cols):
            row[c] = float(Y[k, j].detach().cpu())
        rows.append(row)
    d = pd.DataFrame(rows)
    d.to_csv(
        Path(out_dir, f"pareto_front_bo_iter_{bo_iter:03d}.csv"),
        index=False,
    )
    return d


# =============================================================================
# Main BO
# =============================================================================

def run(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    ensure_dir(args.out_dir)
    ensure_dir(args.decoder_dir)
    ensure_dir(args.pareto_dir)

    model, flow, ckpt, ddim_steps = load_checkpoint(
        args.flow_checkpoint, device
    )

    if (
        not args.allow_non_score_checkpoint
        and ckpt.get("selection") != "minimum_validation_score_mse"
    ):
        raise RuntimeError(
            "For this fine-tuning framework, prospective BO should use the "
            "minimum-validation-score-MSE checkpoint. "
            f"Found selection={ckpt.get('selection')!r}. "
            "Use --allow-non-score-checkpoint only for a deliberate ablation."
        )

    blackbox_fc = load_blackbox_function(
        args.blackbox_script, args.project_root
    )

    (
        peptides,
        E_all_cpu,
        Y_all_cpu,
        heldout_set,
        leakage_audit,
    ) = load_leakage_safe_training_data(args)

    E_all = E_all_cpu.to(device)
    Y_full = Y_all_cpu.to(device)
    train_set = set(peptides)

    # Fixed native epsilonK box derived once from TRAIN coordinates.
    max_abs = float(E_all.abs().max().cpu())
    bound = max(float(args.epsilonK_bound), 1.05 * max_abs)
    args.resolved_epsilon_bound = bound

    X_all = eps_to_unit(E_all, bound, clamp=False)
    if bool(((X_all < 0) | (X_all > 1)).any().item()):
        raise RuntimeError("Fixed epsilonK box does not contain TRAIN coordinates.")

    print(
        f"Leakage-safe BO train={len(peptides)}; heldout={len(heldout_set)}"
    )
    print(
        f"Native epsilonK norm mean={float(E_all.norm(dim=-1).mean().cpu()):.6f}; "
        f"std={float(E_all.norm(dim=-1).std().cpu()):.6f}; "
        f"max|coord|={max_abs:.6f}"
    )
    print(f"Fixed epsilonK box=[-{bound:.6f}, +{bound:.6f}]")
    print(
        f"GP cube min={float(X_all.min().cpu()):.6f}, "
        f"max={float(X_all.max().cpu()):.6f}"
    )

    tr_mask = pareto_mask_max(Y_full)
    tr_idx = torch.where(tr_mask)[0]
    Y_train_pareto = Y_full[tr_mask]

    pd.DataFrame({
        "global_index": tr_idx.detach().cpu().numpy(),
        "peptide": [peptides[i] for i in tr_idx.detach().cpu().tolist()],
        **{
            c: Y_train_pareto[:, j].detach().cpu().numpy()
            for j, c in enumerate(args.obj_cols)
        },
    }).to_csv(
        Path(args.out_dir, "train_only_pareto_CU_direct_diffusion_realnvp_epsilonK.csv"),
        index=False,
    )

    cache = {
        p: [float(v) for v in y]
        for p, y in zip(
            peptides,
            Y_full.detach().cpu().numpy().tolist(),
        )
    }

    evaluated_set = set()
    generated_set = set()

    labeled_idx = initial_indices(
        Y_full, args.initial_labeled, args.seed + 42
    )
    ii = torch.tensor(labeled_idx, device=device)
    E_lab = E_all[ii]
    X_lab = X_all[ii]
    Y_lab = Y_full[ii]
    evaluated_set.update(peptides[i] for i in labeled_idx)

    # Fixed hypervolume reference.
    ref = Y_lab.min(0).values - float(args.ref_margin)
    initial_hv = float(Hypervolume(ref_point=ref).compute(Y_lab))
    best_hv = initial_hv

    run_cfg = vars(args).copy()
    run_cfg.update({
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_selection": ckpt.get("selection"),
        "checkpoint_metrics": ckpt.get("metrics", {}),
        "bo_space": "native_realnvp_epsilonK_direct_sequence_diffusion",
        "bo_dim": BO_DIM,
        "ddim_steps_from_checkpoint": ddim_steps,
        "sphere_projection_used": False,
        "historical_observations": "train_only",
        "validation_test_used_for_gp": False,
        "validation_test_used_for_cache": False,
        "heldout_exact_rejection": True,
        "fixed_epsilonK_bound": bound,
        "fixed_hv_reference": ref.detach().cpu().tolist(),
        "initial_hypervolume": initial_hv,
        "leakage_audit": leakage_audit,
    })
    Path(args.out_dir, "bo_run_config.json").write_text(
        json.dumps(run_cfg, indent=2, default=str),
        encoding="utf-8",
    )

    hv_rows = []
    all_rows = []

    for bo_iter in range(args.bo_iters):
        print(f"\n=== BO iter {bo_iter + 1}/{args.bo_iters} ===")

        model_gp = fit_mo_gp(
            X_lab.double(),
            Y_lab.double(),
            device,
        )

        # Balanced Pareto center.
        nd = torch.where(pareto_mask_max(Y_lab))[0]
        Ynd = Y_lab[nd]
        lo = Ynd.min(0).values
        span = (Ynd.max(0).values - lo).clamp_min(1e-12)
        balanced = ((Ynd - lo) / span).mean(-1)
        center_i = int(nd[torch.argmax(balanced)].item())
        xc = X_lab[center_i].double()

        r_unit = float(args.tr_radius_epsilonK) / (2.0 * bound)
        bounds = torch.stack([
            (xc - r_unit).clamp(0.0, 1.0),
            (xc + r_unit).clamp(0.0, 1.0),
        ])

        Ynd_gp = Y_lab.double()[pareto_mask_max(Y_lab.double())]
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

        partitioning = NondominatedPartitioning(
            ref_point=ref.double(), Y=Ypart
        )
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
            model=model_gp,
            ref_point=ref.double().detach().cpu().tolist(),
            partitioning=partitioning,
            sampler=sampler,
        ).to(device)

        parts, avals = [], []
        repeats = args.q_batch if args.optimize_q_one_at_a_time else 1
        q_inner = 1 if args.optimize_q_one_at_a_time else args.q_batch

        for _ in range(repeats):
            try:
                u, a = optimize_acqf(
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
                u, a = optimize_acqf(
                    acq,
                    bounds=bounds,
                    q=1,
                    num_restarts=max(1, args.num_restarts // 2),
                    raw_samples=max(16, args.raw_samples // 4),
                    sequential=True,
                )
            parts.append(u.detach())
            avals.append(a.detach().reshape(-1)[0])

        Ucand = torch.cat(parts).clamp(0.0, 1.0)
        Ecand = unit_to_eps(Ucand, bound).float()
        acq_value = float(torch.stack(avals).mean().cpu())

        raw_peps, diag = decode_candidates(
            model, Ecand, bo_iter, args, ddim_steps
        )
        accepted, rejected = filter_novel(
            raw_peps,
            train_set,
            heldout_set,
            evaluated_set,
            generated_set,
        )

        if args.max_blackbox_per_iter > 0:
            accepted = accepted[:args.max_blackbox_per_iter]

        rejected_map = dict(rejected)
        status = diag.copy()

        if not accepted:
            generated_set.update(raw_peps)
            status["accepted_for_blackbox"] = False
            status["rejected_reason"] = status["peptide"].map(
                lambda p: rejected_map.get(p, "not_selected")
            )
            status.to_csv(
                Path(
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
                Path(args.out_dir, "bo_hypervolume_history.csv"),
                index=False,
            )
            continue

        Ynew = evaluate_blackbox(
            accepted, cache, args.obj_cols, blackbox_fc
        ).to(device)

        # Critical discrete-consistency update:
        # accepted peptide -> actual epsilon0 -> RealNVP -> actual epsilonK.
        Enew = torch.cat([
            peptide_to_epsilonK(model, flow, p, ddim_steps)
            for p in accepted
        ], dim=0)

        Xnew = eps_to_unit(Enew, bound, clamp=False)
        outside = ((Xnew < 0) | (Xnew > 1)).any(-1)
        if bool(outside.any().item()):
            bad = [accepted[i] for i in torch.where(outside)[0].tolist()]
            raise RuntimeError(
                "A newly evaluated peptide re-encoded outside the fixed TRAIN "
                f"epsilonK box. Examples={bad[:5]}. Re-run with a larger "
                "--epsilonK-bound. The script refuses to clip actual GP coordinates."
            )

        iter_rows = []
        for i, (p, y) in enumerate(zip(accepted, Ynew)):
            n_dom = sum(dominates(y, y0) for y0 in Y_train_pareto)
            row = {
                "bo_iter": bo_iter,
                "peptide": p,
                "acq_value": acq_value,
                "epsilonK_norm": float(Enew[i].norm().cpu()),
                "epsilonK_max_abs_coordinate": float(Enew[i].abs().max().cpu()),
                "dominated_by_train_pareto": int(
                    any(dominates(y0, y) for y0 in Y_train_pareto)
                ),
                "dominates_any_train_pareto": int(n_dom > 0),
                "n_train_pareto_members_dominated": int(n_dom),
            }
            for j, c in enumerate(args.obj_cols):
                row[c] = float(y[j].cpu())
            iter_rows.append(row)
            all_rows.append(row)

        pd.DataFrame(iter_rows).to_csv(
            Path(
                args.out_dir,
                f"accepted_candidates_scored_bo_iter_{bo_iter:03d}.csv",
            ),
            index=False,
        )

        status["accepted_for_blackbox"] = status["peptide"].isin(accepted)
        status["rejected_reason"] = status["peptide"].map(
            lambda p: "" if p in accepted
            else rejected_map.get(p, "not_selected")
        )
        for j, c in enumerate(args.obj_cols):
            mp = {p: float(y[j].cpu()) for p, y in zip(accepted, Ynew)}
            status[c] = status["peptide"].map(
                lambda p: mp.get(p, np.nan)
            )
        status.to_csv(
            Path(
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
        E_lab = torch.cat([E_lab, Enew], 0)
        X_lab = torch.cat([X_lab, Xnew], 0)
        Y_lab = torch.cat([Y_lab, Ynew], 0)
        evaluated_set.update(accepted)
        generated_set.update(raw_peps)

        hv = float(Hypervolume(ref_point=ref).compute(Y_lab))
        improved = hv > best_hv + args.hv_tol
        if improved:
            best_hv = hv

        pf = save_pareto(
            peptides,
            labeled_idx,
            Y_lab,
            bo_iter,
            args.pareto_dir,
            args.obj_cols,
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
            "max_train_pareto_members_dominated": int(max(
                r["n_train_pareto_members_dominated"] for r in iter_rows
            )),
        }
        hv_rows.append(hv_row)

        pd.DataFrame(hv_rows).to_csv(
            Path(args.out_dir, "bo_hypervolume_history.csv"), index=False
        )
        pd.DataFrame(all_rows).to_csv(
            Path(args.out_dir, "all_accepted_candidates_scored.csv"),
            index=False,
        )

        print(
            f"Accepted {len(accepted)} peptides; HV={hv:.6f}; "
            f"best={best_hv:.6f}; dominates_train={hv_row['n_dominates_train']}; "
            f"max_train_members_dominated="
            f"{hv_row['max_train_pareto_members_dominated']}"
        )

    # Final Pareto and exact domination counts.
    Ycpu = Y_lab.detach().cpu()
    m = pareto_mask_max(Ycpu)
    train_p_cpu = Y_train_pareto.detach().cpu()

    final_rows = []
    for k in torch.where(m)[0].tolist():
        g = int(labeled_idx[k])
        p = peptides[g]
        y = Ycpu[k]
        n_dom = sum(dominates(y, y0) for y0 in train_p_cpu)
        row = {
            "global_index": g,
            "peptide": p,
            "is_novel": p not in train_set,
            "dominates_train_pareto": bool(n_dom > 0),
            "n_train_pareto_members_dominated": int(n_dom),
        }
        for j, c in enumerate(args.obj_cols):
            row[c] = float(y[j])
        final_rows.append(row)

    final_df = pd.DataFrame(final_rows)
    final_path = Path(
        args.out_dir,
        "bo_final_pareto_CU_direct_diffusion_realnvp_epsilonK_qlogehvi_leakage_safe.csv",
    )
    final_df.to_csv(final_path, index=False)

    dom_df = (
        final_df[
            (final_df["is_novel"] == True)
            & (final_df["dominates_train_pareto"] == True)
        ].copy()
        if len(final_df) else pd.DataFrame()
    )
    if len(dom_df):
        dom_df = dom_df.sort_values(
            "n_train_pareto_members_dominated", ascending=False
        )
    dom_path = Path(
        args.out_dir,
        "bo_pareto_dominates_train_CU_direct_diffusion_realnvp_epsilonK_qlogehvi_leakage_safe.csv",
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
        description=(
            "Leakage-safe multi-objective GP/qLogEHVI BO in native 200-D "
            "RealNVP epsilonK space of the direct sequence diffusion framework."
        )
    )
    p.add_argument(
        "--flow-checkpoint",
        required=True,
        help=(
            "Recommended for attached history: "
            "best_val_score_mse_cu_direct_diffusion_noise_flow.pt (epoch 100)."
        ),
    )
    p.add_argument("--data-csv", required=True)
    p.add_argument(
        "--coordinate-csv",
        required=True,
        help=(
            "cu_direct_diffusion_noise_flow_coordinates_for_bo.csv exported by "
            "the SAME fine-tuning run. Its split column is used as the authoritative "
            "leakage-safe manifest."
        ),
    )
    p.add_argument("--blackbox-script", default="black_box_fcn_mo_CU_f.py")
    p.add_argument("--project-root", action="append", default=[])
    p.add_argument("--peptide-col", default="peptide_len10")
    p.add_argument("--obj-cols", nargs=4, default=OBJ_COLS)
    p.add_argument(
        "--allow-non-score-checkpoint",
        action="store_true",
        help="Allow a checkpoint not selected by minimum validation score MSE.",
    )

    p.add_argument(
        "--out-dir",
        default="bo_results_CU_direct_diffusion_realnvp_epsilonK_leakage_safe_v4",
    )
    p.add_argument(
        "--decoder-dir",
        default="bo_results_CU_direct_diffusion_realnvp_epsilonK_leakage_safe_v4/decoder_monitoring",
    )
    p.add_argument(
        "--pareto-dir",
        default="bo_results_CU_direct_diffusion_realnvp_epsilonK_leakage_safe_v4/pareto_fronts",
    )

    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    p.add_argument("--seed", type=int, default=0)

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
        "--acqf", choices=["qehvi", "qlogehvi"], default="qlogehvi"
    )
    p.add_argument("--max-partition-points", type=int, default=20)
    p.add_argument("--ref-margin", type=float, default=0.10)
    p.add_argument("--hv-tol", type=float, default=1e-6)

    p.add_argument(
        "--epsilonK-bound",
        type=float,
        default=4.0,
        help=(
            "Initial symmetric native epsilonK bound. It is automatically enlarged "
            "to 1.05*max_abs(TRAIN epsilonK) before BO."
        ),
    )
    p.add_argument(
        "--tr-radius-epsilonK",
        type=float,
        default=0.75,
        help="Trust-region half-width in raw native epsilonK coordinate units.",
    )

    p.add_argument(
        "--novelty-sigmas",
        nargs="*",
        type=float,
        default=[0.05, 0.10, 0.20],
    )
    p.add_argument("--local-neighbors-per-sigma", type=int, default=4)
    p.add_argument("--max-blackbox-per-iter", type=int, default=8)

    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
