# ============================================================
# Bayesian optimization after RealNVP flow for direct peptide sequence diffusion
#
# This is the direct-diffusion/noise-flow analogue of your older
# BO_gp_after_flow_epoch199_latent_conditioned.py GRU-VAE script.
#
# Main difference from the GRU-VAE BO code:
#   OLD: sequence -> encoder_mu -> planar flow -> zK -> autoregressive decoder
#   NEW: peptide -> optimized epsilon0 preimage -> RealNVP flow -> epsilonK
#        GP/qEHVI is run in epsilonK space, and candidates are decoded by
#        direct DDIM diffusion from epsilonK.
#
# Required inputs from the fine-tuning run:
#   1) best_val_score_mse_cu_direct_diffusion_noise_flow.pt or another
#      cu_direct_sequence_diffusion_noise_flow checkpoint
#   2) cu_direct_diffusion_noise_flow_coordinates_for_bo.csv, preferred
#      because it already contains epsilonK_000 ... epsilonK_199
#   3) blackbox-scored Cu CSV with columns:
#      peptide_len10, chelation_sub, solubility_sub, stability_sub,
#      expression_sub, final_score, split
#
# BO method:
#   - Fits independent exact SingleTaskGP models for the four objectives
#   - Uses qExpectedHypervolumeImprovement (qEHVI)
#   - Uses trust-region acquisition optimization in normalized epsilonK space
#   - Decodes optimized epsilonK directly through the frozen direct diffusion model
#   - Evaluates novel decoded peptides with black_box_fcn_mo_CU_f.blackbox_fc
#   - Saves final Pareto and candidates that dominate the training Pareto
# ============================================================

from __future__ import annotations

import argparse
import json
import math
import os
import random
import gc
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- BO stack ----
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

# Optional black-box evaluator. It is only required if genuinely novel decoded
# peptides are produced and need to be scored.
try:
    from peptide_optimization.src.scripts.black_box_fcn_mo_CU_f import blackbox_fc
except Exception as exc:  # pragma: no cover - useful local error only
    blackbox_fc = None
    _BLACKBOX_IMPORT_ERROR = exc
else:
    _BLACKBOX_IMPORT_ERROR = None

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_I = {a: i for i, a in enumerate(AA)}
I_TO_AA = {i: a for a, i in AA_TO_I.items()}
SEQ_LEN = 10
VOCAB = 20
BO_DIM = SEQ_LEN * VOCAB
OBJ_COLS = ["chelation_sub", "solubility_sub", "stability_sub", "expression_sub"]


# ============================================================
# Reproducibility and basic utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_peptide(x: object) -> Optional[str]:
    p = str(x).strip().upper()
    if len(p) != SEQ_LEN:
        return None
    if any(ch not in AA_TO_I for ch in p):
        return None
    return p


def onehot_encode_peptides(peptides: Sequence[str]) -> torch.Tensor:
    x = torch.zeros((len(peptides), SEQ_LEN, VOCAB), dtype=torch.float32)
    for n, pep in enumerate(peptides):
        pep2 = clean_peptide(pep)
        if pep2 is None:
            raise ValueError(f"Invalid peptide: {pep!r}")
        for t, aa in enumerate(pep2):
            x[n, t, AA_TO_I[aa]] = 1.0
    return x


def decode_tensor_to_peptides(x: torch.Tensor) -> List[str]:
    idx = x.argmax(dim=-1).detach().cpu().tolist()
    return ["".join(I_TO_AA[int(i)] for i in row) for row in idx]


def selected_token_confidence(x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Treat final DDIM x as logits/proxy logits and report argmax softmax confidence."""
    probs = torch.softmax(x, dim=-1)
    conf, idx = probs.max(dim=-1)
    peps = ["".join(I_TO_AA[int(i)] for i in row) for row in idx.detach().cpu().tolist()]
    min_conf = conf.min(dim=-1).values
    return (
        conf.mean(dim=-1).detach().cpu().numpy(),
        min_conf.detach().cpu().numpy(),
        peps,
    )


def levenshtein_edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + int(ca != cb)))
        previous = current
    return int(previous[-1])


def project_to_sphere(eps: torch.Tensor, radius: float) -> torch.Tensor:
    return eps / eps.norm(dim=-1, keepdim=True).clamp(min=1e-8) * float(radius)


def pareto_mask_maximize(Y: torch.Tensor) -> torch.Tensor:
    """Boolean mask of non-dominated rows for maximization."""
    if Y.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool, device=Y.device)
    A = Y.unsqueeze(0)
    B = Y.unsqueeze(1)
    dominates = (A >= B).all(dim=-1) & (A > B).any(dim=-1)
    dominates.fill_diagonal_(False)
    dominated = dominates.any(dim=1)
    return ~dominated


def dominates(y_a: torch.Tensor, y_b: torch.Tensor) -> bool:
    return bool(((y_a >= y_b).all() and (y_a > y_b).any()).item())


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# ============================================================
# Direct sequence diffusion architecture compatible with finetune script
# ============================================================

@dataclass
class SequenceDiffusionConfig:
    hidden_size: int = 32
    n_layers: int = 2
    dropout: float = 0.0
    time_dim: int = 32
    train_steps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 2e-2


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time_dim must be even")
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        exponent = -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
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

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.x_proj(x_t)
        h = h + self.time_mlp(self.time_embedding(t)).unsqueeze(1)
        h, _ = self.gru(h)
        h = F.silu(self.out_norm(h))
        return {"predicted_noise": self.noise_head(h), "x0_logits": self.x0_logits_head(h)}


class DirectSequenceDiffusion(nn.Module):
    def __init__(self, cfg: SequenceDiffusionConfig):
        super().__init__()
        self.cfg = cfg
        self.denoiser = DirectSequenceDenoiser(cfg)
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.train_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    @property
    def bo_radius(self) -> float:
        return math.sqrt(BO_DIM)

    @torch.no_grad()
    def ddim_sample(self, base_noise: torch.Tensor, inference_steps: int = 20) -> torch.Tensor:
        return ddim_sample_with_grad(self, base_noise, inference_steps=inference_steps)


def ddim_sample_with_grad(model: DirectSequenceDiffusion, base_noise: torch.Tensor, inference_steps: int = 20) -> torch.Tensor:
    if base_noise.ndim == 2:
        x = base_noise.view(-1, SEQ_LEN, VOCAB)
    elif base_noise.ndim == 3:
        x = base_noise
    else:
        raise ValueError(f"Expected base_noise [B,200] or [B,10,20], got {tuple(base_noise.shape)}")

    n_train_steps = model.cfg.train_steps
    inference_steps = min(max(int(inference_steps), 1), n_train_steps)
    timesteps = torch.linspace(n_train_steps - 1, 0, inference_steps, device=x.device).round().long()
    timesteps = torch.unique_consecutive(timesteps)

    for step_index, t_scalar in enumerate(timesteps):
        t = torch.full((x.size(0),), int(t_scalar.item()), device=x.device, dtype=torch.long)
        out = model.denoiser(x, t)
        predicted_noise = out["predicted_noise"]
        alpha_bar_t = model.alpha_bars[t_scalar]
        x0_pred = (x - torch.sqrt(1.0 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
        if step_index == len(timesteps) - 1:
            x = x0_pred
            break
        t_prev_scalar = timesteps[step_index + 1]
        alpha_bar_prev = model.alpha_bars[t_prev_scalar]
        x = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1.0 - alpha_bar_prev) * predicted_noise
    return x


# ============================================================
# RealNVP flow compatible with finetune script
# ============================================================

class AffineCouplingBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, mask: torch.Tensor, max_scale: float = 1.5):
        super().__init__()
        self.dim = int(dim)
        self.max_scale = float(max_scale)
        self.register_buffer("mask", mask.float())
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_masked = x * self.mask
        s_t = self.net(x_masked)
        s, t = s_t.chunk(2, dim=-1)
        s = torch.tanh(s) * self.max_scale
        inv_mask = 1.0 - self.mask
        y = x_masked + inv_mask * (x * torch.exp(s) + t)
        logdet = (inv_mask * s).sum(dim=-1)
        return y, logdet

    def inverse(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y_masked = y * self.mask
        s_t = self.net(y_masked)
        s, t = s_t.chunk(2, dim=-1)
        s = torch.tanh(s) * self.max_scale
        inv_mask = 1.0 - self.mask
        x = y_masked + inv_mask * ((y - t) * torch.exp(-s))
        logdet = -(inv_mask * s).sum(dim=-1)
        return x, logdet


class RealNVPFlow(nn.Module):
    def __init__(self, dim: int = BO_DIM, n_layers: int = 6, hidden_dim: int = 256, max_scale: float = 1.5):
        super().__init__()
        masks = []
        for layer in range(n_layers):
            mask_np = np.zeros(dim, dtype=np.float32)
            mask_np[layer % 2 :: 2] = 1.0
            masks.append(torch.tensor(mask_np))
        self.blocks = nn.ModuleList([AffineCouplingBlock(dim, hidden_dim, mask=m, max_scale=max_scale) for m in masks])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = x
        logdet_total = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
        for block in self.blocks:
            h, logdet = block(h)
            logdet_total = logdet_total + logdet
        return h, logdet_total

    def inverse(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = y
        logdet_total = torch.zeros(y.size(0), dtype=y.dtype, device=y.device)
        for block in reversed(self.blocks):
            h, logdet = block.inverse(h)
            logdet_total = logdet_total + logdet
        return h, logdet_total


# ============================================================
# Loading model/checkpoint and coordinate data
# ============================================================

def _infer_flow_architecture(ckpt: Dict, args: argparse.Namespace) -> Tuple[int, int, float]:
    ck_args = ckpt.get("args", {}) or {}
    n_layers = int(args.flow_layers or ck_args.get("flow_layers", 4))
    hidden_dim = int(args.flow_hidden_dim or ck_args.get("flow_hidden_dim", 128))
    max_scale = float(args.flow_max_scale if args.flow_max_scale is not None else ck_args.get("flow_max_scale", 1.5))
    return n_layers, hidden_dim, max_scale


def load_noise_flow_checkpoint(path: str, device: torch.device, args: argparse.Namespace) -> Tuple[DirectSequenceDiffusion, RealNVPFlow, Dict]:
    ckpt = torch.load(path, map_location=device)
    if "model_state_dict" not in ckpt or "flow_state_dict" not in ckpt:
        raise KeyError(
            f"Checkpoint {path} must contain model_state_dict and flow_state_dict. "
            "Use a checkpoint from finetune_cu_direct_sequence_diffusion_noise_flow_h32_cudnn_fix_v2.py."
        )
    cfg_dict = ckpt.get("diffusion_config")
    if cfg_dict is None:
        raise KeyError(f"Checkpoint {path} is missing diffusion_config.")
    cfg = SequenceDiffusionConfig(**cfg_dict)
    model = DirectSequenceDiffusion(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    n_layers, hidden_dim, max_scale = _infer_flow_architecture(ckpt, args)
    flow = RealNVPFlow(dim=BO_DIM, n_layers=n_layers, hidden_dim=hidden_dim, max_scale=max_scale).to(device)
    flow.load_state_dict(ckpt["flow_state_dict"], strict=True)

    for p in model.parameters():
        p.requires_grad_(False)
    for p in flow.parameters():
        p.requires_grad_(False)
    model.eval()
    flow.eval()

    print(f"Loaded direct diffusion + RealNVP flow checkpoint: {path}")
    print("checkpoint_type:", ckpt.get("checkpoint_type"))
    print("selection:", ckpt.get("selection"))
    print("epoch:", ckpt.get("epoch"))
    print(f"flow architecture: layers={n_layers}, hidden_dim={hidden_dim}, max_scale={max_scale}")
    return model, flow, ckpt


def load_scored_dataframe(path: str, peptide_col: str, obj_cols: Sequence[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = [peptide_col] + list(obj_cols)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Input CSV is missing required columns: {missing}")
    df = df.copy()
    df[peptide_col] = df[peptide_col].map(clean_peptide)
    df = df[df[peptide_col].notna()].copy()
    for c in obj_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=list(obj_cols)).reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid scored peptides remained after cleaning.")
    return df


def load_epsilonK_from_coordinate_csv(path: str, peptides: Sequence[str], peptide_col_candidates: Sequence[str]) -> torch.Tensor:
    coord = pd.read_csv(path)
    eps_cols = [f"epsilonK_{i:03d}" for i in range(BO_DIM)]
    missing = [c for c in eps_cols if c not in coord.columns]
    if missing:
        raise KeyError(f"Coordinate CSV is missing epsilonK columns, first missing: {missing[:5]}")

    pep_col = None
    for c in peptide_col_candidates:
        if c in coord.columns:
            pep_col = c
            break
    if pep_col is None:
        raise KeyError(f"Coordinate CSV must have one of these peptide columns: {peptide_col_candidates}")

    coord = coord.copy()
    coord["_pep"] = coord[pep_col].map(clean_peptide)
    coord = coord.dropna(subset=["_pep"]).drop_duplicates(subset=["_pep"], keep="first")
    mapping = {row["_pep"]: row[eps_cols].to_numpy(dtype=np.float32) for _, row in coord.iterrows()}

    missing_peps = [p for p in peptides if p not in mapping]
    if missing_peps:
        raise KeyError(
            f"Coordinate CSV has no epsilonK row for {len(missing_peps)} peptides. "
            f"Examples: {missing_peps[:10]}"
        )
    z = np.stack([mapping[p] for p in peptides], axis=0).astype(np.float32)
    return torch.tensor(z, dtype=torch.float32)


def load_or_compute_epsilonK(args: argparse.Namespace, df: pd.DataFrame, model: DirectSequenceDiffusion, flow: RealNVPFlow, device: torch.device) -> torch.Tensor:
    peptides = df[args.peptide_col].astype(str).tolist()
    if args.coordinate_csv and os.path.exists(args.coordinate_csv):
        print(f"Loading epsilonK coordinates from: {args.coordinate_csv}")
        return load_epsilonK_from_coordinate_csv(args.coordinate_csv, peptides, ["peptide", args.peptide_col, "peptide_len10"])

    if args.preimage_cache and os.path.exists(args.preimage_cache):
        print(f"Coordinate CSV not used. Computing epsilonK from preimage cache: {args.preimage_cache}")
        payload = torch.load(args.preimage_cache, map_location="cpu")
        if "epsilon0" not in payload:
            raise KeyError(f"Preimage cache {args.preimage_cache} does not contain epsilon0")
        eps0 = payload["epsilon0"].float()
        if eps0.shape != (len(peptides), BO_DIM):
            raise ValueError(f"epsilon0 shape {tuple(eps0.shape)} does not match expected {(len(peptides), BO_DIM)}")
        chunks = []
        radius = math.sqrt(BO_DIM)
        with torch.no_grad():
            for start in range(0, eps0.size(0), args.batch_size):
                e0 = eps0[start:start + args.batch_size].to(device)
                eK_raw, _ = flow(e0)
                eK = project_to_sphere(eK_raw, radius) if args.project_to_sphere else eK_raw
                chunks.append(eK.cpu())
        return torch.cat(chunks, dim=0).float()

    raise FileNotFoundError(
        "Could not load BO coordinates. Provide --coordinate-csv pointing to "
        "cu_direct_diffusion_noise_flow_coordinates_for_bo.csv, or provide --preimage-cache."
    )


# ============================================================
# GP/qEHVI and black-box scoring
# ============================================================

def fit_mo_models(Z: torch.Tensor, Y: torch.Tensor, device: torch.device) -> ModelListGP:
    models = []
    for m in range(Y.size(-1)):
        y_m = Y[:, m:m + 1]
        gp_m = SingleTaskGP(Z, y_m, outcome_transform=Standardize(m=1)).to(device)
        gp_m.likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.GreaterThan(1e-6))
        gp_m.likelihood.noise = 1e-4
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp_m.likelihood, gp_m)
        fit_gpytorch_mll(mll)
        models.append(gp_m)
    return ModelListGP(*models)


def evaluate_blackbox(peptides: Sequence[str], cache: Dict[str, List[float]], obj_cols: Sequence[str]) -> torch.Tensor:
    norm_peps = [str(p).strip().upper() for p in peptides]
    uncached = [p for p in norm_peps if p not in cache]
    if uncached:
        if blackbox_fc is None:
            raise RuntimeError(
                "black_box_fcn_mo_CU_f.blackbox_fc could not be imported, and novel peptides need scoring. "
                f"Original import error: {_BLACKBOX_IMPORT_ERROR}"
            )
        scored = blackbox_fc(uncached)
        for _, row in scored.iterrows():
            pep = clean_peptide(row.get("peptide_len10", row.get("peptide", "")))
            if pep is None:
                continue
            cache[pep] = [float(row[c]) for c in obj_cols]
    rows = [cache.get(p, [0.0] * len(obj_cols)) for p in norm_peps]
    return torch.tensor(rows, dtype=torch.float32)


# ============================================================
# Decoding and diagnostics
# ============================================================

@torch.no_grad()
def decode_from_epsilonK(model: DirectSequenceDiffusion, epsK: torch.Tensor, args: argparse.Namespace) -> Tuple[List[str], np.ndarray, np.ndarray]:
    model.eval()
    radius = math.sqrt(BO_DIM)
    z = project_to_sphere(epsK, radius) if args.project_to_sphere else epsK
    x0_hat = model.ddim_sample(z, inference_steps=args.ddim_steps)
    mean_conf, min_conf, peps = selected_token_confidence(x0_hat)
    return peps, mean_conf, min_conf


@torch.no_grad()
def decode_candidates_with_diagnostics(model: DirectSequenceDiffusion, epsK_cand: torch.Tensor, bo_iter: int, args: argparse.Namespace) -> Tuple[List[str], pd.DataFrame]:
    ensure_dir(args.decoder_dir)
    rows: List[Dict[str, object]] = []
    decoded: List[str] = []

    for j in range(epsK_cand.size(0)):
        center = epsK_cand[j:j + 1]
        peps, mean_conf, min_conf = decode_from_epsilonK(model, center, args)
        pep = peps[0]
        decoded.append(pep)
        rows.append({
            "bo_iter": bo_iter,
            "candidate_index": j,
            "decode_rank": 0,
            "decode_type": "argmax_ddim_from_epsilonK",
            "peptide": pep,
            "decoder_confidence_mean": float(mean_conf[0]),
            "decoder_confidence_min": float(min_conf[0]),
            "epsilonK_norm": float(center[0].norm().cpu()),
            "local_noise_l2": 0.0,
            "edit_distance_to_center_peptide": 0,
        })

        # Local noise diagnostics: useful for checking whether nearby epsilonK
        # coordinates decode to similar peptides.
        for r in range(1, args.decoder_samples_per_z):
            noise = torch.randn_like(center) * float(args.local_noise_std)
            local = project_to_sphere(center + noise, math.sqrt(BO_DIM)) if args.project_to_sphere else center + noise
            peps_l, mean_l, min_l = decode_from_epsilonK(model, local, args)
            pep_l = peps_l[0]
            decoded.append(pep_l)
            rows.append({
                "bo_iter": bo_iter,
                "candidate_index": j,
                "decode_rank": r,
                "decode_type": "local_noisy_ddim_from_epsilonK",
                "peptide": pep_l,
                "decoder_confidence_mean": float(mean_l[0]),
                "decoder_confidence_min": float(min_l[0]),
                "epsilonK_norm": float(local[0].norm().cpu()),
                "local_noise_l2": float(noise[0].norm().cpu()),
                "edit_distance_to_center_peptide": levenshtein_edit_distance(pep, pep_l),
            })

    df = pd.DataFrame(rows)
    df["is_duplicate_raw_decode"] = df.duplicated(subset=["peptide"], keep="first")
    df.to_csv(os.path.join(args.decoder_dir, f"decoder_diagnostics_bo_iter_{bo_iter:03d}.csv"), index=False)
    unique = list(dict.fromkeys(decoded))
    return unique, df


def filter_novel_peptides(candidates: Sequence[str], train_set: set, evaluated_set: set, generated_set: set, reject_training: bool, reject_seen: bool) -> Tuple[List[str], List[Tuple[str, str]]]:
    accepted: List[str] = []
    rejected: List[Tuple[str, str]] = []
    for pep in candidates:
        reason = None
        if clean_peptide(pep) is None:
            reason = "invalid_peptide"
        elif reject_training and pep in train_set:
            reason = "in_training"
        elif reject_seen and pep in evaluated_set:
            reason = "already_evaluated"
        elif reject_seen and pep in generated_set:
            reason = "duplicate_generated"
        if reason is None:
            accepted.append(pep)
        else:
            rejected.append((pep, reason))
    return accepted, rejected


# ============================================================
# BO main loop
# ============================================================

def build_initial_labeled_indices(Y: torch.Tensor, n_init: int, seed: int) -> List[int]:
    pareto_idx = torch.where(pareto_mask_maximize(Y))[0].cpu().tolist()
    pareto_set = set(pareto_idx)
    non_pareto = [i for i in range(Y.size(0)) if i not in pareto_set]
    rng = np.random.default_rng(seed)
    n_extra = max(0, int(n_init) - len(pareto_idx))
    extra = rng.choice(non_pareto, size=min(n_extra, len(non_pareto)), replace=False).tolist() if n_extra else []
    return list(dict.fromkeys(pareto_idx + extra))


def save_pareto(df_peptides: Sequence[str], labeled_idx: Sequence[int], Y_lab: torch.Tensor, bo_iter: int, out_dir: str) -> pd.DataFrame:
    ensure_dir(out_dir)
    Y_cpu = Y_lab.detach().cpu()
    mask = pareto_mask_maximize(Y_cpu)
    rows = []
    for k in torch.where(mask)[0].tolist():
        global_i = int(labeled_idx[k])
        pep = df_peptides[global_i] if 0 <= global_i < len(df_peptides) else ""
        objs = Y_cpu[k].tolist()
        rows.append([bo_iter, global_i, pep] + objs)
    df = pd.DataFrame(rows, columns=["bo_iter", "global_index", "peptide"] + OBJ_COLS)
    df.to_csv(os.path.join(out_dir, f"pareto_front_bo_iter_{bo_iter:03d}.csv"), index=False)
    return df


def run_bo(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)
    ensure_dir(args.out_dir)
    ensure_dir(args.decoder_dir)
    ensure_dir(args.pareto_dir)

    model, flow, ckpt = load_noise_flow_checkpoint(args.flow_checkpoint, device, args)
    df = load_scored_dataframe(args.data_csv, args.peptide_col, args.obj_cols)
    peptides = df[args.peptide_col].astype(str).tolist()
    Z_all = load_or_compute_epsilonK(args, df, model, flow, device).to(device)
    Z_all = project_to_sphere(Z_all, math.sqrt(BO_DIM)) if args.project_to_sphere else Z_all
    Y_train_full = torch.tensor(df[args.obj_cols].to_numpy(dtype=np.float32), dtype=torch.float32, device=device)

    # Objective cache starts with all training/scored CSV peptides.
    obj_cache: Dict[str, List[float]] = {
        pep: [float(v) for v in vals]
        for pep, vals in zip(peptides, df[args.obj_cols].to_numpy(dtype=np.float32).tolist())
    }

    train_set = set(peptides)
    evaluated_set: set = set()
    generated_set: set = set()

    train_pareto_mask = pareto_mask_maximize(Y_train_full)
    train_pareto_idx = torch.where(train_pareto_mask)[0]
    Y_train_pareto = Y_train_full[train_pareto_mask]
    train_pareto_df = pd.DataFrame({
        "global_index": train_pareto_idx.detach().cpu().numpy(),
        "peptide": [peptides[i] for i in train_pareto_idx.detach().cpu().tolist()],
        **{c: Y_train_pareto[:, j].detach().cpu().numpy() for j, c in enumerate(args.obj_cols)},
    })
    train_pareto_path = os.path.join(args.out_dir, "train_pareto_cu_direct_diffusion_after_flow_gp.csv")
    train_pareto_df.to_csv(train_pareto_path, index=False)
    print(f"Training Pareto: {len(train_pareto_df)} solutions -> {train_pareto_path}")

    labeled_idx = build_initial_labeled_indices(Y_train_full, args.initial_labeled, args.seed + 42)
    Y_lab = Y_train_full[torch.tensor(labeled_idx, dtype=torch.long, device=device)]
    Z_lab = Z_all[torch.tensor(labeled_idx, dtype=torch.long, device=device)]
    evaluated_set.update(peptides[i] for i in labeled_idx)

    ref_point = (Y_lab.detach().double().min(dim=0).values - float(args.ref_margin)).tolist()
    best_hv = float(Hypervolume(ref_point=torch.tensor(ref_point, device=device, dtype=Y_lab.dtype)).compute(Y_lab))

    config = vars(args).copy()
    config.update({
        "checkpoint_selection": ckpt.get("selection"),
        "checkpoint_epoch": ckpt.get("epoch"),
        "bo_space": "epsilonK_after_RealNVP_flow_direct_diffusion_noise_space",
        "bo_dim": BO_DIM,
        "bo_radius": math.sqrt(BO_DIM),
        "initial_labeled_actual": len(labeled_idx),
        "training_pareto_size": len(train_pareto_df),
    })
    with open(os.path.join(args.out_dir, "bo_run_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    all_candidate_rows: List[Dict[str, object]] = []
    hv_rows: List[Dict[str, object]] = []

    for bo_iter in range(int(args.bo_iters)):
        print(f"\n=== BO iter {bo_iter + 1}/{args.bo_iters} ===")
        Z_lab_bo = Z_lab.detach().double().to(device)
        Y_lab_bo = Y_lab.detach().double().to(device)

        Z_min = Z_lab_bo.min(dim=0).values
        Z_max = Z_lab_bo.max(dim=0).values
        Z_range = (Z_max - Z_min).clamp_min(1e-12)

        def z_to_unit(z: torch.Tensor) -> torch.Tensor:
            return ((z - Z_min) / Z_range).clamp(0.0, 1.0)

        def z_from_unit(u: torch.Tensor) -> torch.Tensor:
            return u * Z_range + Z_min

        Z_unit = z_to_unit(Z_lab_bo)
        mo_model = fit_mo_models(Z_unit, Y_lab_bo, device=device)

        # Trust-region center uses scalarized objectives only for localization.
        scalar = Y_lab.mean(dim=-1)
        best_i = int(torch.argmax(scalar).item())
        u_center = z_to_unit(Z_lab[best_i].detach().double())
        lb = (u_center - float(args.tr_radius_unit)).clamp(0.0, 1.0)
        ub = (u_center + float(args.tr_radius_unit)).clamp(0.0, 1.0)
        bounds = torch.stack([lb, ub], dim=0)

        with torch.no_grad():
            nd_mask = pareto_mask_maximize(Y_lab_bo)
            Y_nd = Y_lab_bo[nd_mask]
            if Y_nd.size(0) > args.max_partition_points:
                keep = torch.topk(Y_nd.mean(dim=-1), k=args.max_partition_points, largest=True).indices
                Y_part = Y_nd[keep]
            else:
                Y_part = Y_nd

        ref_point = (Y_lab_bo.min(dim=0).values - float(args.ref_margin)).tolist()
        partitioning = NondominatedPartitioning(ref_point=torch.tensor(ref_point, device=device), Y=Y_part)
        # Memory-safe acquisition optimization. qEHVI/qLogEHVI can allocate a very
        # large tensor when q_batch, raw_samples, mc_samples, and the number of
        # partition boxes are all large. We therefore optimize one candidate at a
        # time and repeat q_batch times, which is usually much safer on 16 GB GPUs.
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([args.mc_samples])).to(device)
        if args.acqf.lower() == "qlogehvi" and qLogExpectedHypervolumeImprovement is not None:
            acq_cls = qLogExpectedHypervolumeImprovement
        else:
            acq_cls = qExpectedHypervolumeImprovement
        acq = acq_cls(
            model=mo_model,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
        ).to(device)

        u_list: List[torch.Tensor] = []
        acq_values: List[torch.Tensor] = []
        q_inner = 1 if args.optimize_q_one_at_a_time else int(args.q_batch)
        n_repeats = int(args.q_batch) if args.optimize_q_one_at_a_time else 1
        for _cand_i in range(n_repeats):
            try:
                U_part, acq_part = optimize_acqf(
                    acq,
                    bounds=bounds,
                    q=q_inner,
                    num_restarts=args.num_restarts,
                    raw_samples=args.raw_samples,
                    sequential=True,
                )
            except torch.OutOfMemoryError:
                print("[OOM] Acquisition optimization exceeded GPU memory. Retrying with smaller temporary settings.")
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
                safe_mc = max(8, int(args.mc_samples) // 4)
                safe_raw = max(16, int(args.raw_samples) // 4)
                safe_restarts = max(1, int(args.num_restarts) // 2)
                safe_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([safe_mc])).to(device)
                safe_acq = acq_cls(
                    model=mo_model,
                    ref_point=ref_point,
                    partitioning=partitioning,
                    sampler=safe_sampler,
                ).to(device)
                U_part, acq_part = optimize_acqf(
                    safe_acq,
                    bounds=bounds,
                    q=1,
                    num_restarts=safe_restarts,
                    raw_samples=safe_raw,
                    sequential=True,
                )
            u_list.append(U_part.detach())
            acq_values.append(acq_part.detach().reshape(-1)[0])
        U_cand = torch.cat(u_list, dim=0)
        acq_value = torch.stack(acq_values).mean()
        epsK_cand = z_from_unit(U_cand).to(dtype=torch.float32)
        epsK_cand = project_to_sphere(epsK_cand, math.sqrt(BO_DIM)) if args.project_to_sphere else epsK_cand

        cand_peps_raw, decoder_df = decode_candidates_with_diagnostics(model, epsK_cand, bo_iter, args)
        cand_peps, rejected = filter_novel_peptides(
            cand_peps_raw,
            train_set=train_set,
            evaluated_set=evaluated_set,
            generated_set=generated_set,
            reject_training=args.reject_training_peptides,
            reject_seen=args.reject_seen_peptides,
        )

        if len(cand_peps) == 0:
            print("[BO] No novel decoded peptide survived filtering; skipping objective evaluation.")
            generated_set.update(cand_peps_raw)
            hv_rows.append({
                "bo_iter": bo_iter,
                "n_raw_decoded": len(cand_peps_raw),
                "n_accepted": 0,
                "hypervolume": best_hv,
                "best_hypervolume": best_hv,
                "acq_value": float(acq_value.detach().cpu().reshape(-1)[0]),
            })
            continue

        # Keep the epsilonK row corresponding to each accepted peptide. If a
        # peptide came from local noisy diagnostics rather than the main qEHVI
        # candidate, use the nearest available candidate center. This keeps BO
        # updates simple and conservative.
        accepted_eps_rows: List[torch.Tensor] = []
        for pep in cand_peps:
            match = decoder_df.index[decoder_df["peptide"].astype(str) == pep].tolist()
            if match:
                candidate_index = int(decoder_df.loc[match[0], "candidate_index"])
                accepted_eps_rows.append(epsK_cand[candidate_index].detach().clone())
            else:
                accepted_eps_rows.append(epsK_cand[0].detach().clone())
        Z_new = torch.stack(accepted_eps_rows, dim=0).to(device)

        Y_new = evaluate_blackbox(cand_peps, obj_cache, args.obj_cols).to(device)

        comparison_rows = []
        for pep, y in zip(cand_peps, Y_new):
            dominated_by_train = any(dominates(y_ref, y) for y_ref in Y_train_pareto)
            dominates_train_any = any(dominates(y, y_ref) for y_ref in Y_train_pareto)
            row = {
                "bo_iter": bo_iter,
                "peptide": pep,
                "acq_value": float(acq_value.detach().cpu().reshape(-1)[0]),
                "dominated_by_train_pareto": int(dominated_by_train),
                "dominates_any_train_pareto": int(dominates_train_any),
            }
            for j, c in enumerate(args.obj_cols):
                row[c] = float(y[j].detach().cpu())
            comparison_rows.append(row)
            all_candidate_rows.append(row)

        pd.DataFrame(comparison_rows).to_csv(
            os.path.join(args.out_dir, f"accepted_candidates_scored_bo_iter_{bo_iter:03d}.csv"),
            index=False,
        )

        status_df = decoder_df.copy()
        rejected_map = {p: reason for p, reason in rejected}
        status_df["accepted_for_blackbox"] = status_df["peptide"].isin(cand_peps)
        status_df["rejected_reason"] = status_df["peptide"].map(lambda p: "" if p in cand_peps else rejected_map.get(p, "duplicate_raw_decode_or_not_selected"))
        for j, c in enumerate(args.obj_cols):
            val_map = {p: float(y[j].detach().cpu()) for p, y in zip(cand_peps, Y_new)}
            status_df[c] = status_df["peptide"].map(lambda p: val_map.get(p, np.nan))
        status_df.to_csv(os.path.join(args.decoder_dir, f"all_decoder_diagnostics_status_bo_iter_{bo_iter:03d}.csv"), index=False)

        # Append newly evaluated continuous points. Their BO coordinate is the
        # optimized epsilonK that decoded to the candidate.
        base_count = len(peptides)
        new_global_indices = []
        for pep, z_new in zip(cand_peps, Z_new):
            peptides.append(pep)
            new_global_indices.append(len(peptides) - 1)
        Z_all = torch.cat([Z_all, Z_new], dim=0)
        Z_lab = torch.cat([Z_lab, Z_new], dim=0)
        Y_lab = torch.cat([Y_lab, Y_new], dim=0)
        labeled_idx.extend(new_global_indices)
        evaluated_set.update(cand_peps)
        generated_set.update(cand_peps_raw)

        with torch.no_grad():
            hv = float(Hypervolume(ref_point=torch.tensor(ref_point, device=device, dtype=Y_lab.dtype)).compute(Y_lab))
        improved = hv > best_hv + float(args.hv_tol)
        if improved:
            best_hv = hv

        pareto_df = save_pareto(peptides, labeled_idx, Y_lab, bo_iter, args.pareto_dir)
        hv_row = {
            "bo_iter": bo_iter,
            "n_raw_decoded": int(len(cand_peps_raw)),
            "n_accepted": int(len(cand_peps)),
            "n_labeled": int(len(labeled_idx)),
            "pareto_size": int(len(pareto_df)),
            "hypervolume": float(hv),
            "best_hypervolume": float(best_hv),
            "improved": int(improved),
            "acq_value": float(acq_value.detach().cpu().reshape(-1)[0]),
        }
        hv_rows.append(hv_row)
        pd.DataFrame(hv_rows).to_csv(os.path.join(args.out_dir, "bo_hypervolume_history.csv"), index=False)
        pd.DataFrame(all_candidate_rows).to_csv(os.path.join(args.out_dir, "all_accepted_candidates_scored.csv"), index=False)
        print(
            f"Accepted {len(cand_peps)} peptides. HV={hv:.6f}, best={best_hv:.6f}, "
            f"pareto={len(pareto_df)}, dominates_train={sum(r['dominates_any_train_pareto'] for r in comparison_rows)}"
        )

    # Final Pareto over all labeled/evaluated points.
    Y_cpu = Y_lab.detach().cpu()
    final_mask = pareto_mask_maximize(Y_cpu)
    rows = []
    for k in torch.where(final_mask)[0].tolist():
        g = int(labeled_idx[k])
        pep = peptides[g] if 0 <= g < len(peptides) else ""
        vals = Y_cpu[k].tolist()
        row = {"global_index": g, "peptide": pep, "is_novel": pep not in train_set}
        for j, c in enumerate(args.obj_cols):
            row[c] = float(vals[j])
        rows.append(row)
    final_df = pd.DataFrame(rows)
    if len(final_df):
        Y_final = torch.tensor(final_df[list(args.obj_cols)].to_numpy(dtype=np.float32))
        Y_train_pareto_cpu = Y_train_pareto.detach().cpu()
        final_df["dominates_train_pareto"] = [
            any(dominates(Y_final[i], y_ref) for y_ref in Y_train_pareto_cpu)
            for i in range(len(final_df))
        ]
    else:
        final_df["dominates_train_pareto"] = []

    final_path = os.path.join(args.out_dir, "bo_final_pareto_CU_direct_diffusion_after_flow_gp.csv")
    final_df.to_csv(final_path, index=False)
    dom_df = final_df[(final_df.get("is_novel", False) == True) & (final_df.get("dominates_train_pareto", False) == True)].copy() if len(final_df) else pd.DataFrame()
    dom_path = os.path.join(args.out_dir, "bo_pareto_dominates_train_CU_direct_diffusion_after_flow_gp.csv")
    dom_df.to_csv(dom_path, index=False)

    print("\nDone.")
    print(f"Final BO Pareto: {len(final_df)} -> {final_path}")
    print(f"Novel BO Pareto solutions dominating training Pareto: {len(dom_df)} -> {dom_path}")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-objective GP/qEHVI BO after RealNVP flow for direct peptide sequence diffusion.")
    p.add_argument("--flow-checkpoint", default=r"cu_direct_sequence_diffusion_noise_flow_chainmapped_h32_blackbox_scored\best_val_score_mse_cu_direct_diffusion_noise_flow.pt")
    p.add_argument("--data-csv", default="metalpdb_CU_chain_mapped_len10_high_confidence_blackbox_scored_ranked.csv")
    p.add_argument("--coordinate-csv", default=r"cu_direct_sequence_diffusion_noise_flow_chainmapped_h32_blackbox_scored\cu_direct_diffusion_noise_flow_coordinates_for_bo.csv")
    p.add_argument("--preimage-cache", default=r"cu_direct_sequence_diffusion_noise_flow_chainmapped_h32_blackbox_scored\cu_direct_diffusion_epsilon0_preimage_cache.pt")
    p.add_argument("--peptide-col", default="peptide_len10")
    p.add_argument("--obj-cols", nargs=4, default=OBJ_COLS)
    p.add_argument("--out-dir", default="bo_results_CU_direct_diffusion_after_flow_gp")
    p.add_argument("--decoder-dir", default="bo_decoder_monitoring_CU_direct_diffusion_after_flow_gp")
    p.add_argument("--pareto-dir", default="pareto_front_CU_direct_diffusion_after_flow_gp")

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--ddim-steps", type=int, default=20)
    p.add_argument("--project-to-sphere", action=argparse.BooleanOptionalAction, default=True)

    # Pass these only if the checkpoint metadata cannot infer them.
    p.add_argument("--flow-layers", type=int, default=None)
    p.add_argument("--flow-hidden-dim", type=int, default=None)
    p.add_argument("--flow-max-scale", type=float, default=None)

    p.add_argument("--bo-iters", type=int, default=20)
    p.add_argument("--initial-labeled", type=int, default=256)
    p.add_argument("--q-batch", type=int, default=1)
    p.add_argument("--num-restarts", type=int, default=4)
    p.add_argument("--raw-samples", type=int, default=64)
    p.add_argument("--mc-samples", type=int, default=64)
    p.add_argument("--acqf", choices=["qehvi", "qlogehvi"], default="qlogehvi")
    p.add_argument("--optimize-q-one-at-a-time", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tr-radius-unit", type=float, default=0.15)
    p.add_argument("--max-partition-points", type=int, default=30)
    p.add_argument("--ref-margin", type=float, default=0.1)
    p.add_argument("--hv-tol", type=float, default=1e-6)

    p.add_argument("--decoder-samples-per-z", type=int, default=8)
    p.add_argument("--local-noise-std", type=float, default=0.05)
    p.add_argument("--reject-training-peptides", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--reject-seen-peptides", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


if __name__ == "__main__":
    run_bo(parse_args())
