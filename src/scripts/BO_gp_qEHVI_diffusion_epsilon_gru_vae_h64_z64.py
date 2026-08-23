from __future__ import annotations

"""
Multi-objective GP / qEHVI Bayesian optimization in the spherical diffusion
epsilon space of the Cu-finetuned GRU-VAE latent-diffusion model.

This script adapts the workflow of:
  BO_gp_after_flow_gru_vae_h64_z64_latent_conditioned_realnvp_high_confidence.py

to the diffusion model defined in:
  finetune_best_bo_ready_gru_vae_cu_latent_diffusion.py

Generative path
---------------
peptide
  -> GRU-VAE encoder mu
  -> training-statistics standardization h0
  -> deterministic DDIM inversion
  -> epsilon projected to sphere ||epsilon|| = sqrt(latent_dim)

Bayesian optimization is performed in epsilon space.

Candidate generation
--------------------
epsilon
  -> deterministic DDIM sample -> h0
  -> unstandardize -> decoder latent mu
  -> autoregressive GRU decoder
  -> peptide
  -> true Cu black-box objectives

Important geometric detail
--------------------------
The fine-tuned model uses spherical epsilon coordinates (radius sqrt(64)=8).
The GP is trained on u = epsilon / radius, so observed BO coordinates lie on
the unit sphere.

Acquisition optimization uses a sphere-projected wrapper:
    u_raw -> u_raw / ||u_raw||
before qEHVI/qLogEHVI is evaluated.

This prevents acquisition values from being evaluated at invalid off-sphere
coordinates. The final candidate is also explicitly reprojected to the sphere.

Objectives are maximized:
  chelation_sub
  solubility_sub
  stability_sub
  expression_sub
"""

import argparse
import gc
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import gpytorch
from botorch.acquisition.acquisition import AcquisitionFunction
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

try:
    from black_box_fcn_mo_CU_f import blackbox_fc
except Exception as exc:
    blackbox_fc = None
    BLACKBOX_IMPORT_ERROR = exc
else:
    BLACKBOX_IMPORT_ERROR = None


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


# ---------------------------------------------------------------------
# Model definitions -- checkpoint-compatible with fine-tuning script
# ---------------------------------------------------------------------

@dataclass
class ModelConfig:
    hidden_size: int = 64
    latent_dim: int = 64
    n_layers: int = 2
    dropout: float = 0.0
    decoder_conditioning: str = "concat_z_at_every_decoder_step"


@dataclass
class DiffusionConfig:
    hidden_dim: int = 128
    time_dim: int = 32
    n_blocks: int = 4
    train_steps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 2e-2


class GRUEncoder(nn.Module):
    def __init__(self, hidden_size, latent_dim, n_layers, dropout):
        super().__init__()
        self.in_proj = nn.Linear(VOCAB, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.to_mu = nn.Linear(hidden_size, latent_dim)
        self.to_logvar = nn.Linear(hidden_size, latent_dim)

    def forward(self, x_onehot):
        h = self.in_proj(x_onehot)
        out_top, h_n = self.gru(h)
        h_last = h_n[-1]
        mu = self.to_mu(h_last)
        logvar = self.to_logvar(h_last).clamp(-8.0, 8.0)
        return mu, logvar, out_top


class LatentConditionedGRUDecoder(nn.Module):
    def __init__(self, hidden_size, latent_dim, n_layers, dropout):
        super().__init__()
        self.token_embed = nn.Linear(VOCAB, hidden_size)
        self.z_to_h = nn.Linear(latent_dim, n_layers * hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size + latent_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.to_logits = nn.Linear(hidden_size, VOCAB)

    def initial_hidden(self, z):
        return self.z_to_h(z).view(
            self.gru.num_layers, z.size(0), self.gru.hidden_size
        )

    def step(self, z, current_token, h):
        emb = self.token_embed(current_token)
        dec_input = torch.cat([emb, z.unsqueeze(1)], dim=-1)
        out_step, h = self.gru(dec_input, h)
        logits = self.to_logits(out_step[:, -1, :])
        return logits, h


class GRUVAE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.enc = GRUEncoder(
            cfg.hidden_size, cfg.latent_dim, cfg.n_layers, cfg.dropout
        )
        self.dec = LatentConditionedGRUDecoder(
            cfg.hidden_size, cfg.latent_dim, cfg.n_layers, cfg.dropout
        )


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if dim % 2:
            raise ValueError("time_dim must be even")
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        exponent = (
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        freq = torch.exp(exponent)
        angles = t.float().unsqueeze(-1) * freq.unsqueeze(0)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class ResidualDenoiserBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return x + self.fc2(F.silu(self.fc1(self.norm(x))))


class LatentDenoiser(nn.Module):
    def __init__(self, latent_dim, cfg: DiffusionConfig):
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(cfg.time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.time_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.in_proj = nn.Linear(latent_dim, cfg.hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualDenoiserBlock(cfg.hidden_dim)
            for _ in range(cfg.n_blocks)
        ])
        self.out_norm = nn.LayerNorm(cfg.hidden_dim)
        self.out_proj = nn.Linear(cfg.hidden_dim, latent_dim)

    def forward(self, h_t, t):
        h = self.in_proj(h_t) + self.time_mlp(self.time_embedding(t))
        for block in self.blocks:
            h = block(h)
        return self.out_proj(F.silu(self.out_norm(h)))


class LatentDiffusion(nn.Module):
    def __init__(self, latent_dim, cfg: DiffusionConfig):
        super().__init__()
        self.latent_dim = latent_dim
        self.cfg = cfg
        self.denoiser = LatentDenoiser(latent_dim, cfg)

        betas = torch.linspace(
            cfg.beta_start, cfg.beta_end, cfg.train_steps, dtype=torch.float32
        )
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", torch.cumprod(alphas, dim=0))

    @torch.no_grad()
    def ddim_sample(self, base_noise, inference_steps=20):
        self.eval()
        n_train = self.cfg.train_steps
        inference_steps = min(max(int(inference_steps), 1), n_train)
        timesteps = torch.linspace(
            n_train - 1, 0, inference_steps, device=base_noise.device
        ).round().long()
        timesteps = torch.unique_consecutive(timesteps)

        h = base_noise
        for j, t_scalar in enumerate(timesteps):
            t = torch.full(
                (h.size(0),),
                int(t_scalar.item()),
                device=h.device,
                dtype=torch.long,
            )
            pred_noise = self.denoiser(h, t)
            abar_t = self.alpha_bars[t_scalar]
            pred_h0 = (
                h - torch.sqrt(1.0 - abar_t) * pred_noise
            ) / torch.sqrt(abar_t)

            if j == len(timesteps) - 1:
                h = pred_h0
                break

            t_prev = timesteps[j + 1]
            abar_prev = self.alpha_bars[t_prev]
            h = (
                torch.sqrt(abar_prev) * pred_h0
                + torch.sqrt(1.0 - abar_prev) * pred_noise
            )
        return h

    @torch.no_grad()
    def ddim_invert(self, h0, inference_steps=20):
        self.eval()
        n_train = self.cfg.train_steps
        inference_steps = min(max(int(inference_steps), 1), n_train)
        timesteps = torch.linspace(
            0, n_train - 1, inference_steps, device=h0.device
        ).round().long()
        timesteps = torch.unique_consecutive(timesteps)

        h = h0
        for j, t_scalar in enumerate(timesteps[:-1]):
            t = torch.full(
                (h.size(0),),
                int(t_scalar.item()),
                device=h.device,
                dtype=torch.long,
            )
            pred_noise = self.denoiser(h, t)
            abar_t = self.alpha_bars[t_scalar]
            pred_h0 = (
                h - torch.sqrt(1.0 - abar_t) * pred_noise
            ) / torch.sqrt(abar_t)

            t_next = timesteps[j + 1]
            abar_next = self.alpha_bars[t_next]
            h = (
                torch.sqrt(abar_next) * pred_h0
                + torch.sqrt(1.0 - abar_next) * pred_noise
            )
        return h


# ---------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def clean_peptide(x: object) -> Optional[str]:
    pep = str(x).strip().upper()
    if len(pep) != SEQ_LEN:
        return None
    if any(a not in AA_TO_I for a in pep):
        return None
    return pep


def onehot_encode_peptides(peptides: Sequence[str]):
    x = torch.zeros(len(peptides), SEQ_LEN, VOCAB, dtype=torch.float32)
    for i, pep in enumerate(peptides):
        pep2 = clean_peptide(pep)
        if pep2 is None:
            raise ValueError(f"Invalid peptide: {pep!r}")
        for t, aa in enumerate(pep2):
            x[i, t, AA_TO_I[aa]] = 1.0
    return x


def levenshtein_edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(
                current[j - 1] + 1,
                previous[j] + 1,
                previous[j - 1] + int(ca != cb),
            ))
        previous = current
    return int(previous[-1])


def project_to_sphere(x: torch.Tensor, radius: float) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12) * float(radius)


def unit_sphere(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def pareto_mask_maximize(Y: torch.Tensor):
    if Y.numel() == 0:
        return torch.zeros(0, dtype=torch.bool, device=Y.device)
    A = Y.unsqueeze(0)
    B = Y.unsqueeze(1)
    dominates = (A >= B).all(-1) & (A > B).any(-1)
    dominates.fill_diagonal_(False)
    dominated = dominates.any(dim=1)
    return ~dominated


def dominates(y_a: torch.Tensor, y_b: torch.Tensor) -> bool:
    return bool(((y_a >= y_b).all() and (y_a > y_b).any()).item())


# ---------------------------------------------------------------------
# Model loading / epsilon transforms
# ---------------------------------------------------------------------

def load_model_from_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)

    cfg_d = ckpt["model_config"]
    diff_d = ckpt["diffusion_config"]

    cfg = ModelConfig(**cfg_d)
    diff_cfg = DiffusionConfig(**diff_d)

    vae = GRUVAE(cfg).to(device)
    diffusion = LatentDiffusion(cfg.latent_dim, diff_cfg).to(device)

    vae.load_state_dict(ckpt["vae_state_dict"], strict=True)
    diffusion.load_state_dict(ckpt["diffusion_state_dict"], strict=True)

    for p in vae.parameters():
        p.requires_grad_(False)
    for p in diffusion.parameters():
        p.requires_grad_(False)

    vae.eval()
    diffusion.eval()

    latent_mean = ckpt["latent_mean"].to(device).float()
    latent_std = ckpt["latent_std"].to(device).float().clamp_min(1e-6)

    radius = float(
        ckpt.get("recommended_bo_radius", math.sqrt(cfg.latent_dim))
    )
    ddim_steps = int(ckpt.get("args", {}).get("ddim_steps", 20))

    print(f"Loaded checkpoint: {path}")
    print(f"epoch: {ckpt.get('epoch')}")
    print(f"selection: {ckpt.get('checkpoint_selection')}")
    print(f"latent_dim: {cfg.latent_dim}")
    print(f"epsilon sphere radius: {radius:.6f}")
    print(f"DDIM steps: {ddim_steps}")

    return vae, diffusion, ckpt, latent_mean, latent_std, radius, ddim_steps


@torch.no_grad()
def encode_to_epsilon(
    vae,
    diffusion,
    x_onehot,
    latent_mean,
    latent_std,
    radius,
    ddim_steps,
    batch_size=256,
):
    chunks = []
    for start in range(0, x_onehot.size(0), batch_size):
        xb = x_onehot[start:start + batch_size]
        mu, _, _ = vae.enc(xb)
        h0 = (mu - latent_mean) / latent_std
        eps = diffusion.ddim_invert(h0, inference_steps=ddim_steps)
        eps = project_to_sphere(eps, radius)
        chunks.append(eps.detach().cpu())
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def epsilon_to_decoder_mu(
    diffusion,
    epsilon,
    latent_mean,
    latent_std,
    ddim_steps,
):
    h0 = diffusion.ddim_sample(epsilon, inference_steps=ddim_steps)
    return h0 * latent_std + latent_mean


@torch.no_grad()
def autoregressive_decode_one(
    vae,
    z,
    temperature=1.0,
    sample=False,
):
    if z.ndim != 2 or z.size(0) != 1:
        raise ValueError(f"Expected z [1,d], got {tuple(z.shape)}")

    h = vae.dec.initial_hidden(z)
    current = torch.zeros(
        1, 1, VOCAB, dtype=z.dtype, device=z.device
    )
    ids = []
    selected_probs = []

    tau = max(float(temperature), 1e-6)
    for _ in range(SEQ_LEN):
        logits, h = vae.dec.step(z, current, h)
        probs = torch.softmax(logits / tau, dim=-1)

        if sample:
            idx = torch.multinomial(probs, 1).squeeze(-1)
        else:
            idx = probs.argmax(dim=-1)

        selected = probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        ids.append(int(idx.item()))
        selected_probs.append(float(selected.item()))

        current = F.one_hot(
            idx, num_classes=VOCAB
        ).to(dtype=z.dtype).unsqueeze(1)

    pep = "".join(I_TO_AA[i] for i in ids)
    return (
        pep,
        float(np.mean(selected_probs)),
        float(np.min(selected_probs)),
        selected_probs,
    )


@torch.no_grad()
def peptide_to_epsilon(
    vae,
    diffusion,
    peptide,
    latent_mean,
    latent_std,
    radius,
    ddim_steps,
):
    x = onehot_encode_peptides([peptide]).to(latent_mean.device)
    mu, _, _ = vae.enc(x)
    h0 = (mu - latent_mean) / latent_std
    eps = diffusion.ddim_invert(h0, inference_steps=ddim_steps)
    return project_to_sphere(eps, radius)


@torch.no_grad()
def epsilon_roundtrip_diagnostics(
    vae,
    diffusion,
    peptide,
    eps_ref,
    latent_mean,
    latent_std,
    radius,
    ddim_steps,
):
    eps_rt = peptide_to_epsilon(
        vae,
        diffusion,
        peptide,
        latent_mean,
        latent_std,
        radius,
        ddim_steps,
    )
    a = eps_ref.reshape(-1)
    b = eps_rt.reshape(-1)
    l2 = torch.linalg.norm(a - b).item()
    cos = F.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0), dim=-1
    ).item()
    return float(l2), float(cos)


# ---------------------------------------------------------------------
# Decoder monitoring / novelty generation
# ---------------------------------------------------------------------

@torch.no_grad()
def decode_candidates_with_diagnostics(
    vae,
    diffusion,
    E_cand,
    bo_iter,
    args,
    latent_mean,
    latent_std,
    radius,
    ddim_steps,
):
    ensure_dir(args.decoder_dir)
    rows = []
    decoded = []

    for j in range(E_cand.size(0)):
        eps = project_to_sphere(E_cand[j:j+1], radius)
        z = epsilon_to_decoder_mu(
            diffusion,
            eps,
            latent_mean,
            latent_std,
            ddim_steps,
        )

        center_pep, mean_p, min_p, token_p = autoregressive_decode_one(
            vae, z, args.decoder_temperature, sample=False
        )
        rt_l2, rt_cos = epsilon_roundtrip_diagnostics(
            vae, diffusion, center_pep, eps,
            latent_mean, latent_std, radius, ddim_steps
        )

        decoded.append(center_pep)
        rows.append({
            "bo_iter": bo_iter,
            "epsilon_candidate_index": j,
            "decode_rank": 0,
            "decode_type": "argmax",
            "peptide": center_pep,
            "decoder_confidence_mean": mean_p,
            "decoder_confidence_min": min_p,
            "token_probabilities": "|".join(f"{v:.6f}" for v in token_p),
            "epsilon_roundtrip_l2": rt_l2,
            "epsilon_roundtrip_cosine": rt_cos,
            "epsilon_norm": float(eps[0].norm().cpu()),
            "local_noise_l2": 0.0,
            "edit_distance_to_center_peptide": 0,
        })

        # stochastic decoder samples from the same epsilon center
        for r in range(1, args.decoder_samples_per_epsilon):
            pep_s, mean_s, min_s, token_s = autoregressive_decode_one(
                vae, z, args.decoder_temperature, sample=True
            )
            rt_l2_s, rt_cos_s = epsilon_roundtrip_diagnostics(
                vae, diffusion, pep_s, eps,
                latent_mean, latent_std, radius, ddim_steps
            )
            decoded.append(pep_s)
            rows.append({
                "bo_iter": bo_iter,
                "epsilon_candidate_index": j,
                "decode_rank": r,
                "decode_type": "sample",
                "peptide": pep_s,
                "decoder_confidence_mean": mean_s,
                "decoder_confidence_min": min_s,
                "token_probabilities": "|".join(f"{v:.6f}" for v in token_s),
                "epsilon_roundtrip_l2": rt_l2_s,
                "epsilon_roundtrip_cosine": rt_cos_s,
                "epsilon_norm": float(eps[0].norm().cpu()),
                "local_noise_l2": 0.0,
                "edit_distance_to_center_peptide":
                    levenshtein_edit_distance(center_pep, pep_s),
            })

        # sphere-preserving local perturbations around optimized epsilon
        for sigma in args.novelty_sigmas:
            for r in range(args.local_neighbors_per_sigma):
                noise = torch.randn_like(eps) * float(sigma)
                eps_local = project_to_sphere(eps + noise, radius)

                z_local = epsilon_to_decoder_mu(
                    diffusion,
                    eps_local,
                    latent_mean,
                    latent_std,
                    ddim_steps,
                )
                pep_l, mean_l, min_l, token_l = autoregressive_decode_one(
                    vae, z_local, args.decoder_temperature, sample=False
                )
                rt_l2_l, rt_cos_l = epsilon_roundtrip_diagnostics(
                    vae, diffusion, pep_l, eps_local,
                    latent_mean, latent_std, radius, ddim_steps
                )

                decoded.append(pep_l)
                rows.append({
                    "bo_iter": bo_iter,
                    "epsilon_candidate_index": j,
                    "decode_rank": 1000 + r,
                    "decode_type": f"sphere_local_sigma_{sigma:g}",
                    "peptide": pep_l,
                    "decoder_confidence_mean": mean_l,
                    "decoder_confidence_min": min_l,
                    "token_probabilities":
                        "|".join(f"{v:.6f}" for v in token_l),
                    "epsilon_roundtrip_l2": rt_l2_l,
                    "epsilon_roundtrip_cosine": rt_cos_l,
                    "epsilon_norm": float(eps_local[0].norm().cpu()),
                    "local_noise_l2":
                        float(torch.linalg.norm(eps_local - eps).cpu()),
                    "edit_distance_to_center_peptide":
                        levenshtein_edit_distance(center_pep, pep_l),
                })

    df = pd.DataFrame(rows)
    df["is_duplicate_raw_decode"] = df.duplicated(
        subset=["peptide"], keep="first"
    )
    df.to_csv(
        os.path.join(
            args.decoder_dir,
            f"decoder_diagnostics_bo_iter_{bo_iter:03d}.csv",
        ),
        index=False,
    )

    unique = list(dict.fromkeys(decoded))
    return unique, df


# ---------------------------------------------------------------------
# GP / acquisition
# ---------------------------------------------------------------------

def fit_mo_models(U: torch.Tensor, Y: torch.Tensor, device):
    """
    U is epsilon / radius and therefore lies on the unit sphere.
    """
    models = []
    for m in range(Y.size(-1)):
        gp = SingleTaskGP(
            U,
            Y[:, m:m+1],
            outcome_transform=Standardize(m=1),
        ).to(device)

        gp.likelihood.noise_covar.register_constraint(
            "raw_noise", gpytorch.constraints.GreaterThan(1e-6)
        )
        gp.likelihood.noise = 1e-4

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            gp.likelihood, gp
        )
        fit_gpytorch_mll(mll)
        models.append(gp)

    return ModelListGP(*models)


class SphereProjectedAcquisition(AcquisitionFunction):
    """
    Wrap a BoTorch acquisition function so every proposed raw coordinate is
    normalized to the unit sphere before acquisition evaluation.

    optimize_acqf can therefore work with a local box, while the underlying GP
    is always queried at valid spherical epsilon coordinates.
    """
    def __init__(self, base_acq: AcquisitionFunction):
        super().__init__(model=base_acq.model)
        self.base_acq = base_acq

    def forward(self, X):
        Xs = unit_sphere(X)
        return self.base_acq(Xs)


# ---------------------------------------------------------------------
# Black-box / novelty / Pareto
# ---------------------------------------------------------------------

def evaluate_blackbox(
    peptides: Sequence[str],
    cache: Dict[str, List[float]],
    obj_cols: Sequence[str],
):
    norm_peps = [str(p).strip().upper() for p in peptides]
    uncached = [p for p in norm_peps if p not in cache]

    if uncached:
        if blackbox_fc is None:
            raise RuntimeError(
                f"blackbox_fc could not be imported: {BLACKBOX_IMPORT_ERROR}"
            )

        scored = blackbox_fc(uncached)

        if isinstance(scored, torch.Tensor):
            arr = scored.detach().cpu().numpy()
            if arr.shape != (len(uncached), len(obj_cols)):
                raise ValueError(
                    f"blackbox_fc returned tensor shape {arr.shape}; "
                    f"expected {(len(uncached), len(obj_cols))}"
                )
            for p, row in zip(uncached, arr):
                cache[p] = [float(v) for v in row]

        elif isinstance(scored, pd.DataFrame):
            for _, row in scored.iterrows():
                pep = clean_peptide(
                    row.get("peptide_len10", row.get("peptide", ""))
                )
                if pep is not None:
                    cache[pep] = [float(row[c]) for c in obj_cols]
        else:
            raise TypeError(
                f"Unsupported blackbox_fc return type: {type(scored)}"
            )

    missing = [p for p in norm_peps if p not in cache]
    if missing:
        raise RuntimeError(
            f"Black-box did not return scores for {len(missing)} peptide(s), "
            f"e.g. {missing[:5]}"
        )

    return torch.tensor(
        [cache[p] for p in norm_peps],
        dtype=torch.float32,
    )


def filter_novel(
    candidates,
    train_set,
    evaluated_set,
    generated_set,
    args,
):
    accepted = []
    rejected = []

    for pep in candidates:
        reason = None

        if clean_peptide(pep) is None:
            reason = "invalid_peptide"
        elif args.reject_training_peptides and pep in train_set:
            reason = "in_training"
        elif args.reject_seen_peptides and pep in evaluated_set:
            reason = "already_evaluated"
        elif args.reject_seen_peptides and pep in generated_set:
            reason = "duplicate_generated"

        if reason is None:
            accepted.append(pep)
        else:
            rejected.append((pep, reason))

    return accepted, rejected


def build_initial_labeled_indices(Y, n_init, seed):
    pareto_idx = torch.where(pareto_mask_maximize(Y))[0].cpu().tolist()
    pareto_set = set(pareto_idx)
    non_pareto = [
        i for i in range(Y.size(0))
        if i not in pareto_set
    ]

    rng = np.random.default_rng(seed)
    n_extra = max(0, int(n_init) - len(pareto_idx))

    if n_extra:
        extra = rng.choice(
            non_pareto,
            size=min(n_extra, len(non_pareto)),
            replace=False,
        ).tolist()
    else:
        extra = []

    return list(dict.fromkeys(pareto_idx + extra))


def save_pareto(
    peptides,
    labeled_idx,
    Y_lab,
    bo_iter,
    out_dir,
    obj_cols,
):
    ensure_dir(out_dir)
    Y_cpu = Y_lab.detach().cpu()
    mask = pareto_mask_maximize(Y_cpu)

    rows = []
    for k in torch.where(mask)[0].tolist():
        g = int(labeled_idx[k])
        pep = peptides[g] if 0 <= g < len(peptides) else ""

        row = {
            "bo_iter": bo_iter,
            "global_index": g,
            "peptide": pep,
        }
        for j, c in enumerate(obj_cols):
            row[c] = float(Y_cpu[k, j])

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(
            out_dir,
            f"pareto_front_bo_iter_{bo_iter:03d}.csv",
        ),
        index=False,
    )
    return df


# ---------------------------------------------------------------------
# BO
# ---------------------------------------------------------------------

def run_bo(args):
    set_seed(args.seed)
    device = torch.device(args.device)

    ensure_dir(args.out_dir)
    ensure_dir(args.decoder_dir)
    ensure_dir(args.pareto_dir)

    (
        vae,
        diffusion,
        ckpt,
        latent_mean,
        latent_std,
        radius,
        ddim_steps,
    ) = load_model_from_checkpoint(args.diffusion_checkpoint, device)

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------
    df = pd.read_csv(args.data_csv)

    required = [args.peptide_col] + list(args.obj_cols)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Data CSV missing columns: {missing}")

    df = df.copy()
    df[args.peptide_col] = df[args.peptide_col].map(clean_peptide)
    df = df.dropna(
        subset=[args.peptide_col] + list(args.obj_cols)
    ).reset_index(drop=True)

    peptides = df[args.peptide_col].astype(str).tolist()

    X_all = onehot_encode_peptides(peptides).to(device)
    Y_train_full = torch.tensor(
        df[list(args.obj_cols)].to_numpy(dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )

    print(f"Valid Cu rows: {len(peptides)}")

    # ------------------------------------------------------------
    # Encode full Cu set to spherical epsilon
    # ------------------------------------------------------------
    E_all = encode_to_epsilon(
        vae,
        diffusion,
        X_all,
        latent_mean,
        latent_std,
        radius,
        ddim_steps,
        batch_size=args.encode_batch_size,
    ).to(device)

    norm_mean = float(E_all.norm(dim=-1).mean().cpu())
    norm_std = float(E_all.norm(dim=-1).std().cpu())
    print(
        f"Epsilon norms: mean={norm_mean:.6f}, std={norm_std:.6e}, "
        f"target={radius:.6f}"
    )

    if abs(norm_mean - radius) > 1e-3:
        raise RuntimeError(
            "Encoded epsilon coordinates are not on expected sphere."
        )

    # BO GP coordinates = unit-sphere epsilon
    U_all = E_all / radius

    # ------------------------------------------------------------
    # Save training Pareto
    # ------------------------------------------------------------
    train_pareto_mask = pareto_mask_maximize(Y_train_full)
    train_pareto_idx = torch.where(train_pareto_mask)[0]
    Y_train_pareto = Y_train_full[train_pareto_mask]

    train_pareto_df = pd.DataFrame({
        "global_index": train_pareto_idx.cpu().numpy(),
        "peptide": [
            peptides[i] for i in train_pareto_idx.cpu().tolist()
        ],
        **{
            c: Y_train_pareto[:, j].detach().cpu().numpy()
            for j, c in enumerate(args.obj_cols)
        },
    })
    train_pareto_path = os.path.join(
        args.out_dir,
        "train_pareto_CU_gru_vae_diffusion_epsilon.csv",
    )
    train_pareto_df.to_csv(train_pareto_path, index=False)
    print(f"Training Pareto: {len(train_pareto_df)} solutions")

    cache = {
        pep: [float(v) for v in vals]
        for pep, vals in zip(
            peptides,
            df[list(args.obj_cols)].to_numpy(dtype=np.float32).tolist(),
        )
    }

    train_set = set(peptides)
    evaluated_set = set()
    generated_set = set()

    # ------------------------------------------------------------
    # Initial observations
    # ------------------------------------------------------------
    labeled_idx = build_initial_labeled_indices(
        Y_train_full,
        args.initial_labeled,
        args.seed + 42,
    )

    idx_t = torch.tensor(
        labeled_idx, dtype=torch.long, device=device
    )
    Y_lab = Y_train_full[idx_t]
    E_lab = E_all[idx_t]
    U_lab = U_all[idx_t]

    evaluated_set.update(peptides[i] for i in labeled_idx)

    initial_ref = (
        Y_lab.detach().min(dim=0).values
        - float(args.ref_margin)
    )
    best_hv = float(
        Hypervolume(ref_point=initial_ref)
        .compute(Y_lab)
    )

    run_config = vars(args).copy()
    run_config.update({
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_selection": ckpt.get("checkpoint_selection"),
        "bo_space": "diffusion_base_noise_spherical_epsilon",
        "latent_dim": int(vae.cfg.latent_dim),
        "epsilon_radius": float(radius),
        "ddim_steps": int(ddim_steps),
        "training_pareto_size": int(len(train_pareto_df)),
        "initial_labeled_actual": int(len(labeled_idx)),
        "acquisition_geometry":
            "local_box_in_unit_coordinates_with_acq_projected_to_unit_sphere",
    })
    with open(
        os.path.join(args.out_dir, "bo_run_config.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(run_config, f, indent=2)

    hv_rows = []
    all_candidate_rows = []

    # ------------------------------------------------------------
    # BO loop
    # ------------------------------------------------------------
    for bo_iter in range(int(args.bo_iters)):
        print(f"\n=== BO iter {bo_iter + 1}/{args.bo_iters} ===")

        U_gp = U_lab.detach().double().to(device)
        Y_gp = Y_lab.detach().double().to(device)

        mo_model = fit_mo_models(U_gp, Y_gp, device)

        # Select a trust-region center from current nondominated points.
        nd_mask = pareto_mask_maximize(Y_lab)
        nd_idx = torch.where(nd_mask)[0]

        # simple balanced scalarization for center selection
        Y_nd = Y_lab[nd_idx]
        Y_min = Y_nd.min(dim=0).values
        Y_rng = (Y_nd.max(dim=0).values - Y_min).clamp_min(1e-12)
        score_nd = ((Y_nd - Y_min) / Y_rng).mean(dim=-1)
        center_lab_i = int(nd_idx[torch.argmax(score_nd)].item())

        u_center = unit_sphere(
            U_lab[center_lab_i:center_lab_i+1].detach().double()
        )[0]

        # Local raw-coordinate box. Acquisition wrapper projects to sphere.
        lb = (u_center - float(args.tr_radius_unit)).clamp(-1.0, 1.0)
        ub = (u_center + float(args.tr_radius_unit)).clamp(-1.0, 1.0)
        bounds = torch.stack([lb, ub], dim=0)

        # qEHVI partitioning
        with torch.no_grad():
            Y_nd_gp = Y_gp[pareto_mask_maximize(Y_gp)]
            if Y_nd_gp.size(0) > args.max_partition_points:
                normY = (
                    Y_nd_gp - Y_nd_gp.min(dim=0).values
                ) / (
                    Y_nd_gp.max(dim=0).values
                    - Y_nd_gp.min(dim=0).values
                ).clamp_min(1e-12)
                keep = torch.topk(
                    normY.mean(dim=-1),
                    k=args.max_partition_points,
                    largest=True,
                ).indices
                Y_part = Y_nd_gp[keep]
            else:
                Y_part = Y_nd_gp

        ref_tensor = (
            Y_gp.min(dim=0).values - float(args.ref_margin)
        )
        ref_point = ref_tensor.tolist()

        partitioning = NondominatedPartitioning(
            ref_point=ref_tensor,
            Y=Y_part,
        )

        sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([args.mc_samples])
        ).to(device)

        if (
            args.acqf == "qlogehvi"
            and qLogExpectedHypervolumeImprovement is not None
        ):
            acq_cls = qLogExpectedHypervolumeImprovement
        else:
            acq_cls = qExpectedHypervolumeImprovement

        base_acq = acq_cls(
            model=mo_model,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
        ).to(device)

        acq = SphereProjectedAcquisition(base_acq).to(device)

        # optimize acquisition
        raw_parts = []
        acq_vals = []

        repeats = (
            int(args.q_batch)
            if args.optimize_q_one_at_a_time
            else 1
        )
        q_inner = (
            1
            if args.optimize_q_one_at_a_time
            else int(args.q_batch)
        )

        for _ in range(repeats):
            try:
                U_raw, acq_val = optimize_acqf(
                    acq,
                    bounds=bounds,
                    q=q_inner,
                    num_restarts=args.num_restarts,
                    raw_samples=args.raw_samples,
                    sequential=True,
                )
            except torch.OutOfMemoryError:
                print("[OOM] Retrying with smaller acquisition settings.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                safe_sampler = SobolQMCNormalSampler(
                    sample_shape=torch.Size([
                        max(8, args.mc_samples // 4)
                    ])
                ).to(device)

                safe_base = acq_cls(
                    model=mo_model,
                    ref_point=ref_point,
                    partitioning=partitioning,
                    sampler=safe_sampler,
                ).to(device)

                safe_acq = SphereProjectedAcquisition(
                    safe_base
                ).to(device)

                U_raw, acq_val = optimize_acqf(
                    safe_acq,
                    bounds=bounds,
                    q=1,
                    num_restarts=max(1, args.num_restarts // 2),
                    raw_samples=max(16, args.raw_samples // 4),
                    sequential=True,
                )

            raw_parts.append(U_raw.detach())
            acq_vals.append(acq_val.detach().reshape(-1)[0])

        U_raw = torch.cat(raw_parts, dim=0)
        U_cand = unit_sphere(U_raw).to(torch.float32)
        E_cand = U_cand * radius
        acq_value = torch.stack(acq_vals).mean()

        # Diagnostics ensure exact sphere geometry.
        cand_norms = E_cand.norm(dim=-1)
        print(
            f"Proposed epsilon norm mean="
            f"{float(cand_norms.mean().cpu()):.6f}, "
            f"min={float(cand_norms.min().cpu()):.6f}, "
            f"max={float(cand_norms.max().cpu()):.6f}"
        )

        # Decode candidate epsilon centers and novelty neighbors.
        cand_raw, decoder_df = decode_candidates_with_diagnostics(
            vae,
            diffusion,
            E_cand,
            bo_iter,
            args,
            latent_mean,
            latent_std,
            radius,
            ddim_steps,
        )

        cand_peps, rejected = filter_novel(
            cand_raw,
            train_set,
            evaluated_set,
            generated_set,
            args,
        )

        # Fallback: increased stochastic decoder temperature.
        if len(cand_peps) == 0:
            print(
                "[BO] No novel peptide survived filtering; "
                "trying higher decoder temperature."
            )
            old_temp = args.decoder_temperature
            args.decoder_temperature = max(
                old_temp, args.fallback_temperature
            )

            extra, extra_df = decode_candidates_with_diagnostics(
                vae,
                diffusion,
                E_cand,
                bo_iter + 100000,
                args,
                latent_mean,
                latent_std,
                radius,
                ddim_steps,
            )
            args.decoder_temperature = old_temp

            cand_peps, rejected2 = filter_novel(
                extra,
                train_set,
                evaluated_set,
                generated_set,
                args,
            )
            rejected.extend(rejected2)
            decoder_df = pd.concat(
                [decoder_df, extra_df],
                ignore_index=True,
            )

        rejected_map = {p: reason for p, reason in rejected}
        status_df = decoder_df.copy()

        if len(cand_peps) == 0:
            print("[BO] Still no novel peptide; skipping objective evaluation.")

            generated_set.update(cand_raw)

            status_df["accepted_for_blackbox"] = False
            status_df["rejected_reason"] = status_df["peptide"].map(
                lambda p: rejected_map.get(
                    p, "duplicate_or_not_selected"
                )
            )
            status_df.to_csv(
                os.path.join(
                    args.decoder_dir,
                    f"all_decoder_diagnostics_status_bo_iter_{bo_iter:03d}.csv",
                ),
                index=False,
            )

            hv_rows.append({
                "bo_iter": bo_iter,
                "n_raw_decoded": len(cand_raw),
                "n_accepted": 0,
                "n_labeled": len(labeled_idx),
                "hypervolume": best_hv,
                "best_hypervolume": best_hv,
                "acq_value": float(acq_value.cpu()),
            })
            pd.DataFrame(hv_rows).to_csv(
                os.path.join(
                    args.out_dir, "bo_hypervolume_history.csv"
                ),
                index=False,
            )
            continue

        # --------------------------------------------------------
        # True objective evaluation
        # --------------------------------------------------------
        Y_new = evaluate_blackbox(
            cand_peps, cache, args.obj_cols
        ).to(device)

        # Each peptide is represented for subsequent GP training by its
        # OWN re-encoded spherical epsilon coordinate, not merely the
        # acquisition center that happened to generate it.
        E_new = []
        for pep in cand_peps:
            e = peptide_to_epsilon(
                vae,
                diffusion,
                pep,
                latent_mean,
                latent_std,
                radius,
                ddim_steps,
            )
            E_new.append(e[0])

        E_new = torch.stack(E_new, dim=0).to(device)
        E_new = project_to_sphere(E_new, radius)
        U_new = E_new / radius

        comparison_rows = []
        for i, (pep, y) in enumerate(zip(cand_peps, Y_new)):
            dominated_by_train = any(
                dominates(y_ref, y)
                for y_ref in Y_train_pareto
            )
            dominates_train_any = any(
                dominates(y, y_ref)
                for y_ref in Y_train_pareto
            )

            row = {
                "bo_iter": bo_iter,
                "peptide": pep,
                "acq_value": float(acq_value.cpu()),
                "epsilon_norm": float(E_new[i].norm().cpu()),
                "dominated_by_train_pareto": int(dominated_by_train),
                "dominates_any_train_pareto": int(dominates_train_any),
            }
            for j, c in enumerate(args.obj_cols):
                row[c] = float(y[j].cpu())

            comparison_rows.append(row)
            all_candidate_rows.append(row)

        pd.DataFrame(comparison_rows).to_csv(
            os.path.join(
                args.out_dir,
                f"accepted_candidates_scored_bo_iter_{bo_iter:03d}.csv",
            ),
            index=False,
        )

        # attach scores to decoder-monitoring rows
        status_df["accepted_for_blackbox"] = (
            status_df["peptide"].isin(cand_peps)
        )
        status_df["rejected_reason"] = status_df["peptide"].map(
            lambda p: (
                ""
                if p in cand_peps
                else rejected_map.get(
                    p, "duplicate_or_not_selected"
                )
            )
        )

        for j, c in enumerate(args.obj_cols):
            val_map = {
                p: float(y[j].cpu())
                for p, y in zip(cand_peps, Y_new)
            }
            status_df[c] = status_df["peptide"].map(
                lambda p: val_map.get(p, np.nan)
            )

        status_df.to_csv(
            os.path.join(
                args.decoder_dir,
                f"all_decoder_diagnostics_status_bo_iter_{bo_iter:03d}.csv",
            ),
            index=False,
        )

        # Add novel evaluated peptides to GP training data.
        new_global_indices = []
        for pep in cand_peps:
            peptides.append(pep)
            new_global_indices.append(len(peptides) - 1)

        E_lab = torch.cat([E_lab, E_new], dim=0)
        U_lab = torch.cat([U_lab, U_new], dim=0)
        Y_lab = torch.cat([Y_lab, Y_new], dim=0)
        labeled_idx.extend(new_global_indices)

        evaluated_set.update(cand_peps)
        generated_set.update(cand_raw)

        # --------------------------------------------------------
        # Hypervolume / Pareto monitoring
        # --------------------------------------------------------
        current_ref = (
            Y_lab.min(dim=0).values
            - float(args.ref_margin)
        )

        hv = float(
            Hypervolume(ref_point=current_ref)
            .compute(Y_lab)
        )

        improved = hv > best_hv + float(args.hv_tol)
        if improved:
            best_hv = hv

        pareto_df = save_pareto(
            peptides,
            labeled_idx,
            Y_lab,
            bo_iter,
            args.pareto_dir,
            args.obj_cols,
        )

        hv_row = {
            "bo_iter": bo_iter,
            "n_raw_decoded": len(cand_raw),
            "n_accepted": len(cand_peps),
            "n_labeled": len(labeled_idx),
            "pareto_size": len(pareto_df),
            "hypervolume": hv,
            "best_hypervolume": best_hv,
            "improved": int(improved),
            "acq_value": float(acq_value.cpu()),
            "n_dominates_train": int(sum(
                r["dominates_any_train_pareto"]
                for r in comparison_rows
            )),
        }
        hv_rows.append(hv_row)

        pd.DataFrame(hv_rows).to_csv(
            os.path.join(
                args.out_dir, "bo_hypervolume_history.csv"
            ),
            index=False,
        )
        pd.DataFrame(all_candidate_rows).to_csv(
            os.path.join(
                args.out_dir, "all_accepted_candidates_scored.csv"
            ),
            index=False,
        )

        print(
            f"Accepted {len(cand_peps)} peptides. "
            f"HV={hv:.6f}, best={best_hv:.6f}, "
            f"dominates_train={hv_row['n_dominates_train']}"
        )

    # ------------------------------------------------------------
    # Final Pareto
    # ------------------------------------------------------------
    Y_cpu = Y_lab.detach().cpu()
    mask = pareto_mask_maximize(Y_cpu)

    final_rows = []
    for k in torch.where(mask)[0].tolist():
        g = int(labeled_idx[k])
        pep = peptides[g] if 0 <= g < len(peptides) else ""

        row = {
            "global_index": g,
            "peptide": pep,
            "is_novel": pep not in train_set,
        }
        for j, c in enumerate(args.obj_cols):
            row[c] = float(Y_cpu[k, j])
        final_rows.append(row)

    final_df = pd.DataFrame(final_rows)

    if len(final_df):
        Y_final = torch.tensor(
            final_df[list(args.obj_cols)].to_numpy(dtype=np.float32)
        )
        train_pareto_cpu = Y_train_pareto.detach().cpu()

        final_df["dominates_train_pareto"] = [
            any(
                dominates(Y_final[i], y_ref)
                for y_ref in train_pareto_cpu
            )
            for i in range(len(final_df))
        ]

    final_path = os.path.join(
        args.out_dir,
        "bo_final_pareto_CU_gru_vae_diffusion_epsilon_gp.csv",
    )
    final_df.to_csv(final_path, index=False)

    if len(final_df):
        dom_df = final_df[
            (final_df["is_novel"] == True)
            & (final_df["dominates_train_pareto"] == True)
        ].copy()
    else:
        dom_df = pd.DataFrame()

    dom_path = os.path.join(
        args.out_dir,
        "bo_pareto_dominates_train_CU_gru_vae_diffusion_epsilon_gp.csv",
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
            "Multi-objective GP/qEHVI BO in spherical diffusion epsilon "
            "space of the Cu-finetuned GRU-VAE latent-diffusion model."
        )
    )

    p.add_argument(
        "--diffusion-checkpoint",
        required=True,
        help=(
            "Use the best DDIM-inversion .pt checkpoint from the Cu latent-"
            "diffusion fine-tuning run."
        ),
    )
    p.add_argument(
        "--data-csv",
        required=True,
        help="Cu scored peptide CSV used as the initial evaluated database.",
    )
    p.add_argument("--peptide-col", default="peptide_len10")
    p.add_argument("--obj-cols", nargs=4, default=OBJ_COLS)

    p.add_argument(
        "--out-dir",
        default="bo_results_CU_gru_vae_diffusion_epsilon_gp",
    )
    p.add_argument(
        "--decoder-dir",
        default="bo_decoder_monitoring_CU_gru_vae_diffusion_epsilon_gp",
    )
    p.add_argument(
        "--pareto-dir",
        default="pareto_front_CU_gru_vae_diffusion_epsilon_gp",
    )

    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--encode-batch-size", type=int, default=256)

    # BO
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

    # In u = epsilon/radius coordinates.
    p.add_argument(
        "--tr-radius-unit",
        type=float,
        default=0.20,
        help=(
            "Half-width of local acquisition box around a unit-sphere "
            "epsilon center before sphere projection."
        ),
    )

    p.add_argument("--max-partition-points", type=int, default=20)
    p.add_argument("--ref-margin", type=float, default=0.10)
    p.add_argument("--hv-tol", type=float, default=1e-6)

    # Decoder novelty generation.
    p.add_argument(
        "--decoder-samples-per-epsilon",
        type=int,
        default=16,
    )
    p.add_argument("--decoder-temperature", type=float, default=0.90)
    p.add_argument("--fallback-temperature", type=float, default=1.25)

    # These sigmas are in raw epsilon coordinates, whose radius is 8 for Z64.
    p.add_argument(
        "--novelty-sigmas",
        nargs="*",
        type=float,
        default=[0.025, 0.05, 0.10],
        help=(
            "Gaussian epsilon perturbation std before reprojection to the "
            "radius-sqrt(latent_dim) sphere."
        ),
    )
    p.add_argument(
        "--local-neighbors-per-sigma",
        type=int,
        default=4,
    )

    p.add_argument(
        "--reject-training-peptides",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--reject-seen-peptides",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    return p.parse_args()


if __name__ == "__main__":
    run_bo(parse_args())
