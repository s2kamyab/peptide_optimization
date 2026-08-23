from __future__ import annotations

"""
Prospective leakage-safe multi-objective GP/qLogEHVI Bayesian optimization
in the latent-diffusion epsilon space learned by:

    finetune_best_bo_ready_gru_vae_cu_latent_diffusion_leakage_safe_v3_objective_gated.py

This BO implementation is matched to the revised fine-tuning framework.

Fine-tuned generative path
--------------------------
peptide
    -> frozen GRU-VAE encoder mu
    -> training-only standardization h0
    -> deterministic DDIM inversion
    -> epsilon                                      [BO space]

epsilon candidate
    -> deterministic DDIM sampling -> h0
    -> unstandardize -> decoder mu
    -> autoregressive GRU decoder
    -> peptide
    -> Cu black-box objectives

Objectives (all maximized)
--------------------------
    chelation_sub
    solubility_sub
    stability_sub
    expression_sub

Critical consistency choices
----------------------------
1. The fine-tuning framework defaults to
       --no-project-inverted-epsilon-to-sphere
   and the attached 150-epoch history confirms that sphere projection was OFF.
   This BO code now supports both native and fixed-radius spherical epsilon via
   --epsilon-geometry. Native remains the default for the attached history.

2. The exact leakage-safe Cu manifest loader from the fine-tuning script is reused.
   Only Cu TRAIN peptides form historical BO observations. Validation/test remain
   completely excluded from GP fitting, the initial Pareto front, and score cache.

3. The attached history contains no epoch satisfying the hard BO objective/effective-
   dimension gate because epsilon standardized effective dimension remains below the
   default floor of 24. The fine-tuning code therefore falls back to the minimum-
   validation-objective-MSE checkpoint. In the attached history this is epoch 148:
       val objective MSE          ~ 0.005359
       val mean objective Pearson ~ 0.540719
       epsilon standardized PR    ~ 21.2776
       DDIM inversion L2          ~ 0.02674
   Consequently the recommended checkpoint for this BO run is:
       best_val_objective_mse_h64_z64_cu_latent_diffusion.pt
   unless a later fine-tuning run produces a true
       minimum_objective_gated_bo_ready_score
   checkpoint.

4. epsilon is mapped affinely from one fixed symmetric box [-B,+B]^D to [0,1]^D
   for BoTorch. B is expanded once, before BO, to contain all Cu TRAIN epsilon
   coordinates. No clipping of historical coordinates is allowed.

5. A fixed hypervolume reference point is created from the INITIAL BO design and
   remains unchanged across iterations.

6. Every accepted discrete peptide is re-encoded through
       peptide -> mu -> h0 -> DDIM inversion -> epsilon_actual
   before its black-box objectives are appended to the GP dataset.

7. Exact matches to training, validation/test held-out peptides, already evaluated
   peptides, or previously generated peptides are rejected before expensive scoring.
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
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
)
try:
    from botorch.acquisition.multi_objective.logei import (
        qLogExpectedHypervolumeImprovement,
    )
except Exception:
    qLogExpectedHypervolumeImprovement = None

from botorch.acquisition.acquisition import AcquisitionFunction
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
OBJ_COLS = [
    "chelation_sub",
    "solubility_sub",
    "stability_sub",
    "expression_sub",
]


# =============================================================================
# Utilities / dynamic imports
# =============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_load_full(path: str, map_location):
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
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
    x = torch.zeros(
        len(peptides),
        SEQ_LEN,
        VOCAB,
        dtype=torch.float32,
    )
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
        return torch.zeros(
            0,
            dtype=torch.bool,
            device=Y.device,
        )
    # dominates[j, i] = j dominates i
    A = Y.unsqueeze(1)
    B = Y.unsqueeze(0)
    dominates = (
        (A >= B).all(-1)
        & (A > B).any(-1)
    )
    dominates.fill_diagonal_(False)
    dominated = dominates.any(dim=0)
    return ~dominated


def dominates(a: torch.Tensor, b: torch.Tensor) -> bool:
    return bool(
        ((a >= b).all() & (a > b).any()).item()
    )


def epsilon_to_unit(
    eps: torch.Tensor,
    bound: float,
    clamp: bool = True,
):
    x = (eps + bound) / (2.0 * bound)
    return x.clamp(0.0, 1.0) if clamp else x


def unit_to_epsilon(x: torch.Tensor, bound: float):
    return 2.0 * bound * x - bound


def project_to_radius(x: torch.Tensor, radius: float) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12) * float(radius)


def apply_epsilon_geometry(
    eps: torch.Tensor, geometry: str, sphere_radius: float
) -> torch.Tensor:
    if geometry == "native":
        return eps
    if geometry == "sphere":
        return project_to_radius(eps, sphere_radius)
    raise ValueError(f"Unknown epsilon geometry: {geometry!r}")


class SphereProjectedAcquisition(AcquisitionFunction):
    """Evaluate the base acquisition on fixed-radius projected epsilon points."""
    def __init__(self, base_acq, epsilon_bound: float, sphere_radius: float):
        super().__init__(model=base_acq.model)
        self.base_acq = base_acq
        self.epsilon_bound = float(epsilon_bound)
        self.sphere_radius = float(sphere_radius)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        eps = unit_to_epsilon(X, self.epsilon_bound)
        eps = project_to_radius(eps, self.sphere_radius)
        Xp = epsilon_to_unit(eps, self.epsilon_bound, clamp=True)
        return self.base_acq(Xp)


def import_module_py314_safe(path: str):
    name = "latent_diffusion_finetune_module_for_bo"
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not import fine-tuning script: {path}"
        )
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
    candidates = []

    for p in explicit_project_roots or []:
        candidates.append(Path(p).expanduser())

    anchors = [
        Path.cwd(),
        Path(__file__).resolve().parent,
        Path(finetune_script).expanduser().resolve().parent,
        Path(blackbox_script).expanduser().resolve().parent,
    ]

    for a in anchors:
        candidates.append(a)
        candidates.extend(list(a.parents)[:6])

    for c in candidates:
        _add_sys_path_once(c)

    searched = set()
    for c in candidates:
        try:
            c = c.resolve()
        except Exception:
            continue
        if (
            c in searched
            or not c.exists()
            or not c.is_dir()
        ):
            continue
        searched.add(c)

        pkg = c / "peptide_optimization"
        if pkg.is_dir():
            _add_sys_path_once(c)

        if c.name == "peptide_optimization":
            _add_sys_path_once(c.parent)

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
            q = root / name
            if q.is_file():
                return q.resolve()

    raise FileNotFoundError(
        "Could not locate the Cu black-box scorer. "
        f"Requested={requested!r}. "
        "Pass an explicit path with --blackbox-script."
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

    module_name = "cu_blackbox_module_for_diffusion_bo"
    spec = importlib.util.spec_from_file_location(
        module_name,
        str(path),
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not create import spec for {path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        sys.modules.pop(module_name, None)
        if exc.name == "peptide_optimization":
            raise RuntimeError(
                "The black-box scorer was found, but Python cannot import "
                "`peptide_optimization`. Pass --project-root equal to the "
                "DIRECTORY CONTAINING the peptide_optimization package."
            ) from exc
        raise
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    fn = getattr(module, "blackbox_fc", None)
    if fn is None or not callable(fn):
        raise AttributeError(
            f"{path} does not define callable blackbox_fc."
        )

    print(f"Loaded Cu black-box function from: {path}")
    return fn


# =============================================================================
# Fine-tuned checkpoint loading
# =============================================================================

def load_model(
    checkpoint_path: str,
    finetune_script: str,
    device: torch.device,
    require_preferred_checkpoint: bool = True,
):
    module = import_module_py314_safe(finetune_script)
    ckpt = torch_load_full(
        checkpoint_path,
        map_location=device,
    )

    expected_type = "cu_best_bo_ready_gru_vae_latent_diffusion"
    ctype = ckpt.get("checkpoint_type", "")
    if ctype and ctype != expected_type:
        raise ValueError(
            f"Unexpected checkpoint_type={ctype!r}; "
            f"expected {expected_type!r}"
        )

    cfg = module.ModelConfig(**ckpt["model_config"])
    diff_cfg = module.DiffusionConfig(
        **ckpt["diffusion_config"]
    )

    vae = module.GRUVAE(cfg).to(device)
    diffusion = module.LatentDiffusion(
        cfg.latent_dim,
        diff_cfg,
    ).to(device)

    vae.load_state_dict(
        ckpt["vae_state_dict"],
        strict=True,
    )
    diffusion.load_state_dict(
        ckpt["diffusion_state_dict"],
        strict=True,
    )

    vae.eval()
    diffusion.eval()

    for p in vae.parameters():
        p.requires_grad_(False)
    for p in diffusion.parameters():
        p.requires_grad_(False)

    latent_mean = (
        ckpt["latent_mean"]
        .to(device)
        .float()
    )
    latent_std = (
        ckpt["latent_std"]
        .to(device)
        .float()
        .clamp_min(1e-6)
    )

    selection = ckpt.get(
        "checkpoint_selection",
        "",
    )
    preferred = {
        "minimum_objective_gated_bo_ready_score",
        # Legitimate fallback produced by the attached fine-tuner when
        # no checkpoint passes the hard BO gate.
        "minimum_validation_multiobjective_mse",
    }

    if (
        require_preferred_checkpoint
        and selection not in preferred
    ):
        raise RuntimeError(
            "For prospective BO, use either the objective-gated BO-ready "
            "checkpoint or, if no epoch passed the hard gate, the fine-tuner's "
            "minimum-validation-objective-MSE fallback. "
            f"Found checkpoint_selection={selection!r}."
        )

    ckpt_obj_cols = list(
        ckpt.get("objective_cols", [])
    )

    checkpoint_args = ckpt.get("args", {})
    project_to_sphere = bool(
        checkpoint_args.get(
            "project_inverted_epsilon_to_sphere",
            False,
        )
    )
    ddim_steps = int(
        checkpoint_args.get(
            "ddim_steps",
            50,
        )
    )

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"epoch: {ckpt.get('epoch')}")
    print(f"selection: {selection}")
    print(f"latent_dim: {cfg.latent_dim}")
    print(
        "diffusion_config:",
        ckpt.get("diffusion_config"),
    )
    print(
        "checkpoint objectives:",
        ckpt_obj_cols,
    )
    print(
        "bo_coordinate_space:",
        ckpt.get(
            "bo_coordinate_space",
            "diffusion_base_noise_epsilon",
        ),
    )
    print(
        "project_inverted_epsilon_to_sphere:",
        project_to_sphere,
    )
    print(f"DDIM steps: {ddim_steps}")

    if project_to_sphere:
        print(
            "WARNING: This checkpoint used sphere-projected epsilon. "
            "This BO script supports it only if you explicitly pass "
            "--allow-sphere-projected-checkpoint. For the attached history, "
            "projection was OFF and native epsilon is the matched space."
        )

    return (
        module,
        vae,
        diffusion,
        ckpt,
        latent_mean,
        latent_std,
        ddim_steps,
        project_to_sphere,
    )


# =============================================================================
# epsilon transforms / decoding
# =============================================================================

@torch.no_grad()
def encode_batch_to_epsilon(
    vae,
    diffusion,
    x,
    latent_mean,
    latent_std,
    ddim_steps,
    geometry,
    sphere_radius,
    batch_size=256,
):
    out = []
    for s in range(0, len(x), batch_size):
        xb = x[s:s + batch_size]
        mu, _, _ = vae.enc(xb)
        h0 = (
            (mu - latent_mean)
            / latent_std
        )
        eps = diffusion.ddim_invert(
            h0,
            inference_steps=ddim_steps,
        )
        eps = apply_epsilon_geometry(eps, geometry, sphere_radius)
        out.append(eps.detach().cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def peptide_to_epsilon(
    vae,
    diffusion,
    peptide,
    latent_mean,
    latent_std,
    ddim_steps,
    geometry,
    sphere_radius,
):
    x = onehot_encode([peptide]).to(
        latent_mean.device
    )
    mu, _, _ = vae.enc(x)
    h0 = (
        (mu - latent_mean)
        / latent_std
    )
    eps = diffusion.ddim_invert(
        h0,
        inference_steps=ddim_steps,
    )
    return apply_epsilon_geometry(eps, geometry, sphere_radius)


@torch.no_grad()
def epsilon_to_decoder_mu(
    diffusion,
    epsilon,
    latent_mean,
    latent_std,
    ddim_steps,
):
    h0 = diffusion.ddim_sample(
        epsilon,
        inference_steps=ddim_steps,
    )
    return (
        h0 * latent_std
        + latent_mean
    )


@torch.no_grad()
def decode_one(
    vae,
    decoder_mu,
    temperature=1.0,
    sample=False,
):
    h = vae.dec.initial_hidden(decoder_mu)
    cur = torch.zeros(
        1,
        1,
        VOCAB,
        device=decoder_mu.device,
        dtype=decoder_mu.dtype,
    )
    ids = []
    probs_chosen = []
    tau = max(float(temperature), 1e-6)

    for _ in range(SEQ_LEN):
        logits, h = vae.dec.step(
            decoder_mu,
            cur,
            h,
        )
        probs = torch.softmax(
            logits / tau,
            dim=-1,
        )
        idx = (
            torch.multinomial(
                probs,
                1,
            ).squeeze(-1)
            if sample
            else probs.argmax(dim=-1)
        )

        probs_chosen.append(
            float(
                probs.gather(
                    -1,
                    idx[:, None],
                ).item()
            )
        )
        ids.append(int(idx.item()))
        cur = (
            F.one_hot(
                idx,
                num_classes=VOCAB,
            )
            .to(decoder_mu.dtype)
            .unsqueeze(1)
        )

    pep = "".join(I_TO_AA[i] for i in ids)
    return (
        pep,
        float(np.mean(probs_chosen)),
        float(np.min(probs_chosen)),
    )


@torch.no_grad()
def reencode_diagnostic(
    vae,
    diffusion,
    peptide,
    eps_ref,
    latent_mean,
    latent_std,
    ddim_steps,
    geometry,
    sphere_radius,
):
    e2 = peptide_to_epsilon(
        vae,
        diffusion,
        peptide,
        latent_mean,
        latent_std,
        ddim_steps,
        geometry,
        sphere_radius,
    )
    l2 = float(
        torch.linalg.norm(
            e2 - eps_ref
        ).cpu()
    )
    cos = float(
        F.cosine_similarity(
            e2,
            eps_ref,
            dim=-1,
        ).item()
    )
    return l2, cos


@torch.no_grad()
def decode_candidates(
    vae,
    diffusion,
    E,
    bo_iter,
    args,
    latent_mean,
    latent_std,
    ddim_steps,
):
    rows = []
    peps = []

    for j in range(len(E)):
        eps = E[j:j+1]
        mu = epsilon_to_decoder_mu(
            diffusion,
            eps,
            latent_mean,
            latent_std,
            ddim_steps,
        )

        center, conf_mean, conf_min = decode_one(
            vae,
            mu,
            args.decoder_temperature,
            sample=False,
        )
        rt_l2, rt_cos = reencode_diagnostic(
            vae,
            diffusion,
            center,
            eps,
            latent_mean,
            latent_std,
            ddim_steps,
            args.epsilon_geometry,
            args.resolved_sphere_radius,
        )

        peps.append(center)
        rows.append({
            "bo_iter": bo_iter,
            "epsilon_candidate_index": j,
            "decode_type": "argmax",
            "peptide": center,
            "decoder_confidence_mean": conf_mean,
            "decoder_confidence_min": conf_min,
            "peptide_reencode_epsilon_l2": rt_l2,
            "peptide_reencode_epsilon_cosine": rt_cos,
            "epsilon_norm": float(
                eps.norm().cpu()
            ),
            "local_noise_l2": 0.0,
            "edit_distance_to_center": 0,
        })

        # Stochastic samples from the same candidate.
        for _ in range(
            max(
                0,
                args.decoder_samples_per_epsilon - 1,
            )
        ):
            p, cm, cmin = decode_one(
                vae,
                mu,
                args.decoder_temperature,
                sample=True,
            )
            l2, cos = reencode_diagnostic(
                vae,
                diffusion,
                p,
                eps,
                latent_mean,
                latent_std,
                ddim_steps,
                args.epsilon_geometry,
                args.resolved_sphere_radius,
            )
            peps.append(p)
            rows.append({
                "bo_iter": bo_iter,
                "epsilon_candidate_index": j,
                "decode_type": "sample",
                "peptide": p,
                "decoder_confidence_mean": cm,
                "decoder_confidence_min": cmin,
                "peptide_reencode_epsilon_l2": l2,
                "peptide_reencode_epsilon_cosine": cos,
                "epsilon_norm": float(
                    eps.norm().cpu()
                ),
                "local_noise_l2": 0.0,
                "edit_distance_to_center":
                    edit_distance(center, p),
            })

        # Native epsilon-space local perturbations.
        for sigma in args.novelty_sigmas:
            for _ in range(
                args.local_neighbors_per_sigma
            ):
                en = (
                    eps
                    + torch.randn_like(eps)
                    * float(sigma)
                )
                en = apply_epsilon_geometry(
                    en,
                    args.epsilon_geometry,
                    args.resolved_sphere_radius,
                )
                en = en.clamp(
                    -args.resolved_epsilon_bound,
                    args.resolved_epsilon_bound,
                )

                mun = epsilon_to_decoder_mu(
                    diffusion,
                    en,
                    latent_mean,
                    latent_std,
                    ddim_steps,
                )
                p, cm, cmin = decode_one(
                    vae,
                    mun,
                    args.decoder_temperature,
                    sample=False,
                )
                l2, cos = reencode_diagnostic(
                    vae,
                    diffusion,
                    p,
                    en,
                    latent_mean,
                    latent_std,
                    ddim_steps,
                    args.epsilon_geometry,
                    args.resolved_sphere_radius,
                )

                peps.append(p)
                rows.append({
                    "bo_iter": bo_iter,
                    "epsilon_candidate_index": j,
                    "decode_type":
                        f"local_epsilon_sigma_{sigma:g}",
                    "peptide": p,
                    "decoder_confidence_mean": cm,
                    "decoder_confidence_min": cmin,
                    "peptide_reencode_epsilon_l2": l2,
                    "peptide_reencode_epsilon_cosine": cos,
                    "epsilon_norm": float(
                        en.norm().cpu()
                    ),
                    "local_noise_l2": float(
                        torch.linalg.norm(
                            en - eps
                        ).cpu()
                    ),
                    "edit_distance_to_center":
                        edit_distance(center, p),
                })

    d = pd.DataFrame(rows)
    d["is_duplicate_raw_decode"] = (
        d.duplicated(
            "peptide",
            keep="first",
        )
    )
    d.to_csv(
        os.path.join(
            args.decoder_dir,
            f"decoder_diagnostics_bo_iter_{bo_iter:03d}.csv",
        ),
        index=False,
    )

    return list(dict.fromkeys(peps)), d


# =============================================================================
# GP / qLogEHVI
# =============================================================================

def fit_mo_gp(
    X: torch.Tensor,
    Y: torch.Tensor,
    device,
):
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

        mll = (
            gpytorch.mlls
            .ExactMarginalLogLikelihood(
                gp.likelihood,
                gp,
            )
        )
        fit_gpytorch_mll(mll)
        models.append(gp)

    return ModelListGP(*models)


# =============================================================================
# Black-box / novelty
# =============================================================================

def evaluate_blackbox(
    peptides,
    cache,
    obj_cols,
    blackbox_fc,
):
    uncached = [
        p for p in peptides
        if p not in cache
    ]

    if uncached:
        scored = blackbox_fc(uncached)

        if isinstance(scored, torch.Tensor):
            arr = (
                scored
                .detach()
                .cpu()
                .numpy()
            )
            if arr.shape != (
                len(uncached),
                len(obj_cols),
            ):
                raise ValueError(
                    "blackbox_fc returned "
                    f"{arr.shape}; expected "
                    f"{(len(uncached), len(obj_cols))}"
                )
            for p, row in zip(
                uncached,
                arr,
            ):
                cache[p] = [
                    float(v)
                    for v in row
                ]

        elif isinstance(scored, pd.DataFrame):
            for _, row in scored.iterrows():
                p = clean_peptide(
                    row.get(
                        "peptide_len10",
                        row.get("peptide", ""),
                    )
                )
                if p is not None:
                    cache[p] = [
                        float(row[c])
                        for c in obj_cols
                    ]
        else:
            raise TypeError(
                "Unsupported blackbox return type: "
                f"{type(scored)}"
            )

    missing = [
        p for p in peptides
        if p not in cache
    ]
    if missing:
        raise RuntimeError(
            f"No black-box scores returned for {missing[:5]}"
        )

    return torch.tensor(
        [cache[p] for p in peptides],
        dtype=torch.float32,
    )


def filter_novel(
    peps,
    train_set,
    heldout_set,
    evaluated_set,
    generated_set,
    args,
):
    accepted = []
    rejected = []

    for p in peps:
        reason = None

        if clean_peptide(p) is None:
            reason = "invalid"
        elif (
            args.reject_training_peptides
            and p in train_set
        ):
            reason = "in_training"
        elif (
            args.reject_heldout_peptides
            and p in heldout_set
        ):
            reason = "in_validation_or_test_holdout"
        elif (
            args.reject_seen_peptides
            and p in evaluated_set
        ):
            reason = "already_evaluated"
        elif (
            args.reject_seen_peptides
            and p in generated_set
        ):
            reason = "already_generated"

        if reason is None:
            accepted.append(p)
        else:
            rejected.append((p, reason))

    return accepted, rejected


def initial_indices(Y, n_init, seed):
    pareto = (
        torch.where(
            pareto_mask_max(Y)
        )[0]
        .cpu()
        .tolist()
    )

    pset = set(pareto)
    others = [
        i for i in range(len(Y))
        if i not in pset
    ]

    rng = np.random.default_rng(seed)
    need = max(
        0,
        int(n_init) - len(pareto),
    )

    extra = (
        rng.choice(
            others,
            size=min(need, len(others)),
            replace=False,
        ).tolist()
        if need
        else []
    )

    return list(
        dict.fromkeys(
            pareto + extra
        )
    )


def save_pareto(
    peptides,
    labeled_idx,
    Y,
    bo_iter,
    out_dir,
    obj_cols,
):
    mask = pareto_mask_max(
        Y.detach().cpu()
    )

    rows = []
    for k in torch.where(mask)[0].tolist():
        g = int(labeled_idx[k])
        row = {
            "bo_iter": bo_iter,
            "global_index": g,
            "peptide": peptides[g],
        }
        for j, c in enumerate(obj_cols):
            row[c] = float(
                Y[k, j]
                .detach()
                .cpu()
            )
        rows.append(row)

    d = pd.DataFrame(rows)
    d.to_csv(
        os.path.join(
            out_dir,
            f"pareto_front_bo_iter_{bo_iter:03d}.csv",
        ),
        index=False,
    )
    return d


# =============================================================================
# Main
# =============================================================================

def run(args):
    set_seed(args.seed)
    device = torch.device(args.device)

    for p in [
        args.out_dir,
        args.decoder_dir,
        args.pareto_dir,
    ]:
        ensure_dir(p)

    (
        module,
        vae,
        diffusion,
        ckpt,
        latent_mean,
        latent_std,
        ddim_steps,
        projected_to_sphere,
    ) = load_model(
        args.diffusion_checkpoint,
        args.finetune_script,
        device,
        require_preferred_checkpoint=(
            not args.allow_nonpreferred_checkpoint
        ),
    )

    checkpoint_obj_cols = list(
        ckpt.get("objective_cols", [])
    )
    if (
        checkpoint_obj_cols
        and checkpoint_obj_cols
        != list(args.obj_cols)
    ):
        raise ValueError(
            "BO objective columns do not match checkpoint: "
            f"checkpoint={checkpoint_obj_cols}, "
            f"requested={list(args.obj_cols)}"
        )

    checkpoint_geometry = "sphere" if projected_to_sphere else "native"
    if checkpoint_geometry != args.epsilon_geometry:
        print(
            "WARNING: checkpoint epsilon geometry and BO epsilon geometry differ: "
            f"checkpoint={checkpoint_geometry}, BO={args.epsilon_geometry}. "
            "This is a valid post-hoc geometry ablation, but not a fully matched "
            "fine-tuning/BO experiment."
        )
        if args.require_checkpoint_geometry_match:
            raise RuntimeError(
                "Checkpoint/BO epsilon geometry mismatch. Use matching geometry, "
                "or disable --require-checkpoint-geometry-match for an ablation."
            )

    blackbox_fc = load_blackbox_function(
        args.blackbox_script,
        args.finetune_script,
        args.project_root,
    )

    # Reuse the fine-tuner's exact leakage-safe Cu mapping.
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

    # BO uses TRAIN only.
    peptides = [
        peptides_all[i]
        for i in train_idx.tolist()
    ]
    Xseq = x_all_cpu[train_idx].to(device)
    Y_full = y_all_cpu[train_idx].to(device)

    heldout_set = {
        peptides_all[i]
        for i in torch.cat(
            [val_idx, test_idx]
        ).tolist()
    }

    print(
        f"Leakage-safe Cu rows: total={len(peptides_all)} "
        f"BO-train={len(train_idx)} "
        f"val-heldout={len(val_idx)} "
        f"test-heldout={len(test_idx)}"
    )

    sphere_radius = (
        float(args.sphere_radius)
        if args.sphere_radius is not None
        else math.sqrt(int(vae.cfg.latent_dim))
    )
    args.resolved_sphere_radius = sphere_radius

    E_all = encode_batch_to_epsilon(
        vae,
        diffusion,
        Xseq,
        latent_mean,
        latent_std,
        ddim_steps,
        geometry=args.epsilon_geometry,
        sphere_radius=sphere_radius,
        batch_size=args.encode_batch_size,
    ).to(device)

    eps_abs_max = float(
        E_all.abs().max().cpu()
    )
    eps_norm_mean = float(
        E_all.norm(dim=-1).mean().cpu()
    )
    eps_norm_std = float(
        E_all.norm(dim=-1).std().cpu()
    )

    if args.epsilon_geometry == "sphere":
        epsilon_bound = sphere_radius
    else:
        epsilon_bound = float(args.epsilon_bound)
        if (
            args.auto_expand_epsilon_bound
            and eps_abs_max >= epsilon_bound
        ):
            epsilon_bound = 1.05 * eps_abs_max

    args.resolved_epsilon_bound = epsilon_bound

    X_all_unclamped = epsilon_to_unit(
        E_all,
        epsilon_bound,
        clamp=False,
    )

    if bool(
        (
            (X_all_unclamped < 0.0)
            | (X_all_unclamped > 1.0)
        ).any().item()
    ):
        raise RuntimeError(
            "Resolved epsilon bound does not contain all Cu TRAIN coordinates. "
            "Increase --epsilon-bound or keep --auto-expand-epsilon-bound enabled."
        )

    X_all = X_all_unclamped

    print(
        f"epsilon norm mean={eps_norm_mean:.6f}; "
        f"std={eps_norm_std:.6f}; "
        f"max |coordinate|={eps_abs_max:.6f}"
    )
    if args.epsilon_geometry == "sphere":
        print(
            f"Sphere epsilon geometry enabled: ||epsilon||_2 = {sphere_radius:.6f}"
        )
    else:
        print("Native epsilon geometry enabled: no radius projection.")
    print(
        f"Fixed epsilon BO box: "
        f"[-{epsilon_bound:.4f}, +{epsilon_bound:.4f}]"
    )
    print(
        f"GP cube min={float(X_all.min().cpu()):.6f}, "
        f"max={float(X_all.max().cpu()):.6f}"
    )

    # Training-only Pareto.
    tr_mask = pareto_mask_max(Y_full)
    tr_idx_local = torch.where(tr_mask)[0]
    Y_train_pareto = Y_full[tr_mask]

    train_pareto = pd.DataFrame({
        "global_index": tr_idx_local.cpu().numpy(),
        "peptide": [
            peptides[i]
            for i in tr_idx_local.cpu().tolist()
        ],
        **{
            c: (
                Y_train_pareto[:, j]
                .detach()
                .cpu()
                .numpy()
            )
            for j, c in enumerate(args.obj_cols)
        },
    })

    train_pareto_path = os.path.join(
        args.out_dir,
        "train_only_pareto_CU_gru_vae_diffusion_epsilon.csv",
    )
    train_pareto.to_csv(
        train_pareto_path,
        index=False,
    )

    print(
        f"Training Pareto: {len(train_pareto)} solutions"
    )

    cache = {
        p: [float(v) for v in vals]
        for p, vals in zip(
            peptides,
            Y_full
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
            .tolist(),
        )
    }

    train_set = set(peptides)
    evaluated_set = set()
    generated_set = set()

    labeled_idx = initial_indices(
        Y_full,
        args.initial_labeled,
        args.seed + 42,
    )

    idx = torch.tensor(
        labeled_idx,
        device=device,
    )

    Y_lab = Y_full[idx]
    E_lab = E_all[idx]
    X_lab = X_all[idx]

    evaluated_set.update(
        peptides[i]
        for i in labeled_idx
    )

    # FIXED HV reference from initial design.
    ref = (
        Y_lab.min(dim=0).values
        - float(args.ref_margin)
    )
    initial_hv = float(
        Hypervolume(
            ref_point=ref
        ).compute(Y_lab)
    )
    best_hv = initial_hv

    print(
        f"Initial labeled: {len(labeled_idx)} "
        f"(includes all {int(tr_mask.sum())} "
        "train-only Pareto observations)"
    )
    print(
        f"Fixed HV ref point: "
        f"{ref.detach().cpu().tolist()}"
    )
    print(
        f"Initial HV: {initial_hv:.6f}"
    )

    run_config = vars(args).copy()
    run_config.update({
        "checkpoint_epoch":
            ckpt.get("epoch"),
        "checkpoint_selection":
            ckpt.get("checkpoint_selection"),
        "checkpoint_metrics":
            ckpt.get("metrics", {}),
        "bo_space": (
            "diffusion_epsilon_sphere"
            if args.epsilon_geometry == "sphere"
            else "native_non_spherical_diffusion_epsilon"
        ),
        "epsilon_geometry": args.epsilon_geometry,
        "sphere_projection_used": bool(args.epsilon_geometry == "sphere"),
        "sphere_radius": float(sphere_radius),
        "checkpoint_epsilon_geometry": checkpoint_geometry,
        "geometry_matches_checkpoint": bool(
            checkpoint_geometry == args.epsilon_geometry
        ),
        "historical_observations_split":
            "train_only",
        "validation_test_used_for_gp_fit":
            False,
        "validation_test_used_for_blackbox_cache":
            False,
        "heldout_exact_peptide_rejection":
            bool(args.reject_heldout_peptides),
        "pretraining_split_manifest":
            args.pretraining_split_manifest,
        "data_audit":
            data_audit,
        "resolved_epsilon_bound":
            epsilon_bound,
        "epsilon_norm_mean_train":
            eps_norm_mean,
        "epsilon_norm_std_train":
            eps_norm_std,
        "initial_hypervolume":
            initial_hv,
        "fixed_hv_ref":
            ref.detach().cpu().tolist(),
        "warm_start_includes_full_training_pareto":
            True,
    })

    Path(
        args.out_dir,
        "bo_run_config.json",
    ).write_text(
        json.dumps(
            run_config,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    hv_rows = []
    all_rows = []

    # -------------------------------------------------------------------------
    # BO loop
    # -------------------------------------------------------------------------
    for bo_iter in range(args.bo_iters):
        print(
            f"\n=== BO iter "
            f"{bo_iter + 1}/{args.bo_iters} ==="
        )

        Xgp = X_lab.double()
        Ygp = Y_lab.double()

        model = fit_mo_gp(
            Xgp,
            Ygp,
            device,
        )

        # Balanced current Pareto point as trust-region center.
        nd = torch.where(
            pareto_mask_max(Y_lab)
        )[0]
        Ynd = Y_lab[nd]
        lo = Ynd.min(0).values
        span = (
            Ynd.max(0).values - lo
        ).clamp_min(1e-12)
        balanced_score = (
            (Ynd - lo) / span
        ).mean(-1)

        center_i = int(
            nd[
                torch.argmax(
                    balanced_score
                )
            ].item()
        )
        xc = X_lab[
            center_i
        ].double()

        # Convert raw epsilon trust-region radius to unit-cube scale.
        r_unit = (
            float(args.tr_radius_epsilon)
            / (2.0 * epsilon_bound)
        )

        bounds = torch.stack([
            (xc - r_unit).clamp(
                0.0,
                1.0,
            ),
            (xc + r_unit).clamp(
                0.0,
                1.0,
            ),
        ])

        Ynd_gp = Ygp[
            pareto_mask_max(Ygp)
        ]

        if (
            len(Ynd_gp)
            > args.max_partition_points
        ):
            yn = (
                (
                    Ynd_gp
                    - Ynd_gp.min(0).values
                )
                / (
                    Ynd_gp.max(0).values
                    - Ynd_gp.min(0).values
                ).clamp_min(1e-12)
            )
            keep = torch.topk(
                yn.mean(-1),
                args.max_partition_points,
            ).indices
            Ypart = Ynd_gp[keep]
        else:
            Ypart = Ynd_gp

        ref_d = ref.double()

        partitioning = NondominatedPartitioning(
            ref_point=ref_d,
            Y=Ypart,
        )

        sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([
                args.mc_samples
            ])
        ).to(device)

        acq_cls = (
            qLogExpectedHypervolumeImprovement
            if (
                args.acqf == "qlogehvi"
                and qLogExpectedHypervolumeImprovement
                is not None
            )
            else qExpectedHypervolumeImprovement
        )

        base_acq = acq_cls(
            model=model,
            ref_point=(
                ref_d.detach().cpu().tolist()
            ),
            partitioning=partitioning,
            sampler=sampler,
        ).to(device)

        if args.epsilon_geometry == "sphere":
            acq = SphereProjectedAcquisition(
                base_acq=base_acq,
                epsilon_bound=epsilon_bound,
                sphere_radius=sphere_radius,
            ).to(device)
        else:
            acq = base_acq

        parts = []
        avals = []

        repeats = (
            args.q_batch
            if args.optimize_q_one_at_a_time
            else 1
        )
        q_inner = (
            1
            if args.optimize_q_one_at_a_time
            else args.q_batch
        )

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
                    num_restarts=max(
                        1,
                        args.num_restarts // 2,
                    ),
                    raw_samples=max(
                        16,
                        args.raw_samples // 4,
                    ),
                    sequential=True,
                )

            parts.append(xnew.detach())
            avals.append(
                aval.detach().reshape(-1)[0]
            )

        Xcand = (
            torch.cat(parts)
            .clamp(0.0, 1.0)
        )
        Ecand = unit_to_epsilon(
            Xcand,
            epsilon_bound,
        ).float()
        Ecand = apply_epsilon_geometry(
            Ecand,
            args.epsilon_geometry,
            sphere_radius,
        )
        Xcand = epsilon_to_unit(
            Ecand,
            epsilon_bound,
            clamp=True,
        )

        acq_value = float(
            torch.stack(avals)
            .mean()
            .cpu()
        )

        print(
            "Proposed epsilon mean norm="
            f"{float(Ecand.norm(dim=-1).mean().cpu()):.6f}; "
            "max |coord|="
            f"{float(Ecand.abs().max().cpu()):.6f}"
        )

        raw_peps, diag = decode_candidates(
            vae,
            diffusion,
            Ecand,
            bo_iter,
            args,
            latent_mean,
            latent_std,
            ddim_steps,
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
            accepted = accepted[
                :args.max_blackbox_per_iter
            ]

        # Higher-temperature fallback if needed.
        if not accepted:
            print(
                "[BO] No novel peptide survived; "
                "trying higher decoder temperature."
            )
            old = args.decoder_temperature
            args.decoder_temperature = max(
                old,
                args.fallback_temperature,
            )

            extra, diag2 = decode_candidates(
                vae,
                diffusion,
                Ecand,
                bo_iter + 100000,
                args,
                latent_mean,
                latent_std,
                ddim_steps,
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
                accepted = accepted[
                    :args.max_blackbox_per_iter
                ]

            rejected.extend(rejected2)
            diag = pd.concat(
                [diag, diag2],
                ignore_index=True,
            )

        rejected_map = dict(rejected)
        status = diag.copy()

        if not accepted:
            generated_set.update(raw_peps)

            status["accepted_for_blackbox"] = False
            status["rejected_reason"] = (
                status["peptide"].map(
                    lambda p:
                        rejected_map.get(
                            p,
                            "not_selected",
                        )
                )
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
                "pareto_size": int(
                    pareto_mask_max(
                        Y_lab
                    ).sum()
                ),
                "hypervolume": best_hv,
                "best_hypervolume": best_hv,
                "improved": 0,
                "acq_value": acq_value,
                "n_dominates_train": 0,
            })

            pd.DataFrame(
                hv_rows
            ).to_csv(
                os.path.join(
                    args.out_dir,
                    "bo_hypervolume_history.csv",
                ),
                index=False,
            )
            continue

        # True objective evaluation.
        Ynew = evaluate_blackbox(
            accepted,
            cache,
            args.obj_cols,
            blackbox_fc,
        ).to(device)

        # IMPORTANT: re-encode each accepted discrete peptide to its OWN epsilon.
        Enew = torch.cat([
            peptide_to_epsilon(
                vae,
                diffusion,
                p,
                latent_mean,
                latent_std,
                ddim_steps,
                args.epsilon_geometry,
                sphere_radius,
            )
            for p in accepted
        ], dim=0)

        Xnew_unclamped = epsilon_to_unit(
            Enew,
            epsilon_bound,
            clamp=False,
        )

        # A generated peptide may re-encode outside the original fixed BO box.
        # Do not silently clip it because that would assign its objective to the
        # wrong GP coordinate.
        outside = (
            (Xnew_unclamped < 0.0)
            | (Xnew_unclamped > 1.0)
        ).any(dim=-1)

        if bool(outside.any().item()):
            bad = [
                accepted[i]
                for i in torch.where(outside)[0].tolist()
            ]
            raise RuntimeError(
                "A newly evaluated peptide re-encoded outside the fixed epsilon "
                f"BO box: examples={bad[:5]}. Rerun with a larger --epsilon-bound. "
                "The script refuses to clip re-encoded coordinates because that "
                "would break peptide/objective/coordinate consistency."
            )

        Xnew = Xnew_unclamped

        iter_rows = []
        for i, (p, y) in enumerate(
            zip(accepted, Ynew)
        ):
            n_train_dom = sum(
                dominates(
                    y,
                    y0,
                )
                for y0 in Y_train_pareto
            )

            row = {
                "bo_iter": bo_iter,
                "peptide": p,
                "acq_value": acq_value,
                "epsilon_geometry": args.epsilon_geometry,
                "epsilon_norm": float(
                    Enew[i].norm().cpu()
                ),
                "sphere_radius": (
                    float(sphere_radius)
                    if args.epsilon_geometry == "sphere"
                    else float("nan")
                ),
                "epsilon_max_abs_coordinate": float(
                    Enew[i].abs().max().cpu()
                ),
                "dominated_by_train_pareto": int(
                    any(
                        dominates(y0, y)
                        for y0 in Y_train_pareto
                    )
                ),
                "dominates_any_train_pareto":
                    int(n_train_dom > 0),
                "n_train_pareto_members_dominated":
                    int(n_train_dom),
            }

            for j, c in enumerate(
                args.obj_cols
            ):
                row[c] = float(y[j].cpu())

            iter_rows.append(row)
            all_rows.append(row)

        pd.DataFrame(
            iter_rows
        ).to_csv(
            os.path.join(
                args.out_dir,
                f"accepted_candidates_scored_bo_iter_{bo_iter:03d}.csv",
            ),
            index=False,
        )

        status["accepted_for_blackbox"] = (
            status["peptide"].isin(
                accepted
            )
        )
        status["rejected_reason"] = (
            status["peptide"].map(
                lambda p:
                    ""
                    if p in accepted
                    else rejected_map.get(
                        p,
                        "not_selected",
                    )
            )
        )

        for j, c in enumerate(
            args.obj_cols
        ):
            mp = {
                p: float(y[j].cpu())
                for p, y in zip(
                    accepted,
                    Ynew,
                )
            }
            status[c] = (
                status["peptide"].map(
                    lambda p:
                        mp.get(
                            p,
                            np.nan,
                        )
                )
            )

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
            new_global.append(
                len(peptides) - 1
            )

        labeled_idx.extend(new_global)
        E_lab = torch.cat(
            [E_lab, Enew],
            dim=0,
        )
        X_lab = torch.cat(
            [X_lab, Xnew],
            dim=0,
        )
        Y_lab = torch.cat(
            [Y_lab, Ynew],
            dim=0,
        )

        evaluated_set.update(accepted)
        generated_set.update(raw_peps)

        # FIXED reference point.
        hv = float(
            Hypervolume(
                ref_point=ref
            ).compute(Y_lab)
        )

        improved = (
            hv
            > best_hv
            + args.hv_tol
        )
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
                r["dominates_any_train_pareto"]
                for r in iter_rows
            )),
            "max_train_pareto_members_dominated":
                int(max(
                    r[
                        "n_train_pareto_members_dominated"
                    ]
                    for r in iter_rows
                )),
        }
        hv_rows.append(hv_row)

        pd.DataFrame(
            hv_rows
        ).to_csv(
            os.path.join(
                args.out_dir,
                "bo_hypervolume_history.csv",
            ),
            index=False,
        )

        pd.DataFrame(
            all_rows
        ).to_csv(
            os.path.join(
                args.out_dir,
                "all_accepted_candidates_scored.csv",
            ),
            index=False,
        )

        print(
            f"Accepted {len(accepted)} peptides; "
            f"HV={hv:.6f}; "
            f"best={best_hv:.6f}; "
            f"dominates_train="
            f"{hv_row['n_dominates_train']}; "
            f"max_train_members_dominated="
            f"{hv_row['max_train_pareto_members_dominated']}"
        )

    # -------------------------------------------------------------------------
    # Final Pareto + exact domination counts
    # -------------------------------------------------------------------------
    Ycpu = Y_lab.detach().cpu()
    mask = pareto_mask_max(Ycpu)

    train_pareto_cpu = (
        Y_train_pareto
        .detach()
        .cpu()
    )

    final_rows = []
    for k in torch.where(mask)[0].tolist():
        g = labeled_idx[k]
        p = peptides[g]
        y = Ycpu[k]

        n_dom = sum(
            dominates(y, y0)
            for y0 in train_pareto_cpu
        )

        row = {
            "global_index": g,
            "peptide": p,
            "is_novel":
                p not in train_set,
            "dominates_train_pareto":
                bool(n_dom > 0),
            "n_train_pareto_members_dominated":
                int(n_dom),
        }

        for j, c in enumerate(
            args.obj_cols
        ):
            row[c] = float(y[j])

        final_rows.append(row)

    final_df = pd.DataFrame(final_rows)

    final_path = os.path.join(
        args.out_dir,
        "bo_final_pareto_CU_gru_vae_diffusion_epsilon_qlogehvi_leakage_safe.csv",
    )
    final_df.to_csv(
        final_path,
        index=False,
    )

    dom_df = (
        final_df[
            (final_df["is_novel"] == True)
            & (
                final_df[
                    "dominates_train_pareto"
                ] == True
            )
        ].copy()
        if len(final_df)
        else pd.DataFrame()
    )

    if len(dom_df):
        dom_df = dom_df.sort_values(
            "n_train_pareto_members_dominated",
            ascending=False,
        )

    dom_path = os.path.join(
        args.out_dir,
        "bo_pareto_dominates_train_CU_gru_vae_diffusion_epsilon_qlogehvi_leakage_safe.csv",
    )
    dom_df.to_csv(
        dom_path,
        index=False,
    )

    print("\nDone.")
    print(
        f"Final BO Pareto: "
        f"{len(final_df)} -> {final_path}"
    )
    print(
        "Novel Pareto solutions dominating training Pareto: "
        f"{len(dom_df)} -> {dom_path}"
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Leakage-safe multi-objective GP/qLogEHVI BO with selectable "
            "native or fixed-radius spherical latent-diffusion epsilon geometry."
        )
    )

    p.add_argument(
        "--diffusion-checkpoint",
        required=True,
        help=(
            "Recommended for the attached history: "
            "best_val_objective_mse_h64_z64_cu_latent_diffusion.pt "
            "(epoch 148 fallback because no epoch passed the hard BO gate)."
        ),
    )
    p.add_argument(
        "--finetune-script",
        required=True,
        help=(
            "Path to "
            "finetune_best_bo_ready_gru_vae_cu_latent_diffusion_"
            "leakage_safe_v3_objective_gated.py"
        ),
    )
    p.add_argument(
        "--data-csv",
        required=True,
    )
    p.add_argument(
        "--pretraining-split-manifest",
        required=True,
        help=(
            "Same leakage-safe manifest used in latent-diffusion fine-tuning."
        ),
    )
    p.add_argument(
        "--peptide-col",
        default="peptide_len10",
    )
    p.add_argument(
        "--obj-cols",
        nargs=4,
        default=OBJ_COLS,
    )

    p.add_argument(
        "--duplicate-policy",
        choices=[
            "mean",
            "first",
            "error",
        ],
        default="mean",
    )
    p.add_argument(
        "--unmapped-policy",
        choices=[
            "drop",
            "error",
        ],
        default="drop",
    )

    p.add_argument(
        "--blackbox-script",
        default="black_box_fcn_mo_CU_f.py",
    )
    p.add_argument(
        "--project-root",
        action="append",
        default=[],
    )

    p.add_argument(
        "--allow-nonpreferred-checkpoint",
        action="store_true",
        help=(
            "Deliberately permit a checkpoint not selected by the objective-gated "
            "BO-ready criterion or its minimum-objective-MSE fallback."
        ),
    )
    p.add_argument(
        "--epsilon-geometry",
        choices=["native", "sphere"],
        default="native",
        help=(
            "Choose BO geometry: native DDIM epsilon or fixed-radius spherical "
            "epsilon."
        ),
    )
    p.add_argument(
        "--sphere-radius",
        type=float,
        default=None,
        help=(
            "Radius for --epsilon-geometry sphere. Default sqrt(latent_dim), "
            "which is 8 for Z64."
        ),
    )
    p.add_argument(
        "--require-checkpoint-geometry-match",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If True, reject BO geometry that differs from the geometry recorded "
            "in the fine-tuning checkpoint. Default False permits post-hoc ablations."
        ),
    )

    p.add_argument(
        "--out-dir",
        default=(
            "bo_results_CU_gru_vae_diffusion_epsilon_"
            "qlogehvi_leakage_safe"
        ),
    )
    p.add_argument(
        "--decoder-dir",
        default=(
            "bo_decoder_monitoring_CU_gru_vae_diffusion_epsilon_"
            "qlogehvi_leakage_safe"
        ),
    )
    p.add_argument(
        "--pareto-dir",
        default=(
            "pareto_front_CU_gru_vae_diffusion_epsilon_"
            "qlogehvi_leakage_safe"
        ),
    )

    p.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    p.add_argument(
        "--encode-batch-size",
        type=int,
        default=256,
    )

    # BO.
    p.add_argument(
        "--bo-iters",
        type=int,
        default=20,
    )
    p.add_argument(
        "--initial-labeled",
        type=int,
        default=256,
    )
    p.add_argument(
        "--q-batch",
        type=int,
        default=2,
    )
    p.add_argument(
        "--optimize-q-one-at-a-time",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--num-restarts",
        type=int,
        default=4,
    )
    p.add_argument(
        "--raw-samples",
        type=int,
        default=64,
    )
    p.add_argument(
        "--mc-samples",
        type=int,
        default=32,
    )
    p.add_argument(
        "--acqf",
        choices=[
            "qehvi",
            "qlogehvi",
        ],
        default="qlogehvi",
    )

    # Epsilon geometry / search box.
    p.add_argument(
        "--epsilon-bound",
        type=float,
        default=4.0,
        help=(
            "Initial symmetric bound for native epsilon. In sphere mode the "
            "box is automatically [-R,+R]."
        ),
    )
    p.add_argument(
        "--auto-expand-epsilon-bound",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--tr-radius-epsilon",
        type=float,
        default=0.75,
        help=(
            "Trust-region half-width in raw epsilon coordinate units."
        ),
    )

    p.add_argument(
        "--max-partition-points",
        type=int,
        default=20,
    )
    p.add_argument(
        "--ref-margin",
        type=float,
        default=0.10,
    )
    p.add_argument(
        "--hv-tol",
        type=float,
        default=1e-6,
    )

    # Decoder / novelty.
    p.add_argument(
        "--decoder-samples-per-epsilon",
        type=int,
        default=16,
    )
    p.add_argument(
        "--decoder-temperature",
        type=float,
        default=0.90,
    )
    p.add_argument(
        "--fallback-temperature",
        type=float,
        default=1.25,
    )
    p.add_argument(
        "--novelty-sigmas",
        nargs="*",
        type=float,
        default=[
            0.05,
            0.10,
            0.20,
        ],
        help=(
            "Gaussian perturbation std in epsilon coordinates. In sphere mode, "
            "the perturbed point is projected back to radius R."
        ),
    )
    p.add_argument(
        "--local-neighbors-per-sigma",
        type=int,
        default=4,
    )
    p.add_argument(
        "--max-blackbox-per-iter",
        type=int,
        default=8,
        help=(
            "Maximum novel candidates sent to expensive black-box evaluation "
            "per BO iteration; <=0 disables the cap."
        ),
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
    )
    p.add_argument(
        "--reject-seen-peptides",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
