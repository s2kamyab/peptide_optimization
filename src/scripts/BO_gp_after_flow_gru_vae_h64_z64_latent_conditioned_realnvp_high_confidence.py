
from __future__ import annotations

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
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.utils.multi_objective.hypervolume import Hypervolume

try:
    from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
except Exception:
    qLogExpectedHypervolumeImprovement = None

from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement

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
OBJ_COLS = ["chelation_sub", "solubility_sub", "stability_sub", "expression_sub"]


@dataclass
class ModelConfig:
    hidden_size: int = 64
    latent_dim: int = 64
    n_layers: int = 2
    dropout: float = 0.0
    decoder_conditioning: str = "concat_z_at_every_decoder_step"


@dataclass
class RealNVPConfig:
    n_layers: int = 4
    hidden_dim: int = 128
    max_scale: float = 1.5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def clean_peptide(x: object) -> Optional[str]:
    pep = str(x).strip().upper()
    if len(pep) != SEQ_LEN:
        return None
    if any(ch not in AA_TO_I for ch in pep):
        return None
    return pep


def onehot_encode_peptides(peptides: Sequence[str]) -> torch.Tensor:
    x = torch.zeros((len(peptides), SEQ_LEN, VOCAB), dtype=torch.float32)
    for n, pep in enumerate(peptides):
        pep2 = clean_peptide(pep)
        if pep2 is None:
            raise ValueError(f"Invalid peptide: {pep!r}")
        for t, aa in enumerate(pep2):
            x[n, t, AA_TO_I[aa]] = 1.0
    return x


def levenshtein_edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + int(ca != cb)))
        prev = cur
    return int(prev[-1])


class GRUEncoder(nn.Module):
    def __init__(self, hidden_size: int, latent_dim: int, n_layers: int, dropout: float):
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

    def forward(self, x_onehot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.in_proj(x_onehot)
        out_top, h_n = self.gru(h)
        h_last = h_n[-1]
        mu = self.to_mu(h_last)
        logvar = self.to_logvar(h_last).clamp(min=-8.0, max=8.0)
        return mu, logvar, out_top


class LatentConditionedGRUDecoder(nn.Module):
    def __init__(self, hidden_size: int, latent_dim: int, n_layers: int, dropout: float):
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

    def forward(self, z: torch.Tensor, x_onehot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = z.size(0)
        x_shift = torch.zeros_like(x_onehot)
        x_shift[:, 1:, :] = x_onehot[:, :-1, :]
        emb = self.token_embed(x_shift)
        z_repeat = z.unsqueeze(1).expand(-1, x_onehot.size(1), -1)
        dec_input = torch.cat([emb, z_repeat], dim=-1)
        h0 = self.z_to_h(z).view(self.gru.num_layers, batch_size, self.gru.hidden_size)
        out_top, _ = self.gru(dec_input, h0)
        logits = self.to_logits(out_top)
        return logits, out_top

    def initial_hidden(self, z: torch.Tensor) -> torch.Tensor:
        return self.z_to_h(z).view(self.gru.num_layers, z.size(0), self.gru.hidden_size)

    def step(self, z: torch.Tensor, current_token: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.token_embed(current_token)
        z_step = z.unsqueeze(1)
        dec_input = torch.cat([emb, z_step], dim=-1)
        out_step, h = self.gru(dec_input, h)
        logits = self.to_logits(out_step[:, -1, :])
        return logits, h


class AffineCouplingBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, mask: torch.Tensor, max_scale: float = 1.5):
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.max_scale = float(max_scale)
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
    def __init__(self, dim: int, n_layers: int, hidden_dim: int, max_scale: float):
        super().__init__()
        masks = []
        for layer in range(int(n_layers)):
            mask = torch.zeros(dim)
            mask[layer % 2 :: 2] = 1.0
            masks.append(mask)
        self.blocks = nn.ModuleList([AffineCouplingBlock(dim, hidden_dim, mask, max_scale) for mask in masks])

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


class FlowGRUVAE(nn.Module):
    def __init__(self, cfg: ModelConfig, flow_cfg: RealNVPConfig):
        super().__init__()
        self.cfg = cfg
        self.flow_cfg = flow_cfg
        self.enc = GRUEncoder(cfg.hidden_size, cfg.latent_dim, cfg.n_layers, cfg.dropout)
        self.dec = LatentConditionedGRUDecoder(cfg.hidden_size, cfg.latent_dim, cfg.n_layers, cfg.dropout)
        self.flow = RealNVPFlow(cfg.latent_dim, flow_cfg.n_layers, flow_cfg.hidden_dim, flow_cfg.max_scale)
        self.score_head = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_size),
            nn.SiLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.SiLU(),
            nn.Linear(cfg.hidden_size, 1),
        )


def pareto_mask_maximize(Y: torch.Tensor) -> torch.Tensor:
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


def load_model_from_checkpoint(path: str, device: torch.device) -> Tuple[FlowGRUVAE, Dict]:
    ckpt = torch.load(path, map_location=device)
    cfg_dict = ckpt.get("model_config", {})
    flow_dict = ckpt.get("flow_config", {})
    cfg = ModelConfig(
        hidden_size=int(cfg_dict.get("hidden_size", 64)),
        latent_dim=int(cfg_dict.get("latent_dim", 64)),
        n_layers=int(cfg_dict.get("n_layers", 2)),
        dropout=float(cfg_dict.get("dropout", 0.0)),
        decoder_conditioning=str(cfg_dict.get("decoder_conditioning", "concat_z_at_every_decoder_step")),
    )
    flow_cfg = RealNVPConfig(
        n_layers=int(flow_dict.get("n_layers", 4)),
        hidden_dim=int(flow_dict.get("hidden_dim", 128)),
        max_scale=float(flow_dict.get("max_scale", 1.5)),
    )
    model = FlowGRUVAE(cfg, flow_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    print(f"Loaded checkpoint: {path}")
    print("checkpoint_type:", ckpt.get("checkpoint_type", "unknown"))
    print("selection:", ckpt.get("checkpoint_selection", "unknown"))
    print("epoch:", ckpt.get("epoch", "unknown"))
    print("model_config:", cfg)
    print("flow_config:", flow_cfg)
    metrics = ckpt.get("metrics", {})
    if metrics:
        print("metrics:", metrics)
    return model, ckpt


@torch.no_grad()
def encode_to_zK(model: FlowGRUVAE, x_onehot: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    model.eval()
    chunks = []
    for start in range(0, x_onehot.size(0), batch_size):
        xb = x_onehot[start:start + batch_size]
        mu, _, _ = model.enc(xb)
        zK, _ = model.flow(mu)
        chunks.append(zK.detach().cpu())
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def autoregressive_decode_one(
    model: FlowGRUVAE,
    z: torch.Tensor,
    temperature: float = 1.0,
    sample: bool = False,
) -> Tuple[str, float, float, List[float]]:
    if z.ndim != 2 or z.size(0) != 1:
        raise ValueError(f"Expected z [1,d], got {tuple(z.shape)}")
    tau = max(float(temperature), 1e-6)
    h = model.dec.initial_hidden(z)
    current_token = torch.zeros(1, 1, VOCAB, dtype=z.dtype, device=z.device)
    ids, probs_selected = [], []
    for _ in range(SEQ_LEN):
        logits, h = model.dec.step(z, current_token, h)
        probs = torch.softmax(logits / tau, dim=-1)
        if sample:
            idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            idx = probs.argmax(dim=-1)
        p = probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        ids.append(int(idx.item()))
        probs_selected.append(float(p.item()))
        current_token = F.one_hot(idx, num_classes=VOCAB).to(dtype=z.dtype).unsqueeze(1)
    pep = "".join(I_TO_AA[i] for i in ids)
    return pep, float(np.mean(probs_selected)), float(np.min(probs_selected)), probs_selected


@torch.no_grad()
def compute_roundtrip_distance(model: FlowGRUVAE, peptide: str, z_ref: torch.Tensor) -> Tuple[float, float]:
    x = onehot_encode_peptides([peptide]).to(z_ref.device)
    mu, _, _ = model.enc(x)
    z_rt, _ = model.flow(mu)
    a = z_ref.reshape(-1)
    b = z_rt.reshape(-1)
    l2 = torch.linalg.norm(a - b).item()
    cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item()
    return float(l2), float(cos)


@torch.no_grad()
def decode_candidates_with_diagnostics(
    model: FlowGRUVAE,
    Z_cand: torch.Tensor,
    bo_iter: int,
    args: argparse.Namespace,
) -> Tuple[List[str], pd.DataFrame]:
    ensure_dir(args.decoder_dir)
    rows = []
    decoded = []
    for j in range(Z_cand.size(0)):
        z = Z_cand[j:j + 1]
        pep, mean_prob, min_prob, token_probs = autoregressive_decode_one(model, z, args.decoder_temperature, sample=False)
        rt_l2, rt_cos = compute_roundtrip_distance(model, pep, z[0])
        decoded.append(pep)
        rows.append({
            "bo_iter": bo_iter,
            "z_candidate_index": j,
            "decode_rank": 0,
            "decode_type": "argmax",
            "peptide": pep,
            "decoder_confidence_mean": mean_prob,
            "decoder_confidence_min": min_prob,
            "token_probabilities": "|".join(f"{v:.6f}" for v in token_probs),
            "roundtrip_l2": rt_l2,
            "roundtrip_cosine": rt_cos,
            "z_norm": float(z[0].norm().detach().cpu()),
            "local_noise_l2": 0.0,
            "edit_distance_to_center_peptide": 0,
        })

        for r in range(1, args.decoder_samples_per_z):
            pep_s, mean_s, min_s, token_s = autoregressive_decode_one(model, z, args.decoder_temperature, sample=True)
            rt_l2_s, rt_cos_s = compute_roundtrip_distance(model, pep_s, z[0])
            decoded.append(pep_s)
            rows.append({
                "bo_iter": bo_iter,
                "z_candidate_index": j,
                "decode_rank": r,
                "decode_type": "sample",
                "peptide": pep_s,
                "decoder_confidence_mean": mean_s,
                "decoder_confidence_min": min_s,
                "token_probabilities": "|".join(f"{v:.6f}" for v in token_s),
                "roundtrip_l2": rt_l2_s,
                "roundtrip_cosine": rt_cos_s,
                "z_norm": float(z[0].norm().detach().cpu()),
                "local_noise_l2": 0.0,
                "edit_distance_to_center_peptide": levenshtein_edit_distance(pep, pep_s),
            })

        # Local latent perturbations are an explicit novelty/diversity fallback diagnostic.
        for radius in args.novelty_radii:
            for r in range(args.local_neighbors_per_radius):
                noise = torch.randn_like(z) * float(radius)
                z_local = z + noise
                pep_l, mean_l, min_l, token_l = autoregressive_decode_one(model, z_local, args.decoder_temperature, sample=False)
                rt_l2_l, rt_cos_l = compute_roundtrip_distance(model, pep_l, z_local[0])
                decoded.append(pep_l)
                rows.append({
                    "bo_iter": bo_iter,
                    "z_candidate_index": j,
                    "decode_rank": 1000 + int(100 * radius) + r,
                    "decode_type": f"local_noise_radius_{radius:g}",
                    "peptide": pep_l,
                    "decoder_confidence_mean": mean_l,
                    "decoder_confidence_min": min_l,
                    "token_probabilities": "|".join(f"{v:.6f}" for v in token_l),
                    "roundtrip_l2": rt_l2_l,
                    "roundtrip_cosine": rt_cos_l,
                    "z_norm": float(z_local[0].norm().detach().cpu()),
                    "local_noise_l2": float(noise[0].norm().detach().cpu()),
                    "edit_distance_to_center_peptide": levenshtein_edit_distance(pep, pep_l),
                })

    df = pd.DataFrame(rows)
    df["is_duplicate_raw_decode"] = df.duplicated(subset=["peptide"], keep="first")
    df.to_csv(os.path.join(args.decoder_dir, f"decoder_diagnostics_bo_iter_{bo_iter:03d}.csv"), index=False)
    unique = list(dict.fromkeys(decoded))
    return unique, df


def fit_mo_models(Z: torch.Tensor, Y: torch.Tensor, device: torch.device) -> ModelListGP:
    models = []
    for m in range(Y.size(-1)):
        gp = SingleTaskGP(Z, Y[:, m:m + 1], outcome_transform=Standardize(m=1)).to(device)
        gp.likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.GreaterThan(1e-6))
        gp.likelihood.noise = 1e-4
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        models.append(gp)
    return ModelListGP(*models)


def evaluate_blackbox(peptides: Sequence[str], cache: Dict[str, List[float]], obj_cols: Sequence[str]) -> torch.Tensor:
    norm = [str(p).strip().upper() for p in peptides]
    uncached = [p for p in norm if p not in cache]
    if uncached:
        if blackbox_fc is None:
            raise RuntimeError(f"blackbox_fc could not be imported: {BLACKBOX_IMPORT_ERROR}")
        scored = blackbox_fc(uncached)
        for _, row in scored.iterrows():
            pep = clean_peptide(row.get("peptide_len10", row.get("peptide", "")))
            if pep is not None:
                cache[pep] = [float(row[c]) for c in obj_cols]
    rows = [cache.get(p, [0.0] * len(obj_cols)) for p in norm]
    return torch.tensor(rows, dtype=torch.float32)


def filter_novel(candidates: Sequence[str], train_set: set, evaluated_set: set, generated_set: set, args: argparse.Namespace) -> Tuple[List[str], List[Tuple[str, str]]]:
    accepted, rejected = [], []
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


def build_initial_labeled_indices(Y: torch.Tensor, n_init: int, seed: int) -> List[int]:
    pareto_idx = torch.where(pareto_mask_maximize(Y))[0].cpu().tolist()
    pareto_set = set(pareto_idx)
    non_pareto = [i for i in range(Y.size(0)) if i not in pareto_set]
    rng = np.random.default_rng(seed)
    n_extra = max(0, int(n_init) - len(pareto_idx))
    extra = rng.choice(non_pareto, size=min(n_extra, len(non_pareto)), replace=False).tolist() if n_extra else []
    return list(dict.fromkeys(pareto_idx + extra))


def save_pareto(peptides: Sequence[str], labeled_idx: Sequence[int], Y_lab: torch.Tensor, bo_iter: int, out_dir: str, obj_cols: Sequence[str]) -> pd.DataFrame:
    ensure_dir(out_dir)
    Y_cpu = Y_lab.detach().cpu()
    mask = pareto_mask_maximize(Y_cpu)
    rows = []
    for k in torch.where(mask)[0].tolist():
        g = int(labeled_idx[k])
        pep = peptides[g] if 0 <= g < len(peptides) else ""
        row = {"bo_iter": bo_iter, "global_index": g, "peptide": pep}
        vals = Y_cpu[k].tolist()
        for j, c in enumerate(obj_cols):
            row[c] = float(vals[j])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f"pareto_front_bo_iter_{bo_iter:03d}.csv"), index=False)
    return df


def run_bo(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)
    ensure_dir(args.out_dir)
    ensure_dir(args.decoder_dir)
    ensure_dir(args.pareto_dir)

    model, ckpt = load_model_from_checkpoint(args.flow_checkpoint, device)

    df = pd.read_csv(args.data_csv)
    required = [args.peptide_col] + list(args.obj_cols)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Data CSV missing columns: {missing}")
    df = df.copy()
    df[args.peptide_col] = df[args.peptide_col].map(clean_peptide)
    df = df.dropna(subset=[args.peptide_col] + list(args.obj_cols)).reset_index(drop=True)
    peptides = df[args.peptide_col].astype(str).tolist()
    X_all = onehot_encode_peptides(peptides).to(device)
    Y_train_full = torch.tensor(df[list(args.obj_cols)].to_numpy(dtype=np.float32), dtype=torch.float32, device=device)

    Z_all = encode_to_zK(model, X_all, batch_size=args.encode_batch_size).to(device)
    Y_train_pareto = Y_train_full[pareto_mask_maximize(Y_train_full)]
    train_pareto_idx = torch.where(pareto_mask_maximize(Y_train_full))[0]
    train_pareto_df = pd.DataFrame({
        "global_index": train_pareto_idx.cpu().numpy(),
        "peptide": [peptides[i] for i in train_pareto_idx.cpu().tolist()],
        **{c: Y_train_pareto[:, j].detach().cpu().numpy() for j, c in enumerate(args.obj_cols)},
    })
    train_pareto_df.to_csv(os.path.join(args.out_dir, "train_pareto_cu_gru_vae_realnvp_after_flow_gp.csv"), index=False)
    print(f"Training Pareto: {len(train_pareto_df)} solutions")

    cache: Dict[str, List[float]] = {
        pep: [float(v) for v in vals]
        for pep, vals in zip(peptides, df[list(args.obj_cols)].to_numpy(dtype=np.float32).tolist())
    }
    train_set = set(peptides)
    evaluated_set = set()
    generated_set = set()

    labeled_idx = build_initial_labeled_indices(Y_train_full, args.initial_labeled, args.seed + 42)
    Y_lab = Y_train_full[torch.tensor(labeled_idx, dtype=torch.long, device=device)]
    Z_lab = Z_all[torch.tensor(labeled_idx, dtype=torch.long, device=device)]
    evaluated_set.update(peptides[i] for i in labeled_idx)

    ref_point = (Y_lab.detach().double().min(dim=0).values - float(args.ref_margin)).tolist()
    best_hv = float(Hypervolume(ref_point=torch.tensor(ref_point, dtype=Y_lab.dtype, device=device)).compute(Y_lab))

    run_config = vars(args).copy()
    run_config.update({
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_selection": ckpt.get("checkpoint_selection"),
        "bo_space": "post_flow_zK_equals_RealNVP_encoder_mu",
        "latent_dim": int(model.cfg.latent_dim),
        "training_pareto_size": int(len(train_pareto_df)),
        "initial_labeled_actual": int(len(labeled_idx)),
    })
    with open(os.path.join(args.out_dir, "bo_run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    all_candidate_rows = []
    hv_rows = []

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

        scalar = Y_lab.mean(dim=-1)
        best_i = int(torch.argmax(scalar).item())
        u_center = z_to_unit(Z_lab[best_i].detach().double())
        lb = (u_center - float(args.tr_radius_unit)).clamp(0.0, 1.0)
        ub = (u_center + float(args.tr_radius_unit)).clamp(0.0, 1.0)
        bounds = torch.stack([lb, ub], dim=0)

        with torch.no_grad():
            nd = pareto_mask_maximize(Y_lab_bo)
            Y_nd = Y_lab_bo[nd]
            if Y_nd.size(0) > args.max_partition_points:
                keep = torch.topk(Y_nd.mean(dim=-1), k=args.max_partition_points, largest=True).indices
                Y_part = Y_nd[keep]
            else:
                Y_part = Y_nd

        ref_point = (Y_lab_bo.min(dim=0).values - float(args.ref_margin)).tolist()
        partitioning = NondominatedPartitioning(ref_point=torch.tensor(ref_point, device=device), Y=Y_part)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([args.mc_samples])).to(device)
        acq_cls = qLogExpectedHypervolumeImprovement if (args.acqf == "qlogehvi" and qLogExpectedHypervolumeImprovement is not None) else qExpectedHypervolumeImprovement
        acq = acq_cls(model=mo_model, ref_point=ref_point, partitioning=partitioning, sampler=sampler).to(device)

        u_parts = []
        acq_vals = []
        repeats = int(args.q_batch) if args.optimize_q_one_at_a_time else 1
        q_inner = 1 if args.optimize_q_one_at_a_time else int(args.q_batch)
        for _ in range(repeats):
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
                print("[OOM] Retrying acquisition with smaller temporary settings.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                safe_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([max(8, args.mc_samples // 4)])).to(device)
                safe_acq = acq_cls(model=mo_model, ref_point=ref_point, partitioning=partitioning, sampler=safe_sampler).to(device)
                U_part, acq_part = optimize_acqf(
                    safe_acq,
                    bounds=bounds,
                    q=1,
                    num_restarts=max(1, args.num_restarts // 2),
                    raw_samples=max(16, args.raw_samples // 4),
                    sequential=True,
                )
            u_parts.append(U_part.detach())
            acq_vals.append(acq_part.detach().reshape(-1)[0])

        U_cand = torch.cat(u_parts, dim=0)
        acq_value = torch.stack(acq_vals).mean()
        Z_cand = z_from_unit(U_cand).to(dtype=torch.float32)

        cand_raw, decoder_df = decode_candidates_with_diagnostics(model, Z_cand, bo_iter, args)
        cand_peps, rejected = filter_novel(cand_raw, train_set, evaluated_set, generated_set, args)

        if len(cand_peps) == 0:
            print("[BO] No novel peptide survived filtering; increasing decoder temperature fallback for this iteration.")
            # Final fallback: sample same centers at higher temperature once.
            extra = []
            old_temp = args.decoder_temperature
            args.decoder_temperature = max(old_temp, args.fallback_temperature)
            extra, extra_df = decode_candidates_with_diagnostics(model, Z_cand, bo_iter + 100000, args)
            args.decoder_temperature = old_temp
            cand_peps, rejected2 = filter_novel(extra, train_set, evaluated_set, generated_set, args)
            rejected.extend(rejected2)
            decoder_df = pd.concat([decoder_df, extra_df], ignore_index=True)

        status_df = decoder_df.copy()
        rejected_map = {p: reason for p, reason in rejected}

        if len(cand_peps) == 0:
            print("[BO] Still no novel peptide; skipping objective evaluation.")
            generated_set.update(cand_raw)
            status_df["accepted_for_blackbox"] = False
            status_df["rejected_reason"] = status_df["peptide"].map(lambda p: rejected_map.get(p, "duplicate_or_not_selected"))
            status_df.to_csv(os.path.join(args.decoder_dir, f"all_decoder_diagnostics_status_bo_iter_{bo_iter:03d}.csv"), index=False)
            hv_rows.append({"bo_iter": bo_iter, "n_raw_decoded": len(cand_raw), "n_accepted": 0, "hypervolume": best_hv, "best_hypervolume": best_hv, "acq_value": float(acq_value.cpu())})
            pd.DataFrame(hv_rows).to_csv(os.path.join(args.out_dir, "bo_hypervolume_history.csv"), index=False)
            continue

        Y_new = evaluate_blackbox(cand_peps, cache, args.obj_cols).to(device)

        # Attach each accepted peptide to the first optimized center that produced it.
        z_rows = []
        for pep in cand_peps:
            matches = decoder_df.index[decoder_df["peptide"].astype(str) == pep].tolist()
            if matches:
                cand_i = int(decoder_df.loc[matches[0], "z_candidate_index"]) if "z_candidate_index" in decoder_df.columns else 0
                cand_i = min(max(cand_i, 0), Z_cand.size(0) - 1)
                z_rows.append(Z_cand[cand_i].detach().clone())
            else:
                z_rows.append(Z_cand[0].detach().clone())
        Z_new = torch.stack(z_rows, dim=0).to(device)

        comparison_rows = []
        for pep, y in zip(cand_peps, Y_new):
            dominated_by_train = any(dominates(y_ref, y) for y_ref in Y_train_pareto)
            dominates_train_any = any(dominates(y, y_ref) for y_ref in Y_train_pareto)
            row = {
                "bo_iter": bo_iter,
                "peptide": pep,
                "acq_value": float(acq_value.detach().cpu()),
                "dominated_by_train_pareto": int(dominated_by_train),
                "dominates_any_train_pareto": int(dominates_train_any),
            }
            for j, c in enumerate(args.obj_cols):
                row[c] = float(y[j].detach().cpu())
            comparison_rows.append(row)
            all_candidate_rows.append(row)

        pd.DataFrame(comparison_rows).to_csv(os.path.join(args.out_dir, f"accepted_candidates_scored_bo_iter_{bo_iter:03d}.csv"), index=False)

        status_df["accepted_for_blackbox"] = status_df["peptide"].isin(cand_peps)
        status_df["rejected_reason"] = status_df["peptide"].map(lambda p: "" if p in cand_peps else rejected_map.get(p, "duplicate_or_not_selected"))
        for j, c in enumerate(args.obj_cols):
            val_map = {p: float(y[j].detach().cpu()) for p, y in zip(cand_peps, Y_new)}
            status_df[c] = status_df["peptide"].map(lambda p: val_map.get(p, np.nan))
        status_df.to_csv(os.path.join(args.decoder_dir, f"all_decoder_diagnostics_status_bo_iter_{bo_iter:03d}.csv"), index=False)

        new_global_indices = []
        for pep in cand_peps:
            peptides.append(pep)
            new_global_indices.append(len(peptides) - 1)
        Z_lab = torch.cat([Z_lab, Z_new], dim=0)
        Y_lab = torch.cat([Y_lab, Y_new], dim=0)
        labeled_idx.extend(new_global_indices)
        evaluated_set.update(cand_peps)
        generated_set.update(cand_raw)

        with torch.no_grad():
            hv = float(Hypervolume(ref_point=torch.tensor(ref_point, dtype=Y_lab.dtype, device=device)).compute(Y_lab))
        improved = hv > best_hv + float(args.hv_tol)
        if improved:
            best_hv = hv

        pareto_df = save_pareto(peptides, labeled_idx, Y_lab, bo_iter, args.pareto_dir, args.obj_cols)
        hv_row = {
            "bo_iter": bo_iter,
            "n_raw_decoded": int(len(cand_raw)),
            "n_accepted": int(len(cand_peps)),
            "n_labeled": int(len(labeled_idx)),
            "pareto_size": int(len(pareto_df)),
            "hypervolume": float(hv),
            "best_hypervolume": float(best_hv),
            "improved": int(improved),
            "acq_value": float(acq_value.detach().cpu()),
            "n_dominates_train": int(sum(r["dominates_any_train_pareto"] for r in comparison_rows)),
        }
        hv_rows.append(hv_row)
        pd.DataFrame(hv_rows).to_csv(os.path.join(args.out_dir, "bo_hypervolume_history.csv"), index=False)
        pd.DataFrame(all_candidate_rows).to_csv(os.path.join(args.out_dir, "all_accepted_candidates_scored.csv"), index=False)
        print(f"Accepted {len(cand_peps)} peptides. HV={hv:.6f}, best={best_hv:.6f}, dominates_train={hv_row['n_dominates_train']}")

    Y_cpu = Y_lab.detach().cpu()
    mask = pareto_mask_maximize(Y_cpu)
    final_rows = []
    for k in torch.where(mask)[0].tolist():
        g = int(labeled_idx[k])
        pep = peptides[g] if 0 <= g < len(peptides) else ""
        row = {"global_index": g, "peptide": pep, "is_novel": pep not in train_set}
        vals = Y_cpu[k].tolist()
        for j, c in enumerate(args.obj_cols):
            row[c] = float(vals[j])
        final_rows.append(row)
    final_df = pd.DataFrame(final_rows)
    if len(final_df):
        Y_final = torch.tensor(final_df[list(args.obj_cols)].to_numpy(dtype=np.float32))
        Y_train_pareto_cpu = Y_train_pareto.detach().cpu()
        final_df["dominates_train_pareto"] = [
            any(dominates(Y_final[i], y_ref) for y_ref in Y_train_pareto_cpu)
            for i in range(len(final_df))
        ]
    else:
        final_df["dominates_train_pareto"] = []

    final_path = os.path.join(args.out_dir, "bo_final_pareto_CU_gru_vae_realnvp_after_flow_gp.csv")
    final_df.to_csv(final_path, index=False)
    dom_df = final_df[(final_df.get("is_novel", False) == True) & (final_df.get("dominates_train_pareto", False) == True)].copy() if len(final_df) else pd.DataFrame()
    dom_path = os.path.join(args.out_dir, "bo_pareto_dominates_train_CU_gru_vae_realnvp_after_flow_gp.csv")
    dom_df.to_csv(dom_path, index=False)

    print("\nDone.")
    print(f"Final BO Pareto: {len(final_df)} -> {final_path}")
    print(f"Novel BO Pareto solutions dominating training Pareto: {len(dom_df)} -> {dom_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-objective GP/qEHVI BO in after-RealNVP zK space of latent-conditioned GRU-VAE.")
    p.add_argument("--flow-checkpoint", default=r"transfer_gru_vae_realnvp_checkpoints_h64_z64_latent_conditioned_high_confidence_data\best_roundtrip_h64_z64_latent_conditioned_realnvp_roundtrip.pt")
    p.add_argument("--data-csv", default="metalpdb_CU_chain_mapped_len10_high_confidence_blackbox_scored_ranked.csv")
    p.add_argument("--peptide-col", default="peptide_len10")
    p.add_argument("--obj-cols", nargs=4, default=OBJ_COLS)
    p.add_argument("--out-dir", default="bo_results_CU_gru_vae_h64_z64_realnvp_after_flow_gp")
    p.add_argument("--decoder-dir", default="bo_decoder_monitoring_CU_gru_vae_h64_z64_realnvp_after_flow_gp")
    p.add_argument("--pareto-dir", default="pareto_front_CU_gru_vae_h64_z64_realnvp_after_flow_gp")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--encode-batch-size", type=int, default=256)

    p.add_argument("--bo-iters", type=int, default=20)
    p.add_argument("--initial-labeled", type=int, default=256)
    p.add_argument("--q-batch", type=int, default=2)
    p.add_argument("--optimize-q-one-at-a-time", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num-restarts", type=int, default=4)
    p.add_argument("--raw-samples", type=int, default=64)
    p.add_argument("--mc-samples", type=int, default=32)
    p.add_argument("--acqf", choices=["qehvi", "qlogehvi"], default="qlogehvi")
    p.add_argument("--tr-radius-unit", type=float, default=0.35)
    p.add_argument("--max-partition-points", type=int, default=20)
    p.add_argument("--ref-margin", type=float, default=0.1)
    p.add_argument("--hv-tol", type=float, default=1e-6)

    p.add_argument("--decoder-samples-per-z", type=int, default=16)
    p.add_argument("--decoder-temperature", type=float, default=0.9)
    p.add_argument("--fallback-temperature", type=float, default=1.25)
    p.add_argument("--novelty-radii", nargs="*", type=float, default=[0.10, 0.25, 0.50])
    p.add_argument("--local-neighbors-per-radius", type=int, default=4)
    p.add_argument("--reject-training-peptides", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--reject-seen-peptides", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


if __name__ == "__main__":
    run_bo(parse_args())
