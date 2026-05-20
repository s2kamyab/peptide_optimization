# ============================================================
# Semi-supervised GRU-VAE (latent normalizing flow) + Deep-kernel SVGP
# + Multi-objective BO with qEHVI (trust region + recentering)
#
# Data: fixed length peptide sequences (len=10), one-hot over 20 AAs
# ============================================================
import os
import wandb
from datetime import datetime
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

from botorch.utils.multi_objective.hypervolume import Hypervolume
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, no Tkinter
import matplotlib.pyplot as plt

# ---- GP / BO stack ----
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.mlls import VariationalELBO
# from botorch.acquisition.multi_objective.monte_carlo import qLogExpectedHypervolumeImprovement
from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.models.transforms.outcome import Standardize

from black_box_fcn_mo_CU_f_updated import blackbox_fc
# ============================================================
# Config
# ============================================================
os.environ["WANDB_MODE"] = "offline"
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_I = {a: i for i, a in enumerate(AA)}
I_TO_AA = {i: a for a, i in AA_TO_I.items()}

SEQ_LEN = 10
VOCAB = 20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

# ---- Architecture knobs (same size everywhere) ----
H = 32                # SAME hidden size across all layers
LATENT_DIM = 32
N_GRU_LAYERS = 2
N_FLOWS = 4           # number of planar flows
WARMUP_EPOCHS = 2
# ---- Training knobs ----
BATCH_SIZE = 256
LR = 1e-3
KL_BETA = 0.01          # you may want KL warmup

# ---- BO knobs ----
TR_RADIUS = 2.0         # Linf trust region radius in latent space
Q_BATCH = 1           # qEHVI batch size
N_RESTARTS = 1
RAW_SAMPLES = 64
BO_ITERS = 50
TRAIN_EPOCHS_PER_ITER = 1

DATA_CSV = "metalpdb_binding_windows_len10_CU_scored_ranked.csv"
PEP_COL = "peptide_len10"


# ============================================================
# Utilities: one-hot encode/decode
# ============================================================

def onehot_encode_peptides(peps: List[str]) -> torch.Tensor:
    """Return [N, L, V] one-hot tensor."""
    x = torch.zeros((len(peps), SEQ_LEN, VOCAB), dtype=DTYPE)
    for n, s in enumerate(peps):
        s = s.strip().upper()
        if len(s) != SEQ_LEN:
            raise ValueError(f"Expected len {SEQ_LEN}, got {len(s)} for peptide={s}")
        for t, ch in enumerate(s):
            x[n, t, AA_TO_I.get(ch, 0)] = 1.0
    return x

@torch.no_grad()
def decode_logits_to_peptide(logits: torch.Tensor) -> str:
    """logits: [L,V] -> argmax peptide str."""
    idx = logits.argmax(dim=-1).tolist()
    return "".join(I_TO_AA[i] for i in idx)
# ============================================================
# Monitoring utilities (ported from train_seqVAE_one_hot_masked_latent_flow.py)
# - These produce matplotlib figures on disk and then log them to W&B.
# - In this BO script we only cache *top-layer* GRU activations (enc/dec),
#   so hidden-layer plots are for "layer_top".
# ============================================================

def _mkdir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def _aa_labels():
    return list(AA)

@torch.no_grad()
def plot_token_accuracy_per_position(logits: torch.Tensor, targets: torch.Tensor, out_dir: str, name: str):
    pred = logits.argmax(dim=-1)  # (B,L)
    accs = [(pred[:, t] == targets[:, t]).float().mean().item() for t in range(SEQ_LEN)]
    plt.figure()
    plt.plot(range(1, SEQ_LEN + 1), accs)
    plt.xlabel("Position (1-10)")
    plt.ylabel("Token accuracy")
    plt.title(f"{name}: Token Accuracy per Position")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_token_acc_per_pos.png"), dpi=200)
    plt.close()
    return accs

@torch.no_grad()
def compute_token_accuracy_per_position(logits: torch.Tensor, targets: torch.Tensor) -> List[float]:
    pred = logits.argmax(dim=-1)  # (B,L)
    return [(pred[:, t] == targets[:, t]).float().mean().item() for t in range(SEQ_LEN)]

@torch.no_grad()
def plot_recon_confidence(logits: torch.Tensor, targets: torch.Tensor, out_dir: str, name: str):
    probs = logits.softmax(dim=-1)
    conf = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B,L)
    conf_np = conf.detach().cpu().numpy().ravel()
    plt.figure()
    plt.hist(conf_np, bins=60)
    plt.xlabel("P(true token)")
    plt.ylabel("Count")
    plt.title(f"{name}: Reconstruction Confidence")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_recon_confidence_hist.png"), dpi=200)
    plt.close()

@torch.no_grad()
def compute_confusion_Tx20x20(logits: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
    pred = logits.argmax(dim=-1)  # (B,L)
    B = targets.size(0)
    conf = np.zeros((SEQ_LEN, VOCAB, VOCAB), dtype=np.int64)
    pred_np = pred.detach().cpu().numpy()
    true_np = targets.detach().cpu().numpy()
    for t in range(SEQ_LEN):
        for b in range(B):
            conf[t, true_np[b, t], pred_np[b, t]] += 1
    return conf

def plot_confusion_heatmap_20x20(conf_20x20: np.ndarray, out_dir: str, name: str):
    labels = _aa_labels()
    plt.figure(figsize=(7, 6))
    plt.imshow(conf_20x20, aspect="auto")
    plt.colorbar(label="Count")
    plt.xticks(range(VOCAB), labels, rotation=90)
    plt.yticks(range(VOCAB), labels)
    plt.xlabel("Predicted AA")
    plt.ylabel("True AA")
    plt.title(name)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}.png"), dpi=200)
    plt.close()

@torch.no_grad()
def plot_confusion_per_position(logits: torch.Tensor, targets: torch.Tensor, out_dir: str, name: str):
    conf = compute_confusion_Tx20x20(logits, targets)  # (L,20,20)
    np.save(os.path.join(out_dir, f"{name}_confusion_Tx20x20.npy"), conf)
    for t in range(SEQ_LEN):
        plot_confusion_heatmap_20x20(
            conf[t],
            out_dir=out_dir,
            name=f"{name}_confusion_pos{t+1:02d}_20x20"
        )

@torch.no_grad()
def plot_most_frequent_error_token_per_position(logits: torch.Tensor, targets: torch.Tensor, out_dir: str, name: str):
    pred = logits.argmax(dim=-1)  # (B,L)
    pred_np = pred.detach().cpu().numpy()
    true_np = targets.detach().cpu().numpy()

    labels = _aa_labels()
    rows = []
    err_tokens, err_counts, err_rates = [], [], []

    for t in range(SEQ_LEN):
        mask_err = (pred_np[:, t] != true_np[:, t])
        n_err = int(mask_err.sum())
        n_all = pred_np.shape[0]
        err_rate = n_err / max(1, n_all)

        if n_err == 0:
            rows.append({"position": t + 1, "most_freq_error_token": "-", "count": 0, "error_rate": err_rate})
            err_tokens.append("-"); err_counts.append(0); err_rates.append(err_rate)
            continue

        wrong_preds = pred_np[mask_err, t]
        counts = np.bincount(wrong_preds, minlength=VOCAB)
        j = int(counts.argmax())
        rows.append({
            "position": t + 1,
            "most_freq_error_token": labels[j],
            "count": int(counts[j]),
            "error_rate": err_rate,
        })
        err_tokens.append(labels[j]); err_counts.append(int(counts[j])); err_rates.append(err_rate)

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"{name}_most_freq_error_per_pos.csv"), index=False)

    plt.figure(figsize=(9, 4))
    xs = np.arange(1, SEQ_LEN + 1)
    plt.bar(xs, err_counts)
    plt.xticks(xs)
    plt.xlabel("Position (1-10)")
    plt.ylabel("Count of most frequent wrong token")
    plt.title(f"{name}: Most Frequent Error Token per Position")
    for x, c, tok, er in zip(xs, err_counts, err_tokens, err_rates):
        plt.text(x, c + 0.5, f"{tok}\n{er:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_most_freq_error_per_pos.png"), dpi=200)
    plt.close()

@torch.no_grad()
def plot_logdet_vs_znorm_scatter(z0: torch.Tensor, zK: torch.Tensor, logdet: torch.Tensor, out_dir: str, name: str):
    z0n = z0.norm(dim=1).detach().cpu().numpy()
    zKn = zK.norm(dim=1).detach().cpu().numpy()
    ld = logdet.detach().cpu().numpy()

    pd.DataFrame({"z0_norm": z0n, "zK_norm": zKn, "logdet": ld}).to_csv(
        os.path.join(out_dir, f"{name}_logdet_vs_znorm.csv"), index=False
    )

    plt.figure()
    plt.scatter(z0n, ld, s=8)
    plt.xlabel("||z0||")
    plt.ylabel("sum log|det J|")
    plt.title(f"{name}: logdet vs ||z0||")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_logdet_vs_z0norm.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.scatter(zKn, ld, s=8)
    plt.xlabel("||zK||")
    plt.ylabel("sum log|det J|")
    plt.title(f"{name}: logdet vs ||zK||")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_logdet_vs_zKnorm.png"), dpi=200)
    plt.close()

@torch.no_grad()
def plot_hidden_norms_over_time(cache: Dict[str, torch.Tensor], out_dir: str, name: str):
    plt.figure()
    for k, v in cache.items():
        norms = v.norm(dim=-1).mean(dim=0).detach().cpu().numpy()  # (L,)
        plt.plot(range(1, SEQ_LEN + 1), norms, label=k)
    plt.xlabel("Time step (1-10)")
    plt.ylabel("Mean hidden L2 norm")
    plt.title(f"{name}: Hidden Norms over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_hidden_norms_over_time.png"), dpi=200)
    plt.close()

@torch.no_grad()
def plot_hidden_mean_std_over_time(cache: Dict[str, torch.Tensor], out_dir: str, name: str):
    for k, v in cache.items():
        v_cpu = v.detach().cpu()  # (B,L,H)
        mean_t = v_cpu.mean(dim=(0, 2)).numpy()  # (L,)
        std_t = v_cpu.std(dim=(0, 2)).numpy()    # (L,)

        plt.figure()
        plt.plot(range(1, SEQ_LEN + 1), mean_t, label="mean")
        plt.plot(range(1, SEQ_LEN + 1), std_t, label="std")
        plt.xlabel("Time step (1-10)")
        plt.ylabel("Activation")
        plt.title(f"{name}: {k} mean/std over time")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}_{k}_mean_std_over_time.png"), dpi=200)
        plt.close()

@torch.no_grad()
def plot_representation_spectrum(cache: Dict[str, torch.Tensor], out_dir: str, name: str, max_svs: int = 30):
    for k, v in cache.items():
        x = v.detach().cpu().reshape(-1, v.shape[-1]).numpy()
        x = x - x.mean(axis=0, keepdims=True)
        try:
            s = np.linalg.svd(x, full_matrices=False, compute_uv=False)
        except Exception:
            continue
        s = s[:max_svs]
        plt.figure()
        plt.plot(range(1, len(s) + 1), s)
        plt.xlabel("Component")
        plt.ylabel("Singular value")
        plt.title(f"{name}: {k} representation spectrum")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}_{k}_spectrum.png"), dpi=200)
        plt.close()

@torch.no_grad()
def plot_latent_distributions(mu, logvar, z0, zK, out_dir: str, name: str):
    def _hist(arr, title, fname):
        plt.figure()
        plt.hist(arr, bins=60)
        plt.xlabel(title)
        plt.ylabel("Count")
        plt.title(f"{name}: {title}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=200)
        plt.close()

    _hist(mu.detach().cpu().numpy().ravel(), "mu", f"{name}_mu_hist.png")
    _hist(logvar.detach().cpu().numpy().ravel(), "logvar", f"{name}_logvar_hist.png")
    _hist(z0.detach().cpu().numpy().ravel(), "z0", f"{name}_z0_hist.png")
    _hist(zK.detach().cpu().numpy().ravel(), "zK", f"{name}_zK_hist.png")

    z0n = z0.norm(dim=1).detach().cpu().numpy()
    zKn = zK.norm(dim=1).detach().cpu().numpy()
    _hist(z0n, "||z0||", f"{name}_z0_norm_hist.png")
    _hist(zKn, "||zK||", f"{name}_zK_norm_hist.png")

@torch.no_grad()
def plot_flow_diagnostics(flow_preacts: List[torch.Tensor], logdet: torch.Tensor, out_dir: str, name: str):
    for i, a in enumerate(flow_preacts):
        a_np = a.detach().cpu().numpy()
        t = np.tanh(a_np)
        tp = 1.0 - t * t  # tanh'(a)

        plt.figure()
        plt.hist(a_np, bins=60)
        plt.xlabel("a = w^T z + b")
        plt.ylabel("Count")
        plt.title(f"{name}: Flow {i} pre-activation a")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}_flow{i}_a_hist.png"), dpi=200)
        plt.close()

        plt.figure()
        plt.hist(tp, bins=60)
        plt.xlabel("tanh'(a)")
        plt.ylabel("Count")
        plt.title(f"{name}: Flow {i} tanh'(a) (saturation proxy)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}_flow{i}_tanhprime_hist.png"), dpi=200)
        plt.close()

    ld = logdet.detach().cpu().numpy()
    plt.figure()
    plt.hist(ld, bins=60)
    plt.xlabel("sum log|det J|")
    plt.ylabel("Count")
    plt.title(f"{name}: logdet distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_logdet_hist.png"), dpi=200)
    plt.close()

def _wandb_log_dir(step: int, prefix: str) -> str:
    # Wandb typically runs from repo root; keep outputs local & structured.
    return _mkdir(os.path.join("wandb_monitoring", prefix, f"step_{step:07d}"))

@torch.no_grad()
def wandb_log_monitoring(out: Dict[str, torch.Tensor], x_onehot: torch.Tensor, step: int, prefix: str = "train", step_key: str = "global_step"):
    """Lightweight monitoring: scalar token accuracy only (no image uploads)."""
    y_ids = x_onehot.argmax(dim=-1)
    accs = compute_token_accuracy_per_position(out["logits"], y_ids)
    wandb_payload = {
        step_key: step,
        f"{prefix}/token_acc": float(np.mean(accs)),
    }
    wandb.log(wandb_payload)



# ============================================================
# Normalizing flow: Planar Flow
# z' = z + u * tanh(w^T z + b)
# log|det J| = log|1 + u^T psi(z)| where psi = (1 - tanh^2(.)) * w
# ============================================================

class PlanarFlow(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim) * 0.02)
        self.u = nn.Parameter(torch.randn(dim) * 0.02)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z: [B, d]
        returns:
          z_new: [B, d]
          logabsdet: [B]  (per-sample log|det|)
          a: [B]          (pre-activation a = w^T z + b) for diagnostics
        """
        a = z @ self.w + self.b          # [B]
        h = torch.tanh(a)                # [B]
        z_new = z + h.unsqueeze(-1) * self.u  # [B,d]

        psi = (1.0 - h**2).unsqueeze(-1) * self.w.unsqueeze(0)  # [B,d]
        det_term = 1.0 + (psi * self.u.unsqueeze(0)).sum(dim=-1)  # [B]
        logabsdet = torch.log(torch.abs(det_term) + 1e-8)  # [B]
        return z_new, logabsdet, a


class FlowSequence(nn.Module):
    def __init__(self, dim: int, n_flows: int):
        super().__init__()
        self.flows = nn.ModuleList([PlanarFlow(dim) for _ in range(n_flows)])

    def forward(self, z0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        z0: [B,d]
        returns:
          zK: [B,d]
          sum_logabsdet: [B]
          preacts: list of a tensors (each [B]) for diagnostics
        """
        z = z0
        sum_logdet = torch.zeros(z0.size(0), device=z0.device)
        preacts: List[torch.Tensor] = []
        for f in self.flows:
            z, logdet, a = f(z)
            sum_logdet = sum_logdet + logdet
            preacts.append(a)
        return z, sum_logdet, preacts


class GRUEncoder(nn.Module):
    def __init__(self, hidden_size: int, latent_dim: int, n_layers: int):
        super().__init__()
        self.in_proj = nn.Linear(VOCAB, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
        )
        self.to_mu = nn.Linear(hidden_size, latent_dim)
        self.to_logvar = nn.Linear(hidden_size, latent_dim)

    def forward(self, x_onehot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          mu: [B,d]
          logvar: [B,d]
          out_top: [B,L,H]  (top-layer GRU outputs for monitoring)
          hN: [n_layers,B,H]
        """
        x = self.in_proj(x_onehot)       # [B,L,H]
        out_top, hN = self.gru(x)        # out_top: [B,L,H], hN: [n_layers,B,H]
        h_last = hN[-1]                  # [B,H]
        mu = self.to_mu(h_last)          # [B,d]
        logvar = self.to_logvar(h_last)  # [B,d]
        return mu, logvar, out_top, hN


class GRUDecoder(nn.Module):
    def __init__(self, hidden_size: int, latent_dim: int, n_layers: int):
        super().__init__()
        # teacher forcing inputs are one-hot tokens -> embed to H
        self.token_embed = nn.Linear(VOCAB, hidden_size)

        # map z -> initial hidden state for ALL layers (same size)
        self.z_to_h = nn.Linear(latent_dim, n_layers * hidden_size)

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
        )
        self.to_logits = nn.Linear(hidden_size, VOCAB)

    def forward(self, z: torch.Tensor, x_onehot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z: [B,d] (use zK)
        x_onehot: [B,L,V] input tokens for teacher forcing
        returns:
          logits: [B,L,V]
          out_top: [B,L,H] top-layer GRU outputs for monitoring
        """
        B = z.size(0)

        # teacher forcing: feed x shifted right with BOS=zero
        x_shift = torch.zeros_like(x_onehot)
        x_shift[:, 1:, :] = x_onehot[:, :-1, :]

        emb = self.token_embed(x_shift)  # [B,L,H]

        h0 = self.z_to_h(z).view(self.gru.num_layers, B, self.gru.hidden_size)  # [n_layers,B,H]
        out_top, _ = self.gru(emb, h0)   # [B,L,H]
        logits = self.to_logits(out_top) # [B,L,V]
        return logits, out_top


class FlowGRUVAE(nn.Module):
    def __init__(self, hidden_size: int, latent_dim: int, n_layers: int, n_flows: int):
        super().__init__()
        self.enc = GRUEncoder(hidden_size, latent_dim, n_layers)
        self.dec = GRUDecoder(hidden_size, latent_dim, n_layers)
        self.flow = FlowSequence(latent_dim, n_flows)

    def reparam(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_onehot: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar, enc_out_top, _ = self.enc(x_onehot)      # [B,d], [B,d], [B,L,H]
        z0 = self.reparam(mu, logvar)                        # [B,d]
        zK, sum_logdet, flow_preacts = self.flow(z0)         # [B,d], [B], list([B])
        logits, dec_out_top = self.dec(zK, x_onehot)         # [B,L,V], [B,L,H]
        return {
            "mu": mu, "logvar": logvar,
            "z0": z0, "zK": zK, "sum_logdet": sum_logdet,
            "flow_preacts": flow_preacts,
            "logits": logits,
            # monitoring caches (top-layer only)
            "enc_cache": {"layer_top": enc_out_top},
            "dec_cache": {"layer_top": dec_out_top},
        }


def vae_flow_loss(
    x_onehot: torch.Tensor,
    logits: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    sum_logdet: torch.Tensor,
    beta: float = KL_BETA
) -> torch.Tensor:
    """
    ELBO with flow:
      recon = CE(x, p(x|zK))
      KL = KL(q(z0|x) || p(z0)) - E_q[log|det(dzK/dz0)|]
    """
    target = x_onehot.argmax(dim=-1)  # [B,L]
    recon = F.cross_entropy(logits.reshape(-1, VOCAB), target.reshape(-1), reduction="mean")

    # KL(q(z0|x) || N(0,I)) for diagonal Gaussian
    kl0 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)  # [B]
    # Flow correction: KL(zK) = KL(z0) - logdet
    kl = (kl0 - sum_logdet).mean()

    return recon + beta * kl


# ============================================================
# Deep-kernel SVGP: GP consumes latent features from encoder (we use mu or zK)
# Gradients flow back to encoder parameters -> "deep kernel"
# ============================================================

class LatentSVGP(ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor):
        q = CholeskyVariationalDistribution(inducing_points.size(0))
        strat = VariationalStrategy(self, inducing_points, q, learn_inducing_locations=True)
        super().__init__(strat)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=inducing_points.size(-1))
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


# ============================================================
# Module-level objective cache  peptide -> [obj0, obj1, obj2, obj3]
# Avoids re-running ColabFold/A3D for previously evaluated peptides.
# Pre-populated from the training CSV at startup (see main()).
# ============================================================
_OBJ_CACHE: Dict[str, List[float]] = {}
_OBJ_COLS = ["chelation_sub", "solubility_sub", "stability_sub", "expression_sub"]

# ============================================================
# Black-box objectives (with in-memory cache + on-disk file check)
# ============================================================

def black_box_objectives(peptides: List[str]) -> torch.Tensor:
    """
    Multi-objective black-box evaluator for BO candidates.

    First checks _OBJ_CACHE for each peptide (in-memory cache).
    Calls blackbox_fc only for uncached peptides; blackbox_fc in turn
    checks whether PDB / A3D .csv files already exist on disk and skips
    ColabFold/A3D when they do.

    Returns [N, 4] float32 tensor (all objectives are maximized).
    """
    norm_peps = [p.strip().upper() for p in peptides]
    uncached = [p for p in norm_peps if p not in _OBJ_CACHE]

    if uncached:
        t = blackbox_fc(uncached)
        for _, row in t.iterrows():
            pep_key = str(row["peptide_len10"]).strip().upper()
            _OBJ_CACHE[pep_key] = [float(row[c]) for c in _OBJ_COLS]

    rows = []
    for p in norm_peps:
        rows.append(_OBJ_CACHE.get(p, [0.0, 0.0, 0.0, 0.0]))
    return torch.tensor(rows, dtype=torch.float32)



# ============================================================
# Pareto utilities (for evaluated points)
# ============================================================

def pareto_mask_maximize(Y: torch.Tensor) -> torch.Tensor:
    """Return boolean mask of Pareto-optimal points for MAXIMIZATION.
    Y: [N, M]
    """
    if Y.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool, device=Y.device)
    # Broadcasting: A[i,j,:] = Y[j,:], B[i,j,:] = Y[i,:]
    A = Y.unsqueeze(0)
    B = Y.unsqueeze(1)
    ge = (A >= B).all(dim=-1)  # [N,N]
    gt = (A >  B).any(dim=-1)  # [N,N]
    dominates = ge & gt
    # ignore self-dominance
    dominates.fill_diagonal_(False)
    dominated = dominates.any(dim=1)   # i is dominated by any j
    return ~dominated

def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def build_pareto_table(
    peptides: List[str],
    labeled_idx: List[int],
    Y_lab: torch.Tensor,
    bo_iter: int,
    out_dir: str = "pareto_front",
) -> Tuple[wandb.Table, pd.DataFrame]:
    """Build & persist Pareto front table for the CURRENT evaluated set (labeled_idx/Y_lab)."""
    Y = Y_lab.detach().cpu()
    mask = pareto_mask_maximize(Y).cpu().numpy()
    pos = np.where(mask)[0].tolist()

    rows = []
    for k in pos:
        g = int(labeled_idx[k])  # global index in X_all/peptides
        pep = peptides[g] if 0 <= g < len(peptides) else ""
        objs = Y[k].tolist()
        rows.append([bo_iter, g, pep] + objs)

    df = pd.DataFrame(rows, columns=["bo_iter", "global_index", "peptide", "obj0", "obj1", "obj2", "obj3"])
    _ensure_dir(out_dir)
    df.to_csv(os.path.join(out_dir, f"pareto_front_bo_iter_{bo_iter:03d}.csv"), index=False)

    table = wandb.Table(columns=df.columns.tolist())
    for r in rows:
        table.add_data(*r)
    return table, df

# ============================================================
# Semi-supervised joint training: VAE on all samples + SVGP on labeled subset
# ============================================================

def pick_initial_labeled_indices(n_total: int, n_init: int = 256, seed: int = 0) -> List[int]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=min(n_init, n_total), replace=False)
    return idx.tolist()

def train_one_epoch_joint(
    model: FlowGRUVAE,
    gp: LatentSVGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    optimizer: torch.optim.Optimizer,
    X_all: torch.Tensor,              # [N,L,V]
    labeled_idx: List[int],
    y_labeled: torch.Tensor,          # [N_lab] scalar supervision for SVGP
    train_indices: Optional[torch.Tensor] = None,  # global indices used for reconstruction training
    global_step: int = 0,
    batch_size: int = BATCH_SIZE,
    log_every: int = 20,
    val_indices: Optional[torch.Tensor] = None,
):
    """
    Minibatch over X.
    - For ALL items in batch: VAE flow ELBO
    - For labeled subset within batch: SVGP variational ELBO (negative added)
    """
    model.train()
    gp.train()
    likelihood.train()

    if train_indices is None:
        train_indices = torch.arange(X_all.size(0), device=X_all.device)
    else:
        train_indices = train_indices.to(device=X_all.device)

    n_all = train_indices.numel()
    perm = train_indices[torch.randperm(n_all, device=X_all.device)]

    # SVGP objective uses total number of labeled points as num_data
    mll = VariationalELBO(likelihood, gp, num_data=len(labeled_idx))

    labeled_set = set(labeled_idx)
    labeled_pos = {g: k for k, g in enumerate(labeled_idx)}  # global index -> position in y_labeled

    for start in range(0, n_all, batch_size):
        optimizer.zero_grad()

        idx = perm[start:start + batch_size]
        xb = X_all[idx]  # [B,L,V]

        out = model(xb)
        # loss_vae = vae_flow_loss(
        #     xb, out["logits"], out["mu"], out["logvar"], out["sum_logdet"], beta=KL_BETA
        # )
        # ---- recon + KL (split) ----
        target = xb.argmax(dim=-1)  # [B, L]
        recon = F.cross_entropy(out["logits"].reshape(-1, VOCAB), target.reshape(-1), reduction="mean")

        kl = -0.5 * torch.sum(
            1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp(),
            dim=-1
        ).mean()

        kl_w = 0.01  # keep same as before (or log if you anneal later)
        loss_vae = recon + kl_w * kl

        # Which in batch are labeled?
        idx_cpu = idx.detach().cpu().tolist()
        lab_globals = [g for g in idx_cpu if g in labeled_set]

        loss_svgp = torch.tensor(0.0, device=X_all.device)
        if len(lab_globals) > 0:
            lab_globals_t = torch.tensor(lab_globals, device=X_all.device)
            lab_local = torch.tensor([labeled_pos[g] for g in lab_globals], device=X_all.device)
            yb = y_labeled[lab_local]  # [B_lab]

            # Deep kernel: latent features from encoder.
            # You can choose mu, z0, or zK. Using mu is common/stable.
            mu_lab, _, _, _ = model.enc(X_all[lab_globals_t])  # [B_lab,d]

            pred = gp(mu_lab)
            loss_svgp = -mll(pred, yb)

        loss = loss_vae + 1.0 * loss_svgp
        loss.backward()
        optimizer.step()
        # ---- W&B logging ----
        if (global_step % log_every) == 0:
            lr = optimizer.param_groups[0]["lr"]
            train_pos_acc = compute_token_accuracy_per_position(out["logits"], target)
            # Use global_step in payload (not as wandb internal step) so that
            # train/* and val/bo/* can advance on independent x-axes.
            wandb.log(
                {
                    "global_step": global_step,
                    "train/recon_ce": float(recon.detach().cpu()),
                    "train/kl": float(kl.detach().cpu()),
                    "train/vae_loss": float(loss_vae.detach().cpu()),
                    "train/total_loss": float(loss.detach().cpu()),
                    "train/lr": lr,
                    "train/token_acc": float(np.mean(train_pos_acc)),
                },
            )
            log_token_accuracy_curves(train_pos_acc, prefix="train", step=global_step, step_key="global_step")

            # ---- Monitoring plots/images (log at same cadence as loss logs) ----
            try:
                wandb_log_monitoring(out, xb, step=global_step, prefix="train", step_key="global_step")
            except Exception as e:
                # don't crash training on monitoring
                wandb.log({"monitoring/error": str(e), "global_step": global_step})

    if val_indices is not None:
        try:
            evaluate_vae_on_indices(
                model=model,
                X_all=X_all,
                eval_indices=val_indices,
                batch_size=batch_size,
                step=global_step,
                prefix="val",
                log_monitoring_batch=False,
                step_key="global_step",
            )
        except Exception as e:
            wandb.log({"val/error": str(e), "global_step": global_step})


        global_step += 1
        print(global_step, f"loss: {loss.item():.4f} (recon: {recon.item():.4f}, kl: {kl.item():.4f}, svgp: {loss_svgp.item():.4f})")
    return global_step


@torch.no_grad()
def encode_to_latent_mu(model: FlowGRUVAE, X: torch.Tensor) -> torch.Tensor:
    model.eval()
    mu, _, _, _ = model.enc(X)
    return mu


def log_token_accuracy_curves(position_accs: List[float], prefix: str, step: int, step_key: str = "global_step"):
    table = wandb.Table(columns=["position", "token_accuracy"])
    for pos, acc in enumerate(position_accs, start=1):
        table.add_data(pos, float(acc))
    wandb.log({
        step_key: step,
        f"{prefix}/token_acc_by_position": wandb.plot.line(
            table,
            "position",
            "token_accuracy",
            title=f"{prefix.capitalize()} Token Accuracy by Position",
        ),
    })
    wandb.log({
        step_key: step,
        **{f"{prefix}/token_acc_pos_{pos}": float(acc) for pos, acc in enumerate(position_accs, start=1)}
    })


# ============================================================
# Validation / evaluation utilities
# ============================================================

@torch.no_grad()
def evaluate_vae_on_indices(
    model: FlowGRUVAE,
    X_all: torch.Tensor,
    eval_indices: torch.Tensor,
    batch_size: int = BATCH_SIZE,
    step: int = 0,
    prefix: str = "val",
    log_monitoring_batch: bool = True,
    step_key: str = "bo_iter",
):
    """Evaluate VAE reconstruction + KL on a held-out set of global indices."""
    model.eval()
    eval_indices = eval_indices.to(device=X_all.device)
    n = eval_indices.numel()

    sum_recon = 0.0
    sum_kl = 0.0
    n_batches = 0
    token_acc_sums = np.zeros(SEQ_LEN, dtype=np.float64)

    monitoring_done = False
    monitor_out = None
    monitor_xb = None

    for start in range(0, n, batch_size):
        idx = eval_indices[start:start + batch_size]
        xb = X_all[idx]
        out = model(xb)

        target = xb.argmax(dim=-1)  # [B,L]
        recon = F.cross_entropy(out["logits"].reshape(-1, VOCAB), target.reshape(-1), reduction="mean")

        kl = -0.5 * torch.sum(
            1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp(),
            dim=-1
        ).mean()

        sum_recon += float(recon.detach().cpu())
        sum_kl += float(kl.detach().cpu())
        token_acc_sums += np.asarray(compute_token_accuracy_per_position(out["logits"], target), dtype=np.float64)
        n_batches += 1

        if log_monitoring_batch and (not monitoring_done):
            monitor_out = out
            monitor_xb = xb
            monitoring_done = True

    mean_recon = sum_recon / max(1, n_batches)
    mean_kl = sum_kl / max(1, n_batches)
    vae_loss = mean_recon + 0.01 * mean_kl  # keep consistent with training kl_w in this script
    mean_pos_acc = (token_acc_sums / max(1, n_batches)).tolist()
    token_acc = float(np.mean(mean_pos_acc))

    if monitor_out is not None and monitor_xb is not None:
        wandb_log_monitoring(monitor_out, monitor_xb, step=step, prefix=prefix, step_key=step_key)
    wandb.log(
        {
            step_key: step,
            f"{prefix}/recon_ce": mean_recon,
            f"{prefix}/kl": mean_kl,
            f"{prefix}_loss": vae_loss,
            f"{prefix}/num_seq": int(n),
            f"{prefix}/token_acc": token_acc,
        },
    )
    log_token_accuracy_curves(mean_pos_acc, prefix=prefix, step=step, step_key=step_key)

    return mean_recon, mean_kl, vae_loss
# ============================================================
# Multi-objective BO models (BoTorch): independent GPs per objective on latent space
# ============================================================

# def fit_mo_models(Z: torch.Tensor, Y: torch.Tensor) -> ModelListGP:
#     models = []
#     M = Y.size(-1)
#     for m in range(M):
#         # y_m = Y[:, m:m+1]
#         # gp_m = SingleTaskGP(Z, y_m, outcome_transform=Standardize(m=1))
#         # mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp_m.likelihood, gp_m)
#         y_m = Y[:, m:m+1]
#         gp_m = SingleTaskGP(
#             Z, y_m,
#             outcome_transform=Standardize(m=1),
#         )
#         gp_m.likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.GreaterThan(1e-6))
#         gp_m.likelihood.noise = 1e-4
#         mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp_m.likelihood, gp_m)
#         fit_gpytorch_mll(mll)
#         models.append(gp_m)
#     return ModelListGP(*models)

def fit_mo_models(Z: torch.Tensor, Y: torch.Tensor, device=DEVICE) -> ModelListGP:
    models = []
    M = Y.size(-1)
    for m in range(M):
        y_m = Y[:, m:m+1]
        gp_m = SingleTaskGP(
            Z, y_m,
            outcome_transform=Standardize(m=1),
        ).to(device)  # ✅ Move to GPU
        gp_m.likelihood.noise_covar.register_constraint(
            "raw_noise", gpytorch.constraints.GreaterThan(1e-6)
        )
        gp_m.likelihood.noise = 1e-4
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp_m.likelihood, gp_m)
        fit_gpytorch_mll(mll)
        models.append(gp_m)
    return ModelListGP(*models)



def make_tr_bounds(center: torch.Tensor, radius: float) -> Tuple[torch.Tensor, torch.Tensor]:
    return center - radius, center + radius

def nondominated_mask(Y: torch.Tensor) -> torch.Tensor:
    """
    Y: [N, M], maximize all objectives
    returns bool mask of nondominated points
    """
    N = Y.size(0)
    is_nd = torch.ones(N, dtype=torch.bool, device=Y.device)
    for i in range(N):
        if not is_nd[i]:
            continue
        dominates_i = (Y >= Y[i]).all(dim=1) & (Y > Y[i]).any(dim=1)
        if dominates_i.any():
            is_nd[i] = False
    return is_nd

def filter_novel_peptides(
    cand_peps,
    train_peptide_set,
    evaluated_peptide_set,
    generated_peptide_set,
    reject_training=True,
    reject_seen=True,
):
    accepted = []
    rejected = []

    for pep in cand_peps:
        reason = None
        if reject_training and pep in train_peptide_set:
            reason = "in_training"
        elif reject_seen and pep in evaluated_peptide_set:
            reason = "already_evaluated"
        elif reject_seen and pep in generated_peptide_set:
            reason = "duplicate_generated"

        if reason is None:
            accepted.append(pep)
        else:
            rejected.append((pep, reason))

    return accepted, rejected

@torch.no_grad()
def decode_multiple_candidates(model, Z_cand, n_samples_per_z=8, temperature=1.0):
    all_peps = []
    for j in range(Z_cand.size(0)):
        z = Z_cand[j:j+1]
        x_dummy = torch.zeros((1, SEQ_LEN, VOCAB), device=Z_cand.device)
        logits, _ = model.dec(z, x_dummy)   # [1, L, V]
        logits = logits[0] / temperature

        # argmax
        all_peps.append(decode_logits_to_peptide(logits))

        # stochastic samples
        probs = torch.softmax(logits, dim=-1)
        for _ in range(n_samples_per_z - 1):
            idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
            pep = "".join(I_TO_AA[i.item()] for i in idx)
            all_peps.append(pep)

    return list(dict.fromkeys(all_peps))   # unique preserve order

def dominates(y_a: torch.Tensor, y_b: torch.Tensor) -> bool:
    return bool(((y_a >= y_b).all() and (y_a > y_b).any()).item())
# ============================================================
# Main loop: load data, initialize, BO iterations with recentering
# ============================================================

def main():
    global_step = 0
    # ---- Load peptides from CSV ----
    df = pd.read_csv(DATA_CSV)
    peptides = df[PEP_COL].astype(str).tolist()

    # ---- Build X_all ----
    X_all = onehot_encode_peptides(peptides).to(DEVICE)

    # Load pre-computed objectives directly from the training CSV.
    # This avoids re-running ColabFold/A3D for training sequences whose
    # scores are already stored in the file.
    train_obj_cols = ["chelation_sub", "solubility_sub", "stability_sub", "expression_sub"]
    Y_train_full = torch.tensor(
        df[train_obj_cols].to_numpy(),
        dtype=torch.float32,
        device=DEVICE,
    )

    # Pre-populate the objective cache with all training data so that
    # black_box_objectives never re-evaluates any training peptide.
    for pep, vals in zip(peptides, df[train_obj_cols].values.tolist()):
        _OBJ_CACHE[pep.strip().upper()] = [float(v) for v in vals]

    train_peptide_set = set(peptides)
    train_peptide_to_idx = {p: i for i, p in enumerate(peptides)}

    train_pareto_mask = nondominated_mask(Y_train_full)
    train_pareto_idx = torch.where(train_pareto_mask)[0]
    Y_train_pareto = Y_train_full[train_pareto_mask]
    train_pareto_peptides = [peptides[i] for i in train_pareto_idx.tolist()]


    # ---- Train/Val split for reconstruction (fixed on the initial dataset only) ----
    rng = np.random.default_rng(0)
    n0 = X_all.size(0)
    all_idx0 = np.arange(n0)
    rng.shuffle(all_idx0)
    val_frac = float(os.getenv("VAL_FRAC", "0.10"))
    n_val = max(1, int(round(val_frac * n0)))
    val_idx = torch.tensor(all_idx0[:n_val], device=DEVICE, dtype=torch.long)
    train_idx = torch.tensor(all_idx0[n_val:], device=DEVICE, dtype=torch.long)

    # ---- Save training Pareto front to disk ----
    os.makedirs("bo_results", exist_ok=True)
    train_pareto_save_df = pd.DataFrame({
        "peptide": train_pareto_peptides,
        "chelation_sub": Y_train_pareto[:, 0].cpu().numpy(),
        "solubility_sub": Y_train_pareto[:, 1].cpu().numpy(),
        "stability_sub": Y_train_pareto[:, 2].cpu().numpy(),
        "expression_sub": Y_train_pareto[:, 3].cpu().numpy(),
    })
    train_pareto_save_df.to_csv("bo_results/train_pareto_cu_updated.csv", index=False)
    print(f"Training Pareto: {len(train_pareto_peptides)} solutions -> bo_results/train_pareto_cu_updated.csv")

    # ---- Labeled subset: training Pareto front + random non-Pareto for diversity ----
    # Seeding the surrogate with the training Pareto ensures the trust region
    # starts in the high-quality region of latent space from iteration 1.
    train_pareto_list = train_pareto_idx.tolist()
    pareto_set = set(train_pareto_list)
    non_pareto_indices = [i for i in range(len(peptides)) if i not in pareto_set]
    n_extra = max(0, 256 - len(train_pareto_list))
    rng_init = np.random.default_rng(42)
    extra_idx = rng_init.choice(
        non_pareto_indices,
        size=min(n_extra, len(non_pareto_indices)),
        replace=False,
    ).tolist()
    labeled_idx = list(dict.fromkeys(train_pareto_list + extra_idx))

    # Y_lab from pre-loaded Y_train_full — no need to call black_box_objectives
    Y_lab = Y_train_full[torch.tensor(labeled_idx, device=DEVICE, dtype=torch.long)]

    M = 4  # only 4 objective
    evaluated_peptide_set = set(peptides[i] for i in labeled_idx)   # initially labeled
    generated_peptide_set = set()

    # For SVGP joint training we need a scalar label.
    # Common choices:
    #   - pick one objective (e.g., 0)
    #   - weighted sum
    # Here: objective 0 as supervision signal.
    y_sup = Y_lab[:, 0].contiguous()  # [N_lab]

    # ---- Initialize GRU-VAE + flow ----
    model = FlowGRUVAE(hidden_size=H, latent_dim=LATENT_DIM, n_layers=N_GRU_LAYERS, n_flows=N_FLOWS).to(DEVICE)

    # -------------------------------
    # W&B init
    # -------------------------------
    WANDB_PROJECT = os.getenv("WANDB_PROJECT", "ankibind-lolbo-peptides")
    WANDB_ENTITY  = os.getenv("WANDB_ENTITY", None)  # optional

    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=f"lolbo-peptides-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={
            "seq_len": SEQ_LEN,
            "vocab": VOCAB,
            "latent_dim": LATENT_DIM,
            "tr_radius": TR_RADIUS,
            "q_batch": Q_BATCH,
            "bo_iters": BO_ITERS,
            "train_epochs_per_iter": TRAIN_EPOCHS_PER_ITER,
            "device": DEVICE,
            "dtype": str(DTYPE),
        },
    )

    wandb.log({"ref/train_pareto_size": int(train_pareto_mask.sum().item())})

    # record split info in config
    wandb.config.update({"val_frac": val_frac, "n0_initial": int(n0), "n_val_initial": int(n_val)}, allow_val_change=True)

    # ------------------------------------------------------------------
    # Define separate x-axes for training metrics vs. BO/validation
    # metrics so that advancing global_step inside the BO loop never
    # creates a backwards-step conflict for val/* and bo/* logs.
    #   train/*  -> x-axis: global_step  (batch-level counter)
    #   val/*    -> x-axis: bo_iter      (BO iteration counter, 0-indexed)
    #   bo/*     -> x-axis: bo_iter
    # ------------------------------------------------------------------
    wandb.define_metric("global_step")
    wandb.define_metric("bo_iter")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("val/*",   step_metric="global_step")
    wandb.define_metric("bo/*",    step_metric="bo_iter")


    # # Optional: track gradients/weights (can be slow for large models)
    # wandb.watch(model, log="gradients", log_freq=200)
    # ---- Initialize SVGP inducing points from current encoder ----
    with torch.no_grad():
        Z_init = encode_to_latent_mu(model, X_all[torch.tensor(labeled_idx, device=DEVICE)])
    inducing = Z_init[: min(64, Z_init.size(0))].clone()

    gp = LatentSVGP(inducing_points=inducing).to(DEVICE)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(gp.parameters()) + list(likelihood.parameters()),
        lr=LR
    )
    # ---- Warmup end-to-end once (or a few epochs) ----
    for ep in range(WARMUP_EPOCHS):   # e.g. 2
        global_step = train_one_epoch_joint(
            model, gp, likelihood, optimizer,
            X_all=X_all,
            train_indices=train_idx,
            labeled_idx=labeled_idx,
            y_labeled=y_sup,
            batch_size=BATCH_SIZE,
            global_step=global_step,
            log_every=10,
            val_indices=val_idx,
        )
        # evaluate_vae_on_indices(model, X_all, val_idx, step=it, prefix="val",
        #                         log_monitoring_batch=(ep == 0))

    # ---- BO state (LoLBO-like) ----
    e2e_freq = 10          # how many consecutive fails before end-to-end update
    hv_tol = 1e-6          # small tolerance to avoid noise counting as improvement
    fails_since_last_e2e = 0

    # hypervolume baseline (compute once using current Y_lab)
    ref_point = (Y_lab.detach().double().min(dim=0).values - 0.1).tolist()
    ref_t = torch.tensor(ref_point, device=DEVICE, dtype=Y_lab.dtype)
    best_hv = float(Hypervolume(ref_point=ref_t).compute(Y_lab))

    # bo_iter is a clean 0-indexed BO iteration counter, fully independent of
    # global_step (which is a batch-level counter used only for train/* logs).
    Y_new = Y_lab[0]
    for bo_iter in range(BO_ITERS):

        print(f"\n=== BO iter {bo_iter+1}/{BO_ITERS} ===")
        # evaluate_vae_on_indices(model, X_all, val_idx, step=bo_iter, prefix="val",
        #                         log_monitoring_batch=True)
        # =========================================================
        # 1) Decide update type: end-to-end vs surrogate-only
        # =========================================================
        if fails_since_last_e2e >= e2e_freq:
            # ---- End-to-end update: train VAE+SVGP (and implicit recentering via re-encode) ----
            print(f"[BO] Stagnating for {fails_since_last_e2e} iters -> end-to-end update")
            for ep in range(TRAIN_EPOCHS_PER_ITER):  # you can set >1 here if you want stronger refresh
                global_step = train_one_epoch_joint(
                    model, gp, likelihood, optimizer,
                    X_all=X_all,
                    train_indices=train_idx,
                    labeled_idx=labeled_idx,
                    y_labeled=y_sup,
                    batch_size=BATCH_SIZE,
                    global_step=global_step,
                    log_every=10,
                    val_indices=val_idx,
                )
            fails_since_last_e2e = 0   # reset after an e2e refresh

        # ---- Always "recenter": compute embeddings with current encoder ----
        X_lab = X_all[torch.tensor(labeled_idx, device=DEVICE)]
        Z_lab = encode_to_latent_mu(model, X_lab)  # [N_lab,d]

        # =========================================================
        # 2) Surrogate-only update (MO GP) on current data
        # =========================================================
        Z_lab_bo = Z_lab.detach().double().to(DEVICE)
        Y_lab_bo = Y_lab.detach().double().to(DEVICE)

        Z_min = Z_lab_bo.min(dim=0).values
        Z_max = Z_lab_bo.max(dim=0).values
        Z_range = (Z_max - Z_min).clamp_min(1e-12)

        def z_to_unit(z): return ((z - Z_min) / Z_range).clamp(0.0, 1.0)
        def z_from_unit(u): return u * Z_range + Z_min

        Z_unit = z_to_unit(Z_lab_bo)
        # mo_model = fit_mo_models(Z_unit, Y_lab_bo)   # <-- surrogate update
        # In main loop:
        mo_model = fit_mo_models(Z_unit, Y_lab_bo, device=DEVICE)

        # =========================================================
        # 3) Acquisition
        # =========================================================
        # Trust region center chosen from scalarization (only for center)
        scalar = Y_lab.mean(dim=-1)
        best_i = torch.argmax(scalar).item()
        z_center = Z_lab[best_i].detach()

        # TR bounds in unit space
        u_center = z_to_unit(z_center.double())
        lb = (u_center - TR_RADIUS / Z_range).clamp(0.0, 1.0)
        ub = (u_center + TR_RADIUS / Z_range).clamp(0.0, 1.0)
        bounds = torch.stack([lb, ub], dim=0)

        # reference point (keep consistent across iterations)
        ref_point = (Y_lab_bo.min(dim=0).values - 0.1).tolist()

        partitioning = NondominatedPartitioning(
            ref_point=torch.tensor(ref_point, device=DEVICE),
            Y=Y_lab_bo
        )
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([32]))
        acq = qExpectedHypervolumeImprovement(
            model=mo_model,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
        ).to(DEVICE)

        U_cand, _ = optimize_acqf(
            acq,
            bounds=bounds,
            q=Q_BATCH,
            num_restarts=N_RESTARTS,
            raw_samples=RAW_SAMPLES,
            sequential=True,
        )
        Z_cand = z_from_unit(U_cand).to(dtype=DTYPE)

        # =========================================================
        # 4) Decode -> evaluate objectives -> update data
        # =========================================================
        model.eval()
        with torch.no_grad():
            x_dummy = torch.zeros((Z_cand.size(0), SEQ_LEN, VOCAB), device=DEVICE)
            logits, _ = model.dec(Z_cand, x_dummy)

        
        # cand_peps = [decode_logits_to_peptide(logits[j]) for j in range(logits.size(0))]
        # cand_peps_raw = [decode_logits_to_peptide(logits[j]) for j in range(logits.size(0))]
        cand_peps_raw = decode_multiple_candidates(model, Z_cand, n_samples_per_z=8, temperature=0.9)
        cand_peps, rejected = filter_novel_peptides(
            cand_peps_raw,
            train_peptide_set=train_peptide_set,
            evaluated_peptide_set=evaluated_peptide_set,
            generated_peptide_set=generated_peptide_set,
            reject_training=True,
            reject_seen=True,
        )
        if len(cand_peps) == 0:
            wandb.log({
                "bo/rejected_all_candidates": 1,
                "bo/rejected_in_training": sum(r == "in_training" for _, r in rejected),
                "bo/rejected_already_evaluated": sum(r == "already_evaluated" for _, r in rejected),
                "bo/rejected_duplicate_generated": sum(r == "duplicate_generated" for _, r in rejected),
            })
            continue

        

        Y_new = black_box_objectives(cand_peps).to(DEVICE)

        comparison_rows = []
        for pep, y in zip(cand_peps, Y_new):
            dominated_by_train = False
            dominates_train_any = False

            for y_ref in Y_train_pareto:
                if dominates(y_ref, y):
                    dominated_by_train = True
                    break

            for y_ref in Y_train_pareto:
                if dominates(y, y_ref):
                    dominates_train_any = True
                    break

            comparison_rows.append({
                "peptide": pep,
                "obj0": float(y[0].item()),
                "obj1": float(y[1].item()),
                "obj2": float(y[2].item()),
                "obj3": float(y[3].item()),
                "dominated_by_train_pareto": int(dominated_by_train),
                "dominates_any_train_pareto": int(dominates_train_any),
            })
        n_dominated_by_train_pareto = sum(r["dominated_by_train_pareto"] for r in comparison_rows)
        n_dominates_train_pareto = sum(r["dominates_any_train_pareto"] for r in comparison_rows)
        y_new_cpu = Y_new.detach().cpu().numpy()
        # wandb.log({"bo/candidates": cand_table}, step=it)

        # add new points (dedupe)
        new_indices = []
        for pep in cand_peps:
            if pep in peptides:
                new_indices.append(peptides.index(pep))
            else:
                peptides.append(pep)
                X_all = torch.cat([X_all, onehot_encode_peptides([pep]).to(DEVICE)], dim=0)
                new_indices.append(len(peptides) - 1)

        labeled_idx.extend(new_indices)
        Y_lab = torch.cat([Y_lab, Y_new], dim=0)
        evaluated_peptide_set.update(cand_peps)
        generated_peptide_set.update(cand_peps_raw)

        # =========================================================
        # 5) Check improvement (hypervolume)
        # =========================================================
        with torch.no_grad():
            ref_t = torch.tensor(ref_point, device=DEVICE, dtype=Y_lab.dtype)
            hv = float(Hypervolume(ref_point=ref_t).compute(Y_lab))

        improved = hv > (best_hv + hv_tol)
        if improved:
            best_hv = hv
            fails_since_last_e2e = 0
        else:
            fails_since_last_e2e += 1

        # wandb.log(
        #     {
        #         "bo/iter": it,
        #         "bo/hypervolume": hv,
        #         "bo/best_hypervolume": best_hv,
        #         "bo/improved": int(improved),
        #         "bo/fails_since_last_e2e": fails_since_last_e2e,
        #         "bo/num_labeled": len(labeled_idx),
        #     },
        #     step=it,
        # )
        # candidates table + BO metrics — both use bo_iter as the x-axis
        # (defined via wandb.define_metric above); no step= argument so
        # wandb's internal counter always advances monotonically.

        # ---- Pareto front over ALL evaluated points (Y_lab) ----
        _, pareto_df = build_pareto_table(
            peptides=peptides,
            labeled_idx=labeled_idx,
            Y_lab=Y_lab,
            bo_iter=bo_iter,
        )
        pareto_size = int(len(pareto_df))
        wandb.log({
            "bo_iter": bo_iter,
            "bo/pareto_size": pareto_size,
            "bo/n_new_candidates": len(cand_peps),
            "bo/n_dominated_by_train_pareto": n_dominated_by_train_pareto,
            "bo/n_dominates_train_pareto": n_dominates_train_pareto,
            "bo/iter": bo_iter,
            "bo/hypervolume": hv,
            "bo/best_hypervolume": best_hv,
            "bo/improved": int(improved),
            "bo/fails_since_last_e2e": fails_since_last_e2e,
            "bo/num_labeled": len(labeled_idx),
        })

        # Update scalar supervision for SVGP (if you keep this design)
        y_sup = Y_lab[:, 0].contiguous()

        all_eval_peptides = []
        all_eval_Y = []

        # training full set
        all_eval_peptides.extend(peptides[:n0])
        all_eval_Y.append(Y_train_full)

        # BO evaluated set beyond training
        # if you want only truly novel ones, keep a separate store for them
        accepted_peps_final = []
        accepted_Y_final = []

        for pep, y in zip(cand_peps, Y_new):
            if pep in train_peptide_set:
                continue
            if pep in evaluated_peptide_set:
                continue

            dominated_by_train = any(dominates(y_ref, y) for y_ref in Y_train_pareto)
            if dominated_by_train:
                continue

            accepted_peps_final.append(pep)
            accepted_Y_final.append(y)

        # os.makedirs("bo_analysis", exist_ok=True)

        print(f"Added {len(new_indices)} labeled points. HV={hv:.6f} (best={best_hv:.6f}), improved={improved}")

    # ============================================================
    # Save final results
    # ============================================================
    os.makedirs("bo_results", exist_ok=True)

    # -- Final BO Pareto front (all evaluated points) --
    Y_all_cpu = Y_lab.detach().cpu()
    bo_pareto_mask_final = pareto_mask_maximize(Y_all_cpu)
    bo_pareto_rows = []
    for k in torch.where(bo_pareto_mask_final)[0].tolist():
        g = int(labeled_idx[k])
        pep = peptides[g] if 0 <= g < len(peptides) else ""
        objs = Y_all_cpu[k].tolist()
        bo_pareto_rows.append([g, pep] + objs)

    final_bo_pareto_df = pd.DataFrame(
        bo_pareto_rows,
        columns=["global_index", "peptide",
                 "chelation_sub", "solubility_sub", "stability_sub", "expression_sub"],
    )
    final_bo_pareto_df["is_novel"] = final_bo_pareto_df["peptide"].apply(
        lambda p: p not in train_peptide_set
    )
    final_bo_pareto_df.to_csv("bo_results/bo_final_pareto_CU_updated.csv", index=False)
    print(f"Final BO Pareto: {len(final_bo_pareto_df)} solutions -> bo_results/bo_final_pareto_CU_updated.csv")

    # -- BO Pareto solutions that dominate at least one training Pareto point --
    Y_train_pareto_cpu = Y_train_pareto.cpu()
    if len(final_bo_pareto_df) > 0:
        Y_bo_par = torch.tensor(
            final_bo_pareto_df[["chelation_sub", "solubility_sub",
                                 "stability_sub", "expression_sub"]].to_numpy(),
            dtype=torch.float32,
        )
        dom_train_mask = [
            any(dominates(Y_bo_par[i], y_ref) for y_ref in Y_train_pareto_cpu)
            for i in range(len(final_bo_pareto_df))
        ]
        final_bo_pareto_df["dominates_train_pareto"] = dom_train_mask
        final_bo_pareto_df.to_csv("bo_results/bo_final_pareto_CU_updated.csv", index=False)

        novel_dom_df = final_bo_pareto_df[
            final_bo_pareto_df["dominates_train_pareto"] & final_bo_pareto_df["is_novel"]
        ].copy()
        novel_dom_df.to_csv("bo_results/bo_pareto_dominates_train_CU_updated.csv", index=False)
        n_dominating = int(sum(dom_train_mask))
        print(f"BO solutions dominating training Pareto: {n_dominating} "
              f"-> bo_results/bo_pareto_dominates_train_CU_updated.csv")
    else:
        n_dominating = 0

    wandb.log({
        "final/bo_pareto_size": len(final_bo_pareto_df),
        "final/bo_novel_pareto_size": int(final_bo_pareto_df["is_novel"].sum()) if len(final_bo_pareto_df) > 0 else 0,
        "final/bo_pareto_dominates_train": n_dominating,
        "final/train_pareto_size": len(train_pareto_peptides),
        "final/best_hypervolume": best_hv,
    })
    wandb.finish()

    print("\nDone.")


if __name__ == "__main__":
    main()