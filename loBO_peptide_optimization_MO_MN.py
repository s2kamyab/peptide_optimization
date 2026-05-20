# ============================================================
# Semi-supervised VAE + Deep-kernel SVGP + MO-BO (qEHVI) for peptides
# Fixed length = 10, alphabet size = 20 (standard AAs)
# ============================================================

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- GP / BO stack ---
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.mlls import VariationalELBO

from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from botorch.models.transforms.outcome import Standardize

# ============================================================
# Config
# ============================================================

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_I = {a: i for i, a in enumerate(AA)}
I_TO_AA = {i: a for a, i in AA_TO_I.items()}

SEQ_LEN = 10
VOCAB = 20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

# ---------------- BO / trust region ----------------
LATENT_DIM = 32
TR_RADIUS = 2.0              # trust region radius in latent space (Linf ball)
Q_BATCH = 8                  # qEHVI batch size
N_RESTARTS = 10
RAW_SAMPLES = 256
BO_ITERS = 15                # number of BO iterations
TRAIN_EPOCHS_PER_ITER = 1    # keep small; you can increase

# ============================================================
# Utilities: encoding / decoding
# ============================================================

def onehot_encode_peptides(peps: List[str]) -> torch.Tensor:
    """
    Returns tensor shape [N, SEQ_LEN, VOCAB] with one-hot encoding.
    """
    x = torch.zeros((len(peps), SEQ_LEN, VOCAB), dtype=DTYPE)
    for n, s in enumerate(peps):
        s = s.strip().upper()
        assert len(s) == SEQ_LEN, f"Expected len {SEQ_LEN}, got {len(s)}"
        for t, ch in enumerate(s):
            x[n, t, AA_TO_I.get(ch, 0)] = 1.0
    return x

@torch.no_grad()
def decode_logits_to_peptide(logits: torch.Tensor) -> str:
    """
    logits: [SEQ_LEN, VOCAB]
    returns argmax peptide string length 10
    """
    idx = logits.argmax(dim=-1).tolist()
    return "".join(I_TO_AA[i] for i in idx)

# ============================================================
# VAE: encoder and decoder (simple MLP; replace with Transformer if you want)
# ============================================================

class Encoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        in_dim = SEQ_LEN * VOCAB
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.to_mu = nn.Linear(256, latent_dim)
        self.to_logvar = nn.Linear(256, latent_dim)

    def forward(self, x_onehot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x_onehot: [N, L, V]
        h = self.net(x_onehot.reshape(x_onehot.size(0), -1))
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        out_dim = SEQ_LEN * VOCAB
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # returns logits [N, L, V]
        out = self.net(z)
        return out.view(z.size(0), SEQ_LEN, VOCAB)

class VAE(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.enc = Encoder(latent_dim)
        self.dec = Decoder(latent_dim)

    def reparam(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_onehot: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar = self.enc(x_onehot)
        z = self.reparam(mu, logvar)
        logits = self.dec(z)
        return {"mu": mu, "logvar": logvar, "z": z, "logits": logits}

def vae_loss(x_onehot: torch.Tensor, logits: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Standard VAE loss:
      recon = cross-entropy over tokens
      KL = KL(q(z|x) || N(0,I))
    """
    # target token indices: [N, L]
    target = x_onehot.argmax(dim=-1)
    recon = F.cross_entropy(logits.reshape(-1, VOCAB), target.reshape(-1), reduction="mean")

    # KL per sample, mean
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
    return recon + 0.01 * kl  # KL weight; tune / warmup if needed

# ============================================================
# Deep-kernel SVGP: GP sees z = encoder(x) as its input features
# We train it jointly with VAE on labeled subset.
# ============================================================

class LatentSVGP(ApproximateGP):
    """
    Variational GP on latent space z (dimension = LATENT_DIM).
    This model expects inputs z directly.
    We will feed z = encoder(x) for labeled points, and backprop to encoder.
    """
    def __init__(self, inducing_points: torch.Tensor):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=inducing_points.size(-1))
        )

    def forward(self, x):
        mean = self.mean_module(x)
        cov = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, cov)

# ============================================================
# Black-box (you must connect yours here)
# ============================================================

def black_box_objectives(peptides: List[str]) -> torch.Tensor:
    """
    Replace this with your imported blackbox, e.g. from black_box_fcn_mo_MN.py.

    Must return tensor [N, M], higher is better for each objective.
    """
    # ----- PLACEHOLDER: random objectives -----
    # Example: 3 objectives
    y = torch.randn(len(peptides), 3)
    return y

# ============================================================
# Helper: build initial labeled set
# ============================================================

def pick_initial_labeled_indices(n_total: int, n_init: int = 128, seed: int = 0) -> List[int]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=min(n_init, n_total), replace=False)
    return idx.tolist()

# ============================================================
# Training step: joint VAE + SVGP
# ============================================================

def train_one_epoch_joint(
    vae: VAE,
    gp: LatentSVGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    optimizer: torch.optim.Optimizer,
    x_all: torch.Tensor,                 # [N_all, L, V]
    labeled_idx: List[int],
    y_labeled: torch.Tensor,             # [N_lab]
    batch_size: int = 256,
):
    vae.train()
    gp.train()
    likelihood.train()

    n_all = x_all.size(0)
    # create minibatches over all X, but SVGP loss only uses labeled points inside the batch
    perm = torch.randperm(n_all, device=x_all.device)

    # SVGP objective
    mll = VariationalELBO(likelihood, gp, num_data=len(labeled_idx))

    labeled_set = set(labeled_idx)
    labeled_pos = {i: k for k, i in enumerate(labeled_idx)}  # map global idx -> pos in y_labeled

    for start in range(0, n_all, batch_size):
        optimizer.zero_grad()

        idx = perm[start : start + batch_size]
        xb = x_all[idx]  # [B, L, V]

        out = vae(xb)
        loss_vae = vae_loss(xb, out["logits"], out["mu"], out["logvar"])

        # Which items in this batch are labeled?
        batch_global = idx.detach().cpu().tolist()
        mask = [i in labeled_set for i in batch_global]

        loss_svgp = torch.tensor(0.0, device=x_all.device)
        if any(mask):
            # Extract labeled points from batch
            lab_global = [i for i, m in zip(batch_global, mask) if m]
            lab_local = [labeled_pos[i] for i in lab_global]

            # z features come from encoder => deep kernel, gradients flow to encoder
            mu_lab, _ = vae.enc(x_all[torch.tensor(lab_global, device=x_all.device)])
            yb = y_labeled[torch.tensor(lab_local, device=x_all.device)]

            # Variational GP expects 1D regression targets
            pred = gp(mu_lab)
            loss_svgp = -mll(pred, yb)

        # total joint loss
        loss = loss_vae + 1.0 * loss_svgp
        loss.backward()
        optimizer.step()

# ============================================================
# BO models: fit independent GPs for each objective on latent z (no joint training here)
# This part uses BoTorch to compute qEHVI.
# ============================================================

def fit_mo_models(Z: torch.Tensor, Y: torch.Tensor) -> ModelListGP:
    """
    Z: [N, d]
    Y: [N, M]
    returns ModelListGP of M independent SingleTaskGPs
    """
    models = []
    M = Y.size(-1)
    for m in range(M):
        y_m = Y[:, m : m + 1]
        gp_m = SingleTaskGP(Z, y_m, outcome_transform=Standardize(m=1))
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp_m.likelihood, gp_m)
        fit_gpytorch_mll(mll)
        models.append(gp_m)
    return ModelListGP(*models)

# ============================================================
# Trust region + recentering
# ============================================================

@torch.no_grad()
def encode_peptides_to_latent(vae: VAE, x_onehot: torch.Tensor) -> torch.Tensor:
    vae.eval()
    mu, _ = vae.enc(x_onehot)
    return mu

def make_tr_bounds(center: torch.Tensor, radius: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Linf trust region bounds in latent space.
    center: [d]
    returns lower, upper each [d]
    """
    lower = center - radius
    upper = center + radius
    return lower, upper

# ============================================================
# Main: load data, initialize, run BO loop
# ============================================================

def main():
    # ---- Load peptides ----
    df = pd.read_csv("metalpdb_binding_windows_len10_MN.csv")
    peptides = df["peptide_len10"].astype(str).tolist()

    # ---- Build X (one-hot) ----
    X_all = onehot_encode_peptides(peptides).to(DEVICE)

    # ---- Initialize labeled subset ----
    labeled_idx = pick_initial_labeled_indices(len(peptides), n_init=256, seed=0)
    y_init = black_box_objectives([peptides[i] for i in labeled_idx]).to(DEVICE)  # [N_lab, M]
    M = y_init.size(-1)

    # We'll store labeled set as:
    #   labeled_idx: indices into peptides
    #   Y_lab: objective tensor aligned with labeled_idx order
    Y_lab = y_init.clone()

    # ---- Initialize VAE + SVGP (single-objective for joint training)
    # For joint training, pick ONE supervised target (or a scalarization).
    # You can:
    #   - use objective 0 only
    #   - or weighted sum
    sup_target = Y_lab[:, 0].contiguous()   # [N_lab]

    vae = VAE(latent_dim=LATENT_DIM).to(DEVICE)

    # Inducing points: sample random peptides, encode, use their mu
    with torch.no_grad():
        init_mu = encode_peptides_to_latent(vae, X_all[torch.tensor(labeled_idx, device=DEVICE)])
    inducing = init_mu[: min(64, init_mu.size(0))].clone()

    gp = LatentSVGP(inducing_points=inducing).to(DEVICE)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE)

    # Joint optimizer (VAE + GP + likelihood)
    optimizer = torch.optim.Adam(
        list(vae.parameters()) + list(gp.parameters()) + list(likelihood.parameters()),
        lr=1e-3
    )

    # ---- Pre-train a bit ----
    for ep in range(2):
        train_one_epoch_joint(vae, gp, likelihood, optimizer, X_all, labeled_idx, sup_target, batch_size=256)

    # ============================================================
    # BO loop (multi-objective, qEHVI), with recentering
    # ============================================================

    for it in range(BO_ITERS):
        print(f"\n=== BO iter {it+1}/{BO_ITERS} ===")

        # ---- Re-encode ALL currently labeled points (RECENTERING STEP) ----
        X_lab = X_all[torch.tensor(labeled_idx, device=DEVICE)]
        Z_lab = encode_peptides_to_latent(vae, X_lab)     # [N_lab, d]

        # ---- Define trust region center:
        # choose the best point under some scalarization (for center only).
        # (You can also choose Pareto-best by hypervolume contributions.)
        scalar = Y_lab.mean(dim=-1)                        # simple scalarization
        best_i = torch.argmax(scalar).item()
        z_center = Z_lab[best_i].detach()

        lb, ub = make_tr_bounds(z_center, TR_RADIUS)

        # ---- Fit BoTorch MO models on (Z_lab, Y_lab) ----
        # qEHVI assumes maximization
        model = fit_mo_models(Z_lab, Y_lab)

        # ---- Build qEHVI acquisition ----
        # Choose a reference point slightly below observed minima (in standardized space it’s OK to do in raw)
        ref_point = (Y_lab.min(dim=0).values - 0.1).tolist()

        partitioning = NondominatedPartitioning(ref_point=torch.tensor(ref_point, device=DEVICE), Y=Y_lab)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))

        acq = qExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
        )

        # ---- Optimize acquisition inside trust region bounds ----
        # botorch expects bounds shape [2, d]
        bounds = torch.stack([lb, ub], dim=0)

        # qEHVI batch candidates in latent space
        Z_cand, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=Q_BATCH,
            num_restarts=N_RESTARTS,
            raw_samples=RAW_SAMPLES,
            options={"batch_limit": 5, "maxiter": 200},
        )  # Z_cand: [q, d]

        # ---- Decode latent points to peptides ----
        vae.eval()
        with torch.no_grad():
            logits = vae.dec(Z_cand)  # [q, L, V]
        cand_peps = [decode_logits_to_peptide(logits[j]) for j in range(logits.size(0))]

        # ---- Evaluate black-box (multi-objective) ----
        Y_new = black_box_objectives(cand_peps).to(DEVICE)  # [q, M]

        # ---- Add to labeled set (avoid duplicates if desired) ----
        # Here we append blindly; you may want to deduplicate by peptide string.
        new_indices = []
        for pep in cand_peps:
            # if peptide already exists in dataset, reuse its index; else append as "new"
            # simplest: if exists in dataset
            if pep in peptides:
                new_indices.append(peptides.index(pep))
            else:
                # optional: treat as new sample outside dataset
                peptides.append(pep)
                X_all = torch.cat([X_all, onehot_encode_peptides([pep]).to(DEVICE)], dim=0)
                new_indices.append(len(peptides) - 1)

        labeled_idx.extend(new_indices)
        Y_lab = torch.cat([Y_lab, Y_new], dim=0)

        # Update supervised scalar target used by joint VAE+SVGP training
        sup_target = Y_lab[:, 0].contiguous()

        print("Added", len(new_indices), "new labeled points. Total labeled:", len(labeled_idx))

        # ---- Joint retraining step (VAE + SVGP) ----
        for ep in range(TRAIN_EPOCHS_PER_ITER):
            train_one_epoch_joint(
                vae, gp, likelihood, optimizer,
                x_all=X_all,
                labeled_idx=labeled_idx,
                y_labeled=sup_target,
                batch_size=256
            )

    print("\nDone.")

if __name__ == "__main__":
    main()