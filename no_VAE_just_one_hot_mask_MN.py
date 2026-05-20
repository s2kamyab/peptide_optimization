# bo_optimize_peptide_space_mo_fast.py
# Multi-objective Bayesian optimization directly in peptide space (NO VAE),
# optimized for speed while keeping the SAME core method:
#   - 4 independent GPs (ModelListGP)
#   - qEHVI acquisition (Monte Carlo)
#   - discrete candidate set (mutation heuristic)
#   - true black-box objectives via compute_objectives_mn
#
# Key speedups (do NOT change the conceptual method):
#   1) Batch black-box evaluations: evaluate Q peptides per BO step in ONE call.
#   2) Two-stage candidate screening: cheap posterior-mean scalarization reduces
#      candidates before qEHVI evaluation.
#   3) No spam prints in candidate loop + safe non-stalling candidate generation.
#   4) Caching: objective cache + feature cache.
#
# pip install torch botorch gpytorch pandas numpy
#
# Assumes:
#   - black_box_fcn_mo_MN.py provides compute_objectives_mn(batch, reverse_translate_strategy="first")
#   - metalpdb_binding_windows_len10_MN.csv includes column peptide_len10
#
# Outputs:
#   - peptide_vae_out/bo_mo_noVAE_MN.csv           (full log)
#   - peptide_vae_out/bo_mo_noVAE_MN_pareto.csv    (nondominated subset)

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

# Optional: helps GP fitting stability (often reduces optimizer work)
from botorch.models.transforms.outcome import Standardize

# Your black-box objectives
from black_box_fcn_mo_MN import compute_objectives_mn


# -----------------------------
# Config
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CSV_DATASET = "metalpdb_binding_windows_len10_MN.csv"
SEQ_COL = "peptide_len10"

OUT_DIR = "peptide_vae_out"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_LOG = os.path.join(OUT_DIR, "bo_mo_noVAE_MN.csv")
OUT_PARETO = os.path.join(OUT_DIR, "bo_mo_noVAE_MN_pareto.csv")

MAX_LEN = 10
MIN_LEN = 3

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_I = {a: i for i, a in enumerate(AA)}
VOCAB = len(AA)

# BO settings
N_INIT = 32
N_STEPS = 50

# Discrete candidate set size generated per BO step
CANDIDATES_PER_STEP = 2048

# Speed knobs (keep method; reduce overhead)
Q_BATCH = 4                # evaluate 4 new peptides per BO step (batched true eval)
SCREEN_K = 256             # qEHVI computed only on top SCREEN_K by cheap GP-mean proxy
QMC_SAMPLES = 128          # 64-256; lower = faster, still qEHVI
REF_MARGIN = 0.05          # ref_point = mins - margin

# Parent selection
ELITE_POOL = 256
ELITE_FRAC_FOR_PARENTS = 0.25
MUTATIONS_PER_CHILD = 1

# Candidate generation safety
MIN_ELITES = 64
MAX_TRIES = 250_000
ALLOW_RANDOM_FILL = True

# GP fitting budget (smaller = faster)
GP_MAXITER = 75  # 50-150 typical

RNG = random.Random(0)
np.random.seed(0)
torch.manual_seed(0)


# -----------------------------
# Utilities: sequence cleaning, mutation
# -----------------------------
def clean_seq(s: str) -> str:
    s = (s or "").strip().upper()
    s = "".join([c for c in s if c in AA])
    return s[:MAX_LEN]


def mutate_sequence(seq: str, n_mut: int = 1) -> str:
    """Length-preserving substitutions by default."""
    seq = clean_seq(seq)
    if len(seq) < 1:
        L = RNG.randint(MIN_LEN, MAX_LEN)
        return "".join(RNG.choice(AA) for _ in range(L))

    s = list(seq[:MAX_LEN])
    if len(s) < MIN_LEN:
        s = s + [RNG.choice(AA) for _ in range(MIN_LEN - len(s))]

    for _ in range(max(1, int(n_mut))):
        i = RNG.randrange(len(s))
        old = s[i]
        new = RNG.choice(AA)
        while new == old:
            new = RNG.choice(AA)
        s[i] = new
    return "".join(s)


def propose_candidates_from_pool(
    pool: List[str],
    n_candidates: int,
    n_mut: int,
    elite_frac: float,
    min_elites: int = MIN_ELITES,
    max_tries: int = MAX_TRIES,
    allow_random_fill: bool = ALLOW_RANDOM_FILL,
    log_every: int = 20_000,
) -> List[str]:
    """
    Generate unique mutated candidates from pool without stalling:
      - expands mutation radius if acceptance stalls
      - optional random fill
    """
    pool = [clean_seq(x) for x in pool if clean_seq(x)]
    pool = list(dict.fromkeys(pool))
    if not pool:
        return []

    n_elites = max(min_elites, int(len(pool) * elite_frac))
    n_elites = min(n_elites, len(pool))
    elites = pool[:n_elites]

    cand: List[str] = []
    seen = set(pool)

    tries = 0
    cur_n_mut = max(1, int(n_mut))

    while len(cand) < n_candidates and tries < max_tries:
        tries += 1
        parent = RNG.choice(elites)
        child = mutate_sequence(parent, n_mut=cur_n_mut)

        if child not in seen and MIN_LEN <= len(child) <= MAX_LEN:
            seen.add(child)
            cand.append(child)

        # throttle logging + expand radius if stalling
        if log_every and tries % log_every == 0 and len(cand) < n_candidates:
            # Increase mutation count gradually: 1 -> 2 -> 3
            cur_n_mut = min(cur_n_mut + 1, 3)

    if allow_random_fill and len(cand) < n_candidates:
        # Fill remainder with random peptides (keeps BO moving)
        while len(cand) < n_candidates:
            L = RNG.randint(MIN_LEN, MAX_LEN)
            child = "".join(RNG.choice(AA) for _ in range(L))
            child = clean_seq(child)
            if child and child not in seen:
                seen.add(child)
                cand.append(child)

    return cand


# -----------------------------
# Feature: one-hot + mask (with caching)
# -----------------------------
def onehot_and_mask(seq: str, max_len: int = MAX_LEN) -> Tuple[np.ndarray, np.ndarray]:
    seq = clean_seq(seq)[:max_len]
    x = np.zeros((max_len, VOCAB), dtype=np.float32)
    m = np.zeros((max_len,), dtype=np.float32)
    for pos, ch in enumerate(seq):
        x[pos, AA_TO_I[ch]] = 1.0
        m[pos] = 1.0
    return x, m


def featurize(seq: str) -> np.ndarray:
    x, m = onehot_and_mask(seq)
    x = x * m[:, None]
    feat = np.concatenate([x.reshape(-1), m], axis=0)
    return feat.astype(np.float32)


def featurize_many(seqs: List[str], feat_cache: Dict[str, np.ndarray]) -> np.ndarray:
    out = []
    for s in seqs:
        s = clean_seq(s)
        if s in feat_cache:
            out.append(feat_cache[s])
        else:
            v = featurize(s)
            feat_cache[s] = v
            out.append(v)
    return np.stack(out, axis=0)


# -----------------------------
# True objective evaluation (batched; same "stable ref batch" trick)
# -----------------------------
def stable_objectives_many(
    seqs: List[str],
    binding_sites: str,
    ref_candidates: List[Dict[str, Any]],
) -> Dict[str, np.ndarray]:
    """
    Evaluate MANY candidates in one compute_objectives_mn call.
    Returns dict peptide->objectives(4,). Dropped candidates get [-1,-1,-1,-1].

    NOTE: We assume compute_objectives_mn returns rows (at least) for the survivors,
    and we match by 'sequence' value in the tail corresponding to candidates.
    """
    cleaned = [clean_seq(s) for s in seqs]
    cleaned = [s for s in cleaned if s]
    cleaned = list(dict.fromkeys(cleaned))
    if not cleaned:
        return {}

    batch = list(ref_candidates) + [
        {"sequence": s, "binding_sites": binding_sites, "DNA_sequence": "", "metal": "MN"}
        for s in cleaned
    ]

    # default: penalize
    out = {s: np.array([-1.0, -1.0, -1.0, -1.0], dtype=float) for s in cleaned}

    df_obj = compute_objectives_mn(batch, reverse_translate_strategy="first")
    if df_obj is None or len(df_obj) == 0:
        return out

    # Consider only candidate-side rows:
    tail = df_obj.tail(len(cleaned)).copy()
    tail["sequence"] = tail["sequence"].astype(str).str.strip().str.upper()

    for s in cleaned:
        hit = tail[tail["sequence"] == s]
        if hit.empty:
            continue
        r = hit.iloc[-1]
        out[s] = np.array(
            [
                float(r["chelation_sub"]),
                float(r["solubility_sub"]),
                float(r["stability_sub"]),
                float(r["expression_sub"]),
            ],
            dtype=float,
        )
    return out


def is_non_dominated_max(Y: np.ndarray) -> np.ndarray:
    """O(N^2) nondominated mask for maximization."""
    n = Y.shape[0]
    nd = np.ones(n, dtype=bool)
    for i in range(n):
        if not nd[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if np.all(Y[j] >= Y[i]) and np.any(Y[j] > Y[i]):
                nd[i] = False
                break
    return nd


# -----------------------------
# MO-BO in peptide space (discrete qEHVI)
# -----------------------------
@dataclass
class EvalRow:
    iter: int
    peptide: str
    obj_chelation: float
    obj_solubility: float
    obj_stability: float
    obj_expression: float


def mo_bo_no_vae_fast(
    df_all: pd.DataFrame,
    ref_candidates: List[Dict[str, Any]],
    binding_sites: str = "",
    n_init: int = N_INIT,
    n_steps: int = N_STEPS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      df_log: all evaluated points with objectives
      df_pareto: nondominated subset of df_log
    """

    # ---- initial design from dataset
    all_seqs = df_all[SEQ_COL].astype(str).map(clean_seq)
    all_seqs = all_seqs[all_seqs.map(lambda s: MIN_LEN <= len(s) <= MAX_LEN)]
    uniq = all_seqs[all_seqs != ""].dropna().drop_duplicates().tolist()

    if len(uniq) < n_init:
        raise ValueError(f"Not enough unique peptides for init: have {len(uniq)}, need {n_init}")

    init_peps = RNG.sample(uniq, k=n_init)

    # caches
    obj_cache: Dict[str, np.ndarray] = {}
    feat_cache: Dict[str, np.ndarray] = {}

    def eval_many(peps: List[str]) -> Dict[str, np.ndarray]:
        peps = [clean_seq(p) for p in peps]
        peps = [p for p in peps if p and p not in obj_cache]
        if not peps:
            return {}
        res = stable_objectives_many(peps, binding_sites, ref_candidates)
        for p, y in res.items():
            obj_cache[p] = y
        return res

    # ---- evaluate init in one batch (fast + stable)
    eval_many(init_peps)

    log_rows: List[EvalRow] = []
    for i, p in enumerate(init_peps):
        y = obj_cache[clean_seq(p)]
        log_rows.append(
            EvalRow(
                iter=i,
                peptide=clean_seq(p),
                obj_chelation=float(y[0]),
                obj_solubility=float(y[1]),
                obj_stability=float(y[2]),
                obj_expression=float(y[3]),
            )
        )

    # tensors
    X = torch.tensor(featurize_many([r.peptide for r in log_rows], feat_cache), dtype=torch.double, device=DEVICE)
    Y = torch.tensor(
        np.stack([obj_cache[r.peptide] for r in log_rows], axis=0),
        dtype=torch.double,
        device=DEVICE,
    )  # (N,4)

    # ---- BO loop
    for t in range(n_steps):
        # Fit 4 independent GPs
        models = []
        for j in range(Y.shape[1]):
            models.append(
                SingleTaskGP(
                    X,
                    Y[:, j : j + 1],
                    outcome_transform=Standardize(m=1),
                )
            )
        model_mo = ModelListGP(*models).to(DEVICE)

        mll = SumMarginalLogLikelihood(model_mo.likelihood, model_mo)
        fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": GP_MAXITER}})

        # ref point
        y_mins = Y.min(dim=0).values
        ref_point = (y_mins - REF_MARGIN).tolist()

        partitioning = NondominatedPartitioning(
            ref_point=torch.tensor(ref_point, dtype=torch.double, device=DEVICE),
            Y=Y,
        )

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([QMC_SAMPLES]))
        acq = qExpectedHypervolumeImprovement(
            model=model_mo,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
        )

        # parent pool from evaluated elites (scalarization for parents only)
        Y_np = Y.detach().cpu().numpy()
        scalar = Y_np.mean(axis=1)
        elite_idx = np.argsort(-scalar)[: min(ELITE_POOL, len(scalar))]
        parent_pool = [log_rows[i].peptide for i in elite_idx]
        parent_pool = list(dict.fromkeys(parent_pool))

        # generate candidates
        cand_peps = propose_candidates_from_pool(
            parent_pool,
            n_candidates=CANDIDATES_PER_STEP,
            n_mut=MUTATIONS_PER_CHILD,
            elite_frac=ELITE_FRAC_FOR_PARENTS,
        )

        # filter out already evaluated
        cand_peps = [p for p in cand_peps if p not in obj_cache]
        if not cand_peps:
            print(f"[{t+1:03d}/{n_steps}] No new candidates; stopping early.")
            break

        # features
        X_cand = torch.tensor(featurize_many(cand_peps, feat_cache), dtype=torch.double, device=DEVICE)

        # ---- SCREENING: cheap proxy (posterior mean avg) -> keep top SCREEN_K
        with torch.no_grad():
            means = []
            for j in range(4):
                pj = model_mo.models[j].posterior(X_cand).mean.squeeze(-1)  # (n,)
                means.append(pj)
            mean_scalar = torch.stack(means, dim=1).mean(dim=1)  # (n,)

        k = min(SCREEN_K, X_cand.shape[0])
        top_idx = torch.topk(mean_scalar, k=k).indices
        X_small = X_cand[top_idx]
        cand_small = [cand_peps[i] for i in top_idx.detach().cpu().numpy().tolist()]

        # qEHVI on screened set
        acq_vals = acq(X_small.unsqueeze(1)).detach().cpu().numpy().reshape(-1)

        # pick top Q_BATCH to evaluate (batched true eval)
        q = min(Q_BATCH, len(cand_small))
        pick = np.argsort(-acq_vals)[:q]
        peps_next = [cand_small[i] for i in pick]

        # evaluate in ONE black-box call
        eval_many(peps_next)

        # append all newly evaluated
        new_X = torch.tensor(featurize_many(peps_next, feat_cache), dtype=torch.double, device=DEVICE)
        new_Y_np = np.stack([obj_cache[p] for p in peps_next], axis=0)
        new_Y = torch.tensor(new_Y_np, dtype=torch.double, device=DEVICE)

        X = torch.cat([X, new_X], dim=0)
        Y = torch.cat([Y, new_Y], dim=0)

        base_iter = n_init + t * Q_BATCH
        for qi, p in enumerate(peps_next):
            y = obj_cache[p]
            log_rows.append(
                EvalRow(
                    iter=base_iter + qi,
                    peptide=p,
                    obj_chelation=float(y[0]),
                    obj_solubility=float(y[1]),
                    obj_stability=float(y[2]),
                    obj_expression=float(y[3]),
                )
            )

        if (t + 1) % 5 == 0:
            nd = is_non_dominated_max(Y.detach().cpu().numpy())
            print(
                f"[step {t+1:03d}/{n_steps}] evaluated={Y.shape[0]} "
                f"pareto={int(nd.sum())} last_batch={peps_next}"
            )

    # outputs
    df_log = pd.DataFrame([r.__dict__ for r in log_rows])
    Y_all = df_log[["obj_chelation", "obj_solubility", "obj_stability", "obj_expression"]].to_numpy(dtype=float)
    nd_mask = is_non_dominated_max(Y_all)
    df_pareto = df_log.loc[nd_mask].reset_index(drop=True)
    return df_log, df_pareto


def main():
    df = pd.read_csv(CSV_DATASET).dropna(subset=[SEQ_COL]).copy()
    df[SEQ_COL] = df[SEQ_COL].astype(str).map(clean_seq)

    # Reference candidates: fixed batch for stable normalization (same idea as your latent code)
    ref_seqs = df[SEQ_COL].dropna()
    ref_seqs = ref_seqs[ref_seqs.map(lambda s: MIN_LEN <= len(s) <= MAX_LEN)]
    ref_seqs = ref_seqs[ref_seqs != ""].drop_duplicates()

    ref_sample = ref_seqs.sample(n=min(2000, len(ref_seqs)), random_state=0).tolist()
    ref = [{"sequence": s, "binding_sites": "", "DNA_sequence": "", "metal": "MN"} for s in ref_sample]

    df_log, df_pareto = mo_bo_no_vae_fast(
        df,
        ref_candidates=ref,
        binding_sites="",
        n_init=N_INIT,
        n_steps=N_STEPS,
    )

    df_log.to_csv(OUT_LOG, index=False)
    df_pareto.to_csv(OUT_PARETO, index=False)

    print(f"\nSaved full log -> {OUT_LOG}   (rows={len(df_log)})")
    print(f"Saved Pareto front -> {OUT_PARETO}   (rows={len(df_pareto)})")
    print("\nTop Pareto (first 10):")
    print(df_pareto.head(10))


if __name__ == "__main__":
    main()
