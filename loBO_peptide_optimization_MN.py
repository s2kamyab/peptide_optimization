# bo_optimize_seqvae.py
# pip install torch botorch gpytorch pandas numpy biopython

import os
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from train_VAE_GRU_MN_peptide_len10 import PeptideDataset, collate_fn, encode_to_z
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from black_box_fcn_MN import compute_final_scores_mn as compute_final_scores
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement, LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from train_VAE_GRU_MN_peptide_len10 import PeptideTokenizer, SeqVAE, encode_to_z
from torch.utils.data import Dataset, DataLoader
# -------------------------
# 1) Load your SeqVAE model definition (must match training)
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def levenshtein(a: str, b: str) -> int:
    """Classic DP Levenshtein edit distance."""
    a = (a or "").strip().upper()
    b = (b or "").strip().upper()
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    # Ensure b is shorter for slightly less memory
    if len(b) > len(a):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def find_closest_peptide_rows(
    df_all: pd.DataFrame,
    queries: List[str],
    seq_col: str = "peptide_len10",
    top_k: int = 5,
) -> pd.DataFrame:
    """
    For each query peptide, find top_k closest rows in df_all[seq_col]
    by Levenshtein distance. Returns a long dataframe.
    """
    # Pre-clean once
    df_all = df_all.copy()
    df_all[seq_col] = df_all[seq_col].astype(str).str.strip().str.upper()

    results = []
    # Optional speed-up: precompute unique sequences and map back
    uniq_seqs = df_all[seq_col].fillna("").unique().tolist()

    for qi, q in enumerate(queries):
        q = (q or "").strip().upper()
        if not q:
            continue

        # distances to unique seqs
        dists = [(s, levenshtein(q, s)) for s in uniq_seqs if s]
        dists.sort(key=lambda x: x[1])
        best_seqs = dists[: max(1, top_k)]

        # pull matching rows for those best sequences
        # for s, dist in best_seqs:
        #     # hit_rows = df_all[df_all[seq_col] == s].copy()
        #     hit_rows = df_all[df_all[seq_col] == s].head(1).copy()  # keep just one row per peptide
        #     # If multiple rows have same peptide, keep them all (or .head(1) if you prefer)
        #     hit_rows.insert(0, "query_peptide", q)
        #     hit_rows.insert(1, "edit_distance", dist)
        #     hit_rows.insert(2, "rank_within_query", None)  # fill later
        #     results.append(hit_rows)

        for rank, (s, dist) in enumerate(best_seqs, start=1):
            # keep ONLY one representative row per matched peptide string
            hit_row = df_all[df_all[seq_col] == s].head(1).copy()

            if hit_row.empty:
                continue

            hit_row.insert(0, "query_peptide", q)
            hit_row.insert(1, "edit_distance", dist)
            hit_row.insert(2, "rank_within_query", rank)

            results.append(hit_row)


    if not results:
        return pd.DataFrame()

    out = pd.concat(results, ignore_index=True)

    # rank within each query by (distance, then peptide string)
    out["rank_within_query"] = (
        out.sort_values(["query_peptide", "edit_distance", seq_col])
           .groupby("query_peptide")
           .cumcount() + 1
    )
    return out.sort_values(["query_peptide", "edit_distance", "rank_within_query"]).reset_index(drop=True)



@torch.no_grad()
def z_to_peptides(model: SeqVAE, tok: PeptideTokenizer, z: torch.Tensor, max_pep_len: int = 10, min_len: int = 3) -> List[str]:

# def z_to_peptides(model: SeqVAE, tok: PeptideTokenizer, z: torch.Tensor, max_pep_len: int = 10) -> List[str]:
    model.eval()
    z = z.to(next(model.parameters()).device)
    B = z.size(0)

    x = torch.full((B, 1), tok.BOS, dtype=torch.long, device=z.device)
    h = model._init_dec_hidden(z)
    z_bias = model.z_to_emb(z)  # (B, E)

    out_ids = []
    max_steps = max_pep_len + 1
    # for _ in range(max_steps):
    for step in range(max_steps):
        emb_tok = model.emb(x[:, -1:])
        emb_tok = model.emb_drop(emb_tok + z_bias.unsqueeze(1))
        o, h = model.dec_gru(emb_tok, h)
        logits = model.out(model.dec_drop(o[:, 0, :]))
        # block EOS too early so decode() doesn't return ""
        if step < int(min_len):
            logits[:, tok.EOS] = -1e9

        nxt = torch.argmax(logits, dim=-1)
        out_ids.append(nxt)
        x = torch.cat([x, nxt.unsqueeze(1)], dim=1)
    
    out_mat = torch.stack(out_ids, dim=1)
    return [tok.decode(out_mat[b].tolist(), stop_at_eos=True) for b in range(B)]


# -------------------------
# 2) Stable objective: normalize vs fixed reference set
# -------------------------
# Re-use your compute_final_scores (paste your function here).
from typing import Sequence, Union, Optional
# <<< paste compute_final_scores(...) from earlier section >>>

# def stable_score(seq: str, binding_sites: str, ref_candidates: List[Dict[str, Any]]) -> float:
#     """
#     To avoid batch-dependence: score candidate together with a fixed reference batch,
#     then return the candidate's score (the last element).
#     """
#     batch = list(ref_candidates) + [{"sequence": seq, "binding_sites": binding_sites, "DNA_sequence": "", "metal": "MN"}]
#     scores = compute_final_scores(batch, reverse_translate_strategy="first")
#     if not scores:
#         return 0.0
#     return float(scores[-1])

def stable_score(seq: str, binding_sites: str, ref_candidates: List[Dict[str, Any]]) -> float:
    """
    Score candidate together with a fixed reference batch, but ensure we are
    actually returning the candidate score (candidate may be dropped if invalid).
    """
    seq_norm = (seq or "").strip().upper()

    batch = list(ref_candidates) + [{
        "sequence": seq_norm,
        "binding_sites": binding_sites,
        "DNA_sequence": "",
        "metal": "MN",
    }]

    scores, df = compute_final_scores(
        batch,
        reverse_translate_strategy="first",
        return_details=True,
    )

    if df is None or len(df) == 0:
        return 0.0

    # Candidate must survive filtering; if dropped, penalize
    if str(df["sequence"].iloc[-1]).strip().upper() != seq_norm:
        return 0.0

    return float(df["final_score"].iloc[-1])



# -------------------------
# 3) BO loop in z-space (EI)
# -------------------------
def bayesopt_in_latent(
    model: SeqVAE,
    tok: PeptideTokenizer,
    ref_candidates: List[Dict[str, Any]],
    binding_sites: str = "",          # if you want fixed label pattern, pass it here
    n_init: int = 32,
    n_steps: int = 50,
    bounds_val: float = 3.0,          # z search box [-3, 3] not used
    seed: int = 0,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    z_dim = model.z_dim
    # Estimate posterior support from reference candidates (or training set if you have it)
    # NOTE: ref_candidates are sequences; we need a dataset/loader to encode properly.
    # Minimal workaround: build a tiny dataset from a sample of ref candidates and encode to mu.

    sample_seqs = [c["sequence"] for c in ref_candidates[:2000] if c.get("sequence")]
    df_tmp = pd.DataFrame({"peptide_len10": sample_seqs})
    ds_tmp = PeptideDataset(df_tmp, tok, max_len=10)
    loader_tmp = DataLoader(ds_tmp, batch_size=256, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, tok.PAD), num_workers=0)

    Z_mu = encode_to_z(model, loader_tmp, use_mean=True)  # (N, z_dim)
    z_mean = torch.tensor(Z_mu.mean(axis=0), dtype=torch.float32, device=DEVICE)
    z_std  = torch.tensor(Z_mu.std(axis=0) + 1e-6, dtype=torch.float32, device=DEVICE)
    # Search bounds around posterior mean +/- k std
    k = 2.5
    lower = (z_mean - k * z_std).double()
    upper = (z_mean + k * z_std).double()
    bounds = torch.stack([lower, upper], dim=0).to(DEVICE)

    # Init points near posterior support (not uniform in [-3,3])
    eps = torch.randn(n_init, z_dim, device=DEVICE)
    Z = (z_mean + k * z_std * eps).float()

    # bounds = torch.tensor([[-bounds_val] * z_dim, [bounds_val] * z_dim], dtype=torch.double, device=DEVICE)

    # # --- initial design ---
    # Z = (torch.rand(n_init, z_dim, device=DEVICE) * 2 - 1) * bounds_val  # uniform in box
    # peps = z_to_peptides(model, tok, Z.float(), max_pep_len=10)
    peps = z_to_peptides(model, tok, Z.float(), max_pep_len=10, min_len=3)


    y = []
    for p in peps:
        y.append(stable_score(p, binding_sites, ref_candidates))
    Y = torch.tensor(y, dtype=torch.double, device=DEVICE).unsqueeze(-1)

    log_rows = []
    for i in range(n_init):
        log_rows.append({"iter": i, "peptide": peps[i], "score": float(Y[i].item())})

    # --- BO iterations ---
    for t in range(n_steps):
        # Fit GP
        train_X = Z.double()
        train_Y = Y
        gp = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        best_f = train_Y.max()
        # acq = ExpectedImprovement(gp, best_f=best_f)
        acq = LogExpectedImprovement(gp, best_f=best_f)

        # Optimize EI
        z_next, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=256,
        )
        z_next = z_next.detach()

        pep_next = z_to_peptides(model, tok, z_next.float(), max_pep_len=10)[0]
        y_next = stable_score(pep_next, binding_sites, ref_candidates)

        # append
        Z = torch.cat([Z, z_next.float()], dim=0)
        Y = torch.cat([Y, torch.tensor([[y_next]], dtype=torch.double, device=DEVICE)], dim=0)

        log_rows.append({"iter": n_init + t, "peptide": pep_next, "score": float(y_next)})

        # optional: print progress
        if (t + 1) % 5 == 0:
            print(f"[{t+1:03d}/{n_steps}] best={float(Y.max().item()):.4f}  last={y_next:.4f}  pep={pep_next}")

    df_log = pd.DataFrame(log_rows).sort_values("score", ascending=False).reset_index(drop=True)
    best = df_log.iloc[0].to_dict()
    return best, df_log


if __name__ == "__main__":
    # Load checkpoint
    ckpt_path = "peptide_vae_out/seqvae_best.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    tok = PeptideTokenizer()
    model = SeqVAE(vocab_size=len(tok.vocab), pad_id=tok.PAD, emb_dim=128, hid_dim=256, z_dim=64, num_layers=2, dropout=0.2).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Build reference candidates from your training CSV (sample e.g. 2000)
    csv_path = "metalpdb_binding_windows_len10_MN.csv"
    df = pd.read_csv(csv_path).dropna(subset=["peptide_len10"]).copy()
    df["peptide_len10"] = df["peptide_len10"].astype(str).str.strip().str.upper()
    ref = [{"sequence": s, "binding_sites": "", "DNA_sequence": "", "metal": "MN"} for s in df["peptide_len10"].sample(n=min(2000, len(df)), random_state=0).tolist()]

    best, log = bayesopt_in_latent(model, tok, ref_candidates=ref, binding_sites="", n_init=32, n_steps=50)
    print("\nBEST FOUND:")
    print(best)
    print("\nTop-10:")
    print(log.head(10))
    log.to_csv("bo_optimized_peptides.csv", index=False)
        # -------------------------
    # 4) Find closest dataset rows to top-10 BO peptides
    # -------------------------
    top10_peps = log.head(10)["peptide"].astype(str).str.strip().str.upper().tolist()

    # Load full dataset (keep all columns so you get full row context)
    df_all = pd.read_csv(csv_path).copy()

    closest = find_closest_peptide_rows(
        df_all=df_all,
        queries=top10_peps,
        seq_col="peptide_len10",
        top_k=5,   # closest 5 rows per BO peptide (change if you want)
    )
    for q in top10_peps:
        sub = closest[closest["query_peptide"] == q]
        print("\n", q)
        print(sub[["query_peptide", "edit_distance", "rank_within_query", "peptide_len10"]])


    out_path = "bo_top10_closest_rows.csv"
    closest.to_csv(out_path, index=False)

    print(f"\nSaved closest dataset rows for top-10 peptides -> {out_path}")
    if len(closest) > 0:
        print(closest[["query_peptide", "edit_distance", "rank_within_query", "peptide_len10"]].head(25))
