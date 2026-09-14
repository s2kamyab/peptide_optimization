from __future__ import annotations

"""
Audit matched to:
    pretrain_peptide_direct_sequence_diffusion_chain_mapped_parts_h32_leakage_safe.py

The H32 framework uses raw 200-D diffusion base noise epsilon constrained to
||epsilon||_2 = sqrt(200) as the intended search/BO coordinate. It does NOT
contain DDIM inversion or PCA z_bo. Therefore this audit intentionally keeps the
H32 framework unchanged and checks:

1) exact-sequence leakage after reproducing the pretrainer's conflict-exclusion rule,
2) optional group leakage after exact-sequence filtering,
3) train-validation and train-test near-neighbor leakage (Hamming distance),
4) generation diversity and exact memorization from spherical epsilon samples,
5) geometry of sampled spherical epsilon coordinates,
6) local smoothness directly in spherical epsilon space using DDIM decoding,
7) soft-output smoothness using token-wise Jensen-Shannon divergence.

These tests establish split integrity and generator/search-coordinate locality,
not Cu-objective smoothness.
"""

import argparse
import glob
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_I = {a: i for i, a in enumerate(AA)}
I_TO_AA = {i: a for i, a in enumerate(AA)}
SEQ_LEN = 10
VOCAB = 20
EPS_DIM = SEQ_LEN * VOCAB
EPS_RADIUS = math.sqrt(EPS_DIM)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_peptide(x) -> Optional[str]:
    p = str(x).strip().upper()
    if len(p) != SEQ_LEN or any(a not in AA_TO_I for a in p):
        return None
    return p


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[j-1] + 1, prev[j] + 1, prev[j-1] + int(ca != cb)))
        prev = cur
    return int(prev[-1])


def import_training_module(path: str):
    name = "h32_direct_sequence_diffusion_training_module"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def torch_load_full(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_model(module, checkpoint: str, device: torch.device):
    ckpt = torch_load_full(checkpoint, device)
    expected = "all_metal_direct_peptide_sequence_diffusion_search_cost_sampling"
    ctype = ckpt.get("checkpoint_type", "")
    if ctype and ctype != expected:
        raise ValueError(f"Unexpected checkpoint_type={ctype!r}; expected {expected!r}")
    c = ckpt.get("diffusion_config", {})
    cfg = module.SequenceDiffusionConfig(
        hidden_size=int(c.get("hidden_size", 32)),
        n_layers=int(c.get("n_layers", 2)),
        dropout=float(c.get("dropout", 0.0)),
        time_dim=int(c.get("time_dim", 32)),
        train_steps=int(c.get("train_steps", 100)),
        beta_start=float(c.get("beta_start", 1e-4)),
        beta_end=float(c.get("beta_end", 2e-2)),
    )
    model = module.DirectSequenceDiffusion(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    if int(model.bo_dim) != EPS_DIM:
        raise ValueError(f"Expected BO dimension {EPS_DIM}, got {model.bo_dim}")
    return model, ckpt, cfg


def discover_files(parts_dir: str, pattern: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(parts_dir, pattern)))
    files = [f for f in files if "failures" not in os.path.basename(f).lower()]
    if not files:
        raise FileNotFoundError(os.path.join(parts_dir, pattern))
    return files


def build_safe_split_map(files, peptide_col, split_col, chunksize, allowed_splits):
    allowed = {str(s).strip().lower() for s in allowed_splits}
    peptide_splits: Dict[str, set] = {}
    n_valid_rows = 0
    for file_i, path in enumerate(files, 1):
        header = pd.read_csv(path, nrows=0).columns
        if peptide_col not in header or split_col not in header:
            raise KeyError(f"{path} must contain {peptide_col!r} and {split_col!r}")
        for chunk in pd.read_csv(path, usecols=[peptide_col, split_col], chunksize=int(chunksize)):
            for p_raw, s_raw in zip(chunk[peptide_col].tolist(), chunk[split_col].tolist()):
                pep = clean_peptide(p_raw)
                if pep is None:
                    continue
                split = str(s_raw).strip().lower()
                if split not in allowed:
                    continue
                peptide_splits.setdefault(pep, set()).add(split)
                n_valid_rows += 1
        if file_i % 10 == 0 or file_i == len(files):
            n_conf = sum(len(v) > 1 for v in peptide_splits.values())
            print(f"[SAFE-SPLIT] scanned {file_i}/{len(files)} files; unique peptides={len(peptide_splits)}; conflicts={n_conf}")

    safe_map, conflict_rows = {}, []
    for pep, splits in peptide_splits.items():
        ss = sorted(splits)
        if len(ss) == 1:
            safe_map[pep] = ss[0]
        else:
            conflict_rows.append({"peptide": pep, "n_splits": len(ss), "splits": "|".join(ss)})
    conflicts = pd.DataFrame(conflict_rows, columns=["peptide", "n_splits", "splits"])
    split_counts = {s: 0 for s in allowed}
    for s in safe_map.values():
        split_counts[s] += 1
    audit = {
        "source_valid_rows_scanned": int(n_valid_rows),
        "unique_exact_peptides_seen": int(len(peptide_splits)),
        "safe_unique_peptides": int(len(safe_map)),
        "excluded_cross_split_exact_peptides": int(len(conflicts)),
        "safe_split_counts": {k: int(v) for k, v in sorted(split_counts.items())},
        "policy": "retain peptide only when assigned to one unique split; exclude cross-split exact conflicts",
        "random_resplitting_used": False,
    }
    return safe_map, conflicts, audit


def split_sets(safe_map, train_name, val_name, test_name):
    tn, vn, sn = train_name.lower(), val_name.lower(), test_name.lower()
    return {
        "train": {p for p, s in safe_map.items() if s == tn},
        "validation": {p for p, s in safe_map.items() if s == vn},
        "test": {p for p, s in safe_map.items() if s == sn},
    }


def load_safe_rows_for_group_audit(files, safe_map, peptide_col, group_cols, chunksize, max_rows):
    rows, total = [], 0
    for path in files:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        present = [c for c in group_cols if c in header and c != peptide_col]
        usecols = [peptide_col] + present
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=int(chunksize)):
            chunk = chunk.copy()
            chunk["peptide"] = chunk[peptide_col].map(clean_peptide)
            chunk = chunk[chunk["peptide"].notna()]
            if chunk.empty:
                continue
            chunk["split"] = chunk["peptide"].map(safe_map)
            chunk = chunk[chunk["split"].notna()]
            if chunk.empty:
                continue
            rows.append(chunk)
            total += len(chunk)
            if max_rows > 0 and total >= max_rows:
                break
        if max_rows > 0 and total >= max_rows:
            break
    if not rows:
        return pd.DataFrame(columns=["peptide", "split"])
    df = pd.concat(rows, ignore_index=True)
    return df.iloc[:max_rows].copy() if max_rows > 0 else df


def group_leakage_audit(df, group_cols, train_name, val_name, test_name):
    out = {}
    tn, vn, sn = train_name.lower(), val_name.lower(), test_name.lower()
    for col in group_cols:
        if col not in df.columns:
            continue
        tmp = df[df[col].notna()]
        if tmp.empty:
            continue
        grouped = tmp.groupby(col)["split"].agg(lambda x: set(map(str, x)))
        tv = grouped.map(lambda s: tn in s and vn in s)
        tt = grouped.map(lambda s: tn in s and sn in s)
        vt = grouped.map(lambda s: vn in s and sn in s)
        out[col] = {
            "n_groups": int(len(grouped)),
            "n_groups_train_validation": int(tv.sum()),
            "fraction_groups_train_validation": float(tv.mean()),
            "n_groups_train_test": int(tt.sum()),
            "fraction_groups_train_test": float(tt.mean()),
            "n_groups_validation_test": int(vt.sum()),
            "fraction_groups_validation_test": float(vt.mean()),
        }
    return out


def encode_int(peptides):
    return np.asarray([[AA_TO_I[a] for a in p] for p in peptides], dtype=np.int8)


def nearest_hamming(reference, query, reference_sample, query_sample, seed):
    rng = np.random.default_rng(seed)
    ref = np.asarray(sorted(reference), dtype=object)
    qry = np.asarray(sorted(query), dtype=object)
    if reference_sample > 0 and len(ref) > reference_sample:
        ref = rng.choice(ref, reference_sample, replace=False)
    if query_sample > 0 and len(qry) > query_sample:
        qry = rng.choice(qry, query_sample, replace=False)
    if len(ref) == 0 or len(qry) == 0:
        return np.asarray([], dtype=int)
    ref_arr, qry_arr = encode_int(ref), encode_int(qry)
    nearest = []
    for q in qry_arr:
        best = SEQ_LEN + 1
        for s in range(0, len(ref_arr), 20000):
            d = (ref_arr[s:s+20000] != q[None, :]).sum(axis=1)
            best = min(best, int(d.min()))
            if best == 0:
                break
        nearest.append(best)
    return np.asarray(nearest, dtype=int)


def hamming_summary(nearest):
    if len(nearest) == 0:
        return {"n_query_sampled": 0}
    return {
        "n_query_sampled": int(len(nearest)),
        "nearest_hamming_mean": float(nearest.mean()),
        "nearest_hamming_median": float(np.median(nearest)),
        "fraction_hamming_0": float(np.mean(nearest == 0)),
        "fraction_hamming_le_1": float(np.mean(nearest <= 1)),
        "fraction_hamming_le_2": float(np.mean(nearest <= 2)),
        "fraction_hamming_le_3": float(np.mean(nearest <= 3)),
    }


def random_sphere(n, dim, radius, device):
    eps = torch.randn(n, dim, device=device)
    return eps / eps.norm(dim=-1, keepdim=True).clamp_min(1e-8) * float(radius)


def project_sphere(x, radius):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8) * float(radius)


def decode_with_probs(x):
    probs = torch.softmax(x, dim=-1)
    idx = x.argmax(dim=-1).detach().cpu().tolist()
    peps = ["".join(I_TO_AA[int(i)] for i in row) for row in idx]
    return peps, probs


def js_divergence(p, q, eps=1e-8):
    p, q = p.clamp_min(eps), q.clamp_min(eps)
    m = 0.5 * (p + q)
    return 0.5 * ((p * (p.log() - m.log())).sum(-1) + (q * (q.log() - m.log())).sum(-1))


def coordinate_geometry(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        return {}
    xc = x - x.mean(0, keepdims=True)
    cov = (xc.T @ xc) / max(1, len(x) - 1)
    eig = np.linalg.eigvalsh(cov).clip(min=0.0)
    total = max(float(eig.sum()), eps)
    pr = total * total / max(float((eig**2).sum()), eps)
    frac = eig[::-1] / total
    cum = np.cumsum(frac)
    z = xc / np.maximum(xc.std(0, keepdims=True), eps)
    corr = (z.T @ z) / len(z)
    off = corr - np.diag(np.diag(corr))
    denom = max(1, x.shape[1] * (x.shape[1] - 1))
    norms = np.linalg.norm(x, axis=1)
    return {
        "dimension": int(x.shape[1]),
        "radius_target": float(EPS_RADIUS),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "effective_dim_pr": float(pr),
        "pc1_variance_fraction": float(frac[0]),
        "pcs90": int(np.searchsorted(cum, 0.90) + 1),
        "pcs95": int(np.searchsorted(cum, 0.95) + 1),
        "mean_abs_offdiag_corr": float(np.abs(off).sum() / denom),
        "max_abs_offdiag_corr": float(np.abs(off).max()),
    }


@torch.no_grad()
def generation_audit(model, n_samples, batch_size, ddim_steps, device, train_set, val_set, test_set):
    rows, eps_chunks = [], []
    for start in range(0, n_samples, batch_size):
        n = min(batch_size, n_samples - start)
        eps = random_sphere(n, EPS_DIM, EPS_RADIUS, device)
        x0 = model.ddim_sample(eps, inference_steps=ddim_steps)
        peps, probs = decode_with_probs(x0)
        ef = eps.cpu().numpy().astype(np.float32)
        eps_chunks.append(ef)
        conf = probs.max(dim=-1).values.mean(dim=-1).cpu().numpy()
        for i, p in enumerate(peps):
            rows.append({
                "sample_index": start + i,
                "peptide": p,
                "epsilon_norm": float(np.linalg.norm(ef[i])),
                "mean_token_confidence": float(conf[i]),
                "exact_match_train": int(p in train_set),
                "exact_match_validation": int(p in val_set),
                "exact_match_test": int(p in test_set),
            })
    df = pd.DataFrame(rows)
    eps_np = np.concatenate(eps_chunks, axis=0)
    counts = df["peptide"].value_counts()
    summary = {
        "n_samples": int(len(df)),
        "n_unique_generated": int(df["peptide"].nunique()),
        "unique_fraction": float(df["peptide"].nunique() / max(1, len(df))),
        "dominant_fraction": float(counts.iloc[0] / len(df)) if len(df) else float("nan"),
        "exact_train_match_fraction": float(df["exact_match_train"].mean()),
        "exact_validation_match_fraction": float(df["exact_match_validation"].mean()),
        "exact_test_match_fraction": float(df["exact_match_test"].mean()),
        "mean_token_confidence": float(df["mean_token_confidence"].mean()),
        "epsilon_geometry": coordinate_geometry(eps_np),
    }
    return df, summary, eps_np


@torch.no_grad()
def local_smoothness(model, n_centers, sigmas, neighbors, ddim_steps, seed, device):
    set_seed(seed)
    centers = random_sphere(n_centers, EPS_DIM, EPS_RADIUS, device)
    xc = model.ddim_sample(centers, inference_steps=ddim_steps)
    center_peps, center_probs = decode_with_probs(xc)
    rows = []
    for sigma in sigmas:
        for i in range(n_centers):
            ec = centers[i:i+1]
            for k in range(neighbors):
                delta = torch.randn_like(ec) * float(sigma)
                en = project_sphere(ec + delta, EPS_RADIUS)
                xn = model.ddim_sample(en, inference_steps=ddim_steps)
                pn, probn = decode_with_probs(xn)
                cos = F.cosine_similarity(ec, en, dim=-1).clamp(-1.0, 1.0)
                ed = levenshtein(center_peps[i], pn[0])
                rows.append({
                    "center_index": i,
                    "center_peptide": center_peps[i],
                    "sigma": float(sigma),
                    "neighbor_index": k,
                    "raw_perturbation_l2": float(delta.norm().cpu()),
                    "spherical_epsilon_l2": float((en - ec).norm().cpu()),
                    "angular_distance_radians": float(torch.acos(cos).cpu()),
                    "neighbor_peptide": pn[0],
                    "sequence_edit": int(ed),
                    "identical": int(ed == 0),
                    "mean_token_js_divergence": float(js_divergence(center_probs[i:i+1], probn).mean().cpu()),
                })
    d = pd.DataFrame(rows)
    sm = d.groupby("sigma").agg(
        n=("sequence_edit", "size"),
        raw_perturbation_l2_mean=("raw_perturbation_l2", "mean"),
        spherical_epsilon_l2_mean=("spherical_epsilon_l2", "mean"),
        spherical_epsilon_l2_median=("spherical_epsilon_l2", "median"),
        angular_distance_mean=("angular_distance_radians", "mean"),
        sequence_edit_mean=("sequence_edit", "mean"),
        sequence_edit_median=("sequence_edit", "median"),
        identical_fraction=("identical", "mean"),
        js_mean=("mean_token_js_divergence", "mean"),
        js_median=("mean_token_js_divergence", "median"),
    ).reset_index()
    g = {}
    if len(sm) >= 2:
        x = sm["spherical_epsilon_l2_mean"].to_numpy()
        ye = sm["sequence_edit_mean"].to_numpy()
        yj = sm["js_mean"].to_numpy()
        g = {
            "across_sigma_pearson_epsilon_l2_vs_edit": float(np.corrcoef(x, ye)[0,1]),
            "across_sigma_spearman_epsilon_l2_vs_edit": float(pd.Series(x).corr(pd.Series(ye), method="spearman")),
            "across_sigma_pearson_epsilon_l2_vs_js": float(np.corrcoef(x, yj)[0,1]),
            "across_sigma_spearman_epsilon_l2_vs_js": float(pd.Series(x).corr(pd.Series(yj), method="spearman")),
        }
    return d, sm, g


def save_plots(sm, nearest_val, nearest_test, out_dir):
    if len(sm):
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(sm.spherical_epsilon_l2_mean, sm.sequence_edit_mean, marker="o")
        ax.set_xlabel("Mean spherical epsilon perturbation L2")
        ax.set_ylabel("Mean decoded sequence edit distance")
        ax.set_title("H32 spherical epsilon perturbation vs sequence change")
        ax.grid(True, alpha=.25)
        fig.tight_layout(); fig.savefig(out_dir/"spherical_epsilon_l2_vs_sequence_edit.png", dpi=300); plt.close(fig)

        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(sm.spherical_epsilon_l2_mean, sm.js_mean, marker="o")
        ax.set_xlabel("Mean spherical epsilon perturbation L2")
        ax.set_ylabel("Mean token Jensen-Shannon divergence")
        ax.set_title("H32 spherical epsilon perturbation vs soft output change")
        ax.grid(True, alpha=.25)
        fig.tight_layout(); fig.savefig(out_dir/"spherical_epsilon_l2_vs_output_js.png", dpi=300); plt.close(fig)

    for name, arr in [("validation", nearest_val), ("test", nearest_test)]:
        if len(arr):
            fig, ax = plt.subplots(figsize=(8,5))
            ax.hist(arr, bins=np.arange(-.5, SEQ_LEN+1.5, 1))
            ax.set_xlabel("Nearest sampled train Hamming distance")
            ax.set_ylabel(f"{name.title()} peptide count")
            ax.set_title(f"Train-{name} near-duplicate audit")
            fig.tight_layout(); fig.savefig(out_dir/f"train_{name}_nearest_hamming.png", dpi=300); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--training-script", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--parts-dir", required=True)
    p.add_argument("--file-pattern", default="metalpdb_ALL_chain_mapped_len10_high_confidence_part_*.csv")
    p.add_argument("--peptide-col", default="peptide_len10")
    p.add_argument("--split-col", default="split")
    p.add_argument("--train-split", default="train")
    p.add_argument("--validation-split", default="validation")
    p.add_argument("--test-split", default="test")
    p.add_argument("--split-audit-chunksize", type=int, default=100000)
    p.add_argument("--group-cols", nargs="*", default=["pdb_id","pdb","pdbid","structure_id","chain_id","chain","protein_id","uniprot_id","source_pdb","source_chain"])
    p.add_argument("--max-group-audit-rows", type=int, default=0)
    p.add_argument("--near-val-sample", type=int, default=500)
    p.add_argument("--near-test-sample", type=int, default=500)
    p.add_argument("--near-train-sample", type=int, default=100000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ddim-steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--generation-samples", type=int, default=2048)
    p.add_argument("--n-centers", type=int, default=128)
    p.add_argument("--sigmas", type=float, nargs="+", default=[.01,.025,.05,.10,.20])
    p.add_argument("--neighbors-per-sigma", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="direct_sequence_diffusion_h32_spherical_epsilon_audit")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    module = import_training_module(args.training_script)
    model, ckpt, cfg = load_model(module, args.checkpoint, device)
    files = discover_files(args.parts_dir, args.file_pattern)

    safe_map, conflicts, split_audit = build_safe_split_map(
        files, args.peptide_col, args.split_col, args.split_audit_chunksize,
        [args.train_split, args.validation_split, args.test_split]
    )
    conflicts.to_csv(out_dir/"cross_split_exact_peptide_conflicts.csv", index=False)
    sets = split_sets(safe_map, args.train_split, args.validation_split, args.test_split)
    exact_audit = {
        "n_train_unique": len(sets["train"]),
        "n_validation_unique": len(sets["validation"]),
        "n_test_unique": len(sets["test"]),
        "exact_train_validation_overlap": len(sets["train"] & sets["validation"]),
        "exact_train_test_overlap": len(sets["train"] & sets["test"]),
        "exact_validation_test_overlap": len(sets["validation"] & sets["test"]),
    }
    if any(exact_audit[k] for k in ["exact_train_validation_overlap","exact_train_test_overlap","exact_validation_test_overlap"]):
        raise RuntimeError(f"Exact leakage remains: {exact_audit}")

    safe_rows = load_safe_rows_for_group_audit(
        files, safe_map, args.peptide_col, args.group_cols,
        args.split_audit_chunksize, args.max_group_audit_rows
    )
    group_audit = group_leakage_audit(
        safe_rows, args.group_cols, args.train_split, args.validation_split, args.test_split
    )

    nearest_val = nearest_hamming(sets["train"], sets["validation"], args.near_train_sample, args.near_val_sample, args.seed)
    nearest_test = nearest_hamming(sets["train"], sets["test"], args.near_train_sample, args.near_test_sample, args.seed+1)
    pd.DataFrame({"nearest_train_hamming": nearest_val}).to_csv(out_dir/"validation_nearest_train_hamming.csv", index=False)
    pd.DataFrame({"nearest_train_hamming": nearest_test}).to_csv(out_dir/"test_nearest_train_hamming.csv", index=False)
    near_audit = {"train_validation": hamming_summary(nearest_val), "train_test": hamming_summary(nearest_test)}

    gd, gs, eps_np = generation_audit(
        model, args.generation_samples, args.batch_size, args.ddim_steps, device,
        sets["train"], sets["validation"], sets["test"]
    )
    gd.to_csv(out_dir/"spherical_epsilon_generation_details.csv", index=False)

    sd, ss, sg = local_smoothness(
        model, args.n_centers, args.sigmas, args.neighbors_per_sigma,
        args.ddim_steps, args.seed, device
    )
    sd.to_csv(out_dir/"spherical_epsilon_smoothness_neighbor_details.csv", index=False)
    ss.to_csv(out_dir/"spherical_epsilon_smoothness_summary.csv", index=False)

    report = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "checkpoint_type": ckpt.get("checkpoint_type", ""),
        "diffusion_config": {
            "hidden_size": int(cfg.hidden_size), "n_layers": int(cfg.n_layers),
            "time_dim": int(cfg.time_dim), "train_steps": int(cfg.train_steps),
            "beta_start": float(cfg.beta_start), "beta_end": float(cfg.beta_end),
        },
        "matched_framework": {
            "bo_coordinate": "raw_direct_diffusion_base_noise_epsilon",
            "bo_dimension": EPS_DIM,
            "constraint": "hypersphere",
            "radius": EPS_RADIUS,
            "ddim_steps_used_for_audit": args.ddim_steps,
            "ddim_inversion_used": False,
            "pca_used": False,
            "reason": "The H32 pretrainer uses raw spherical epsilon directly and defines neither DDIM inversion nor PCA z_bo.",
        },
        "safe_split_policy": split_audit,
        "exact_sequence_leakage": exact_audit,
        "group_leakage_after_exact_filter": group_audit,
        "near_neighbor_leakage": near_audit,
        "sampled_spherical_epsilon_geometry": coordinate_geometry(eps_np),
        "generation_audit": gs,
        "local_spherical_epsilon_smoothness_global": sg,
        "interpretation_notes": [
            "Exact peptide leakage is tested after reproducing the H32 conflict-exclusion rule.",
            "Group overlap is reported separately because exact-sequence disjointness does not imply PDB/chain/protein disjointness.",
            "Local smoothness is tested directly in the actual 200-D spherical epsilon search space; no PCA or DDIM inversion is introduced.",
            "Generator smoothness does not establish Cu-objective smoothness.",
        ],
    }
    with open(out_dir/"audit_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    lines = [
        "H32 DIRECT SEQUENCE DIFFUSION: SPHERICAL-EPSILON + LEAKAGE AUDIT",
        "="*88,
        f"checkpoint_epoch={report['checkpoint_epoch']}",
        f"hidden_size={cfg.hidden_size}",
        f"epsilon_dim={EPS_DIM}",
        f"epsilon_radius={EPS_RADIUS:.6f}",
        f"ddim_steps={args.ddim_steps}",
        "", "EXACT-SEQUENCE LEAKAGE", "-"*88, json.dumps(exact_audit, indent=2),
        "", "SAFE-SPLIT POLICY", "-"*88, json.dumps(split_audit, indent=2),
        "", "GROUP LEAKAGE AFTER EXACT FILTER", "-"*88, json.dumps(group_audit, indent=2),
        "", "NEAR-NEIGHBOR LEAKAGE", "-"*88, json.dumps(near_audit, indent=2),
        "", "SPHERICAL EPSILON GENERATION", "-"*88, json.dumps(gs, indent=2),
        "", "LOCAL SPHERICAL EPSILON SMOOTHNESS", "-"*88, ss.to_string(index=False),
        "", json.dumps(sg, indent=2),
        "", "CAUTION", "-"*88,
        "These tests quantify split integrity, memorization, generation diversity, and local smoothness in the H32 spherical epsilon coordinate. They do not establish Cu-objective smoothness.",
    ]
    (out_dir/"audit_report.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    save_plots(ss, nearest_val, nearest_test, out_dir)
    print("\n".join(lines))
    print(f"\nSaved audit outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
