from __future__ import annotations

"""
Audit a pretrained BO-ready GRU-VAE for:

A) latent-space health and decoder smoothness in encoder-mu space
B) train/validation leakage risk in the pretraining split

This script is designed for checkpoints produced by:
    pretrain_gru_vae_bo_ready_h64_z64_with_validation.py

IMPORTANT INTERPRETATION
------------------------
This script tests:
  * exact train/validation peptide separation
  * duplicate consistency
  * optional source/group leakage when metadata columns are available
  * near-duplicate sequence similarity across train/validation
  * latent activity / effective dimensionality
  * local smoothness of standardized encoder-mu coordinates
  * hard decoded sequence changes under latent perturbations
  * soft output-distribution changes (Jensen-Shannon divergence)

It does NOT by itself establish Bayesian-optimization OBJECTIVE smoothness.
For BO smoothness you must additionally evaluate the true black-box objective
vector f(decode(z)) for center and perturbed coordinates.

Recommended use:
  run this on the BEST validation checkpoint, not the last epoch checkpoint.
"""

import argparse
import glob
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_I = {a: i for i, a in enumerate(AA)}
I_TO_AA = {i: a for i, a in enumerate(AA)}
SEQ_LEN = 10
VOCAB = 20


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_peptide(x) -> Optional[str]:
    p = str(x).strip().upper()
    if len(p) != SEQ_LEN:
        return None
    if any(a not in AA_TO_I for a in p):
        return None
    return p


def onehot_encode(peptides: Sequence[str]) -> torch.Tensor:
    x = torch.zeros(len(peptides), SEQ_LEN, VOCAB, dtype=torch.float32)
    for i, p in enumerate(peptides):
        for j, aa in enumerate(p):
            x[i, j, AA_TO_I[aa]] = 1.0
    return x


def levenshtein(a: str, b: str) -> int:
    # Fixed length=10, but generic implementation retained.
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(
                current[j - 1] + 1,
                previous[j] + 1,
                previous[j - 1] + int(ca != cb),
            ))
        previous = current
    return previous[-1]


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def deterministic_is_val(peptide: str, val_fraction: float, split_seed: int) -> bool:
    key = f"{split_seed}:{peptide}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    u = int.from_bytes(digest[:8], "big", signed=False) / float(2**64)
    return u < float(val_fraction)


def import_training_module(path: str):
    """Import the training script safely on Python 3.14+.

    dataclasses resolves annotations through sys.modules[cls.__module__].
    When a module is constructed manually with module_from_spec(), it must be
    registered in sys.modules before exec_module() runs; otherwise Python 3.14
    can fail inside @dataclass with:
        AttributeError: 'NoneType' object has no attribute '__dict__'
    """
    module_name = "gru_vae_training_module"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import training script: {path}")

    module = importlib.util.module_from_spec(spec)

    # Critical for Python 3.14 dataclasses / postponed annotations.
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        # Avoid leaving a partially initialized module behind.
        sys.modules.pop(module_name, None)
        raise

    return module


def load_checkpoint_model(training_module, checkpoint: str, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device)
    cfg_saved = ckpt.get("model_config", {})
    cfg = training_module.ModelConfig(
        hidden_size=int(cfg_saved.get("hidden_size", 64)),
        latent_dim=int(cfg_saved.get("latent_dim", 64)),
        n_layers=int(cfg_saved.get("n_layers", 2)),
        dropout=float(cfg_saved.get("dropout", 0.0)),
    )
    model = training_module.GRUVAE(cfg).to(device)
    state = ckpt.get("model_state_dict", ckpt.get("vae_state_dict", ckpt))
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, ckpt, cfg


def discover_files(parts_dir: str, file_pattern: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(parts_dir, file_pattern)))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found under {parts_dir!r} with {file_pattern!r}"
        )
    return [f for f in files if "failures" not in os.path.basename(f).lower()]


def load_peptides_and_optional_groups(
    files: Sequence[str],
    peptide_col: str,
    group_cols: Sequence[str],
    max_rows: int,
) -> pd.DataFrame:
    rows = []
    total = 0

    for path in files:
        # Read header first so absent optional columns do not fail.
        header = pd.read_csv(path, nrows=0)
        cols = [peptide_col] + [c for c in group_cols if c in header.columns]
        if peptide_col not in header.columns:
            raise ValueError(f"{peptide_col!r} missing in {path}")

        for chunk in pd.read_csv(path, usecols=cols, chunksize=100000):
            chunk = chunk.copy()
            chunk["peptide"] = chunk[peptide_col].map(clean_peptide)
            chunk = chunk[chunk["peptide"].notna()]
            if chunk.empty:
                continue
            chunk["source_file"] = os.path.basename(path)
            rows.append(chunk)
            total += len(chunk)
            if max_rows > 0 and total >= max_rows:
                break

        if max_rows > 0 and total >= max_rows:
            break

    if not rows:
        raise ValueError("No valid length-10 peptides found.")

    df = pd.concat(rows, ignore_index=True)
    if max_rows > 0:
        df = df.iloc[:max_rows].copy()
    return df


def split_dataframe(
    df: pd.DataFrame,
    val_fraction: float,
    split_seed: int,
) -> pd.DataFrame:
    out = df.copy()
    out["split"] = [
        "val" if deterministic_is_val(p, val_fraction, split_seed) else "train"
        for p in out["peptide"]
    ]
    return out


def leakage_audit(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    near_val_sample: int,
    near_train_sample: int,
    seed: int,
):
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]

    train_unique = set(train["peptide"])
    val_unique = set(val["peptide"])
    exact_overlap = train_unique & val_unique

    # Duplicate consistency: identical sequence should always hash to one split.
    split_counts = df.groupby("peptide")["split"].nunique()
    inconsistent_duplicates = int((split_counts > 1).sum())

    report = {
        "n_rows": int(len(df)),
        "n_train_rows": int(len(train)),
        "n_val_rows": int(len(val)),
        "n_unique_peptides": int(df["peptide"].nunique()),
        "n_train_unique_peptides": int(len(train_unique)),
        "n_val_unique_peptides": int(len(val_unique)),
        "exact_train_val_peptide_overlap": int(len(exact_overlap)),
        "duplicate_sequences_assigned_to_both_splits": inconsistent_duplicates,
        "duplicate_row_fraction": float(1.0 - df["peptide"].nunique() / max(1, len(df))),
    }

    # Group/source leakage: if same PDB/chain/protein group has peptides in both splits.
    group_results = {}
    for col in group_cols:
        if col not in df.columns:
            continue
        temp = df[df[col].notna()].copy()
        if temp.empty:
            continue
        per_group = temp.groupby(col)["split"].agg(lambda s: set(s))
        both = per_group.map(lambda s: "train" in s and "val" in s)
        group_results[col] = {
            "n_groups": int(len(per_group)),
            "n_groups_present_in_both_train_and_val": int(both.sum()),
            "fraction_groups_present_in_both": float(both.mean()) if len(both) else float("nan"),
        }
    report["group_leakage"] = group_results

    # Approximate near-duplicate leakage audit by sampling unique sequences.
    rng = np.random.default_rng(seed)
    train_list = np.array(sorted(train_unique), dtype=object)
    val_list = np.array(sorted(val_unique), dtype=object)

    if near_train_sample > 0 and len(train_list) > near_train_sample:
        train_list = rng.choice(train_list, size=near_train_sample, replace=False)
    if near_val_sample > 0 and len(val_list) > near_val_sample:
        val_list = rng.choice(val_list, size=near_val_sample, replace=False)

    # Encode chars as integers to efficiently compute Hamming distances in chunks.
    train_arr = np.array([[AA_TO_I[a] for a in p] for p in train_list], dtype=np.int8)
    val_arr = np.array([[AA_TO_I[a] for a in p] for p in val_list], dtype=np.int8)

    nearest = []
    chunk_size = 20000
    for v in val_arr:
        best = SEQ_LEN + 1
        for start in range(0, len(train_arr), chunk_size):
            t = train_arr[start:start + chunk_size]
            d = (t != v[None, :]).sum(axis=1)
            local = int(d.min())
            if local < best:
                best = local
            if best == 0:
                break
        nearest.append(best)

    nearest = np.asarray(nearest, dtype=int)
    report["near_duplicate_audit"] = {
        "n_train_unique_sampled": int(len(train_arr)),
        "n_val_unique_sampled": int(len(val_arr)),
        "nearest_train_hamming_mean": float(nearest.mean()) if len(nearest) else float("nan"),
        "nearest_train_hamming_median": float(np.median(nearest)) if len(nearest) else float("nan"),
        "fraction_val_with_train_neighbor_hamming_0": float(np.mean(nearest == 0)) if len(nearest) else float("nan"),
        "fraction_val_with_train_neighbor_hamming_le_1": float(np.mean(nearest <= 1)) if len(nearest) else float("nan"),
        "fraction_val_with_train_neighbor_hamming_le_2": float(np.mean(nearest <= 2)) if len(nearest) else float("nan"),
    }
    return report, nearest


@torch.no_grad()
def encode_mu(model, peptides: Sequence[str], batch_size: int, device: torch.device):
    mus = []
    for start in range(0, len(peptides), batch_size):
        x = onehot_encode(peptides[start:start + batch_size]).to(device)
        mu, _, _ = model.enc(x)
        mus.append(mu.detach().cpu())
    return torch.cat(mus, dim=0)


@torch.no_grad()
def autoregressive_decode_with_probs(model, z: torch.Tensor):
    model.eval()
    batch = z.size(0)
    h = model.dec.initial_hidden(z)
    current = torch.zeros(batch, 1, VOCAB, device=z.device, dtype=z.dtype)
    probs_steps = []
    idx_steps = []

    for _ in range(SEQ_LEN):
        logits, h = model.dec.step(z, current, h)
        probs = torch.softmax(logits, dim=-1)
        idx = probs.argmax(dim=-1)
        probs_steps.append(probs.unsqueeze(1))
        idx_steps.append(idx.unsqueeze(1))
        current = F.one_hot(idx, num_classes=VOCAB).to(z.dtype).unsqueeze(1)

    probs_all = torch.cat(probs_steps, dim=1)
    idx_all = torch.cat(idx_steps, dim=1)
    peptides = [
        "".join(I_TO_AA[int(i)] for i in row)
        for row in idx_all.detach().cpu().tolist()
    ]
    return peptides, probs_all


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8):
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
    kl_qm = (q * (q.log() - m.log())).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


@torch.no_grad()
def latent_health(model, train_peptides, val_peptides, batch_size, device):
    mu_train = encode_mu(model, train_peptides, batch_size, device)
    mu_val = encode_mu(model, val_peptides, batch_size, device)

    mean = mu_train.mean(dim=0)
    std = mu_train.std(dim=0, unbiased=False).clamp_min(1e-8)

    h_train = (mu_train - mean) / std
    h_val = (mu_val - mean) / std

    # Effective dimensionality from covariance eigenvalues.
    cov = torch.cov(h_train.T)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0)
    participation_ratio = float((eig.sum() ** 2 / eig.pow(2).sum().clamp_min(1e-12)).item())

    health = {
        "latent_dim": int(mu_train.shape[1]),
        "train_mu_global_abs_mean": float(mu_train.mean(dim=0).abs().mean()),
        "train_mu_dim_std_mean": float(std.mean()),
        "train_mu_dim_std_min": float(std.min()),
        "train_mu_dim_std_median": float(std.median()),
        "train_mu_dim_std_max": float(std.max()),
        "val_standardized_mean_abs": float(h_val.mean(dim=0).abs().mean()),
        "val_standardized_std_mean": float(h_val.std(dim=0, unbiased=False).mean()),
        "effective_dimension_participation_ratio": participation_ratio,
        "fraction_dims_train_std_lt_0p1": float((std < 0.1).float().mean()),
        "fraction_dims_train_std_lt_0p5": float((std < 0.5).float().mean()),
    }
    return health, mean, std, h_train, h_val


@torch.no_grad()
def local_smoothness(
    model,
    val_peptides,
    latent_mean,
    latent_std,
    sigmas,
    n_centers,
    neighbors_per_sigma,
    batch_size,
    seed,
    device,
):
    rng = np.random.default_rng(seed)
    if len(val_peptides) > n_centers:
        selected = rng.choice(len(val_peptides), size=n_centers, replace=False)
    else:
        selected = np.arange(len(val_peptides))

    centers_pep = [val_peptides[i] for i in selected]
    mu = encode_mu(model, centers_pep, batch_size, device).to(device)
    mean = latent_mean.to(device)
    std = latent_std.to(device)
    h = (mu - mean) / std

    center_decoded, center_probs = autoregressive_decode_with_probs(model, mu)

    details = []
    latent_dim = h.size(1)

    torch_gen = torch.Generator(device="cpu")
    torch_gen.manual_seed(seed + 999)

    for sigma in sigmas:
        for i in range(len(centers_pep)):
            h0 = h[i:i+1]
            mu0 = mu[i:i+1]
            p0 = center_decoded[i]
            prob0 = center_probs[i:i+1]

            for k in range(neighbors_per_sigma):
                delta = torch.randn(
                    (1, latent_dim),
                    generator=torch_gen,
                    dtype=torch.float32,
                    device="cpu",
                ).to(device) * float(sigma)

                hn = h0 + delta
                mun = hn * std + mean
                decoded, probs = autoregressive_decode_with_probs(model, mun)
                p1 = decoded[0]

                dh = float(torch.linalg.norm(hn - h0).cpu())
                dmu = float(torch.linalg.norm(mun - mu0).cpu())
                edit = levenshtein(p0, p1)
                js = float(js_divergence(prob0, probs).mean().cpu())
                l1prob = float((prob0 - probs).abs().mean().cpu())

                details.append({
                    "center_index": int(i),
                    "source_val_peptide": centers_pep[i],
                    "center_decoded": p0,
                    "sigma": float(sigma),
                    "neighbor_index": int(k),
                    "standardized_latent_l2": dh,
                    "raw_mu_l2": dmu,
                    "decoded_neighbor": p1,
                    "sequence_edit": int(edit),
                    "identical": int(edit == 0),
                    "mean_token_js_divergence": js,
                    "mean_probability_l1": l1prob,
                })

    details = pd.DataFrame(details)

    summary = (
        details.groupby("sigma")
        .agg(
            n=("sequence_edit", "size"),
            latent_l2_mean=("standardized_latent_l2", "mean"),
            latent_l2_median=("standardized_latent_l2", "median"),
            sequence_edit_mean=("sequence_edit", "mean"),
            sequence_edit_median=("sequence_edit", "median"),
            identical_fraction=("identical", "mean"),
            js_mean=("mean_token_js_divergence", "mean"),
            js_median=("mean_token_js_divergence", "median"),
            probability_l1_mean=("mean_probability_l1", "mean"),
        )
        .reset_index()
    )

    # Across-scale monotonicity: a useful summary for local smoothness.
    if len(summary) >= 2:
        x = summary["latent_l2_mean"].to_numpy()
        y_edit = summary["sequence_edit_mean"].to_numpy()
        y_js = summary["js_mean"].to_numpy()
        pearson_edit = float(np.corrcoef(x, y_edit)[0, 1])
        pearson_js = float(np.corrcoef(x, y_js)[0, 1])
        rank_x = pd.Series(x).rank().to_numpy()
        rank_edit = pd.Series(y_edit).rank().to_numpy()
        rank_js = pd.Series(y_js).rank().to_numpy()
        spearman_edit = float(np.corrcoef(rank_x, rank_edit)[0, 1])
        spearman_js = float(np.corrcoef(rank_x, rank_js)[0, 1])
    else:
        pearson_edit = pearson_js = spearman_edit = spearman_js = float("nan")

    global_summary = {
        "across_sigma_pearson_latent_vs_edit": pearson_edit,
        "across_sigma_spearman_latent_vs_edit": spearman_edit,
        "across_sigma_pearson_latent_vs_js": pearson_js,
        "across_sigma_spearman_latent_vs_js": spearman_js,
    }
    return details, summary, global_summary


def save_plots(summary: pd.DataFrame, nearest: np.ndarray, out_dir: Path):
    # Separate figures, no subplots.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary["latent_l2_mean"], summary["sequence_edit_mean"], marker="o")
    ax.set_xlabel("Mean standardized latent perturbation L2")
    ax.set_ylabel("Mean decoded sequence edit distance")
    ax.set_title("GRU-VAE latent perturbation vs sequence change")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "latent_l2_vs_sequence_edit.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary["latent_l2_mean"], summary["js_mean"], marker="o")
    ax.set_xlabel("Mean standardized latent perturbation L2")
    ax.set_ylabel("Mean token Jensen-Shannon divergence")
    ax.set_title("GRU-VAE latent perturbation vs soft decoder change")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "latent_l2_vs_decoder_js.png", dpi=300)
    plt.close(fig)

    if len(nearest):
        fig, ax = plt.subplots(figsize=(8, 5))
        bins = np.arange(-0.5, SEQ_LEN + 1.5, 1)
        ax.hist(nearest, bins=bins)
        ax.set_xlabel("Nearest sampled train Hamming distance")
        ax.set_ylabel("Validation peptide count")
        ax.set_title("Approximate train-validation near-duplicate audit")
        fig.tight_layout()
        fig.savefig(out_dir / "train_val_nearest_hamming.png", dpi=300)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--training-script", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--parts-dir", required=True)
    p.add_argument(
        "--file-pattern",
        default="metalpdb_ALL_chain_mapped_len10_high_confidence_part_*.csv",
    )
    p.add_argument("--peptide-col", default="peptide_len10")
    p.add_argument("--validation-fraction", type=float, default=0.10)
    p.add_argument("--validation-split-seed", type=int, default=12345)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=[0.01, 0.025, 0.05, 0.10, 0.20],
        help="Gaussian perturbation std in TRAIN-standardized mu coordinates.",
    )
    p.add_argument("--n-centers", type=int, default=128)
    p.add_argument("--neighbors-per-sigma", type=int, default=5)

    p.add_argument(
        "--group-cols",
        nargs="*",
        default=[
            "pdb_id", "pdb", "pdbid", "structure_id",
            "chain_id", "chain", "protein_id", "uniprot_id",
            "source_pdb", "source_chain",
        ],
        help="Metadata columns to inspect for source/group leakage if present.",
    )
    p.add_argument(
        "--near-val-sample", type=int, default=500,
        help="Validation unique peptides sampled for near-duplicate Hamming audit.",
    )
    p.add_argument(
        "--near-train-sample", type=int, default=100000,
        help="Train unique peptides sampled for near-duplicate Hamming audit. 0=all.",
    )
    p.add_argument(
        "--max-audit-rows", type=int, default=0,
        help="Optional row cap for leakage audit; 0=all rows.",
    )
    p.add_argument(
        "--max-latent-train", type=int, default=50000,
        help="Maximum unique training peptides for latent-statistics computation.",
    )
    p.add_argument(
        "--max-latent-val", type=int, default=10000,
        help="Maximum unique validation peptides for latent-statistics computation.",
    )
    p.add_argument("--out-dir", default="gru_vae_latent_smoothness_leakage_audit")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    module = import_training_module(args.training_script)
    model, ckpt, cfg = load_checkpoint_model(module, args.checkpoint, device)

    files = discover_files(args.parts_dir, args.file_pattern)
    raw = load_peptides_and_optional_groups(
        files=files,
        peptide_col=args.peptide_col,
        group_cols=args.group_cols,
        max_rows=args.max_audit_rows,
    )
    split_df = split_dataframe(
        raw,
        val_fraction=args.validation_fraction,
        split_seed=args.validation_split_seed,
    )

    leakage_report, nearest = leakage_audit(
        split_df,
        group_cols=args.group_cols,
        near_val_sample=args.near_val_sample,
        near_train_sample=args.near_train_sample,
        seed=args.seed,
    )

    # Unique peptides are enough for latent geometry diagnostics.
    train_peptides = sorted(split_df.loc[split_df["split"] == "train", "peptide"].unique())
    val_peptides = sorted(split_df.loc[split_df["split"] == "val", "peptide"].unique())

    rng = np.random.default_rng(args.seed)
    if args.max_latent_train > 0 and len(train_peptides) > args.max_latent_train:
        train_peptides = rng.choice(
            np.array(train_peptides, dtype=object),
            size=args.max_latent_train,
            replace=False,
        ).tolist()
    if args.max_latent_val > 0 and len(val_peptides) > args.max_latent_val:
        val_peptides = rng.choice(
            np.array(val_peptides, dtype=object),
            size=args.max_latent_val,
            replace=False,
        ).tolist()

    health, latent_mean, latent_std, _, _ = latent_health(
        model, train_peptides, val_peptides, args.batch_size, device
    )

    details, smooth_summary, smooth_global = local_smoothness(
        model=model,
        val_peptides=val_peptides,
        latent_mean=latent_mean,
        latent_std=latent_std,
        sigmas=args.sigmas,
        n_centers=args.n_centers,
        neighbors_per_sigma=args.neighbors_per_sigma,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
    )

    details.to_csv(out_dir / "latent_smoothness_neighbor_details.csv", index=False)
    smooth_summary.to_csv(out_dir / "latent_smoothness_summary.csv", index=False)
    pd.DataFrame({"nearest_train_hamming": nearest}).to_csv(
        out_dir / "near_duplicate_hamming_sample.csv", index=False
    )

    full_report = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "model_config": {
            "hidden_size": int(cfg.hidden_size),
            "latent_dim": int(cfg.latent_dim),
            "n_layers": int(cfg.n_layers),
            "dropout": float(cfg.dropout),
        },
        "split": {
            "method": "deterministic_sha256_by_exact_peptide_sequence",
            "validation_fraction": args.validation_fraction,
            "validation_split_seed": args.validation_split_seed,
        },
        "leakage_audit": leakage_report,
        "latent_health": health,
        "latent_smoothness_global": smooth_global,
        "interpretation_notes": [
            "Exact peptide leakage should be zero because identical peptide strings hash to the same split.",
            "Near-identical peptides can still occur across splits; inspect the Hamming-distance audit.",
            "If source PDB/chain/protein metadata appear in both splits, source-group leakage remains possible even with exact peptide separation.",
            "Validation is used every epoch and for checkpoint selection, so it is a model-selection set, not an untouched final test set.",
            "Latent perturbation diagnostics establish decoder/representation locality, not black-box objective smoothness.",
        ],
    }
    with open(out_dir / "audit_report.json", "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    save_plots(smooth_summary, nearest, out_dir)

    # Human-readable report.
    lines = []
    lines.append("GRU-VAE LATENT SMOOTHNESS + LEAKAGE AUDIT")
    lines.append("=" * 78)
    lines.append(f"checkpoint_epoch={int(ckpt.get('epoch', -1))}")
    lines.append("")
    lines.append("LEAKAGE")
    lines.append("-" * 78)
    for k, v in leakage_report.items():
        if k not in {"group_leakage", "near_duplicate_audit"}:
            lines.append(f"{k}={v}")
    lines.append(f"group_leakage={json.dumps(leakage_report['group_leakage'])}")
    lines.append(f"near_duplicate_audit={json.dumps(leakage_report['near_duplicate_audit'])}")
    lines.append("")
    lines.append("LATENT HEALTH")
    lines.append("-" * 78)
    for k, v in health.items():
        lines.append(f"{k}={v}")
    lines.append("")
    lines.append("SMOOTHNESS ACROSS PERTURBATION SCALES")
    lines.append("-" * 78)
    lines.append(smooth_summary.to_string(index=False))
    lines.append("")
    for k, v in smooth_global.items():
        lines.append(f"{k}={v}")
    lines.append("")
    lines.append("CAUTION")
    lines.append("-" * 78)
    lines.append(
        "These tests quantify latent/decoder locality. They do NOT establish "
        "smoothness of Cu black-box objectives. Objective smoothness requires "
        "scoring decoded center and neighbor peptides with the same true objective function."
    )
    (out_dir / "audit_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\nSaved audit outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
