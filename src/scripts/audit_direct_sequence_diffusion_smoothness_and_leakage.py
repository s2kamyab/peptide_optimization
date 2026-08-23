from __future__ import annotations

"""
Audit direct peptide sequence diffusion trained by
pretrain_peptide_direct_sequence_diffusion_bo_ready_h96.py.

The GRU-VAE audit is not appropriate for this model because there is no encoder
mu/h0, KL posterior, or autoregressive VAE decoder. This audit instead checks:

1) the EXISTING split column used by direct pretraining (train/validation/test),
2) exact/group/near-neighbor leakage,
3) DDIM inversion -> sampling roundtrip quality,
4) raw inverted-epsilon geometry,
5) train-only PCA-whitened z_bo geometry matching the pretrainer,
6) local smoothness in z_bo after inverse PCA -> epsilon -> DDIM -> peptide.

These diagnostics establish generator/coordinate locality, not Cu objective
smoothness. Objective smoothness must be evaluated with the downstream scorer.
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
from typing import Dict, List, Optional, Sequence

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
    if len(p) != SEQ_LEN or any(a not in AA_TO_I for a in p):
        return None
    return p


def onehot(peptides: Sequence[str]) -> torch.Tensor:
    x = torch.zeros(len(peptides), SEQ_LEN, VOCAB, dtype=torch.float32)
    for i, p in enumerate(peptides):
        for j, aa in enumerate(p):
            x[i, j, AA_TO_I[aa]] = 1.0
    return x


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
    name = "direct_sequence_diffusion_training_module"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # important for Python 3.14 + dataclasses
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
        hidden_size=int(c.get("hidden_size", 96)),
        n_layers=int(c.get("n_layers", 2)),
        dropout=float(c.get("dropout", 0.0)),
        time_dim=int(c.get("time_dim", 32)),
        train_steps=int(c.get("train_steps", 100)),
        beta_start=float(c.get("beta_start", 1e-5)),
        beta_end=float(c.get("beta_end", 8e-3)),
    )
    model = module.DirectSequenceDiffusion(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, ckpt, cfg


def discover_files(parts_dir, pattern):
    files = sorted(glob.glob(os.path.join(parts_dir, pattern)))
    files = [f for f in files if "failures" not in os.path.basename(f).lower()]
    if not files:
        raise FileNotFoundError(os.path.join(parts_dir, pattern))
    return files


def load_dataframe(files, peptide_col, split_col, group_cols, max_rows):
    rows, total = [], 0
    for path in files:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        if peptide_col not in header or split_col not in header:
            raise KeyError(f"Required columns {peptide_col!r}, {split_col!r} missing in {path}")
        cols = [peptide_col, split_col] + [c for c in group_cols if c in header and c not in {peptide_col, split_col}]
        for chunk in pd.read_csv(path, usecols=cols, chunksize=100000):
            chunk = chunk.copy()
            chunk["peptide"] = chunk[peptide_col].map(clean_peptide)
            chunk = chunk[chunk["peptide"].notna()]
            if chunk.empty:
                continue
            chunk["split"] = chunk[split_col].astype(str).str.strip().str.lower()
            rows.append(chunk)
            total += len(chunk)
            if max_rows > 0 and total >= max_rows:
                break
        if max_rows > 0 and total >= max_rows:
            break
    if not rows:
        raise RuntimeError("No valid peptides found")
    df = pd.concat(rows, ignore_index=True)
    return df.iloc[:max_rows].copy() if max_rows > 0 else df


def leakage_audit(df, train_name, val_name, test_name, group_cols, near_val_sample, near_train_sample, seed):
    train_name, val_name, test_name = train_name.lower(), val_name.lower(), test_name.lower()
    sets = {
        "train": set(df.loc[df["split"] == train_name, "peptide"]),
        "validation": set(df.loc[df["split"] == val_name, "peptide"]),
        "test": set(df.loc[df["split"] == test_name, "peptide"]),
    }
    split_nunique = df.groupby("peptide")["split"].nunique()
    report = {
        "n_rows": int(len(df)),
        "n_unique_peptides": int(df["peptide"].nunique()),
        "n_train_unique": len(sets["train"]),
        "n_validation_unique": len(sets["validation"]),
        "n_test_unique": len(sets["test"]),
        "exact_train_validation_overlap": len(sets["train"] & sets["validation"]),
        "exact_train_test_overlap": len(sets["train"] & sets["test"]),
        "exact_validation_test_overlap": len(sets["validation"] & sets["test"]),
        "peptides_assigned_to_multiple_splits": int((split_nunique > 1).sum()),
    }
    groups = {}
    for col in group_cols:
        if col not in df.columns:
            continue
        tmp = df[df[col].notna()]
        if tmp.empty:
            continue
        s = tmp.groupby(col)["split"].agg(lambda x: set(map(str, x)))
        tv = s.map(lambda x: train_name in x and val_name in x)
        tt = s.map(lambda x: train_name in x and test_name in x)
        groups[col] = {
            "n_groups": int(len(s)),
            "n_groups_train_validation": int(tv.sum()),
            "fraction_groups_train_validation": float(tv.mean()),
            "n_groups_train_test": int(tt.sum()),
            "fraction_groups_train_test": float(tt.mean()),
        }
    report["group_leakage"] = groups

    rng = np.random.default_rng(seed)
    tr = np.asarray(sorted(sets["train"]), dtype=object)
    va = np.asarray(sorted(sets["validation"]), dtype=object)
    if near_train_sample > 0 and len(tr) > near_train_sample:
        tr = rng.choice(tr, near_train_sample, replace=False)
    if near_val_sample > 0 and len(va) > near_val_sample:
        va = rng.choice(va, near_val_sample, replace=False)
    tr_arr = np.asarray([[AA_TO_I[a] for a in p] for p in tr], dtype=np.int8)
    va_arr = np.asarray([[AA_TO_I[a] for a in p] for p in va], dtype=np.int8)
    nearest = []
    for v in va_arr:
        best = SEQ_LEN + 1
        for s in range(0, len(tr_arr), 20000):
            d = (tr_arr[s:s+20000] != v[None, :]).sum(axis=1)
            best = min(best, int(d.min()))
            if best == 0:
                break
        nearest.append(best)
    nearest = np.asarray(nearest, dtype=int)
    report["near_duplicate_audit"] = {
        "n_train_unique_sampled": len(tr_arr),
        "n_validation_unique_sampled": len(va_arr),
        "nearest_train_hamming_mean": float(nearest.mean()) if len(nearest) else float("nan"),
        "nearest_train_hamming_median": float(np.median(nearest)) if len(nearest) else float("nan"),
        "fraction_validation_hamming_0": float(np.mean(nearest == 0)) if len(nearest) else float("nan"),
        "fraction_validation_hamming_le_1": float(np.mean(nearest <= 1)) if len(nearest) else float("nan"),
        "fraction_validation_hamming_le_2": float(np.mean(nearest <= 2)) if len(nearest) else float("nan"),
    }
    return report, nearest


def geometry(x: np.ndarray, eps=1e-8) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        return {}
    xc = x - x.mean(0, keepdims=True)
    cov = (xc.T @ xc) / max(1, len(x) - 1)
    eig = np.linalg.eigvalsh(cov).clip(min=0)
    total = max(float(eig.sum()), eps)
    pr = total * total / max(float((eig**2).sum()), eps)
    frac = eig[::-1] / total
    cum = np.cumsum(frac)
    z = xc / np.maximum(xc.std(0, keepdims=True), eps)
    corr = (z.T @ z) / len(z)
    off = corr - np.diag(np.diag(corr))
    denom = max(1, x.shape[1] * (x.shape[1] - 1))
    return {
        "dimension": int(x.shape[1]),
        "effective_dim_pr": float(pr),
        "pc1_variance_fraction": float(frac[0]),
        "pcs90": int(np.searchsorted(cum, .90) + 1),
        "pcs95": int(np.searchsorted(cum, .95) + 1),
        "mean_abs_offdiag_corr": float(np.abs(off).sum() / denom),
        "max_abs_offdiag_corr": float(np.abs(off).max()),
    }


def fit_pca(x: np.ndarray, k: int, eps=1e-6):
    k = min(int(k), x.shape[1], max(1, x.shape[0]-1))
    mean = x.mean(0, keepdims=True)
    xc = x - mean
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    comps = vt[:k].astype(np.float32)
    scores = xc @ comps.T
    score_std = np.maximum(scores.std(0, keepdims=True).astype(np.float32), eps)
    ev = (s**2) / max(1, x.shape[0]-1)
    return {
        "mean": mean.astype(np.float32),
        "components": comps,
        "score_std": score_std,
        "explained_variance_ratio": (ev[:k] / max(float(ev.sum()), eps)).astype(np.float32),
    }


def pca_project(eps, pca):
    return ((eps - pca["mean"]) @ pca["components"].T) / pca["score_std"]


def pca_inverse(z, pca):
    return (z * pca["score_std"]) @ pca["components"] + pca["mean"]


@torch.no_grad()
def collect_eps(model, peptides, batch_size, ddim_steps, device):
    rows = []
    for s in range(0, len(peptides), batch_size):
        x0 = onehot(peptides[s:s+batch_size]).to(device)
        e = model.ddim_invert(x0, inference_steps=ddim_steps).reshape(x0.size(0), -1)
        rows.append(e.cpu().numpy().astype(np.float32))
    return np.concatenate(rows, axis=0)


def decode_with_probs(x):
    probs = torch.softmax(x, dim=-1)
    idx = x.argmax(-1).cpu().tolist()
    peps = ["".join(I_TO_AA[int(i)] for i in row) for row in idx]
    return peps, probs


def js_divergence(p, q, eps=1e-8):
    p, q = p.clamp_min(eps), q.clamp_min(eps)
    m = .5*(p+q)
    return .5*((p*(p.log()-m.log())).sum(-1)+(q*(q.log()-m.log())).sum(-1))


@torch.no_grad()
def raw_roundtrip(model, peptides, batch_size, ddim_steps, device):
    rows, eps_rows = [], []
    for s in range(0, len(peptides), batch_size):
        p = peptides[s:s+batch_size]
        x0 = onehot(p).to(device)
        eps = model.ddim_invert(x0, inference_steps=ddim_steps)
        xb = model.ddim_sample(eps, inference_steps=ddim_steps)
        dec, _ = decode_with_probs(xb)
        l2 = torch.linalg.norm(x0.reshape(x0.size(0),-1)-xb.reshape(xb.size(0),-1), dim=-1)
        ef = eps.reshape(eps.size(0),-1)
        eps_rows.append(ef.cpu().numpy().astype(np.float32))
        for i, src in enumerate(p):
            ed = levenshtein(src, dec[i])
            rows.append({"source_peptide":src,"decoded_peptide":dec[i],"sequence_edit":ed,
                         "identical":int(ed==0),"x0_roundtrip_l2":float(l2[i].cpu()),
                         "epsilon_norm":float(ef[i].norm().cpu())})
    d = pd.DataFrame(rows)
    eps_np = np.concatenate(eps_rows, axis=0)
    summary = {
        "n": len(d),
        "sequence_edit_mean": float(d.sequence_edit.mean()),
        "sequence_edit_median": float(d.sequence_edit.median()),
        "identical_fraction": float(d.identical.mean()),
        "decoded_unique_fraction": float(d.decoded_peptide.nunique()/max(1,len(d))),
        "x0_roundtrip_l2_mean": float(d.x0_roundtrip_l2.mean()),
        "epsilon_norm_mean": float(d.epsilon_norm.mean()),
        "epsilon_norm_std": float(d.epsilon_norm.std(ddof=0)),
        "epsilon_geometry": geometry(eps_np),
    }
    return d, summary, eps_np


@torch.no_grad()
def zbo_roundtrip(model, peptides, eps_np, pca, batch_size, ddim_steps, device, project_sphere):
    z = pca_project(eps_np, pca).astype(np.float32)
    eb = pca_inverse(z, pca).astype(np.float32)
    rows=[]; radius=math.sqrt(SEQ_LEN*VOCAB)
    for s in range(0,len(peptides),batch_size):
        p=peptides[s:s+batch_size]
        e=torch.tensor(eb[s:s+batch_size],dtype=torch.float32,device=device)
        if project_sphere:
            e=e/e.norm(dim=-1,keepdim=True).clamp_min(1e-8)*radius
        xb=model.ddim_sample(e,inference_steps=ddim_steps)
        dec,_=decode_with_probs(xb)
        for i,src in enumerate(p):
            ed=levenshtein(src,dec[i]); rows.append({"source_peptide":src,"decoded_from_z_bo":dec[i],"sequence_edit":ed,"identical":int(ed==0)})
    d=pd.DataFrame(rows)
    return d, {
        "bo_latent_dim": int(z.shape[1]),
        "pca_explained_variance_fraction": float(pca["explained_variance_ratio"].sum()),
        "z_bo_geometry": geometry(z),
        "sequence_edit_mean": float(d.sequence_edit.mean()),
        "sequence_edit_median": float(d.sequence_edit.median()),
        "identical_fraction": float(d.identical.mean()),
    }, z


@torch.no_grad()
def local_smoothness(model, center_peptides, center_eps, pca, sigmas, neighbors, ddim_steps, seed, device, project_sphere):
    z0=pca_project(center_eps,pca).astype(np.float32)
    eps0_np=pca_inverse(z0,pca).astype(np.float32)
    eps0=torch.tensor(eps0_np,dtype=torch.float32,device=device)
    radius=math.sqrt(SEQ_LEN*VOCAB)
    if project_sphere:
        eps0=eps0/eps0.norm(dim=-1,keepdim=True).clamp_min(1e-8)*radius
    xc=model.ddim_sample(eps0,inference_steps=ddim_steps)
    pc,probc=decode_with_probs(xc)
    rng=np.random.default_rng(seed); rows=[]
    for sigma in sigmas:
        for i in range(len(center_peptides)):
            for k in range(neighbors):
                zn=z0[i:i+1]+rng.normal(0,float(sigma),size=z0[i:i+1].shape).astype(np.float32)
                en_np=pca_inverse(zn,pca).astype(np.float32)
                en=torch.tensor(en_np,dtype=torch.float32,device=device)
                if project_sphere:
                    en=en/en.norm(dim=-1,keepdim=True).clamp_min(1e-8)*radius
                xn=model.ddim_sample(en,inference_steps=ddim_steps)
                pn,probn=decode_with_probs(xn)
                ed=levenshtein(pc[i],pn[0])
                rows.append({"center_index":i,"source_peptide":center_peptides[i],"center_decoded":pc[i],
                             "sigma":float(sigma),"neighbor_index":k,"z_bo_l2":float(np.linalg.norm(zn-z0[i:i+1])),
                             "epsilon_l2":float(np.linalg.norm(en_np-eps0_np[i:i+1])),"decoded_neighbor":pn[0],
                             "sequence_edit":ed,"identical":int(ed==0),
                             "mean_token_js_divergence":float(js_divergence(probc[i:i+1],probn).mean().cpu())})
    d=pd.DataFrame(rows)
    sm=d.groupby("sigma").agg(n=("sequence_edit","size"),z_bo_l2_mean=("z_bo_l2","mean"),
        z_bo_l2_median=("z_bo_l2","median"),epsilon_l2_mean=("epsilon_l2","mean"),
        sequence_edit_mean=("sequence_edit","mean"),sequence_edit_median=("sequence_edit","median"),
        identical_fraction=("identical","mean"),js_mean=("mean_token_js_divergence","mean"),
        js_median=("mean_token_js_divergence","median")).reset_index()
    if len(sm)>=2:
        x=sm.z_bo_l2_mean.to_numpy(); ye=sm.sequence_edit_mean.to_numpy(); yj=sm.js_mean.to_numpy()
        g={"across_sigma_pearson_zbo_vs_edit":float(np.corrcoef(x,ye)[0,1]),
           "across_sigma_spearman_zbo_vs_edit":float(pd.Series(x).corr(pd.Series(ye),method="spearman")),
           "across_sigma_pearson_zbo_vs_js":float(np.corrcoef(x,yj)[0,1]),
           "across_sigma_spearman_zbo_vs_js":float(pd.Series(x).corr(pd.Series(yj),method="spearman"))}
    else:g={}
    return d,sm,g


def save_plots(sm, nearest, out_dir):
    fig,ax=plt.subplots(figsize=(8,5)); ax.plot(sm.z_bo_l2_mean,sm.sequence_edit_mean,marker="o")
    ax.set_xlabel("Mean z_bo perturbation L2"); ax.set_ylabel("Mean decoded sequence edit distance")
    ax.set_title("Direct diffusion z_bo perturbation vs sequence change"); ax.grid(True,alpha=.25)
    fig.tight_layout(); fig.savefig(out_dir/"zbo_l2_vs_sequence_edit.png",dpi=300); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5)); ax.plot(sm.z_bo_l2_mean,sm.js_mean,marker="o")
    ax.set_xlabel("Mean z_bo perturbation L2"); ax.set_ylabel("Mean token Jensen-Shannon divergence")
    ax.set_title("Direct diffusion z_bo perturbation vs soft output change"); ax.grid(True,alpha=.25)
    fig.tight_layout(); fig.savefig(out_dir/"zbo_l2_vs_output_js.png",dpi=300); plt.close(fig)
    if len(nearest):
        fig,ax=plt.subplots(figsize=(8,5)); ax.hist(nearest,bins=np.arange(-.5,SEQ_LEN+1.5,1))
        ax.set_xlabel("Nearest sampled train Hamming distance"); ax.set_ylabel("Validation peptide count")
        ax.set_title("Train-validation near-duplicate audit"); fig.tight_layout()
        fig.savefig(out_dir/"train_validation_nearest_hamming.png",dpi=300); plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--training-script",required=True)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--parts-dir",required=True)
    p.add_argument("--file-pattern",default="metalpdb_ALL_chain_mapped_len10_high_confidence_part_*.csv")
    p.add_argument("--peptide-col",default="peptide_len10"); p.add_argument("--split-col",default="split")
    p.add_argument("--train-split",default="train"); p.add_argument("--validation-split",default="validation"); p.add_argument("--test-split",default="test")
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size",type=int,default=256); p.add_argument("--ddim-steps",type=int,default=50)
    p.add_argument("--bo-latent-dim",type=int,default=64); p.add_argument("--pca-fit-max-train",type=int,default=50000)
    p.add_argument("--max-validation-eval",type=int,default=10000); p.add_argument("--seed",type=int,default=0)
    p.add_argument("--sigmas",type=float,nargs="+",default=[.01,.025,.05,.10,.20]); p.add_argument("--n-centers",type=int,default=128)
    p.add_argument("--neighbors-per-sigma",type=int,default=5)
    p.add_argument("--project-generated-epsilon-to-sphere",action=argparse.BooleanOptionalAction,default=False)
    p.add_argument("--group-cols",nargs="*",default=["pdb_id","pdb","pdbid","structure_id","chain_id","chain","protein_id","uniprot_id","source_pdb","source_chain"])
    p.add_argument("--near-val-sample",type=int,default=500); p.add_argument("--near-train-sample",type=int,default=100000)
    p.add_argument("--max-audit-rows",type=int,default=0); p.add_argument("--out-dir",default="direct_sequence_diffusion_smoothness_leakage_audit")
    args=p.parse_args(); set_seed(args.seed); device=torch.device(args.device)
    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    module=import_training_module(args.training_script); model,ckpt,cfg=load_model(module,args.checkpoint,device)
    files=discover_files(args.parts_dir,args.file_pattern)
    df=load_dataframe(files,args.peptide_col,args.split_col,args.group_cols,args.max_audit_rows)
    leak,nearest=leakage_audit(df,args.train_split,args.validation_split,args.test_split,args.group_cols,args.near_val_sample,args.near_train_sample,args.seed)
    train_peps=sorted(df.loc[df.split==args.train_split.lower(),"peptide"].unique())
    val_peps=sorted(df.loc[df.split==args.validation_split.lower(),"peptide"].unique())
    rng=np.random.default_rng(args.seed)
    if args.pca_fit_max_train>0 and len(train_peps)>args.pca_fit_max_train: train_peps=rng.choice(np.asarray(train_peps,dtype=object),args.pca_fit_max_train,replace=False).tolist()
    if args.max_validation_eval>0 and len(val_peps)>args.max_validation_eval: val_peps=rng.choice(np.asarray(val_peps,dtype=object),args.max_validation_eval,replace=False).tolist()
    train_eps=collect_eps(model,train_peps,args.batch_size,args.ddim_steps,device); pca=fit_pca(train_eps,args.bo_latent_dim)
    np.savez(out_dir/"audit_direct_diffusion_bo_pca_whitener.npz",mean=pca["mean"],components=pca["components"],score_std=pca["score_std"],explained_variance_ratio=pca["explained_variance_ratio"])
    inv_d,inv_s,val_eps=raw_roundtrip(model,val_peps,args.batch_size,args.ddim_steps,device); inv_d.to_csv(out_dir/"ddim_inversion_roundtrip_details.csv",index=False)
    z_d,z_s,val_z=zbo_roundtrip(model,val_peps,val_eps,pca,args.batch_size,args.ddim_steps,device,args.project_generated_epsilon_to_sphere); z_d.to_csv(out_dir/"pca_bo_roundtrip_details.csv",index=False)
    idx=rng.choice(len(val_peps),min(args.n_centers,len(val_peps)),replace=False); centers=[val_peps[i] for i in idx]
    sd,ss,sg=local_smoothness(model,centers,val_eps[idx],pca,args.sigmas,args.neighbors_per_sigma,args.ddim_steps,args.seed,device,args.project_generated_epsilon_to_sphere)
    sd.to_csv(out_dir/"zbo_smoothness_neighbor_details.csv",index=False); ss.to_csv(out_dir/"zbo_smoothness_summary.csv",index=False)
    pd.DataFrame({"nearest_train_hamming":nearest}).to_csv(out_dir/"near_duplicate_hamming_sample.csv",index=False)
    report={"checkpoint":args.checkpoint,"checkpoint_epoch":int(ckpt.get("epoch",-1)),"checkpoint_type":ckpt.get("checkpoint_type",""),
      "split":{"method":"existing_split_column_from_chain_mapped_parts","split_col":args.split_col,"train":args.train_split,"validation":args.validation_split,"test":args.test_split},
      "leakage_audit":leak,"train_raw_epsilon_geometry":geometry(train_eps),"validation_raw_epsilon_roundtrip":inv_s,"pca_bo_coordinate":z_s,"local_z_bo_smoothness_global":sg,
      "interpretation_notes":["This audit uses the direct pretrainer's existing split column; it does not recreate a peptide-hash split.","PCA/whitening is fitted on training inverted epsilon only, matching the direct pretrainer.","Local z_bo smoothness establishes generator/coordinate locality, not Cu objective smoothness."]}
    with open(out_dir/"audit_report.json","w",encoding="utf-8") as f: json.dump(report,f,indent=2)
    lines=["DIRECT SEQUENCE DIFFUSION BO-READINESS + LEAKAGE AUDIT","="*84,f"checkpoint_epoch={report['checkpoint_epoch']}","","LEAKAGE","-"*84,json.dumps(leak,indent=2),"","TRAIN RAW EPSILON GEOMETRY","-"*84,json.dumps(report["train_raw_epsilon_geometry"],indent=2),"","VALIDATION DDIM ROUNDTRIP","-"*84,json.dumps(inv_s,indent=2),"","PCA-WHITENED z_bo","-"*84,json.dumps(z_s,indent=2),"","LOCAL z_bo SMOOTHNESS","-"*84,ss.to_string(index=False),"",json.dumps(sg,indent=2),"","CAUTION","-"*84,"These tests quantify split integrity, DDIM/PCA roundtrip quality, and local generator smoothness; they do not establish Cu objective smoothness."]
    (out_dir/"audit_report.txt").write_text("\n".join(lines)+"\n",encoding="utf-8"); save_plots(ss,nearest,out_dir)
    print("\n".join(lines)); print(f"\nSaved audit outputs to: {out_dir.resolve()}")

if __name__=="__main__": main()
