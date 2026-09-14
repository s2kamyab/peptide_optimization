from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

try:
    from scipy.stats import chi2, kstest, skew, kurtosis
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_I = {a: i for i, a in enumerate(AA)}
SEQ_LEN = 10
VOCAB = 20


def clean_peptide(x):
    p = str(x).strip().upper()
    if len(p) != SEQ_LEN or any(a not in AA_TO_I for a in p):
        return None
    return p


def onehot_encode(peptides: Sequence[str]) -> torch.Tensor:
    x = torch.zeros(len(peptides), SEQ_LEN, VOCAB, dtype=torch.float32)
    for i, pep in enumerate(peptides):
        for t, aa in enumerate(pep):
            x[i, t, AA_TO_I[aa]] = 1.0
    return x


def import_module(path: str):
    path = str(Path(path).expanduser().resolve())
    name = "latent_diffusion_finetune_for_trust_audit"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def torch_load_full(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.no_grad()
def load_model(module, checkpoint, device):
    ckpt = torch_load_full(checkpoint, device)
    cfg = module.ModelConfig(**ckpt["model_config"])
    diff_cfg = module.DiffusionConfig(**ckpt["diffusion_config"])
    vae = module.GRUVAE(cfg).to(device)
    diffusion = module.LatentDiffusion(cfg.latent_dim, diff_cfg).to(device)
    vae.load_state_dict(ckpt["vae_state_dict"], strict=True)
    diffusion.load_state_dict(ckpt["diffusion_state_dict"], strict=True)
    vae.eval()
    diffusion.eval()
    latent_mean = ckpt["latent_mean"].to(device).float()
    latent_std = ckpt["latent_std"].to(device).float().clamp_min(1e-6)
    ddim_steps = int((ckpt.get("args", {}) or {}).get("ddim_steps", 50))
    return vae, diffusion, ckpt, latent_mean, latent_std, ddim_steps


def resolve_cols(df, prefixes, dim):
    for prefix in prefixes:
        cols = [f"{prefix}{i:02d}" for i in range(dim)]
        if all(c in df.columns for c in cols):
            return cols, prefix
    raise KeyError(f"Could not find complete vector for prefixes {prefixes}")


@torch.no_grad()
def encode_peptides_native(
    vae, diffusion, peptides, latent_mean, latent_std, ddim_steps, device, batch_size
):
    E, H = [], []
    for s in range(0, len(peptides), batch_size):
        batch = peptides[s:s+batch_size]
        x = onehot_encode(batch).to(device)
        mu, _, _ = vae.enc(x)
        h0 = (mu - latent_mean) / latent_std
        eps = diffusion.ddim_invert(h0, inference_steps=ddim_steps)
        E.append(eps.cpu())
        H.append(h0.cpu())
    return torch.cat(E), torch.cat(H)


@torch.no_grad()
def roundtrip_l2(diffusion, E, H, ddim_steps, device, batch_size):
    vals = []
    for s in range(0, len(E), batch_size):
        e = E[s:s+batch_size].to(device)
        h = H[s:s+batch_size].to(device)
        hb = diffusion.ddim_sample(e, inference_steps=ddim_steps)
        vals.append(torch.linalg.norm(hb-h, dim=-1).cpu())
    return torch.cat(vals).numpy()


def stdnormal_summary(E):
    n, d = E.shape
    means = E.mean(0)
    stds = E.std(0, ddof=0)
    r2 = (E**2).sum(1)
    logp = -0.5 * (d*np.log(2*np.pi) + r2)
    out = {
        "n": n,
        "dim": d,
        "mean_abs_coordinate_mean": float(np.abs(means).mean()),
        "max_abs_coordinate_mean": float(np.abs(means).max()),
        "coordinate_std_mean": float(stds.mean()),
        "mean_abs_std_minus_1": float(np.abs(stds-1).mean()),
        "epsilon_norm_mean": float(np.sqrt(r2).mean()),
        "epsilon_norm_std": float(np.sqrt(r2).std(ddof=0)),
        "squared_norm_mean": float(r2.mean()),
        "expected_squared_norm_N01": float(d),
        "mean_logp_per_dim_N01": float((logp/d).mean()),
    }
    if SCIPY_OK:
        lo, hi = chi2.ppf(0.005, d), chi2.ppf(0.995, d)
        out["chi2_0.5pct"] = float(lo)
        out["chi2_99.5pct"] = float(hi)
        out["fraction_inside_0.5_99.5pct_radial_shell"] = float(np.mean((r2>=lo)&(r2<=hi)))
    return out


def dim_diagnostics(E):
    rows = []
    for j in range(E.shape[1]):
        x = E[:, j]
        row = {"dimension": j, "mean": x.mean(), "std": x.std(ddof=0)}
        if SCIPY_OK:
            row["skew"] = skew(x, bias=False)
            row["excess_kurtosis"] = kurtosis(x, fisher=True, bias=False)
            ks, p = kstest(x, "norm")
            row["ks_stat_vs_N01"] = ks
            row["ks_p_vs_N01"] = p
            row["ks_pass_p_ge_0.05"] = bool(p >= 0.05)
        rows.append(row)
    return pd.DataFrame(rows)


def fit_empirical_gaussian(E, shrinkage):
    mu = E.mean(0)
    X = E-mu
    cov = np.cov(X, rowvar=False, ddof=1)
    diag = np.diag(np.diag(cov))
    cov = (1-shrinkage)*cov + shrinkage*diag
    cov += np.eye(cov.shape[0]) * max(np.diag(cov).mean(), 1e-8) * 1e-6
    inv = np.linalg.pinv(cov)
    return mu, inv


def maha_sq(E, mu, inv):
    X = E-mu
    return np.einsum("ni,ij,nj->n", X, inv, X)


def empirical_percentile(ref, values):
    r = np.sort(np.asarray(ref, float))
    v = np.asarray(values, float)
    return np.searchsorted(r, v, side="right") / max(1, len(r))


def nearest_dist(query, train, self_query=False, chunk=256):
    q = torch.tensor(query, dtype=torch.float32)
    t = torch.tensor(train, dtype=torch.float32)
    out = []
    for s in range(0, len(q), chunk):
        qb = q[s:s+chunk]
        d = torch.cdist(qb, t)
        if self_query:
            rr = torch.arange(len(qb))
            cc = torch.arange(s, s+len(qb))
            d[rr, cc] = float("inf")
        out.append(d.min(1).values)
    return torch.cat(out).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--finetune-script", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--coordinate-csv", required=True)
    p.add_argument("--bo-candidates", required=True)
    p.add_argument("--bo-pareto", default=None)
    p.add_argument("--peptide-col", default="peptide")
    p.add_argument("--out-dir", default="latent_diffusion_bo_trustworthiness_audit")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--chi-low", type=float, default=0.005)
    p.add_argument("--chi-high", type=float, default=0.995)
    p.add_argument("--mahalanobis-quantile", type=float, default=0.995)
    p.add_argument("--nn-quantile", type=float, default=0.995)
    p.add_argument("--roundtrip-quantile", type=float, default=0.995)
    p.add_argument("--covariance-shrinkage", type=float, default=0.05)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    module = import_module(args.finetune_script)
    vae, diffusion, ckpt, latent_mean, latent_std, ddim_steps = load_model(
        module, args.checkpoint, device
    )
    d = int(vae.cfg.latent_dim)

    ref = pd.read_csv(args.coordinate_csv)
    eps_cols, eps_prefix = resolve_cols(ref, ["epsilon_raw_", "epsilon_"], d)
    if "split" not in ref.columns:
        raise KeyError("coordinate CSV must include split")

    if eps_prefix != "epsilon_raw_":
        print("WARNING: epsilon_raw_* not found. If epsilon_* is sphere-projected, N(0,I) radial likelihood is not valid.")

    E_all = ref[eps_cols].to_numpy(float)
    split = ref["split"].astype(str).to_numpy()
    tr = split == "train"
    ho = np.isin(split, ["val","test"])
    E_train = E_all[tr]
    E_hold = E_all[ho]

    pd.DataFrame([
        {"group":"train", **stdnormal_summary(E_train)},
        {"group":"val_test", **stdnormal_summary(E_hold)},
        {"group":"all", **stdnormal_summary(E_all)},
    ]).to_csv(out/"reference_standard_normal_summary.csv", index=False)

    ddiag = dim_diagnostics(E_train)
    ddiag.to_csv(out/"reference_dimension_diagnostics.csv", index=False)

    mu_emp, inv_cov = fit_empirical_gaussian(E_train, args.covariance_shrinkage)
    tr_maha = maha_sq(E_train, mu_emp, inv_cov)
    tr_nn = nearest_dist(E_train, E_train, self_query=True)

    if "ddim_h_l2" in ref.columns:
        rt_ref = pd.to_numeric(ref.loc[tr, "ddim_h_l2"], errors="coerce").dropna().to_numpy(float)
    else:
        pcol = "peptide" if "peptide" in ref.columns else "peptide_len10"
        peps_ref = ref.loc[tr, pcol].map(clean_peptide).dropna().tolist()
        Er, Hr = encode_peptides_native(vae,diffusion,peps_ref,latent_mean,latent_std,ddim_steps,device,args.batch_size)
        rt_ref = roundtrip_l2(diffusion,Er,Hr,ddim_steps,device,args.batch_size)

    maha_thr = float(np.quantile(tr_maha, args.mahalanobis_quantile))
    nn_thr = float(np.quantile(tr_nn, args.nn_quantile))
    rt_thr = float(np.quantile(rt_ref, args.roundtrip_quantile))

    cand = pd.read_csv(args.bo_candidates)
    pcol = args.peptide_col if args.peptide_col in cand.columns else next(
        (c for c in ["peptide","peptide_len10","sequence"] if c in cand.columns), None
    )
    if pcol is None:
        raise KeyError("No peptide column found in BO candidate CSV")

    cand["_peptide_clean"] = cand[pcol].map(clean_peptide)
    cand = cand[cand["_peptide_clean"].notna()].drop_duplicates("_peptide_clean").reset_index(drop=True)
    peps = cand["_peptide_clean"].tolist()

    E_t, H_t = encode_peptides_native(vae,diffusion,peps,latent_mean,latent_std,ddim_steps,device,args.batch_size)
    E = E_t.numpy().astype(float)
    rt = roundtrip_l2(diffusion,E_t,H_t,ddim_steps,device,args.batch_size)

    r2 = (E**2).sum(1)
    norms = np.sqrt(r2)
    logp = -0.5 * (d*np.log(2*np.pi)+r2)
    radial_z = (r2-d)/math.sqrt(2*d)

    if SCIPY_OK:
        chi_pct = chi2.cdf(r2,d)
        radial_ok = (chi_pct >= args.chi_low) & (chi_pct <= args.chi_high)
    else:
        chi_pct = np.full(len(E), np.nan)
        radial_ok = np.abs(radial_z) <= 3

    m = maha_sq(E,mu_emp,inv_cov)
    m_pct = empirical_percentile(tr_maha,m)
    m_ok = m <= maha_thr

    nn = nearest_dist(E,E_train)
    nn_pct = empirical_percentile(tr_nn,nn)
    nn_ok = nn <= nn_thr

    rt_ok = rt <= rt_thr
    trusted = radial_ok & m_ok & nn_ok & rt_ok
    score = (radial_ok.astype(float)+m_ok.astype(float)+nn_ok.astype(float)+rt_ok.astype(float))/4

    trust = pd.DataFrame({
        "epsilon_norm": norms,
        "epsilon_squared_norm": r2,
        "standard_normal_radial_z": radial_z,
        "standard_normal_chi2_percentile": chi_pct,
        "standard_normal_logp": logp,
        "standard_normal_logp_per_dim": logp/d,
        "empirical_mahalanobis_sq": m,
        "empirical_mahalanobis_train_percentile": m_pct,
        "nearest_train_epsilon_l2": nn,
        "nearest_train_distance_train_percentile": nn_pct,
        "ddim_native_roundtrip_h_l2": rt,
        "passes_standard_normal_radial": radial_ok,
        "passes_empirical_mahalanobis": m_ok,
        "passes_nearest_train_distance": nn_ok,
        "passes_roundtrip_fidelity": rt_ok,
        "trust_score_0_to_1": score,
        "passes_all_trust_filters": trusted,
    })

    audited = pd.concat([cand,trust],axis=1)
    for j in range(d):
        audited[f"native_epsilon_{j:02d}"] = E[:,j]

    audited.to_csv(out/"candidate_trustworthiness.csv", index=False)
    audited[audited["passes_all_trust_filters"]].to_csv(out/"trustworthy_bo_candidates.csv", index=False)
    audited[~audited["passes_all_trust_filters"]].to_csv(out/"rejected_low_trust_bo_candidates.csv", index=False)

    flags = [
        "passes_standard_normal_radial",
        "passes_empirical_mahalanobis",
        "passes_nearest_train_distance",
        "passes_roundtrip_fidelity",
        "passes_all_trust_filters",
    ]
    rows = []
    for f in flags:
        rows.append({
            "criterion": f,
            "n_pass": int(audited[f].sum()),
            "n_fail": int((~audited[f]).sum()),
            "fraction_pass": float(audited[f].mean()),
        })
    pd.DataFrame(rows).to_csv(out/"candidate_trust_summary.csv", index=False)

    thresholds = {
        "latent_dim": d,
        "epsilon_columns_used": eps_prefix,
        "native_likelihood_interpretation_valid": eps_prefix == "epsilon_raw_",
        "chi_low": args.chi_low,
        "chi_high": args.chi_high,
        "mahalanobis_threshold": maha_thr,
        "nearest_train_l2_threshold": nn_thr,
        "roundtrip_l2_threshold": rt_thr,
        "train_epsilon_norm_mean": float(np.linalg.norm(E_train,axis=1).mean()),
        "train_epsilon_norm_std": float(np.linalg.norm(E_train,axis=1).std(ddof=0)),
        "expected_standard_normal_radius_approx": math.sqrt(d),
    }
    (out/"reference_distance_thresholds.json").write_text(json.dumps(thresholds,indent=2))

    pareto_summary = None
    if args.bo_pareto:
        pf = pd.read_csv(args.bo_pareto)
        pp = next((c for c in [pcol,"peptide","peptide_len10"] if c in pf.columns), None)
        if pp is None:
            raise KeyError("No peptide column in --bo-pareto CSV")
        pf["_peptide_clean"] = pf[pp].map(clean_peptide)
        keep_cols = ["_peptide_clean"] + list(trust.columns)
        pf2 = pf.merge(pd.concat([cand[["_peptide_clean"]],trust],axis=1)[keep_cols],
                       on="_peptide_clean", how="left", validate="m:1")
        pf2.to_csv(out/"bo_pareto_with_trust_metrics.csv", index=False)
        pf_tr = pf2[pf2["passes_all_trust_filters"] == True].copy()
        pf_tr.to_csv(out/"trustworthy_bo_pareto.csv", index=False)
        pareto_summary = {
            "n_input": int(len(pf2)),
            "n_trusted": int(len(pf_tr)),
            "fraction_trusted": float(len(pf_tr)/max(1,len(pf2))),
        }

    ks_fraction = (
        float(ddiag["ks_pass_p_ge_0.05"].mean())
        if SCIPY_OK and "ks_pass_p_ge_0.05" in ddiag.columns else None
    )

    summary = {
        "checkpoint_epoch": int(ckpt.get("epoch",-1)),
        "checkpoint_selection": ckpt.get("checkpoint_selection",""),
        "ddim_steps": ddim_steps,
        "epsilon_space_audited": "native DDIM epsilon before optional sphere projection",
        "reference_train": stdnormal_summary(E_train),
        "reference_val_test": stdnormal_summary(E_hold),
        "fraction_train_dimensions_passing_KS_p_ge_0.05": ks_fraction,
        "thresholds": thresholds,
        "bo_candidates": {
            "n": int(len(audited)),
            "n_trusted": int(trusted.sum()),
            "fraction_trusted": float(trusted.mean()),
            "mean_trust_score": float(score.mean()),
        },
        "pareto_post_filter": pareto_summary,
        "interpretation": {
            "radial": "Tests whether ||epsilon||^2 lies in a plausible chi-square(d) region under N(0,I).",
            "empirical": "Mahalanobis and nearest-neighbor checks detect OOD candidates relative to actual Cu training epsilon geometry.",
            "roundtrip": "Rejects candidates whose peptide->epsilon->h0 reconstruction is unusually poor relative to training.",
            "sphere_warning": "Do not assess Gaussian likelihood after fixed-radius sphere projection because projection erases radial information."
        }
    }
    (out/"trustworthiness_audit_summary.json").write_text(json.dumps(summary,indent=2))

    print("\nAUDIT COMPLETE")
    print(f"Candidates: {len(audited)}")
    print(f"Trusted: {trusted.sum()} ({100*trusted.mean():.1f}%)")
    print(f"Outputs: {out.resolve()}")


if __name__ == "__main__":
    main()
