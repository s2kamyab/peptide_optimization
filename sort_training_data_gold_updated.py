import os
import pathlib
import re
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.Data import CodonTable
from run_a3d import run_a3d_on_pdb
from run_colabfold import run_colabfold_wsl

CSV_IN = "metalpdb_Au_binding_windows_len10.csv"
CSV_OUT = "metalpdb_binding_windows_len10_AU_scored_ranked.csv"

# A3D (Windows executable) - keep as you had it
A3D_EXE = r"py27venv\Scripts\aggrescan.exe"

AA_OK = set("ACDEFGHIKLMNPQRSTVWY")


# ---------- reverse translate ----------
def _aa_to_codons(table_id: int = 1):
    table = CodonTable.unambiguous_dna_by_id[table_id]
    aa_to_codons = defaultdict(list)
    for codon, aa in table.forward_table.items():
        aa_to_codons[aa].append(codon)
    return dict(aa_to_codons)


def reverse_translate(protein: str, strategy: str = "first") -> str:
    rnd = random.Random(0)
    protein = protein.strip().upper()
    aa2codons = _aa_to_codons()

    def choose_codon(aa: str) -> str:
        if aa not in aa2codons:
            return "NNN"
        opts = aa2codons[aa]
        if strategy == "first":
            return opts[0]
        if strategy == "random":
            return rnd.choice(opts)
        raise ValueError("strategy must be one of {'first','random'}")

    return "".join(choose_codon(aa) for aa in protein)


# ---------- helpers ----------
def _minmax(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(0.5, index=s.index)
    return (s - mn) / (mx - mn)


def _minmax_reverse(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(0.5, index=s.index)
    return (mx - s) / (mx - mn)


def _gc_fraction(dna: str) -> float:
    dna = (dna or "").upper()
    if not dna:
        return 0.0
    return (dna.count("G") + dna.count("C")) / len(dna)


def _longest_homopolymer(dna: str) -> int:
    dna = (dna or "").upper()
    if not dna:
        return 0
    best = cur = 1
    for i in range(1, len(dna)):
        if dna[i] == dna[i - 1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def _parse_binding_sites(s):
    if isinstance(s, list):
        return [int(x) for x in s]
    if isinstance(s, str):
        toks = [t for t in s.strip().split() if t]
        out = []
        for t in toks:
            try:
                out.append(int(t))
            except Exception:
                out.append(0)
        return out
    return []


def _gold_chelate_motif_density(seq: str) -> float:
    seq = (seq or "").upper()
    L = max(len(seq), 1)
    patterns = [
        r"C.C",     # CXC
        r"C..C",    # CXXC
        r"H.H",     # HXH
        r"H..H",    # HXXH
    ]
    motif_hits = sum(len(re.findall(p, seq)) for p in patterns)
    return motif_hits / L


def _gold_surface_motif_density(seq: str) -> float:
    seq = (seq or "").upper()
    L = max(len(seq), 1)
    patterns = [
        r"WW",
        r"W.W",     # WXW
        r"KR",
        r"K.K",     # KXK
        r"R..R",    # RXXR
    ]
    motif_hits = sum(len(re.findall(p, seq)) for p in patterns)
    return motif_hits / L


def _gold_center_density(seq: str) -> float:
    seq = (seq or "").upper()
    L = len(seq)
    if L == 0:
        return 0.0

    center_start = int(np.floor(0.25 * L))
    center_end = int(np.ceil(0.75 * L))
    center_region = seq[center_start:center_end]
    interesting = set("CHDEKRWYF")
    center_hits = sum(aa in interesting for aa in center_region)
    return center_hits / L


def _gold_features(seq: str):
    seq = (seq or "").upper()
    L = max(len(seq), 1)

    return {
        "f_cys": seq.count("C") / L,
        "f_his": seq.count("H") / L,
        "f_acid": (seq.count("D") + seq.count("E")) / L,
        "f_arom": (seq.count("W") + seq.count("Y") + seq.count("F")) / L,
        "f_cat": (seq.count("K") + seq.count("R")) / L,
        "m_chelate": _gold_chelate_motif_density(seq),
        "m_surf": _gold_surface_motif_density(seq),
        "l_center": _gold_center_density(seq),
    }


def run_colabfold_here(seq: str, out_dir: str, threads: int = 8) -> str:
    """
    Runs ColabFold inside WSL (conda env colabfold) and returns the produced PDB file path (Windows path).
    Fast CPU settings:
      - msa_mode single_sequence
      - num_models 1
      - num_recycle 1
      - model_type alphafold2
    """
    pdb_path = run_colabfold_wsl(
        seq=seq,
        out_dir_win=out_dir,
        threads=threads,
        num_models=1,
        num_recycle=1,
        msa_mode="single_sequence",
        model_type="alphafold2",
    )
    return pdb_path


def compute_scores_gold_df(df_in: pd.DataFrame, reverse_translate_strategy: str = "first") -> pd.DataFrame:
    df = df_in.copy()

    # --- clean + filter sequences fast ---
    df["peptide_len10"] = (
        df["peptide_len10"].astype(str).str.strip().str.upper()
          .str.replace("*", "", regex=False)
          .str.replace("U", "", regex=False)
    )
    df = df[df["peptide_len10"].str.match(r"^[ACDEFGHIKLMNPQRSTVWY]+$", na=False)].copy()

    # --- DNA fill ---
    if "DNA_sequence" not in df.columns:
        df["DNA_sequence"] = ""
    dna = df["DNA_sequence"].astype(str).str.strip().str.upper()
    need_dna = dna.eq("")
    if need_dna.any():
        uniq = df.loc[need_dna, "peptide_len10"].unique()
        rt_cache = {s: reverse_translate(s, strategy=reverse_translate_strategy) for s in uniq}
        df.loc[need_dna, "DNA_sequence"] = df.loc[need_dna, "peptide_len10"].map(rt_cache)

    # --- binding site labels ---
    if "binding_site_labels_len10" not in df.columns:
        df["binding_site_labels_len10"] = ""
    labels = [_parse_binding_sites(bs) for bs in df["binding_site_labels_len10"].astype(str).tolist()]

    # --- ProteinAnalysis features (per unique sequence) ---
    uniq_seq = df["peptide_len10"].unique()
    pa_cache = {}
    for s in uniq_seq:
        pa = ProteinAnalysis(s)
        pa_cache[s] = (
            float(pa.molecular_weight()),
            float(pa.aromaticity()),
            float(pa.instability_index()),
            float(pa.gravy()),
        )

    mw, arom, instab, gravy = zip(*df["peptide_len10"].map(pa_cache))
    df["molecular_weight"] = mw
    df["aromaticity"] = arom
    df["instability_index"] = instab
    df["gravy"] = gravy

    # ------------------------------------------------------------------
    # Load A3D.csv per peptide and compute mean(score)
    # ------------------------------------------------------------------
    A3D_ROOT = Path("AU_metaldb_a3ds")
    A3D_FILENAME = "A3D.csv"

    def _read_a3d_mean(seq: str) -> float:
        csv_path = A3D_ROOT / seq / A3D_FILENAME
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            return float("nan")
        try:
            a3d = pd.read_csv(csv_path)
            cols_lower = {c.lower(): c for c in a3d.columns}
            if "score" not in cols_lower:
                return float("nan")
            score_col = cols_lower["score"]
            scores = pd.to_numeric(a3d[score_col], errors="coerce")
            return float(scores.mean()) if scores.notna().any() else float("nan")
        except Exception:
            return float("nan")

    a3d_means = {s: _read_a3d_mean(s) for s in uniq_seq}
    df["a3d_mean"] = df["peptide_len10"].map(a3d_means)

    # --- gold chelation features ---
    gold_rows = [_gold_features(seq) for seq in df["peptide_len10"].tolist()]
    gold_df = pd.DataFrame(gold_rows)
    df = pd.concat([df.reset_index(drop=True), gold_df.reset_index(drop=True)], axis=1)

    # --- optional label-aware center proxy ---
    if labels:
        label_center_proxy = []
        for seq, lab in zip(df["peptide_len10"].tolist(), labels):
            if not lab:
                label_center_proxy.append(np.nan)
                continue
            pos = [i for i, v in enumerate(lab) if int(v) == 1]
            if not pos:
                label_center_proxy.append(np.nan)
                continue
            n = max(len(lab) - 1, 1)
            center = (sum(pos) / len(pos)) / n
            center_score = 1.0 - abs(center - 0.5) / 0.5
            label_center_proxy.append(float(np.clip(center_score, 0.0, 1.0)))
        df["label_center_proxy"] = label_center_proxy
        if df["label_center_proxy"].notna().any():
            df["l_center"] = 0.5 * df["l_center"] + 0.5 * df["label_center_proxy"].fillna(df["l_center"])

    # --- expression helpers ---
    df["gc_frac"] = [_gc_fraction(d) for d in df["DNA_sequence"].tolist()]
    df["longest_run"] = [_longest_homopolymer(d) for d in df["DNA_sequence"].tolist()]

    # ========= Gold chelation terms =========
    mm_f_cys = _minmax(df["f_cys"])
    mm_f_his = _minmax(df["f_his"])
    mm_f_acid = _minmax(df["f_acid"])
    mm_f_arom = _minmax(df["f_arom"])
    mm_f_cat = _minmax(df["f_cat"])
    mm_m_chelate = _minmax(df["m_chelate"])
    mm_m_surf = _minmax(df["m_surf"])
    mm_l_center = _minmax(df["l_center"])

    df["C_sol"] = (
        0.35 * mm_f_cys +
        0.25 * mm_f_his +
        0.15 * mm_f_acid +
        0.20 * mm_m_chelate +
        0.05 * mm_l_center
    )

    df["C_surf"] = (
        0.35 * mm_f_arom +
        0.25 * mm_f_cat +
        0.20 * mm_m_surf +
        0.10 * mm_f_cys +
        0.10 * mm_l_center
    )

    # Tailings / ionic gold ranking uses solution Au3+ chelation
    df["chelation_sub"] = df["C_sol"]

    # ========= Other subscores =========
    a3d_fill = df["a3d_mean"].median() if df["a3d_mean"].notna().any() else 0.0
    a3d_good = _minmax_reverse(df["a3d_mean"].fillna(a3d_fill))
    gravy_good = _minmax_reverse(df["gravy"])
    arom_good = _minmax_reverse(df["aromaticity"])
    df["solubility_sub"] = 0.6 * a3d_good + 0.3 * gravy_good + 0.1 * arom_good

    instab_good = _minmax_reverse(df["instability_index"])
    L_med = df["peptide_len10"].str.len().median()
    len_dev = (df["peptide_len10"].str.len() - L_med).abs() / max(float(L_med), 1.0)
    len_score_raw = (1.0 - len_dev).clip(lower=0.0)
    len_good = _minmax(len_score_raw)
    df["stability_sub"] = 0.7 * instab_good + 0.3 * len_good

    gc_dev = (df["gc_frac"] - 0.5).abs()
    gc_score_raw = (1.0 - gc_dev / 0.5).clip(lower=0.0)
    gc_good = _minmax(gc_score_raw)
    run_good = _minmax_reverse(df["longest_run"])

    if "CAI" in df.columns:
        cai_mm = _minmax(pd.to_numeric(df["CAI"], errors="coerce").fillna(df["CAI"].astype(float).median() if pd.to_numeric(df["CAI"], errors="coerce").notna().any() else 0.0))
    else:
        cai_mm = None

    if "dg_rbs" in df.columns:
        dg_num = pd.to_numeric(df["dg_rbs"], errors="coerce")
        dg_fill = dg_num.median() if dg_num.notna().any() else 0.0
        dg_mm = _minmax((-dg_num).fillna(-dg_fill))
    else:
        dg_mm = None

    if cai_mm is not None and dg_mm is not None:
        df["expression_sub"] = (
            0.25 * gc_good +
            0.15 * run_good +
            0.35 * cai_mm +
            0.25 * dg_mm
        )
    else:
        df["expression_sub"] = 0.7 * gc_good + 0.3 * run_good

    df["final_score"] = (
        0.50 * df["chelation_sub"] +
        0.30 * df["solubility_sub"] +
        0.10 * df["stability_sub"] +
        0.10 * df["expression_sub"]
    )

    return df


def main():
    df_in = pd.read_csv(CSV_IN)
    df_in = df_in.drop_duplicates(subset=["peptide_len10"], keep="first").reset_index(drop=True)

    df_out = compute_scores_gold_df(df_in)
    df_out = df_out.sort_values("final_score", ascending=False).reset_index(drop=True)

    df_out.to_csv(CSV_OUT, index=False)
    print(f"Saved ranked peptides to: {CSV_OUT}")


if __name__ == "__main__":
    main()
