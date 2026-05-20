from __future__ import annotations

import re
import random
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Union, Tuple

import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.Data import CodonTable


# ---------- reverse translate ----------
def _aa_to_codons(table_id: int = 1) -> Dict[str, List[str]]:
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


def _parse_binding_sites(s: Any) -> List[int]:
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


def _count_overlapping_regex(pattern: str, seq: str) -> int:
    return len(re.findall(f"(?=({pattern}))", seq))


def _motif_density(seq: str, patterns: List[str]) -> float:
    seq = (seq or "").upper()
    L = max(len(seq), 1)
    motif_hits = sum(_count_overlapping_regex(p, seq) for p in patterns)
    return motif_hits / L


def _label_center_score(labels: List[int]) -> float:
    if not labels:
        return 0.5

    labels = [int(v) for v in labels]
    pos = [i for i, v in enumerate(labels) if v == 1]
    if not pos:
        return 0.5

    center = (sum(pos) / len(pos)) / max(len(labels) - 1, 1)
    center_score = 1.0 - abs(center - 0.5) / 0.5
    return float(np.clip(center_score, 0.0, 1.0))


def _gold_center_fraction(seq: str) -> float:
    seq = (seq or "").upper()
    L = len(seq)
    if L == 0:
        return 0.0

    interesting = set("CHDEKRWYF")
    start = int(np.floor(0.25 * L))
    end = int(np.ceil(0.75 * L))
    center_window = seq[start:end]
    return sum(aa in interesting for aa in center_window) / L


def _gold_features(seq: str, labels: List[int]) -> Dict[str, float]:
    seq = (seq or "").upper()
    L = max(len(seq), 1)

    frac_cys = seq.count("C") / L
    frac_his = seq.count("H") / L
    frac_acid = (seq.count("D") + seq.count("E")) / L
    frac_arom = (seq.count("W") + seq.count("Y") + seq.count("F")) / L
    frac_cat = (seq.count("K") + seq.count("R")) / L

    chelate_patterns = [
        r"C.C",      # CxC
        r"C..C",     # CxxC
        r"H.H",      # HxH
        r"H..H",     # HxxH
    ]
    surf_patterns = [
        r"WW",
        r"W.W",      # WxW
        r"KR",
        r"K.K",      # KxK
        r"R..R",     # RxxR
    ]

    m_chelate = _motif_density(seq, chelate_patterns)
    m_surf = _motif_density(seq, surf_patterns)

    # The Word file proposes either central clustering of interesting residues
    # or label-position information. We support both and blend them conservatively.
    l_center_seq = _gold_center_fraction(seq)
    l_center_lbl = _label_center_score(labels)
    l_center = 0.5 * l_center_seq + 0.5 * l_center_lbl

    return {
        "f_cys": frac_cys,
        "f_his": frac_his,
        "f_acid": frac_acid,
        "f_arom": frac_arom,
        "f_cat": frac_cat,
        "m_chelate": m_chelate,
        "m_surf": m_surf,
        "l_center": l_center,
    }


CandidateLike = Union[Dict[str, Any], Any]



def compute_final_scores(
    candidates: Sequence[CandidateLike],
    *,
    reverse_translate_strategy: str = "first",
    return_details: bool = False,
) -> Union[List[float], Tuple[List[float], pd.DataFrame]]:
    """
    Compute batch-normalized final_score using the gold formula from the attached Word file.

    Final score:
        0.50 * chelation_sub
      + 0.30 * solubility_sub
      + 0.10 * stability_sub
      + 0.10 * expression_sub

    Gold chelation_sub for tailings / ionic gold uses the solution Au(III) score:
        C_sol =
            0.35 * mm_f_cys
          + 0.25 * mm_f_his
          + 0.15 * mm_f_acid
          + 0.20 * mm_m_chelate
          + 0.05 * mm_l_center

    The surface-binding score is also computed for analysis:
        C_surf =
            0.35 * mm_f_arom
          + 0.25 * mm_f_cat
          + 0.20 * mm_m_surf
          + 0.10 * mm_f_cys
          + 0.10 * mm_l_center

    Expression term:
      - Base version uses GC-band score + homopolymer penalty.
      - If CAI and dg_rbs are provided on candidates, they are incorporated using
        the stronger expression formula described in the Word file.
    """
    rows = []

    for c in candidates:
        seq = (c.get("sequence") if isinstance(c, dict) else getattr(c, "sequence", "")) or ""
        bs = (c.get("binding_sites") if isinstance(c, dict) else getattr(c, "binding_sites", "")) or ""
        dna = (c.get("DNA_sequence") if isinstance(c, dict) else getattr(c, "DNA_sequence", "")) or ""
        metal = (c.get("metal") if isinstance(c, dict) else getattr(c, "metal", "")) or ""
        cai = c.get("CAI") if isinstance(c, dict) else getattr(c, "CAI", np.nan)
        if pd.isna(cai):
            cai = c.get("cai") if isinstance(c, dict) else getattr(c, "cai", np.nan)
        dg_rbs = c.get("dg_rbs") if isinstance(c, dict) else getattr(c, "dg_rbs", np.nan)

        seq = seq.strip().upper().replace("*", "").replace("U", "")
        if not seq or any(ch not in "ACDEFGHIKLMNPQRSTVWY" for ch in seq):
            continue

        dna = dna.strip().upper()
        if not dna:
            dna = reverse_translate(seq, strategy=reverse_translate_strategy)

        labels = _parse_binding_sites(bs)

        pa = ProteinAnalysis(seq)
        mw = float(pa.molecular_weight())
        arom = float(pa.aromaticity())
        instab = float(pa.instability_index())
        gravy = float(pa.gravy())

        agg = (c.get("aggregation_scores") if isinstance(c, dict) else getattr(c, "aggregation_scores", None))
        a3d_mean = np.nan
        if isinstance(agg, list) and len(agg) > 0:
            try:
                a3d_mean = float(np.mean([float(x) for x in agg]))
            except Exception:
                a3d_mean = np.nan

        gold_feats = _gold_features(seq, labels)

        rows.append({
            "sequence": seq,
            "length": int(len(seq)),
            "binding_sites": bs,
            "DNA_sequence": dna,
            "metal": metal.strip().upper(),
            "molecular_weight": mw,
            "aromaticity": arom,
            "instability_index": instab,
            "gravy": gravy,
            "a3d_mean": a3d_mean,
            "gc_frac": _gc_fraction(dna),
            "longest_run": _longest_homopolymer(dna),
            "CAI": pd.to_numeric(cai, errors="coerce"),
            "dg_rbs": pd.to_numeric(dg_rbs, errors="coerce"),
            **gold_feats,
        })

    if not rows:
        return ([], pd.DataFrame()) if return_details else []

    df = pd.DataFrame(rows)

    # ---- Gold chelation terms ----
    mm_f_cys = _minmax(df["f_cys"])
    mm_f_his = _minmax(df["f_his"])
    mm_f_acid = _minmax(df["f_acid"])
    mm_f_arom = _minmax(df["f_arom"])
    mm_f_cat = _minmax(df["f_cat"])
    mm_m_chelate = _minmax(df["m_chelate"])
    mm_m_surf = _minmax(df["m_surf"])
    mm_l_center = _minmax(df["l_center"])

    df["C_sol"] = (
        0.35 * mm_f_cys
        + 0.25 * mm_f_his
        + 0.15 * mm_f_acid
        + 0.20 * mm_m_chelate
        + 0.05 * mm_l_center
    )

    df["C_surf"] = (
        0.35 * mm_f_arom
        + 0.25 * mm_f_cat
        + 0.20 * mm_m_surf
        + 0.10 * mm_f_cys
        + 0.10 * mm_l_center
    )

    # Tailings / soluble gold ranking uses C_sol as the chelation term.
    df["chelation_sub"] = df["C_sol"]

    # ---- solubility_sub (sign-corrected per Word file) ----
    a3d_fill = df["a3d_mean"].median() if df["a3d_mean"].notna().any() else 0.0
    a3d_good = _minmax_reverse(df["a3d_mean"].fillna(a3d_fill))
    gravy_good = _minmax_reverse(df["gravy"])
    arom_good = _minmax_reverse(df["aromaticity"])
    df["solubility_sub"] = 0.6 * a3d_good + 0.3 * gravy_good + 0.1 * arom_good

    # ---- stability_sub (sign-corrected) ----
    instab_good = _minmax_reverse(df["instability_index"])
    L_med = df["length"].median()
    len_dev = (df["length"] - L_med).abs() / max(float(L_med), 1.0)
    len_score_raw = (1.0 - len_dev).clip(lower=0.0)
    len_good = _minmax(len_score_raw)
    df["stability_sub"] = 0.7 * instab_good + 0.3 * len_good

    # ---- expression_sub ----
    gc_dev = (df["gc_frac"] - 0.5).abs()
    gc_score_raw = (1.0 - gc_dev / 0.5).clip(lower=0.0)
    gc_good = _minmax(gc_score_raw)
    run_good = _minmax_reverse(df["longest_run"])

    has_cai = df["CAI"].notna().any()
    has_dg = df["dg_rbs"].notna().any()

    if has_cai or has_dg:
        cai_mm = _minmax(df["CAI"].fillna(df["CAI"].median() if has_cai else 0.5))
        dg_inv = -df["dg_rbs"].fillna(df["dg_rbs"].median() if has_dg else 0.0)
        dg_mm = _minmax(dg_inv)
        df["expression_sub"] = (
            0.25 * gc_good
            + 0.15 * run_good
            + 0.35 * cai_mm
            + 0.25 * dg_mm
        )
    else:
        df["expression_sub"] = 0.7 * gc_good + 0.3 * run_good

    # ---- final score ----
    df["final_score"] = (
        0.50 * df["chelation_sub"]
        + 0.30 * df["solubility_sub"]
        + 0.10 * df["stability_sub"]
        + 0.10 * df["expression_sub"]
    )

    scores = [float(x) for x in df["final_score"].tolist()]
    return (scores, df) if return_details else scores
