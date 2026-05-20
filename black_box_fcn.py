from __future__ import annotations

import re
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Union, Tuple

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

def _chel_features(seq: str, labels: List[int]) -> Dict[str, float]:
    seq = (seq or "").upper()
    L = max(len(seq), 1)

    soft = set("CMHDE")
    arom = set("WYF")
    cat = set("KR")

    frac_soft = sum(aa in soft for aa in seq) / L
    frac_arom = sum(aa in arom for aa in seq) / L
    frac_cat  = sum(aa in cat  for aa in seq) / L

    patterns = [r"H.H", r"C.C", r"H..H", r"C..C"]
    motif_hits = sum(len(re.findall(p, seq)) for p in patterns)
    motif_density = motif_hits / L

    label_fraction = 0.0
    center_score = 0.5
    if labels:
        n1 = sum(1 for v in labels if int(v) == 1)
        label_fraction = n1 / max(len(labels), 1)
        pos = [i for i, v in enumerate(labels) if int(v) == 1]
        if pos:
            center = (sum(pos) / len(pos)) / max(L - 1, 1)
            center_score = 1.0 - abs(center - 0.5) / 0.5

    return {
        "frac_soft": frac_soft,
        "frac_arom": frac_arom,
        "frac_cat": frac_cat,
        "motif_density": motif_density,
        "label_fraction": label_fraction,
        "label_center_score": center_score,
    }


CandidateLike = Union[Dict[str, Any], Any]  # dict or object with attributes


def compute_final_scores(
    candidates: Sequence[CandidateLike],
    *,
    reverse_translate_strategy: str = "first",
    return_details: bool = False,
) -> Union[List[float], Tuple[List[float], pd.DataFrame]]:
    """
    Compute batch-normalized final_score exactly like your service.

    Inputs:
      candidates: each candidate must provide:
        - sequence (str)
        - binding_sites (str or list; can be empty)
        - DNA_sequence (optional; can be empty)
        - metal (optional)

      It will skip invalid sequences (non-AA chars).

    Returns:
      - List[final_score] aligned with the *kept* candidates (invalid ones are dropped),
        OR (scores, df_details) if return_details=True.
    """
    rows = []
    kept_idx = []  # indices of original candidates we kept

    for i, c in enumerate(candidates):
        # support dict or object
        seq = (c.get("sequence") if isinstance(c, dict) else getattr(c, "sequence", "")) or ""
        bs  = (c.get("binding_sites") if isinstance(c, dict) else getattr(c, "binding_sites", "")) or ""
        dna = (c.get("DNA_sequence") if isinstance(c, dict) else getattr(c, "DNA_sequence", "")) or ""
        metal = (c.get("metal") if isinstance(c, dict) else getattr(c, "metal", "")) or ""

        seq = seq.strip().upper().replace("*", "").replace("U", "")
        if not seq or any(ch not in "ACDEFGHIKLMNPQRSTVWY" for ch in seq):
            continue

        dna = dna.strip().upper()
        if not dna:
            dna = reverse_translate(seq, strategy=reverse_translate_strategy)

        labels = _parse_binding_sites(bs)

        pa = ProteinAnalysis(seq)
        if not seq:
            mw = arom = instab = gravy = None  # or np.nan, or skip this record
        else:
            pa = ProteinAnalysis(seq)
            mw = float(pa.molecular_weight())
            arom = float(pa.aromaticity())
            instab = float(pa.instability_index())
            gravy = float(pa.gravy())
        
        # mw = float(pa.molecular_weight())
        # arom = float(pa.aromaticity())
        # instab = float(pa.instability_index())
        # gravy = float(pa.gravy())

        agg = (c.get("aggregation_scores") if isinstance(c, dict) else getattr(c, "aggregation_scores", None))
        a3d_mean = np.nan
        if isinstance(agg, list) and len(agg) > 0:
            try:
                a3d_mean = float(np.mean([float(x) for x in agg]))
            except Exception:
                a3d_mean = np.nan

        feats = _chel_features(seq, labels)

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

            **feats,
        })
        kept_idx.append(i)

    if not rows:
        return ([], pd.DataFrame()) if return_details else []

    df = pd.DataFrame(rows)

    # ---- scoring (same as your service) ----
    che_soft   = _minmax(df["frac_soft"])
    che_arom   = _minmax(df["frac_arom"])
    che_cat    = _minmax(df["frac_cat"])
    che_motif  = _minmax(df["motif_density"])
    che_labels = _minmax(df["label_fraction"])
    che_center = _minmax(df["label_center_score"])

    df["chelation_sub"] = (
        0.35 * che_soft +
        0.15 * che_arom +
        0.15 * che_cat +
        0.15 * che_motif +
        0.10 * che_labels +
        0.10 * che_center
    )

    a3d_fill = df["a3d_mean"].median() if df["a3d_mean"].notna().any() else 0.0
    a3d_good   = _minmax_reverse(df["a3d_mean"].fillna(a3d_fill))
    gravy_good = _minmax_reverse(df["gravy"])
    arom_good  = _minmax_reverse(df["aromaticity"])
    df["solubility_sub"] = 0.6 * a3d_good + 0.3 * gravy_good + 0.1 * arom_good

    instab_good = _minmax_reverse(df["instability_index"])
    L_med = df["length"].median()
    len_dev = (df["length"] - L_med).abs() / max(float(L_med), 1.0)
    len_score_raw = (1.0 - len_dev).clip(lower=0.0)
    len_good = _minmax(len_score_raw)
    df["stability_sub"] = 0.7 * instab_good + 0.3 * len_good

    gc_dev = (df["gc_frac"] - 0.5).abs()
    gc_score_raw = (1.0 - gc_dev / 0.5).clip(lower=0.0)
    gc_good = _minmax(gc_score_raw)
    run_good = _minmax_reverse(df["longest_run"])
    df["expression_sub"] = 0.7 * gc_good + 0.3 * run_good

    df["final_score"] = (
        0.50 * df["chelation_sub"] +
        0.30 * df["solubility_sub"] +
        0.10 * df["stability_sub"] +
        0.10 * df["expression_sub"]
    )

    scores = [float(x) for x in df["final_score"].tolist()]
    return (scores, df) if return_details else scores
