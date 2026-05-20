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



def _cu_motif_density(seq: str) -> float:
    seq = (seq or "").upper()
    L = max(len(seq), 1)
    patterns = [
        r"H.H",      # HxH
        r"H..H",     # HxxH
        r"HE..H",    # HExxH
        r"H..EH",    # HxxEH
    ]
    motif_hits = sum(len(re.findall(p, seq)) for p in patterns)
    return motif_hits / L



def _label_cluster_density(labels: List[int], seq_len: int) -> float:
    if not labels:
        return 0.0

    labels = [int(v) for v in labels]
    n = len(labels)
    frac = sum(labels) / max(n, 1)
    pos = [i for i, v in enumerate(labels) if v == 1]
    if not pos:
        return 0.0

    center = (sum(pos) / len(pos)) / max(n - 1, 1)
    center_score = 1.0 - abs(center - 0.5) / 0.5
    center_score = float(np.clip(center_score, 0.0, 1.0))

    # Single cluster-position feature following the Cu spec.
    return frac * (0.5 + 0.5 * center_score)



def _cu_features(seq: str, labels: List[int]) -> Dict[str, float]:
    seq = (seq or "").upper()
    L = max(len(seq), 1)

    frac_H = seq.count("H") / L
    frac_DE = (seq.count("D") + seq.count("E")) / L
    frac_CM = (seq.count("C") + seq.count("M")) / L
    frac_KR = (seq.count("K") + seq.count("R")) / L
    motif_density = _cu_motif_density(seq)
    label_cluster_density = _label_cluster_density(labels, L)

    return {
        "frac_H": frac_H,
        "frac_DE": frac_DE,
        "frac_CM": frac_CM,
        "frac_KR": frac_KR,
        "cu_motif_density": motif_density,
        "label_cluster_density": label_cluster_density,
    }


CandidateLike = Union[Dict[str, Any], Any]  # dict or object with attributes



def compute_final_scores(
    candidates: Sequence[CandidateLike],
    *,
    reverse_translate_strategy: str = "first",
    return_details: bool = False,
) -> Union[List[float], Tuple[List[float], pd.DataFrame]]:
    """
    Compute batch-normalized final_score using the Cu objective definition.

    Chelation uses the Word document specification:
      0.30 * H
    + 0.25 * (D+E)
    + 0.15 * (C+M)
    + 0.10 * (K+R)
    + 0.10 * Cu motif density
    + 0.10 * label/cluster density

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

    for c in candidates:
        seq = (c.get("sequence") if isinstance(c, dict) else getattr(c, "sequence", "")) or ""
        bs = (c.get("binding_sites") if isinstance(c, dict) else getattr(c, "binding_sites", "")) or ""
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

        cu_feats = _cu_features(seq, labels)

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
            **cu_feats,
        })

    if not rows:
        return ([], pd.DataFrame()) if return_details else []

    df = pd.DataFrame(rows)

    # ---- Cu chelation_sub ----
    h_good = _minmax(df["frac_H"])
    de_good = _minmax(df["frac_DE"])
    cm_good = _minmax(df["frac_CM"])
    kr_good = _minmax(df["frac_KR"])
    motif_good = _minmax(df["cu_motif_density"])
    cluster_good = _minmax(df["label_cluster_density"])

    df["cu_chelation_raw"] = (
        0.30 * df["frac_H"]
        + 0.25 * df["frac_DE"]
        + 0.15 * df["frac_CM"]
        + 0.10 * df["frac_KR"]
        + 0.10 * df["cu_motif_density"]
        + 0.10 * df["label_cluster_density"]
    )

    df["chelation_sub"] = (
        0.30 * h_good
        + 0.25 * de_good
        + 0.15 * cm_good
        + 0.10 * kr_good
        + 0.10 * motif_good
        + 0.10 * cluster_good
    )

    # ---- solubility_sub (reuse with sign-corrected terms) ----
    a3d_fill = df["a3d_mean"].median() if df["a3d_mean"].notna().any() else 0.0
    a3d_good = _minmax_reverse(df["a3d_mean"].fillna(a3d_fill))
    gravy_good = _minmax_reverse(df["gravy"])
    arom_good = _minmax_reverse(df["aromaticity"])
    df["solubility_sub"] = 0.6 * a3d_good + 0.3 * gravy_good + 0.1 * arom_good

    # ---- stability_sub (reuse, sign-corrected) ----
    instab_good = _minmax_reverse(df["instability_index"])
    L_med = df["length"].median()
    len_dev = (df["length"] - L_med).abs() / max(float(L_med), 1.0)
    len_score_raw = (1.0 - len_dev).clip(lower=0.0)
    len_good = _minmax(len_score_raw)
    df["stability_sub"] = 0.7 * instab_good + 0.3 * len_good

    # ---- expression_sub (reuse, cleaned up) ----
    gc_dev = (df["gc_frac"] - 0.5).abs()
    gc_score_raw = (1.0 - gc_dev / 0.5).clip(lower=0.0)
    gc_good = _minmax(gc_score_raw)
    run_good = _minmax_reverse(df["longest_run"])
    df["expression_sub"] = 0.7 * gc_good + 0.3 * run_good

    df["final_score"] = (
        0.50 * df["chelation_sub"]
        + 0.30 * df["solubility_sub"]
        + 0.10 * df["stability_sub"]
        + 0.10 * df["expression_sub"]
    )

    scores = [float(x) for x in df["final_score"].tolist()]
    return (scores, df) if return_details else scores
