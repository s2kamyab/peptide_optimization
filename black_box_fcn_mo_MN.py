from __future__ import annotations

import re
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Union, Tuple

import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.Data import CodonTable
from black_box_fcn import CandidateLike, _parse_binding_sites, _chel_features, _longest_homopolymer, _minmax, _minmax_reverse, _gc_fraction, reverse_translate
from typing import Sequence, Union, Tuple, List
import pandas as pd
import numpy as np

# ... keep your imports + compute_final_scores_mn code ...

def compute_objectives_mn(
    candidates: Sequence[CandidateLike],
    *,
    reverse_translate_strategy: str = "first",
) -> pd.DataFrame:
    """
    Returns a dataframe with per-candidate objective columns:
      - chelation_sub
      - solubility_sub
      - stability_sub
      - expression_sub

    IMPORTANT: these are currently min-max normalized within the evaluated batch,
    so use a fixed reference batch + candidate (like you already do) to make this stable.
    """
    _, df = compute_final_scores_mn(
        candidates,
        reverse_translate_strategy=reverse_translate_strategy,
        return_details=True,
    )
    if df is None or len(df) == 0:
        return pd.DataFrame()

    # Ensure these exist (they do in your implementation)
    obj_cols = ["chelation_sub", "solubility_sub", "stability_sub", "expression_sub"]
    return df[["sequence"] + obj_cols].copy()

def _mn_chelation_score(seq: str) -> float:
    """
    Mn2+ chelation preference (hard-ish Lewis acid):
      D (Asp) > H (His) > E (Glu)

    Returns an unnormalized score in [0, 1] (since it's weighted fractions).
    """
    seq = (seq or "").strip().upper()
    if not seq:
        return 0.0
    L = len(seq)
    f_D = seq.count("D") / L
    f_H = seq.count("H") / L
    f_E = seq.count("E") / L
    return 0.45 * f_D + 0.35 * f_H + 0.20 * f_E


def compute_final_scores_mn(
    candidates: Sequence[CandidateLike],
    *,
    reverse_translate_strategy: str = "first",
    return_details: bool = False,
) -> Union[List[float], Tuple[List[float], pd.DataFrame]]:
    """
    MN-specific version of compute_final_scores:
    - replaces chelation_sub with an Mn chelation score that favors D > H > E
    - keeps solubility/stability/expression exactly as before
    - then combines into final_score (weights adjustable below)
    """
    rows = []
    for i, c in enumerate(candidates):
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

        # physchem
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

    if not rows:
        return ([], pd.DataFrame()) if return_details else []

    df = pd.DataFrame(rows)

    # ----------------------------
    # Mn chelation_sub (NEW)
    # ----------------------------
    # Compute per-row Mn chelation score, then min-max normalize *within batch*
    df["mn_chelation_raw"] = df["sequence"].apply(_mn_chelation_score)
    df["chelation_sub"] = _minmax(df["mn_chelation_raw"])

    # ----------------------------
    # Keep your other subscores (UNCHANGED)
    # ----------------------------
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

    # ----------------------------
    # Final score (keep same weights, or adjust for Mn)
    # ----------------------------
    # Option 1: keep your existing blend
    df["final_score"] = (
        0.50 * df["chelation_sub"] +
        0.30 * df["solubility_sub"] +
        0.10 * df["stability_sub"] +
        0.10 * df["expression_sub"]
    )

    # Option 2 (closer to your text): emphasize chelation vs expression only
    # df["final_score"] = 0.7 * df["chelation_sub"] + 0.3 * df["expression_sub"]

    scores = df["final_score"].astype(float).tolist()
    return (scores, df) if return_details else scores
# black_box_fcn_MN.py

