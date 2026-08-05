#!/usr/bin/env python3
"""
Compute original (pre-normalization) ranges for the Cu peptide objective inputs.

This script is designed for the AnkiBind Cu ranking data. It reports the raw
feature ranges used before fixed min-max normalization, grouped by objective:

  chelation  -> cu_chelation_raw and its sequence components
  solubility -> a3d_mean, gravy, aromaticity
  stability  -> instability_index, len_score_raw
  expression -> gc_frac, gc_score_raw, longest_run

Important: solubility, stability, and expression are composite objectives after
normalization because their raw ingredients have different physical units. The
most defensible "original range" is therefore the range of each raw ingredient.
For convenience, the script also writes optional directional raw proxy ranges,
but those proxy scores mix units and should not be used as the BO objective scale.

Example:
  python compute_raw_objective_ranges_CU.py \
      --input metalpdb_binding_windows_len10_CU_scored_ranked.csv \
      --outdir raw_objective_ranges_CU

For an unscored input CSV, the script can compute sequence-based features and
ProteinAnalysis features, but it will only use existing A3D.csv files under
CU_metaldb_a3ds/<peptide>/A3D.csv; it will not run ColabFold/A3D.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
except Exception:  # pragma: no cover
    ProteinAnalysis = None

AA_OK = set("ACDEFGHIKLMNPQRSTVWY")
SEQ_LEN = 10

RAW_FEATURE_GROUPS: Dict[str, List[str]] = {
    "chelation": [
        "cu_chelation_raw",
        "frac_H",
        "frac_DE",
        "frac_CM",
        "frac_KR",
        "cu_motif_density",
        "label_cluster_density",
    ],
    "solubility": [
        "a3d_mean",
        "gravy",
        "aromaticity",
    ],
    "stability": [
        "instability_index",
        "len_score_raw",
    ],
    "expression": [
        "gc_frac",
        "gc_score_raw",
        "longest_run",
    ],
}

DIRECTION_HIGHER_IS_BETTER = {
    "cu_chelation_raw": True,
    "frac_H": True,
    "frac_DE": True,
    "frac_CM": True,
    "frac_KR": True,
    "cu_motif_density": True,
    "label_cluster_density": True,
    "a3d_mean": False,
    "gravy": False,
    "aromaticity": False,
    "instability_index": False,
    "len_score_raw": True,
    "gc_frac": None,        # best around 0.5, not monotonic
    "gc_score_raw": True,
    "longest_run": False,
}


def parse_binding_sites(value) -> List[int]:
    """Parse binding labels from strings such as '0 0 0 0 1 0 0 0 0 0'."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    s = str(value).strip()
    if not s:
        return []
    toks = re.findall(r"[01]", s)
    return [int(t) for t in toks]


def reverse_translate_first(seq: str) -> str:
    """Simple deterministic reverse translation using one common codon per amino acid."""
    codon = {
        "A": "GCT", "C": "TGT", "D": "GAT", "E": "GAA", "F": "TTT",
        "G": "GGT", "H": "CAT", "I": "ATT", "K": "AAA", "L": "CTT",
        "M": "ATG", "N": "AAT", "P": "CCT", "Q": "CAA", "R": "CGT",
        "S": "TCT", "T": "ACT", "V": "GTT", "W": "TGG", "Y": "TAT",
    }
    return "".join(codon.get(a, "NNN") for a in str(seq).strip().upper())


def gc_fraction(dna: str) -> float:
    dna = str(dna or "").strip().upper()
    dna = re.sub(r"[^ACGT]", "", dna)
    if not dna:
        return float("nan")
    return float((dna.count("G") + dna.count("C")) / len(dna))


def longest_homopolymer(dna: str) -> int:
    dna = str(dna or "").strip().upper()
    dna = re.sub(r"[^ACGT]", "", dna)
    if not dna:
        return 0
    best = 1
    cur = 1
    for i in range(1, len(dna)):
        if dna[i] == dna[i - 1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return int(best)


def cu_motif_density(seq: str) -> float:
    """Cu motif density: HxH, HxxH, HExxH, HxxEH."""
    seq = str(seq or "").strip().upper()
    L = max(len(seq), 1)
    patterns = [r"H.H", r"H..H", r"HE..H", r"H..EH"]
    hits = sum(len(re.findall(p, seq)) for p in patterns)
    return float(hits / L)


def label_cluster_density(labels: List[int], seq_len: int) -> float:
    if not labels:
        return 0.0
    labels = [int(v) if str(v).strip() in {"0", "1"} else 0 for v in labels]
    n = len(labels)
    frac = sum(labels) / max(n, 1)
    positions = [i for i, v in enumerate(labels) if v == 1]
    if not positions:
        return 0.0
    center = (sum(positions) / len(positions)) / max(n - 1, 1)
    center_score = 1.0 - abs(center - 0.5) / 0.5
    center_score = float(np.clip(center_score, 0.0, 1.0))
    return float(frac * (0.5 + 0.5 * center_score))


def read_a3d_mean(seq: str, a3d_root: Path, filename: str = "A3D.csv") -> float:
    path = a3d_root / seq / filename
    if not path.exists() or path.stat().st_size == 0:
        return float("nan")
    try:
        a3d = pd.read_csv(path)
        cols_lower = {c.lower(): c for c in a3d.columns}
        if "score" not in cols_lower:
            return float("nan")
        scores = pd.to_numeric(a3d[cols_lower["score"]], errors="coerce")
        return float(scores.mean()) if scores.notna().any() else float("nan")
    except Exception:
        return float("nan")


def clean_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    if "peptide_len10" not in df.columns:
        raise ValueError("Input CSV must contain a 'peptide_len10' column.")
    out = df.copy()
    out["peptide_len10"] = (
        out["peptide_len10"].astype(str).str.strip().str.upper()
        .str.replace("*", "", regex=False)
        .str.replace("U", "", regex=False)
    )
    valid_aa = out["peptide_len10"].apply(lambda s: bool(s) and set(s).issubset(AA_OK))
    valid_len = out["peptide_len10"].str.len().eq(SEQ_LEN)
    return out[valid_aa & valid_len].drop_duplicates(subset=["peptide_len10"], keep="first").reset_index(drop=True)


def add_missing_raw_features(df: pd.DataFrame, a3d_root: Path) -> pd.DataFrame:
    """Add missing raw feature columns without applying normalization."""
    out = clean_and_filter(df)

    if "binding_site_labels_len10" not in out.columns:
        out["binding_site_labels_len10"] = ""
    labels = [parse_binding_sites(v) for v in out["binding_site_labels_len10"].tolist()]

    # DNA-derived features
    if "DNA_sequence" not in out.columns:
        out["DNA_sequence"] = out["peptide_len10"].map(reverse_translate_first)
    else:
        empty = out["DNA_sequence"].isna() | out["DNA_sequence"].astype(str).str.strip().eq("")
        out.loc[empty, "DNA_sequence"] = out.loc[empty, "peptide_len10"].map(reverse_translate_first)

    if "gc_frac" not in out.columns:
        out["gc_frac"] = out["DNA_sequence"].map(gc_fraction)
    if "longest_run" not in out.columns:
        out["longest_run"] = out["DNA_sequence"].map(longest_homopolymer)
    if "gc_score_raw" not in out.columns:
        gc_dev = (pd.to_numeric(out["gc_frac"], errors="coerce") - 0.5).abs()
        out["gc_score_raw"] = (1.0 - gc_dev / 0.5).clip(lower=0.0, upper=1.0)

    # Sequence-composition Cu features
    seqs = out["peptide_len10"].tolist()
    L = out["peptide_len10"].str.len().replace(0, np.nan)
    if "frac_H" not in out.columns:
        out["frac_H"] = [s.count("H") / max(len(s), 1) for s in seqs]
    if "frac_DE" not in out.columns:
        out["frac_DE"] = [(s.count("D") + s.count("E")) / max(len(s), 1) for s in seqs]
    if "frac_CM" not in out.columns:
        out["frac_CM"] = [(s.count("C") + s.count("M")) / max(len(s), 1) for s in seqs]
    if "frac_KR" not in out.columns:
        out["frac_KR"] = [(s.count("K") + s.count("R")) / max(len(s), 1) for s in seqs]
    if "cu_motif_density" not in out.columns:
        out["cu_motif_density"] = [cu_motif_density(s) for s in seqs]
    if "label_cluster_density" not in out.columns:
        out["label_cluster_density"] = [label_cluster_density(lab, len(seq)) for seq, lab in zip(seqs, labels)]

    if "cu_chelation_raw" not in out.columns:
        out["cu_chelation_raw"] = (
            0.30 * pd.to_numeric(out["frac_H"], errors="coerce").fillna(0.0)
            + 0.25 * pd.to_numeric(out["frac_DE"], errors="coerce").fillna(0.0)
            + 0.15 * pd.to_numeric(out["frac_CM"], errors="coerce").fillna(0.0)
            + 0.10 * pd.to_numeric(out["frac_KR"], errors="coerce").fillna(0.0)
            + 0.10 * pd.to_numeric(out["cu_motif_density"], errors="coerce").fillna(0.0)
            + 0.10 * pd.to_numeric(out["label_cluster_density"], errors="coerce").fillna(0.0)
        )

    # ProteinAnalysis features
    need_pa = [c for c in ["molecular_weight", "aromaticity", "instability_index", "gravy"] if c not in out.columns]
    if need_pa:
        if ProteinAnalysis is None:
            raise ImportError("Biopython is required to compute ProteinAnalysis features. Install with: pip install biopython")
        cache = {}
        for s in out["peptide_len10"].unique():
            pa = ProteinAnalysis(s)
            cache[s] = {
                "molecular_weight": float(pa.molecular_weight()),
                "aromaticity": float(pa.aromaticity()),
                "instability_index": float(pa.instability_index()),
                "gravy": float(pa.gravy()),
            }
        for c in need_pa:
            out[c] = out["peptide_len10"].map(lambda s: cache[s][c])

    # A3D feature from existing files only
    if "a3d_mean" not in out.columns:
        out["a3d_mean"] = [read_a3d_mean(s, a3d_root=a3d_root) for s in seqs]

    if "len_score_raw" not in out.columns:
        lengths = out["peptide_len10"].str.len()
        L_med = float(lengths.median()) if len(lengths) else float(SEQ_LEN)
        len_dev = (lengths - L_med).abs() / max(L_med, 1.0)
        out["len_score_raw"] = (1.0 - len_dev).clip(lower=0.0, upper=1.0)

    return out


def numeric_summary(series: pd.Series) -> Dict[str, float]:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return {
            "n_nonmissing": 0,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "q05": np.nan,
            "q95": np.nan,
        }
    return {
        "n_nonmissing": int(vals.shape[0]),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "mean": float(vals.mean()),
        "median": float(vals.median()),
        "std": float(vals.std(ddof=0)),
        "q05": float(vals.quantile(0.05)),
        "q95": float(vals.quantile(0.95)),
    }


def compute_raw_feature_range_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for objective, features in RAW_FEATURE_GROUPS.items():
        for feature in features:
            if feature not in df.columns:
                rows.append({
                    "objective": objective,
                    "raw_feature": feature,
                    "higher_is_better_before_normalization": DIRECTION_HIGHER_IS_BETTER.get(feature),
                    "n_nonmissing": 0,
                    "min": np.nan,
                    "max": np.nan,
                    "range": np.nan,
                    "mean": np.nan,
                    "median": np.nan,
                    "std": np.nan,
                    "q05": np.nan,
                    "q95": np.nan,
                    "note": "missing column",
                })
                continue
            s = numeric_summary(df[feature])
            rows.append({
                "objective": objective,
                "raw_feature": feature,
                "higher_is_better_before_normalization": DIRECTION_HIGHER_IS_BETTER.get(feature),
                **s,
                "range": (s["max"] - s["min"]) if s["n_nonmissing"] else np.nan,
                "note": "raw/pre-normalization feature range",
            })
    return pd.DataFrame(rows)


def add_directional_raw_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add optional raw directional proxies. These are not normalized and mix units;
    they are useful only for diagnostics, not as final BO objectives.
    """
    out = df.copy()
    # Fill missing a3d with median only for a diagnostic proxy.
    a3d = pd.to_numeric(out.get("a3d_mean", pd.Series(np.nan, index=out.index)), errors="coerce")
    a3d_fill = a3d.median() if a3d.notna().any() else np.nan
    a3d = a3d.fillna(a3d_fill)

    out["chelation_raw_objective"] = pd.to_numeric(out["cu_chelation_raw"], errors="coerce")
    out["solubility_raw_directional_proxy"] = (
        0.6 * (-a3d)
        + 0.3 * (-pd.to_numeric(out["gravy"], errors="coerce"))
        + 0.1 * (-pd.to_numeric(out["aromaticity"], errors="coerce"))
    )
    out["stability_raw_directional_proxy"] = (
        0.7 * (-pd.to_numeric(out["instability_index"], errors="coerce"))
        + 0.3 * pd.to_numeric(out["len_score_raw"], errors="coerce")
    )
    out["expression_raw_directional_proxy"] = (
        0.7 * pd.to_numeric(out["gc_score_raw"], errors="coerce")
        + 0.3 * (-pd.to_numeric(out["longest_run"], errors="coerce"))
    )
    return out


def compute_proxy_range_table(df: pd.DataFrame) -> pd.DataFrame:
    proxy_cols = [
        ("chelation", "chelation_raw_objective"),
        ("solubility", "solubility_raw_directional_proxy"),
        ("stability", "stability_raw_directional_proxy"),
        ("expression", "expression_raw_directional_proxy"),
    ]
    rows = []
    for objective, col in proxy_cols:
        s = numeric_summary(df[col])
        rows.append({
            "objective": objective,
            "raw_objective_or_proxy": col,
            **s,
            "range": (s["max"] - s["min"]) if s["n_nonmissing"] else np.nan,
            "warning": "proxy mixes raw units; use feature ranges for rigorous normalization",
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="metalpdb_binding_windows_len10_CU_scored_ranked.csv",
                        help="Input CSV. Can be scored/ranked or raw windows CSV.")
    parser.add_argument("--outdir", default="raw_objective_ranges_CU",
                        help="Output directory.")
    parser.add_argument("--a3d-root", default="CU_metaldb_a3ds",
                        help="Folder containing existing CU_metaldb_a3ds/<peptide>/A3D.csv files.")
    args = parser.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_in = pd.read_csv(in_path)
    df_raw = add_missing_raw_features(df_in, a3d_root=Path(args.a3d_root))
    df_raw = add_directional_raw_proxies(df_raw)

    raw_features_path = outdir / "cu_raw_features_no_normalization.csv"
    feature_ranges_path = outdir / "cu_raw_feature_ranges_by_objective.csv"
    proxy_ranges_path = outdir / "cu_raw_directional_proxy_ranges.csv"
    json_path = outdir / "cu_raw_feature_ranges_by_objective.json"

    feature_ranges = compute_raw_feature_range_table(df_raw)
    proxy_ranges = compute_proxy_range_table(df_raw)

    df_raw.to_csv(raw_features_path, index=False)
    feature_ranges.to_csv(feature_ranges_path, index=False)
    proxy_ranges.to_csv(proxy_ranges_path, index=False)

    # Compact JSON: feature -> min/max plus objective group.
    feature_json = {}
    for _, row in feature_ranges.iterrows():
        feature_json[row["raw_feature"]] = {
            "objective": row["objective"],
            "higher_is_better_before_normalization": row["higher_is_better_before_normalization"],
            "n_nonmissing": int(row["n_nonmissing"]),
            "min": None if pd.isna(row["min"]) else float(row["min"]),
            "max": None if pd.isna(row["max"]) else float(row["max"]),
            "range": None if pd.isna(row["range"]) else float(row["range"]),
        }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(feature_json, f, indent=2)

    print("Saved:")
    print(f"  raw features:        {raw_features_path}")
    print(f"  raw feature ranges:  {feature_ranges_path}")
    print(f"  raw proxy ranges:    {proxy_ranges_path}")
    print(f"  JSON ranges:         {json_path}")
    print("\nMain raw ranges by objective:")
    print(feature_ranges[["objective", "raw_feature", "n_nonmissing", "min", "max", "range", "higher_is_better_before_normalization"]].to_string(index=False))


if __name__ == "__main__":
    main()
