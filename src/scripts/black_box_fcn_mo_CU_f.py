"""Canonical Cu multi-objective black-box scoring with fixed-range normalization.

This file is now self-contained. It does NOT require a separate
`black_box_fcn_mo_CU_f_updated_batch_normalized.py` file.

Key behavior
------------
1. Computes raw Cu objective features for length-10 peptides.
2. Uses the updated Cu chelation formula:
      0.30*frac_H + 0.25*frac_DE + 0.15*frac_CM
    + 0.10*frac_KR + 0.10*cu_motif_density + 0.10*label_cluster_density
3. Applies fixed-range normalization, not per-batch min-max normalization.
4. Allows the active fixed ranges to be initialized from the training dataframe,
   loaded from JSON, explicitly provided, or defaulted to safe fallback ranges.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from collections import defaultdict
import json
import random
import re

import numpy as np
import pandas as pd
import torch
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.Data import CodonTable


# from black_box_fcn import (
#     _parse_binding_sites,
#     _gc_fraction,
#     _longest_homopolymer,
#     reverse_translate,
#     _cu_features,
# )

from peptide_optimization.src.util.run_esnfold import run_esmfold_hf
from peptide_optimization.src.util.run_a3d import run_a3d_on_pdb

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
    fixed_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    compute_missing_a3d: bool = True,
) -> Union[List[float], Tuple[List[float], pd.DataFrame]]:
    """
    Compute final_score using the same canonical Cu objective evaluation path
    as blackbox_fc(...).

    This wrapper is kept for backward compatibility with older code that calls
    compute_final_scores(...), but it no longer has a separate batch-normalized
    implementation. It uses the same objective definitions, Cu motif logic,
    label-cluster logic, and fixed-range normalization as blackbox_fc(...).
    """
    peptides: List[str] = []
    labels: List[Union[str, List[int]]] = []
    dnas: List[str] = []
    a3d_values: List[float] = []
    metals: List[str] = []

    for c in candidates:
        seq = (c.get("sequence") if isinstance(c, dict) else getattr(c, "sequence", "")) or ""
        bs = (c.get("binding_sites") if isinstance(c, dict) else getattr(c, "binding_sites", "")) or ""
        dna = (c.get("DNA_sequence") if isinstance(c, dict) else getattr(c, "DNA_sequence", "")) or ""
        metal = (c.get("metal") if isinstance(c, dict) else getattr(c, "metal", "")) or ""

        seq = str(seq).strip().upper().replace("*", "").replace("U", "")
        if not seq or any(ch not in "ACDEFGHIKLMNPQRSTVWY" for ch in seq):
            continue

        # Backward-compatible support for old candidate dictionaries that stored
        # A3D residue scores under aggregation_scores.
        agg = (c.get("aggregation_scores") if isinstance(c, dict) else getattr(c, "aggregation_scores", None))
        a3d_mean = np.nan
        if isinstance(agg, list) and len(agg) > 0:
            try:
                a3d_mean = float(np.mean([float(x) for x in agg]))
            except Exception:
                a3d_mean = np.nan

        peptides.append(seq)
        labels.append(bs)
        dnas.append(str(dna).strip().upper())
        a3d_values.append(a3d_mean)
        metals.append(str(metal).strip().upper())

    if not peptides:
        return ([], pd.DataFrame()) if return_details else []

    df = blackbox_fc(
        peptides=peptides,
        binding_site_labels_len10=labels,
        dna_sequences=dnas,
        a3d_mean=a3d_values,
        reverse_translate_strategy=reverse_translate_strategy,
        fixed_ranges=fixed_ranges,
        compute_missing_a3d=compute_missing_a3d,
    )
    if len(df):
        df["metal"] = metals[: len(df)]

    scores = [float(x) for x in df["final_score"].tolist()]
    return (scores, df) if return_details else scores

# ---------------------------------------------------------------------
# Fixed normalization defaults
# ---------------------------------------------------------------------
# These are only fallbacks. For publication-quality or BO-consistent runs,
# call initialize_fixed_objective_ranges_from_training_df(df_train) once
# from your training CSV and reuse those ranges for all candidates.
_DEFAULT_FIXED_RANGES: Dict[str, Tuple[float, float]] = {
    "cu_chelation_raw": (0.0, 0.30050000000000004),
    "a3d_mean": (-2.4, 2.4),
    "gravy": (-4.017, 2.787),
    "aromaticity": (0.0, 0.6599999999999999),
    "instability_index": (-75.509, 227.119),
    "gc_score_raw": (0.11999999999999994, 1.0),
    "longest_run": (1.0, 12.0),
    "len_score_raw": (0.0, 1.0),
}

# Optional file name used when the JSON is present beside this Python file.
_DEFAULT_RANGES_JSON = "cu_objective_fixed_ranges_training_CU_updated_margined.json"

_FIXED_OBJECTIVE_RANGES: Dict[str, Tuple[float, float]] = {}


def fixed_minmax(x, lo: float, hi: float) -> pd.Series:
    """Scale values to [0, 1] using fixed lo/hi values."""
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    denom = max(float(hi) - float(lo), 1e-12)
    return ((x - float(lo)) / denom).clip(0.0, 1.0)


def fixed_minmax_reverse(x, lo: float, hi: float) -> pd.Series:
    """Reverse fixed min-max scale so lower raw values become better."""
    return 1.0 - fixed_minmax(x, lo, hi)


def _safe_numeric_range(s: pd.Series, fallback: Tuple[float, float]) -> Tuple[float, float]:
    vals = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return float(fallback[0]), float(fallback[1])

    lo = float(vals.min())
    hi = float(vals.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return float(fallback[0]), float(fallback[1])

    return lo, hi


def _active_ranges(
    fixed_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, Tuple[float, float]]:
    """Return explicit ranges, loaded/initialized ranges, JSON ranges, or safe defaults."""
    if fixed_ranges is not None:
        return {k: (float(v[0]), float(v[1])) for k, v in fixed_ranges.items()}
    if _FIXED_OBJECTIVE_RANGES:
        return dict(_FIXED_OBJECTIVE_RANGES)

    # Prefer the margined JSON ranges if the file is available beside this script
    # or in the current working directory. This keeps BO runs fixed to the
    # training-derived ranges instead of recomputing batch ranges.
    for candidate in [Path(__file__).with_name(_DEFAULT_RANGES_JSON), Path(_DEFAULT_RANGES_JSON)]:
        if candidate.exists():
            try:
                with candidate.open("r", encoding="utf-8") as f:
                    raw = json.load(f)
                return {k: (float(v[0]), float(v[1])) for k, v in raw.items()}
            except Exception:
                pass

    return dict(_DEFAULT_FIXED_RANGES)


def set_fixed_objective_ranges(ranges: Dict[str, Tuple[float, float]]) -> None:
    """Set fixed ranges explicitly, e.g. from a BO script."""
    global _FIXED_OBJECTIVE_RANGES
    _FIXED_OBJECTIVE_RANGES = {
        k: (float(v[0]), float(v[1])) for k, v in ranges.items()
    }


def get_fixed_objective_ranges() -> Dict[str, Tuple[float, float]]:
    """Return currently active initialized ranges. Empty means defaults are being used."""
    return dict(_FIXED_OBJECTIVE_RANGES)


def save_fixed_objective_ranges(
    json_path: Union[str, Path],
    ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> None:
    """Save fixed ranges to JSON."""
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    active = _active_ranges(ranges)
    with path.open("w", encoding="utf-8") as f:
        json.dump({k: [float(v[0]), float(v[1])] for k, v in active.items()}, f, indent=2)


def load_fixed_objective_ranges(json_path: Union[str, Path]) -> Dict[str, Tuple[float, float]]:
    """Load fixed ranges from JSON and activate them."""
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    ranges = {k: (float(v[0]), float(v[1])) for k, v in raw.items()}
    set_fixed_objective_ranges(ranges)
    return ranges


def _add_derived_objective_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add raw columns required for fixed normalization when possible."""
    out = df.copy()

    if "gc_score_raw" not in out.columns and "gc_frac" in out.columns:
        gc_frac = pd.to_numeric(out["gc_frac"], errors="coerce")
        gc_dev = (gc_frac - 0.5).abs()
        out["gc_score_raw"] = (1.0 - gc_dev / 0.5).clip(lower=0.0, upper=1.0)

    if "len_score_raw" not in out.columns and "peptide_len10" in out.columns:
        lengths = out["peptide_len10"].astype(str).str.len()
        L_med = float(lengths.median()) if len(lengths) else 10.0
        if not np.isfinite(L_med) or L_med <= 0:
            L_med = 10.0
        len_dev = (lengths - L_med).abs() / max(L_med, 1.0)
        out["len_score_raw"] = (1.0 - len_dev).clip(lower=0.0, upper=1.0)

    cu_components = [
        "frac_H",
        "frac_DE",
        "frac_CM",
        "frac_KR",
        "cu_motif_density",
        "label_cluster_density",
    ]
    if "cu_chelation_raw" not in out.columns and all(c in out.columns for c in cu_components):
        out["cu_chelation_raw"] = (
            0.30 * pd.to_numeric(out["frac_H"], errors="coerce").fillna(0.0)
            + 0.25 * pd.to_numeric(out["frac_DE"], errors="coerce").fillna(0.0)
            + 0.15 * pd.to_numeric(out["frac_CM"], errors="coerce").fillna(0.0)
            + 0.10 * pd.to_numeric(out["frac_KR"], errors="coerce").fillna(0.0)
            + 0.10 * pd.to_numeric(out["cu_motif_density"], errors="coerce").fillna(0.0)
            + 0.10 * pd.to_numeric(out["label_cluster_density"], errors="coerce").fillna(0.0)
        )

    return out


def initialize_fixed_objective_ranges_from_training_df(
    df_train: pd.DataFrame,
    *,
    save_json_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute fixed normalization ranges once from the training dataframe.

    The dataframe should preferably contain the raw columns:
      cu_chelation_raw, a3d_mean, gravy, aromaticity, instability_index,
      gc_frac or gc_score_raw, longest_run, len_score_raw.

    If any raw column is unavailable or constant, a safe fallback range is used.
    """
    global _FIXED_OBJECTIVE_RANGES

    work = _add_derived_objective_raw_columns(df_train)
    ranges: Dict[str, Tuple[float, float]] = {}

    for col, fallback in _DEFAULT_FIXED_RANGES.items():
        if col in work.columns:
            ranges[col] = _safe_numeric_range(work[col], fallback)
        else:
            ranges[col] = (float(fallback[0]), float(fallback[1]))

    _FIXED_OBJECTIVE_RANGES = ranges

    if save_json_path is not None:
        save_fixed_objective_ranges(save_json_path, ranges)

    return ranges


def apply_fixed_objective_normalization(
    df_in: pd.DataFrame,
    ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> pd.DataFrame:
    """
    Recompute objective columns using fixed ranges from the Cu margined JSON.

    This is fixed-range normalization, not batch normalization:
      normalized = (x - lo) / (hi - lo), clipped to [0, 1]

    For lower-is-better raw measures, the normalized score is inverted:
      score = 1 - normalized

    The resulting objective columns are all oriented so higher is better:
      chelation_sub, solubility_sub, stability_sub, expression_sub
    """
    active = _active_ranges(ranges)
    df = _add_derived_objective_raw_columns(df_in)

    required = [
        "cu_chelation_raw",
        "a3d_mean",
        "gravy",
        "aromaticity",
        "instability_index",
        "gc_score_raw",
        "longest_run",
        "len_score_raw",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Cannot apply fixed objective normalization. Missing columns: "
            + ", ".join(missing)
        )

    # 1) Cu chelation: higher raw chelation proxy is better.
    df["chelation_sub"] = fixed_minmax(
        df["cu_chelation_raw"],
        *active["cu_chelation_raw"],
    )

    # 2) Solubility / low aggregation:
    # lower A3D, lower GRAVY, and lower aromaticity are better.
    a3d = pd.to_numeric(df["a3d_mean"], errors="coerce")
    if a3d.isna().any():
        lo, hi = active["a3d_mean"]
        a3d = a3d.fillna((lo + hi) / 2.0)

    a3d_good = fixed_minmax_reverse(a3d, *active["a3d_mean"])
    gravy_good = fixed_minmax_reverse(df["gravy"], *active["gravy"])
    arom_good = fixed_minmax_reverse(df["aromaticity"], *active["aromaticity"])
    df["solubility_sub"] = 0.6 * a3d_good + 0.3 * gravy_good + 0.1 * arom_good

    # 3) Structural/developability stability:
    # lower instability index is better; higher length score is better.
    instab_good = fixed_minmax_reverse(
        df["instability_index"],
        *active["instability_index"],
    )
    len_good = fixed_minmax(
        df["len_score_raw"],
        *active["len_score_raw"],
    )
    df["stability_sub"] = 0.7 * instab_good + 0.3 * len_good

    # 4) Expression proxy:
    # higher GC-band score is better; shorter homopolymer run is better.
    gc_good = fixed_minmax(
        df["gc_score_raw"],
        *active["gc_score_raw"],
    )
    run_good = fixed_minmax_reverse(
        df["longest_run"],
        *active["longest_run"],
    )
    df["expression_sub"] = 0.7 * gc_good + 0.3 * run_good

    # Weighted scalar score for ranking only. Multi-objective BO should still
    # use the four separate objective columns.
    df["final_score"] = (
        0.50 * pd.to_numeric(df["chelation_sub"], errors="coerce")
        + 0.30 * pd.to_numeric(df["solubility_sub"], errors="coerce")
        + 0.10 * pd.to_numeric(df["stability_sub"], errors="coerce")
        + 0.10 * pd.to_numeric(df["expression_sub"], errors="coerce")
    )

    return df


# ---------------------------------------------------------------------
# Structure/A3D helpers
# ---------------------------------------------------------------------
def _find_pdb_for_seq(pdb_root: Path, seq: str) -> Optional[Path]:
    seq_dir = pdb_root / f"peptide_{seq}"
    if not seq_dir.exists():
        return None

    preferred = seq_dir / "query_unrelaxed_rank_001_alphafold2_model_1_seed_000.pdb"
    if preferred.exists():
        return preferred

    pdbs = sorted(seq_dir.glob("*.pdb"))
    return pdbs[0] if pdbs else None


def _compute_a3d_mean_from_csv(a3d_csv: Path) -> float:
    t = pd.read_csv(a3d_csv)

    for col in ["score", "a3d_score", "Aggrescan3D_score", "value"]:
        if col in t.columns:
            vals = pd.to_numeric(t[col], errors="coerce").dropna()
            if len(vals):
                return float(vals.mean())

    num_cols = [c for c in t.columns if pd.api.types.is_numeric_dtype(t[c])]
    if num_cols:
        vals = pd.to_numeric(t[num_cols[-1]], errors="coerce").dropna()
        if len(vals):
            return float(vals.mean())

    raise ValueError(f"No numeric score column found in {a3d_csv}")


def add_a3d_mean_to_df(
    df: pd.DataFrame,
    seq_col: str = "peptide_len10",
    *,
    pdb_root: Path = Path("CU_metaldb_pdbs"),
    a3d_root: Path = Path("CU_metaldb_a3ds"),
    a3d_csv_name: str = "A3D.csv",
    overwrite_colabfold: bool = False,
) -> pd.DataFrame:
    """
    Compute A3D mean for sequences where an A3D output exists or can be generated.

    This function returns a copy with an a3d_mean column. It does not preserve
    pre-existing a3d_mean values by itself; blackbox_fc only calls this helper
    on rows where a3d_mean is missing.
    """
    out_df = df.copy()
    a3d_root.mkdir(parents=True, exist_ok=True)
    pdb_root.mkdir(parents=True, exist_ok=True)

    uniq_seq = out_df[seq_col].astype(str).str.strip().str.upper().unique()
    a3d_means: Dict[str, float] = {}

    for i, seq in enumerate(uniq_seq, start=1):
        if not seq:
            continue

        print(f"[{i}/{len(uniq_seq)}] seq={seq}")

        seq_out_dir = pdb_root / f"peptide_{seq}"
        seq_out_dir.mkdir(parents=True, exist_ok=True)

        pdb_file = _find_pdb_for_seq(pdb_root, seq)
        if pdb_file is None or overwrite_colabfold:
            pdb = run_esmfold_hf(
                seq=seq,
                out_dir=str(seq_out_dir),
                num_recycles=3,
                chunk_size=64,
                overwrite=True,
            )
            print("Saved PDB to:", pdb)
            pdb_file = _find_pdb_for_seq(pdb_root, seq)

        if pdb_file is None or not pdb_file.exists():
            print(f"  - skipped: no PDB produced for {seq}")
            continue

        a3d_out_dir = a3d_root / seq
        a3d_out_dir.mkdir(parents=True, exist_ok=True)

        candidate_csvs = [
            a3d_out_dir / a3d_csv_name,
            a3d_out_dir / "A3D.csv"
        ]
        out_csv = next((p for p in candidate_csvs if p.exists() and p.stat().st_size > 0), None)
        print(out_csv)

        if out_csv is None:
            produced_csv = run_a3d_on_pdb(pdb_file, a3d_out_dir)
            if produced_csv is not None and Path(produced_csv).exists():
                out_csv = Path(produced_csv)

        if out_csv is None or not (out_csv.exists() and out_csv.stat().st_size > 0):
            print(f"  - skipped: A3D output missing for {seq}")
            continue

        try:
            a3d_means[seq] = _compute_a3d_mean_from_csv(out_csv)
        except Exception as e:
            print(f"  - skipped: cannot parse A3D for {seq}: {e}")
            continue

    out_df["a3d_mean"] = out_df[seq_col].astype(str).str.strip().str.upper().map(a3d_means)
    return out_df


# ---------------------------------------------------------------------
# Main objective API
# ---------------------------------------------------------------------
def blackbox_fc(
    peptides: Union[str, List[str]],
    binding_site_labels_len10: Union[str, List[str], None] = None,
    dna_sequences: Union[str, List[str], None] = None,
    a3d_mean: Union[float, List[float], None] = None,
    reverse_translate_strategy: str = "first",
    fixed_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    compute_missing_a3d: bool = True,
) -> pd.DataFrame:
    """
    Compute Cu objectives and final score for one or many length-10 peptides.

    Normalization is fixed-range by default. It uses, in order of priority:
      1) fixed_ranges passed to this function,
      2) ranges initialized with initialize_fixed_objective_ranges_from_training_df(...),
      3) _DEFAULT_FIXED_RANGES.

    No per-batch normalization is used.
    """
    if isinstance(peptides, str):
        peptides_list = [peptides]
    else:
        peptides_list = list(peptides)

    n = len(peptides_list)

    def _as_list(x, default=None):
        if x is None:
            return [default] * n
        if isinstance(x, (str, float, int, np.floating, np.integer)):
            return [x] * n
        x = list(x)
        if len(x) != n:
            raise ValueError(f"Expected {n} values, got {len(x)}.")
        return x

    labels_list = _as_list(binding_site_labels_len10, default="")
    dna_list = _as_list(dna_sequences, default="")
    a3d_list = _as_list(a3d_mean, default=np.nan)

    df = pd.DataFrame(
        {
            "peptide_len10": [
                str(s).strip().upper().replace("*", "").replace("U", "")
                for s in peptides_list
            ],
            "binding_site_labels_len10": ["" if s is None else str(s) for s in labels_list],
            "DNA_sequence": ["" if s is None else str(s).strip().upper() for s in dna_list],
            "a3d_mean": pd.to_numeric(pd.Series(a3d_list), errors="coerce"),
        }
    )

    valid_aa = df["peptide_len10"].str.match(r"^[ACDEFGHIKLMNPQRSTVWY]+$", na=False)
    valid_len = df["peptide_len10"].str.len().eq(10)
    df = df[valid_aa & valid_len].copy()
    if df.empty:
        return df

    # Fill DNA if missing.
    need_dna = df["DNA_sequence"].astype(str).str.strip().eq("")
    if need_dna.any():
        uniq = df.loc[need_dna, "peptide_len10"].unique()
        rt_cache = {s: reverse_translate(s, strategy=reverse_translate_strategy) for s in uniq}
        df.loc[need_dna, "DNA_sequence"] = df.loc[need_dna, "peptide_len10"].map(rt_cache)

    labels_parsed = [
        _parse_binding_sites(bs)
        for bs in df["binding_site_labels_len10"].astype(str).tolist()
    ]

    # ProteinAnalysis features per unique peptide.
    uniq_seq = df["peptide_len10"].unique()
    pa_cache: Dict[str, Tuple[float, float, float, float]] = {}
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

    # Cu binding/motif features.
    cu_rows = [
        _cu_features(seq, lab)
        for seq, lab in zip(df["peptide_len10"].tolist(), labels_parsed)
    ]
    cu_df = pd.DataFrame(cu_rows)
    df = pd.concat([df.reset_index(drop=True), cu_df.reset_index(drop=True)], axis=1)

    # DNA expression helper features.
    df["gc_frac"] = [_gc_fraction(d) for d in df["DNA_sequence"].tolist()]
    df["longest_run"] = [_longest_homopolymer(d) for d in df["DNA_sequence"].tolist()]
    gc_dev = (df["gc_frac"] - 0.5).abs()
    df["gc_score_raw"] = (1.0 - gc_dev / 0.5).clip(lower=0.0, upper=1.0)

    # Updated Cu chelation objective.
    df["cu_chelation_raw"] = (
        0.30 * df["frac_H"]
        + 0.25 * df["frac_DE"]
        + 0.15 * df["frac_CM"]
        + 0.10 * df["frac_KR"]
        + 0.10 * df["cu_motif_density"]
        + 0.10 * df["label_cluster_density"]
    )

    # Preserve supplied/precomputed A3D values and compute only missing rows.
    if compute_missing_a3d and df["a3d_mean"].isna().any():
        missing_mask = df["a3d_mean"].isna()
        computed_df = add_a3d_mean_to_df(df.loc[missing_mask].copy())
        computed_map = dict(
            zip(
                computed_df["peptide_len10"].astype(str).str.strip().str.upper(),
                computed_df["a3d_mean"],
            )
        )
        df.loc[missing_mask, "a3d_mean"] = (
            df.loc[missing_mask, "peptide_len10"]
            .astype(str)
            .str.strip()
            .str.upper()
            .map(computed_map)
        )

    # Length score. For strict length-10 peptides this is normally 1.0, but it
    # is retained for consistency with the training sorter and BO scripts.
    L_med = df["peptide_len10"].str.len().median()
    len_dev = (df["peptide_len10"].str.len() - L_med).abs() / max(float(L_med), 1.0)
    df["len_score_raw"] = (1.0 - len_dev).clip(lower=0.0, upper=1.0)

    # Canonical fixed-range normalization using the Cu margined JSON ranges.
    df = apply_fixed_objective_normalization(df, ranges=fixed_ranges)

    return df


def blackbox_fc_mo(
    peptides: Union[str, List[str]],
    binding_site_labels_len10: Union[str, List[str], None] = None,
    dna_sequences: Union[str, List[str], None] = None,
    a3d_mean: Union[float, List[float], None] = None,
    device: Optional[str] = None,
    fixed_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    compute_missing_a3d: bool = True,
) -> torch.Tensor:
    """Return Y: [N, 4] = [chelation_sub, solubility_sub, stability_sub, expression_sub]."""
    df = blackbox_fc(
        peptides=peptides,
        binding_site_labels_len10=binding_site_labels_len10,
        dna_sequences=dna_sequences,
        a3d_mean=a3d_mean,
        fixed_ranges=fixed_ranges,
        compute_missing_a3d=compute_missing_a3d,
    )

    n_in = 1 if isinstance(peptides, str) else len(peptides)
    if len(df) != n_in:
        raise ValueError(
            f"Some peptides were filtered out as invalid. in={n_in}, kept={len(df)}. "
            "Check sequence length, characters, or empty strings."
        )

    Y = torch.tensor(
        df[["chelation_sub", "solubility_sub", "stability_sub", "expression_sub"]].to_numpy(),
        dtype=torch.float32,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    return Y
