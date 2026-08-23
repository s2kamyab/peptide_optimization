import os
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from peptide_optimization.src.util.run_a3d import run_a3d_on_pdb
from peptide_optimization.src.util.run_colabfold import run_colabfold_wsl
from peptide_optimization.src.scripts.black_box_fcn import (
    _parse_binding_sites,
    _gc_fraction,
    _longest_homopolymer,
    reverse_translate,
)

CSV_IN  = "metalpdb_CU_chain_mapped_len10_high_confidence.csv"
CSV_OUT = "metalpdb_binding_windows_len10_CU_chain_mapped_scored_ranked.csv"
RANGES_IN  = "cu_objective_fixed_ranges_training_CU_updated_margined.json"

# A3D (Windows executable) - keep as you had it
A3D_EXE = r"py27venv\Scripts\aggrescan.exe"

AA_OK = set("ACDEFGHIKLMNPQRSTVWY")
SEQ_LEN = 10

# Fallbacks are used only when a raw feature is missing or constant in the
# training dataset. Otherwise, min/max are learned once from the full training
# dataframe and reused for all rows in this scoring run.
_DEFAULT_FIXED_RANGES = {
    "cu_chelation_raw": (0.0, 0.45),
    "a3d_mean": (-2.0, 2.0),
    "gravy": (-4.5, 4.5),
    "aromaticity": (0.0, 0.6),
    "instability_index": (-20.0, 120.0),
    "gc_score_raw": (0.0, 1.0),
    "longest_run": (1.0, 10.0),
    "len_score_raw": (0.0, 1.0),
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


def fixed_minmax(x, lo: float, hi: float) -> pd.Series:
    """Fixed min-max scaling using precomputed training-set bounds."""
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    denom = max(float(hi) - float(lo), 1e-12)
    return ((x - float(lo)) / denom).clip(0.0, 1.0)


def fixed_minmax_reverse(x, lo: float, hi: float) -> pd.Series:
    """Fixed reverse min-max scaling; lower raw value is better."""
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


def compute_fixed_ranges_from_training_df(df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """Compute fixed normalization ranges once from the complete training dataframe."""
    ranges: Dict[str, Tuple[float, float]] = {}
    for col, fallback in _DEFAULT_FIXED_RANGES.items():
        if col in df.columns:
            ranges[col] = _safe_numeric_range(df[col], fallback)
        else:
            ranges[col] = (float(fallback[0]), float(fallback[1]))
    return ranges


def save_fixed_ranges(ranges: Dict[str, Tuple[float, float]], path: str) -> None:
    """Save fixed ranges to JSON. Kept for optional diagnostics/backups."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: [float(v[0]), float(v[1])] for k, v in ranges.items()}, f, indent=2)


def load_fixed_ranges(path: str = RANGES_IN) -> Dict[str, Tuple[float, float]]:
    """
    Load fixed normalization ranges from JSON instead of recomputing them
    from the current dataframe. This keeps training, validation, and BO
    candidates on the same objective scale.
    """
    json_path = Path(path)
    if not json_path.exists():
        # Also try locating the JSON beside this Python file.
        alt_path = Path(__file__).resolve().parent / path
        if alt_path.exists():
            json_path = alt_path
        else:
            raise FileNotFoundError(
                f"Fixed-ranges JSON not found: {path}. "
                f"Place {RANGES_IN} beside this script or pass a valid path."
            )

    with json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    required = list(_DEFAULT_FIXED_RANGES.keys())
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(
            "Fixed-ranges JSON is missing required keys: " + ", ".join(missing)
        )

    ranges: Dict[str, Tuple[float, float]] = {}
    for k in required:
        vals = raw[k]
        if not isinstance(vals, (list, tuple)) or len(vals) != 2:
            raise ValueError(f"Range for {k!r} must be [lower, upper].")
        lo, hi = float(vals[0]), float(vals[1])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            raise ValueError(f"Invalid fixed range for {k!r}: {vals}")
        ranges[k] = (lo, hi)

    return ranges


def _cu_motif_density(seq: str) -> float:
    """
    Cu motif density following the Word specification examples:
    HxH, HxxH, HExxH, and HxxEH.
    """
    seq = (seq or "").strip().upper()
    L = max(len(seq), 1)
    patterns = [
        r"H.H",      # HxH
        r"H..H",     # HxxH
        r"HE..H",    # HExxH
        r"H..EH",    # HxxEH
    ]
    motif_hits = sum(len(re.findall(p, seq)) for p in patterns)
    return float(motif_hits / L)


def _label_cluster_density(labels: List[int], seq_len: int) -> float:
    """
    Center-weighted label/cluster density, consistent with the black-box code
    and the Word file description that allows center-weighted cluster density.
    """
    if not labels:
        return 0.0

    labels = [int(v) if str(v).strip() in {"0", "1"} else 0 for v in labels]
    n = len(labels)
    if n == 0:
        return 0.0

    frac = sum(labels) / max(n, 1)
    pos = [i for i, v in enumerate(labels) if v == 1]
    if not pos:
        return 0.0

    center = (sum(pos) / len(pos)) / max(n - 1, 1)
    center_score = 1.0 - abs(center - 0.5) / 0.5
    center_score = float(np.clip(center_score, 0.0, 1.0))
    return float(frac * (0.5 + 0.5 * center_score))


def _cu_features(seq: str, labels: List[int]) -> Dict[str, float]:
    """
    Cu-specific raw sequence features matching black_box_fcn_mo_CU_f_updated.py
    and the attached Word ranking specification.
    """
    seq = (seq or "").strip().upper()
    L = max(len(seq), 1)

    frac_H = seq.count("H") / L
    frac_DE = (seq.count("D") + seq.count("E")) / L
    frac_CM = (seq.count("C") + seq.count("M")) / L
    frac_KR = (seq.count("K") + seq.count("R")) / L
    motif_density = _cu_motif_density(seq)
    label_cluster_density = _label_cluster_density(labels, L)

    return {
        "frac_H": float(frac_H),
        "frac_DE": float(frac_DE),
        "frac_CM": float(frac_CM),
        "frac_KR": float(frac_KR),
        "cu_motif_density": float(motif_density),
        "label_cluster_density": float(label_cluster_density),
    }

def compute_scores_cu_df(df_in: pd.DataFrame, reverse_translate_strategy="first") -> pd.DataFrame:
    df = df_in.copy()

    # --- clean + filter sequences fast ---
    df["peptide_len10"] = (
        df["peptide_len10"].astype(str).str.strip().str.upper()
          .str.replace("*", "", regex=False)
          .str.replace("U", "", regex=False)
    )
    valid_aa = df["peptide_len10"].str.match(r"^[ACDEFGHIKLMNPQRSTVWY]+$", na=False)
    valid_len = df["peptide_len10"].str.len().eq(SEQ_LEN)
    df = df[valid_aa & valid_len].copy()
    if df.empty:
        return df

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

    # --- Run ColabFold + (optional) A3D per unique sequence ---
    # NOTE: This is the slow part. You may want to parallelize later.
    # a3d_means = {}
    # cnt = 0
    # a3d_out = "CU_metaldb_a3ds/"
    # pdb_path = "CU_metaldb_pdbs/peptide_"
    # for seq in uniq_seq:
    #     print(cnt)
    #     cnt = cnt+1
    #     pdb_out_dir = f"C:\\Users\\shima\\OneDrive\\Documentos\\Leili\\peptide_structure_optimization\\peptide_optimization\\CU_metaldb_pdbs\\peptide_{seq}\\input.fasta"
    #     pdb_path = run_colabfold_here(seq, out_dir=pdb_out_dir, threads=8)
    #
    # # Run A3D
    # cnt = 0
    # a3d_out = Path("CU_metaldb_a3ds")
    # pdb_root = Path("CU_metaldb_pdbs")
    #
    # EXPECTED_CSV = "A3D.csv"
    #
    # for seq in uniq_seq:
    #     print(cnt)
    #     cnt += 1
    #
    #     out_dir = a3d_out / seq
    #     out_csv = out_dir / EXPECTED_CSV
    #
    #     if out_csv.exists() and out_csv.stat().st_size > 0:
    #         continue
    #
    #     pdb_file = pdb_root / f"peptide_{seq}" / "query_unrelaxed_rank_001_alphafold2_model_1_seed_000.pdb"
    #     run_a3d_on_pdb(pdb_file, out_dir)

    # ------------------------------------------------------------------
    # Load A3D.csv per peptide and compute mean(score)
    # ------------------------------------------------------------------
    A3D_ROOT = Path("CU_metaldb_a3ds")
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
            m = float(scores.mean()) if scores.notna().any() else float("nan")
            return m
        except Exception:
            return float("nan")

    a3d_means = {s: _read_a3d_mean(s) for s in uniq_seq}

    # ensure column exists and fill
    df["a3d_mean"] = df["peptide_len10"].map(a3d_means)

    # --- Cu chelation features (per row) ---
    cu_rows = [_cu_features(seq, lab) for seq, lab in zip(df["peptide_len10"].tolist(), labels)]
    cu_df = pd.DataFrame(cu_rows)
    df = pd.concat([df.reset_index(drop=True), cu_df.reset_index(drop=True)], axis=1)

    # --- expression helpers ---
    df["gc_frac"] = [_gc_fraction(d) for d in df["DNA_sequence"].tolist()]
    df["longest_run"] = [_longest_homopolymer(d) for d in df["DNA_sequence"].tolist()]

    # ========= Updated Cu chelation raw objective =========
    df["cu_chelation_raw"] = (
        0.30 * df["frac_H"]
        + 0.25 * df["frac_DE"]
        + 0.15 * df["frac_CM"]
        + 0.10 * df["frac_KR"]
        + 0.10 * df["cu_motif_density"]
        + 0.10 * df["label_cluster_density"]
    )

    # ========= Raw helper columns for fixed normalization =========
    lengths = df["peptide_len10"].str.len()
    L_med = float(lengths.median()) if len(lengths) else float(SEQ_LEN)
    if not np.isfinite(L_med) or L_med <= 0:
        L_med = float(SEQ_LEN)
    len_dev = (lengths - L_med).abs() / max(float(L_med), 1.0)
    df["len_score_raw"] = (1.0 - len_dev).clip(lower=0.0, upper=1.0)

    gc_dev = (df["gc_frac"] - 0.5).abs()
    df["gc_score_raw"] = (1.0 - gc_dev / 0.5).clip(lower=0.0, upper=1.0)

    # Load fixed ranges from the provided JSON file.
    # IMPORTANT: Do not recompute ranges from the current dataframe; otherwise
    # the same peptide may receive different normalized values in different runs.
    fixed_ranges = load_fixed_ranges(RANGES_IN)

    # ========= Fixed-range normalized subscores =========
    df["chelation_sub"] = fixed_minmax(
        df["cu_chelation_raw"],
        *fixed_ranges["cu_chelation_raw"],
    )

    a3d_fill = df["a3d_mean"].median() if df["a3d_mean"].notna().any() else np.mean(fixed_ranges["a3d_mean"])
    a3d_good = fixed_minmax_reverse(df["a3d_mean"].fillna(a3d_fill), *fixed_ranges["a3d_mean"])
    gravy_good = fixed_minmax_reverse(df["gravy"], *fixed_ranges["gravy"])
    arom_good = fixed_minmax_reverse(df["aromaticity"], *fixed_ranges["aromaticity"])
    df["solubility_sub"] = 0.6 * a3d_good + 0.3 * gravy_good + 0.1 * arom_good

    instab_good = fixed_minmax_reverse(df["instability_index"], *fixed_ranges["instability_index"])
    len_good = fixed_minmax(df["len_score_raw"], *fixed_ranges["len_score_raw"])
    df["stability_sub"] = 0.7 * instab_good + 0.3 * len_good

    gc_good = fixed_minmax(df["gc_score_raw"], *fixed_ranges["gc_score_raw"])
    run_good = fixed_minmax_reverse(df["longest_run"], *fixed_ranges["longest_run"])
    df["expression_sub"] = 0.7 * gc_good + 0.3 * run_good

    df["final_score"] = (
        0.50 * df["chelation_sub"]
        + 0.30 * df["solubility_sub"]
        + 0.10 * df["stability_sub"]
        + 0.10 * df["expression_sub"]
    )

    return df


def main():
    df_in = pd.read_csv(CSV_IN)

    # Keep only high-confidence chain-mapped positive windows by default.
    # The corrected extractor writes these columns; older inputs without them
    # are still accepted for backward compatibility.
    if "is_negative" in df_in.columns:
        df_in = df_in[df_in["is_negative"].astype(int).eq(0)].copy()
    if "label_confidence" in df_in.columns:
        df_in = df_in[df_in["label_confidence"].astype(str).str.contains("high_chain_mapped", na=False)].copy()
    if "mapping_status" in df_in.columns:
        df_in = df_in[df_in["mapping_status"].astype(str).str.startswith("accepted", na=False)].copy()

    # Deduplicate after filtering. Keep the first occurrence but preserve the
    # audited metadata columns in the retained row.
    dedup_cols = ["peptide_len10"]
    if "binding_site_labels_len10" in df_in.columns:
        dedup_cols.append("binding_site_labels_len10")
    df_in = df_in.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)

    df_out = compute_scores_cu_df(df_in)
    df_out = df_out.sort_values("final_score", ascending=False).reset_index(drop=True)

    df_out.to_csv(CSV_OUT, index=False)
    print(f"Saved ranked peptides to: {CSV_OUT}")
    print(f"Used fixed normalization ranges from: {RANGES_IN}")
    print(f"Rows scored after high-confidence filtering/deduplication: {len(df_out)}")


if __name__ == "__main__":
    main()
