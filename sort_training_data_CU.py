import os
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from run_a3d import run_a3d_on_pdb
from run_colabfold import run_colabfold_wsl
from black_box_fcn import (
    _parse_binding_sites, _chel_features, _minmax, _minmax_reverse,
    _gc_fraction, _longest_homopolymer, reverse_translate
)

CSV_IN  = "metalpdb_Cu_binding_windows_len10.csv"
CSV_OUT = "metalpdb_binding_windows_len10_CU_scored_ranked.csv"

# A3D (Windows executable) - keep as you had it
A3D_EXE = r"py27venv\Scripts\aggrescan.exe"

AA_OK = set("ACDEFGHIKLMNPQRSTVWY")


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


def _cu_chelation_raw_from_seq(seq_series: pd.Series) -> pd.Series:
    """
    Heuristic Cu chelation score.

    Copper-binding peptides are often enriched in:
      - H (His)
      - C (Cys)
      - M (Met)

    This returns a raw weighted score before batch min-max normalization.
    """
    L = seq_series.str.len().replace(0, np.nan)
    fH = seq_series.str.count("H") / L
    fC = seq_series.str.count("C") / L
    fM = seq_series.str.count("M") / L

    return (0.40 * fH + 0.40 * fC + 0.20 * fM).fillna(0.0)


def compute_scores_cu_df(df_in: pd.DataFrame, reverse_translate_strategy="first") -> pd.DataFrame:
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

    # --- chelation features (per row) ---
    chel_rows = [_chel_features(seq, lab) for seq, lab in zip(df["peptide_len10"].tolist(), labels)]
    chel_df = pd.DataFrame(chel_rows)
    df = pd.concat([df.reset_index(drop=True), chel_df.reset_index(drop=True)], axis=1)

    # --- expression helpers ---
    df["gc_frac"] = [_gc_fraction(d) for d in df["DNA_sequence"].tolist()]
    df["longest_run"] = [_longest_homopolymer(d) for d in df["DNA_sequence"].tolist()]

    # ========= Vectorized Cu chelation raw =========
    seqs = df["peptide_len10"]
    df["cu_chelation_raw"] = _cu_chelation_raw_from_seq(seqs)
    df["chelation_sub"] = _minmax(df["cu_chelation_raw"])

    # ========= Other subscores =========
    a3d_fill = df["a3d_mean"].median() if df["a3d_mean"].notna().any() else 0.0
    a3d_good   = _minmax_reverse(df["a3d_mean"].fillna(a3d_fill))
    gravy_good = _minmax_reverse(df["gravy"])
    arom_good  = _minmax_reverse(df["aromaticity"])
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

    df_out = compute_scores_cu_df(df_in)
    df_out = df_out.sort_values("final_score", ascending=False).reset_index(drop=True)

    df_out.to_csv(CSV_OUT, index=False)
    print(f"Saved ranked peptides to: {CSV_OUT}")


if __name__ == "__main__":
    main()