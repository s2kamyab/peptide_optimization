"""Cu objective implementation updated to match the attached ranking specification."""
from __future__ import annotations

from typing import Dict, List, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from black_box_fcn_cu_updated import (
    _parse_binding_sites,
    _minmax,
    _minmax_reverse,
    _gc_fraction,
    _longest_homopolymer,
    reverse_translate,
    _cu_features,
)

from run_esnfold import run_esmfold_hf
from run_a3d import run_a3d_on_pdb


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
            return float(pd.to_numeric(t[col], errors="coerce").dropna().mean())

    num_cols = [c for c in t.columns if pd.api.types.is_numeric_dtype(t[c])]
    if num_cols:
        return float(pd.to_numeric(t[num_cols[-1]], errors="coerce").dropna().mean())

    raise ValueError(f"No numeric score column found in {a3d_csv}")



def add_a3d_mean_to_df(
    df: pd.DataFrame,
    seq_col: str = "peptide_len10",
    *,
    pdb_root: Path = Path("CU_metaldb_pdbs"),
    a3d_root: Path = Path("CU_metaldb_a3ds"),
    a3d_csv_name: str = "A3D.csv",
    threads: int = 8,
    overwrite_colabfold: bool = False,
) -> pd.DataFrame:
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
        out_csv = a3d_out_dir / a3d_csv_name

        if not (out_csv.exists() and out_csv.stat().st_size > 0):
            produced_csv = run_a3d_on_pdb(pdb_file, a3d_out_dir)
            if produced_csv is not None and Path(produced_csv).exists():
                out_csv = Path(produced_csv)

        if not (out_csv.exists() and out_csv.stat().st_size > 0):
            print(f"  - skipped: A3D output missing for {seq}")
            continue

        try:
            a3d_means[seq] = _compute_a3d_mean_from_csv(out_csv)
        except Exception as e:
            print(f"  - skipped: cannot parse A3D for {seq}: {e}")
            continue

    out_df["a3d_mean"] = out_df[seq_col].astype(str).str.strip().str.upper().map(a3d_means)
    return out_df



def blackbox_fc(
    peptides: Union[str, List[str]],
    binding_site_labels_len10: Union[str, List[str], None] = None,
    dna_sequences: Union[str, List[str], None] = None,
    a3d_mean: Union[float, List[float], None] = None,
    reverse_translate_strategy: str = "first",
) -> pd.DataFrame:
    """Compute Cu objectives and final score for one or many peptides."""

    if isinstance(peptides, str):
        peptides_list = [peptides]
    else:
        peptides_list = list(peptides)

    n = len(peptides_list)

    def _as_list(x, default=None):
        if x is None:
            return [default] * n
        if isinstance(x, str):
            return [x]
        return list(x)

    labels_list = _as_list(binding_site_labels_len10, default="")
    dna_list = _as_list(dna_sequences, default="")
    a3d_list = _as_list(a3d_mean, default=np.nan)

    df = pd.DataFrame(
        {
            "peptide_len10": [str(s).strip().upper().replace("*", "").replace("U", "") for s in peptides_list],
            "binding_site_labels_len10": ["" if s is None else str(s) for s in labels_list],
            "DNA_sequence": ["" if s is None else str(s).strip().upper() for s in dna_list],
            "a3d_mean": pd.to_numeric(pd.Series(a3d_list), errors="coerce"),
        }
    )

    df = df[df["peptide_len10"].str.match(r"^[ACDEFGHIKLMNPQRSTVWY]+$", na=False)].copy()
    if df.empty:
        return df

    need_dna = df["DNA_sequence"].astype(str).str.strip().eq("")
    if need_dna.any():
        uniq = df.loc[need_dna, "peptide_len10"].unique()
        rt_cache = {s: reverse_translate(s, strategy=reverse_translate_strategy) for s in uniq}
        df.loc[need_dna, "DNA_sequence"] = df.loc[need_dna, "peptide_len10"].map(rt_cache)

    labels_parsed = [_parse_binding_sites(bs) for bs in df["binding_site_labels_len10"].astype(str).tolist()]

    uniq_seq = df["peptide_len10"].unique()
    pa_cache: Dict[str, tuple[float, float, float, float]] = {}
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

    cu_rows = [_cu_features(seq, lab) for seq, lab in zip(df["peptide_len10"].tolist(), labels_parsed)]
    cu_df = pd.DataFrame(cu_rows)
    df = pd.concat([df.reset_index(drop=True), cu_df.reset_index(drop=True)], axis=1)

    df["gc_frac"] = [_gc_fraction(d) for d in df["DNA_sequence"].tolist()]
    df["longest_run"] = [_longest_homopolymer(d) for d in df["DNA_sequence"].tolist()]

    # ---- Cu chelation term from the Word specification ----
    df["cu_chelation_raw"] = (
        0.30 * df["frac_H"]
        + 0.25 * df["frac_DE"]
        + 0.15 * df["frac_CM"]
        + 0.10 * df["frac_KR"]
        + 0.10 * df["cu_motif_density"]
        + 0.10 * df["label_cluster_density"]
    )
    df["chelation_sub"] = _minmax(df["cu_chelation_raw"])

    # ---- Solubility / low aggregation ----
    df = add_a3d_mean_to_df(df)
    a3d_fill = df["a3d_mean"].median() if df["a3d_mean"].notna().any() else 0.0
    a3d_good = _minmax_reverse(df["a3d_mean"].fillna(a3d_fill))
    gravy_good = _minmax_reverse(df["gravy"])
    arom_good = _minmax_reverse(df["aromaticity"])
    df["solubility_sub"] = 0.6 * a3d_good + 0.3 * gravy_good + 0.1 * arom_good

    # ---- Structural stability ----
    instab_good = _minmax_reverse(df["instability_index"])
    L_med = df["peptide_len10"].str.len().median()
    len_dev = (df["peptide_len10"].str.len() - L_med).abs() / max(float(L_med), 1.0)
    len_score_raw = (1.0 - len_dev).clip(lower=0.0)
    len_good = _minmax(len_score_raw)
    df["stability_sub"] = 0.7 * instab_good + 0.3 * len_good

    # ---- Ease of expression ----
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

    return df



def blackbox_fc_mo(
    peptides: Union[str, List[str]],
    binding_site_labels_len10: Union[str, List[str], None] = None,
    dna_sequences: Union[str, List[str], None] = None,
    a3d_mean: Union[float, List[float], None] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    """Returns Y: [N, 4] = [chelation_sub, solubility_sub, stability_sub, expression_sub]."""
    df = blackbox_fc(
        peptides=peptides,
        binding_site_labels_len10=binding_site_labels_len10,
        dna_sequences=dna_sequences,
        a3d_mean=a3d_mean,
    )

    n_in = 1 if isinstance(peptides, str) else len(peptides)
    if len(df) != n_in:
        raise ValueError(
            f"Some peptides were filtered out as invalid. in={n_in}, kept={len(df)}. "
            f"Check characters / empty strings."
        )

    Y = torch.tensor(
        df[["chelation_sub", "solubility_sub", "stability_sub", "expression_sub"]].to_numpy(),
        dtype=torch.float32,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    return Y
