
"""
this function computes blackbox function from scratch by calling colabfold and a3dscan models
"""
from __future__ import annotations

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
# from __future__ import annotations

from typing import List, Optional, Union
import torch

# These helpers are imported in your script and used to compute subscores :contentReference[oaicite:1]{index=1}
from black_box_fcn import (
    _parse_binding_sites, _chel_features, _minmax, _minmax_reverse,
    _gc_fraction, _longest_homopolymer, reverse_translate
)


from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

# Your helpers (uploaded)
from run_colabfold import run_colabfold_wsl  # :contentReference[oaicite:0]{index=0}
from run_a3d import run_a3d_on_pdb          # (expects: run_a3d_on_pdb(pdb_file, out_dir))


def _find_pdb_for_seq(pdb_root: Path, seq: str) -> Optional[Path]:
    """
    Find a PDB produced by ColabFold for a given sequence folder.
    Prefer the standard 'query_unrelaxed_rank_001...' name if present,
    otherwise take the first *.pdb found.
    """
    seq_dir = pdb_root / f"peptide_{seq}"
    if not seq_dir.exists():
        return None

    preferred = seq_dir / "query_unrelaxed_rank_001_alphafold2_model_1_seed_000.pdb"
    if preferred.exists():
        return preferred

    pdbs = sorted(seq_dir.glob("*.pdb"))
    return pdbs[0] if pdbs else None


def _compute_a3d_mean_from_csv(a3d_csv: Path) -> float:
    """
    Read A3D.csv (or whatever your A3D script writes) and compute mean score.
    Adjust column name(s) if your A3D output schema differs.
    """
    t = pd.read_csv(a3d_csv)
    # Common patterns: "score", "Aggrescan3D_score", etc.
    for col in ["score", "a3d_score", "Aggrescan3D_score", "value"]:
        if col in t.columns:
            return float(pd.to_numeric(t[col], errors="coerce").dropna().mean())

    # Fallback: try the last numeric column
    num_cols = [c for c in t.columns if pd.api.types.is_numeric_dtype(t[c])]
    if num_cols:
        return float(pd.to_numeric(t[num_cols[-1]], errors="coerce").dropna().mean())

    raise ValueError(f"No numeric score column found in {a3d_csv}")


def add_a3d_mean_to_df(
    df: pd.DataFrame,
    seq_col: str = "peptide_len10",
    *,
    pdb_root: Path = Path("MN_metaldb_pdbs"),
    a3d_root: Path = Path("MN_metaldb_a3ds"),
    a3d_csv_name: str = "A3D.csv",
    threads: int = 8,
    overwrite_colabfold: bool = False,
) -> pd.DataFrame:
    """
    Adds/updates df['a3d_mean'] by:
      1) Running ColabFold for each unique seq (cached by folder existence)
      2) Running A3D for each unique seq (cached by output CSV existence)
      3) Computing mean A3D score and mapping back to rows

    Returns a copy of df with a3d_mean column.
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

        # -------------------------
        # 1) Run ColabFold (cached)
        # -------------------------
        seq_out_dir = pdb_root / f"peptide_{seq}"
        seq_out_dir.mkdir(parents=True, exist_ok=True)

        pdb_file = _find_pdb_for_seq(pdb_root, seq)
        if pdb_file is None or overwrite_colabfold:
            # run_colabfold_wsl expects a Windows out_dir path (it writes input.fasta internally) :contentReference[oaicite:1]{index=1}
            # If you are on Windows, pass str(seq_out_dir.resolve()) which is a Windows path.
            _ = run_colabfold_wsl(
                seq=seq,
                out_dir_win=str(seq_out_dir.resolve()),
                threads=threads,
                overwrite=overwrite_colabfold,
            )
            pdb_file = _find_pdb_for_seq(pdb_root, seq)

        if pdb_file is None or not pdb_file.exists():
            print(f"  - skipped: no PDB produced for {seq}")
            continue

        # -------------------------
        # 2) Run A3D (cached)
        # -------------------------
        a3d_out_dir = a3d_root / seq
        a3d_out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = a3d_out_dir / a3d_csv_name

        if not (out_csv.exists() and out_csv.stat().st_size > 0):
            run_a3d_on_pdb(pdb_file, a3d_out_dir)

        if not (out_csv.exists() and out_csv.stat().st_size > 0):
            print(f"  - skipped: A3D output missing for {seq}")
            continue

        # -------------------------
        # 3) Compute mean score
        # -------------------------
        try:
            a3d_means[seq] = _compute_a3d_mean_from_csv(out_csv)
        except Exception as e:
            print(f"  - skipped: cannot parse A3D for {seq}: {e}")
            continue

    out_df["a3d_mean"] = out_df[seq_col].astype(str).str.strip().str.upper().map(a3d_means)

    return out_df

AA_OK = set("ACDEFGHIKLMNPQRSTVWY")


def blackbox_fc(
    peptides: Union[str, List[str]],
    binding_site_labels_len10: Union[str, List[str], None] = None,
    dna_sequences: Union[str, List[str], None] = None,
    a3d_mean: Union[float, List[float], None] = None,
    reverse_translate_strategy: str = "first",
) -> pd.DataFrame:
    """
    Compute the Mn scoring pipeline and final_score for one or many peptides,
    matching sort_training_data_MN.py. :contentReference[oaicite:2]{index=2}

    Parameters
    ----------
    peptides:
        peptide string or list of peptide strings (expected length=10, but code works for any length).
    binding_site_labels_len10:
        label string(s) like "0 0 0 0 1 0 0 0 0 0". If None, labels are set to all zeros.
        Used by _chel_features(). :contentReference[oaicite:3]{index=3}
    dna_sequences:
        DNA string(s). If None/empty, will be filled by reverse_translate(). 
    a3d_mean:
        A3D mean score(s). If None, set to NaN (then median fill behavior applies).
    reverse_translate_strategy:
        Passed to reverse_translate().

    Returns
    -------
    DataFrame with chelation_sub, solubility_sub, stability_sub, expression_sub, final_score
    plus intermediate columns used in scoring.
    """

    # ---- normalize inputs to lists ----
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

    # ---- build df ----
    df = pd.DataFrame(
        {
            "peptide_len10": [str(s).strip().upper().replace("*", "").replace("U", "") for s in peptides_list],
            "binding_site_labels_len10": ["" if s is None else str(s) for s in labels_list],
            "DNA_sequence": ["" if s is None else str(s).strip().upper() for s in dna_list],
            "a3d_mean": pd.to_numeric(pd.Series(a3d_list), errors="coerce"),
        }
    )

    # ---- filter invalid peptides (same AA set) :contentReference[oaicite:5]{index=5} ----
    df = df[df["peptide_len10"].str.match(r"^[ACDEFGHIKLMNPQRSTVWY]+$", na=False)].copy()
    if df.empty:
        return df

    # ---- fill DNA if missing (reverse_translate cache) :contentReference[oaicite:6]{index=6} ----
    need_dna = df["DNA_sequence"].astype(str).str.strip().eq("")
    if need_dna.any():
        uniq = df.loc[need_dna, "peptide_len10"].unique()
        rt_cache = {s: reverse_translate(s, strategy=reverse_translate_strategy) for s in uniq}
        df.loc[need_dna, "DNA_sequence"] = df.loc[need_dna, "peptide_len10"].map(rt_cache)

    # ---- parse binding site labels for chelation features :contentReference[oaicite:7]{index=7} ----
    labels_parsed = [
        _parse_binding_sites(bs) for bs in df["binding_site_labels_len10"].astype(str).tolist()
    ]

    # ---- ProteinAnalysis features (mw/arom/instab/gravy) :contentReference[oaicite:8]{index=8} ----
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

    # ---- chelation features (per row) :contentReference[oaicite:9]{index=9} ----
    chel_rows = [
        _chel_features(seq, lab) for seq, lab in zip(df["peptide_len10"].tolist(), labels_parsed)
    ]
    chel_df = pd.DataFrame(chel_rows)
    df = pd.concat([df.reset_index(drop=True), chel_df.reset_index(drop=True)], axis=1)

    # ---- expression helpers gc_frac + longest_run :contentReference[oaicite:10]{index=10} ----
    df["gc_frac"] = [_gc_fraction(d) for d in df["DNA_sequence"].tolist()]
    df["longest_run"] = [_longest_homopolymer(d) for d in df["DNA_sequence"].tolist()]

    # ========= Mn chelation raw (vectorized) :contentReference[oaicite:11]{index=11} =========
    seqs = df["peptide_len10"]
    L = seqs.str.len().replace(0, np.nan)
    fD = seqs.str.count("D") / L
    fH = seqs.str.count("H") / L
    fE = seqs.str.count("E") / L
    df["mn_chelation_raw"] = (0.45 * fD + 0.35 * fH + 0.20 * fE).fillna(0.0)
    df["chelation_sub"] = _minmax(df["mn_chelation_raw"])

    # ========= solubility_sub :contentReference[oaicite:12]{index=12} =========
    df = add_a3d_mean_to_df(df)
    a3d_fill = df["a3d_mean"].median() if df["a3d_mean"].notna().any() else 0.0
    a3d_good = _minmax_reverse(df["a3d_mean"].fillna(a3d_fill))
    gravy_good = _minmax_reverse(df["gravy"])
    arom_good = _minmax_reverse(df["aromaticity"])
    df["solubility_sub"] = 0.6 * a3d_good + 0.3 * gravy_good + 0.1 * arom_good

    # ========= stability_sub :contentReference[oaicite:13]{index=13} =========
    instab_good = _minmax_reverse(df["instability_index"])
    L_med = df["peptide_len10"].str.len().median()
    len_dev = (df["peptide_len10"].str.len() - L_med).abs() / max(float(L_med), 1.0)
    len_score_raw = (1.0 - len_dev).clip(lower=0.0)
    len_good = _minmax(len_score_raw)
    df["stability_sub"] = 0.7 * instab_good + 0.3 * len_good

    # ========= expression_sub :contentReference[oaicite:14]{index=14} =========
    gc_dev = (df["gc_frac"] - 0.5).abs()
    gc_score_raw = (1.0 - gc_dev / 0.5).clip(lower=0.0)
    gc_good = _minmax(gc_score_raw)
    run_good = _minmax_reverse(df["longest_run"])
    df["expression_sub"] = 0.7 * gc_good + 0.3 * run_good

    # ========= final_score :contentReference[oaicite:15]{index=15} =========
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
    """
    Multi-objective black-box for BO.
    Returns Y: [N, 4] = [chelation_sub, solubility_sub, stability_sub, expression_sub]
    """
    df = blackbox_fc(
        peptides=peptides,
        binding_site_labels_len10=binding_site_labels_len10,
        dna_sequences=dna_sequences,
        a3d_mean=a3d_mean,
    )

    # If any peptide filtered out as invalid, raise (BO expects aligned outputs)
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