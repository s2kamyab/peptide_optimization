"""Gold objective implementation updated to match the attached ranking specification."""
from __future__ import annotations

import re
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from Bio.Data import CodonTable
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from run_esnfold import run_esmfold_hf
from run_a3d import run_a3d_on_pdb


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
    pdb_root: Path = Path("AU_metaldb_pdbs"),
    a3d_root: Path = Path("AU_metaldb_a3ds"),
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
    cai: Union[float, List[float], None] = None,
    dg_rbs: Union[float, List[float], None] = None,
    reverse_translate_strategy: str = "first",
) -> pd.DataFrame:
    """Compute gold objectives and final score for one or many peptides."""

    if isinstance(peptides, str):
        peptides_list = [peptides]
    else:
        peptides_list = list(peptides)

    n = len(peptides_list)

    def _as_list(x, default=None):
        if x is None:
            return [default] * n
        if isinstance(x, (str, int, float, np.floating)):
            return [x]
        return list(x)

    labels_list = _as_list(binding_site_labels_len10, default="")
    dna_list = _as_list(dna_sequences, default="")
    a3d_list = _as_list(a3d_mean, default=np.nan)
    cai_list = _as_list(cai, default=np.nan)
    dg_list = _as_list(dg_rbs, default=np.nan)

    df = pd.DataFrame(
        {
            "peptide_len10": [str(s).strip().upper().replace("*", "").replace("U", "") for s in peptides_list],
            "binding_site_labels_len10": ["" if s is None else str(s) for s in labels_list],
            "DNA_sequence": ["" if s is None else str(s).strip().upper() for s in dna_list],
            "a3d_mean": pd.to_numeric(pd.Series(a3d_list), errors="coerce"),
            "CAI": pd.to_numeric(pd.Series(cai_list), errors="coerce"),
            "dg_rbs": pd.to_numeric(pd.Series(dg_list), errors="coerce"),
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

    gold_rows = [_gold_features(seq, lab) for seq, lab in zip(df["peptide_len10"].tolist(), labels_parsed)]
    gold_df = pd.DataFrame(gold_rows)
    df = pd.concat([df.reset_index(drop=True), gold_df.reset_index(drop=True)], axis=1)

    df["gc_frac"] = [_gc_fraction(d) for d in df["DNA_sequence"].tolist()]
    df["longest_run"] = [_longest_homopolymer(d) for d in df["DNA_sequence"].tolist()]

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

    df["chelation_sub"] = df["C_sol"]

    # ---- Solubility / low aggregation ----
    if df["a3d_mean"].isna().all():
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
    cai: Union[float, List[float], None] = None,
    dg_rbs: Union[float, List[float], None] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    """Returns Y: [N, 4] = [chelation_sub, solubility_sub, stability_sub, expression_sub]."""
    df = blackbox_fc(
        peptides=peptides,
        binding_site_labels_len10=binding_site_labels_len10,
        dna_sequences=dna_sequences,
        a3d_mean=a3d_mean,
        cai=cai,
        dg_rbs=dg_rbs,
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
