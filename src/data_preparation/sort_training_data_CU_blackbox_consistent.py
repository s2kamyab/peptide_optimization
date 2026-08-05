"""
Score/rank corrected Cu length-10 peptide windows using the same black-box
objective path used by BO_gp_after_flow_epoch199_latent_conditioned.py.

Main difference from older sort_training_data_CU_*.py scripts:
  - This script does NOT reimplement Cu objective formulas locally.
  - It imports blackbox_fc from black_box_fcn_mo_CU_f.py and delegates scoring
    to that function, matching the BO black-box evaluation path.
  - If a3d_mean is missing and --compute-missing-a3d is enabled, blackbox_fc can
    compute missing structures/A3D values using the workspace PDB/A3D pipeline.

Expected workspace files/modules:
  - black_box_fcn_mo_CU_f.py
  - run_esnfold.py / run_a3d.py or whatever dependencies your blackbox module uses
  - Optional cached folders: CU_metaldb_pdbs/, CU_metaldb_a3ds/
  - Optional fixed-ranges JSON: cu_objective_fixed_ranges_training_CU_updated_margined.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

AA = "ACDEFGHIKLMNPQRSTVWY"
SEQ_LEN = 10
OBJ_COLS = ["chelation_sub", "solubility_sub", "stability_sub", "expression_sub"]


def clean_peptide(value: object, seq_len: int = SEQ_LEN) -> Optional[str]:
    pep = str(value).strip().upper().replace("*", "").replace("U", "")
    if len(pep) != seq_len:
        return None
    if any(ch not in AA for ch in pep):
        return None
    return pep


def load_objective_module(module_path: str):
    """Load black_box_fcn_mo_CU_f.py from an explicit file path."""
    path = Path(module_path)
    if not path.exists():
        # allow normal import from sys.path / current directory
        import black_box_fcn_mo_CU_f as module  # type: ignore
        return module

    module_dir = str(path.resolve().parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    spec = importlib.util.spec_from_file_location("black_box_fcn_mo_CU_f", str(path.resolve()))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load objective module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["black_box_fcn_mo_CU_f"] = module
    spec.loader.exec_module(module)
    return module


def maybe_load_ranges(module: Any, ranges_json: Optional[str]) -> Optional[Dict[str, tuple]]:
    """Load and activate fixed objective ranges if the module exposes the loader."""
    if not ranges_json:
        return None
    p = Path(ranges_json)
    if not p.exists():
        print(f"[WARN] fixed-ranges JSON not found: {ranges_json}; using objective module defaults/auto lookup.")
        return None
    if hasattr(module, "load_fixed_objective_ranges"):
        ranges = module.load_fixed_objective_ranges(p)
        print(f"Loaded fixed objective ranges via objective module: {p}")
        return ranges
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    ranges = {k: (float(v[0]), float(v[1])) for k, v in raw.items()}
    if hasattr(module, "set_fixed_objective_ranges"):
        module.set_fixed_objective_ranges(ranges)
        print(f"Activated fixed objective ranges via set_fixed_objective_ranges: {p}")
    else:
        print(f"Loaded fixed ranges from {p}; passing them to blackbox_fc.")
    return ranges


def prepare_input_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.input_csv)

    if args.peptide_col not in df.columns:
        raise KeyError(f"Peptide column {args.peptide_col!r} not found in {args.input_csv}")

    df = df.copy()
    df[args.peptide_col] = df[args.peptide_col].map(lambda x: clean_peptide(x, args.seq_len))
    before = len(df)
    df = df[df[args.peptide_col].notna()].copy()
    print(f"Valid length-{args.seq_len} peptide rows: {len(df)}/{before}")

    # Optional high-confidence filtering, compatible with corrected extractor outputs.
    if args.filter_high_confidence:
        before_filter = len(df)
        if "is_negative" in df.columns:
            df = df[pd.to_numeric(df["is_negative"], errors="coerce").fillna(0).astype(int).eq(0)].copy()
        if "label_confidence" in df.columns:
            df = df[df["label_confidence"].astype(str).str.contains("high_chain_mapped", na=False)].copy()
        if "mapping_status" in df.columns:
            df = df[df["mapping_status"].astype(str).str.startswith("accepted", na=False)].copy()
        print(f"Rows after high-confidence filters: {len(df)}/{before_filter}")

    if args.labels_col not in df.columns:
        print(f"[WARN] labels column {args.labels_col!r} not found; using empty labels.")
        df[args.labels_col] = ""

    if args.dna_col not in df.columns:
        df[args.dna_col] = ""

    if args.deduplicate:
        dedup_cols = [args.peptide_col]
        if args.dedup_include_labels and args.labels_col in df.columns:
            dedup_cols.append(args.labels_col)
        before_dedup = len(df)
        df = df.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)
        print(f"Rows after deduplication on {dedup_cols}: {len(df)}/{before_dedup}")
    else:
        df = df.reset_index(drop=True)

    return df


def evaluate_with_blackbox(df: pd.DataFrame, module: Any, args: argparse.Namespace, fixed_ranges=None) -> pd.DataFrame:
    if not hasattr(module, "blackbox_fc"):
        raise AttributeError("Objective module does not expose blackbox_fc(...).")

    peptides = df[args.peptide_col].astype(str).str.strip().str.upper().tolist()
    labels = df[args.labels_col].fillna("").astype(str).tolist()
    dna = df[args.dna_col].fillna("").astype(str).tolist()

    if args.a3d_col in df.columns:
        a3d_values = pd.to_numeric(df[args.a3d_col], errors="coerce").tolist()
        print(f"Using pre-existing column {args.a3d_col!r} as a3d_mean input.")
    else:
        a3d_values = [np.nan] * len(df)
        print(f"No {args.a3d_col!r} column found; blackbox_fc will use cached/compute_missing_a3d behavior.")

    kwargs: Dict[str, Any] = {
        "peptides": peptides,
        "binding_site_labels_len10": labels,
        "dna_sequences": dna,
        "a3d_mean": a3d_values,
        "compute_missing_a3d": bool(args.compute_missing_a3d),
    }
    if fixed_ranges is not None:
        kwargs["fixed_ranges"] = fixed_ranges

    print("Calling blackbox_fc(...) with:")
    print(f"  peptides: {len(peptides)}")
    print(f"  compute_missing_a3d: {args.compute_missing_a3d}")
    print(f"  fixed_ranges passed: {fixed_ranges is not None}")

    scored = module.blackbox_fc(**kwargs)
    if len(scored) != len(df):
        raise RuntimeError(
            f"blackbox_fc returned {len(scored)} rows for {len(df)} input rows. "
            "This usually means some peptides were filtered as invalid after preprocessing."
        )

    # Avoid duplicated columns from scored dataframe; preserve original metadata first.
    scored = scored.reset_index(drop=True)
    original = df.reset_index(drop=True)
    add_cols = [c for c in scored.columns if c not in original.columns]
    overlap_keep_from_scored = [
        c for c in [
            args.a3d_col,
            "molecular_weight",
            "aromaticity",
            "instability_index",
            "gravy",
            "frac_H",
            "frac_DE",
            "frac_CM",
            "frac_KR",
            "cu_motif_density",
            "label_cluster_density",
            "gc_frac",
            "longest_run",
            "gc_score_raw",
            "len_score_raw",
            "cu_chelation_raw",
            *OBJ_COLS,
            "final_score",
        ]
        if c in scored.columns and c not in add_cols and c not in original.columns
    ]
    out = pd.concat([original, scored[add_cols + overlap_keep_from_scored]], axis=1)

    # Ensure objectives are present.
    missing_obj = [c for c in OBJ_COLS + ["final_score"] if c not in out.columns]
    if missing_obj:
        # If columns overlapped with original, refresh from scored explicitly.
        for c in missing_obj:
            if c in scored.columns:
                out[c] = scored[c].values
        missing_obj = [c for c in OBJ_COLS + ["final_score"] if c not in out.columns]
    if missing_obj:
        raise RuntimeError("Objective columns missing after blackbox_fc: " + ", ".join(missing_obj))

    # User-friendly aliases for downstream code/notebooks.
    out["chelation"] = out["chelation_sub"]
    out["solubility"] = out["solubility_sub"]
    out["stability"] = out["stability_sub"]
    out["expression"] = out["expression_sub"]

    out = out.sort_values("final_score", ascending=False).reset_index(drop=True)
    out["rank_final_score"] = np.arange(1, len(out) + 1)
    return out


def write_summary(df_out: pd.DataFrame, args: argparse.Namespace, fixed_ranges) -> None:
    summary = {
        "input_csv": args.input_csv,
        "output_csv": args.output_csv,
        "rows_scored": int(len(df_out)),
        "objective_module": args.objective_module,
        "ranges_json": args.ranges_json,
        "compute_missing_a3d": bool(args.compute_missing_a3d),
        "used_fixed_ranges_argument": fixed_ranges is not None,
        "objective_means": {c: float(pd.to_numeric(df_out[c], errors="coerce").mean()) for c in OBJ_COLS + ["final_score"]},
        "objective_mins": {c: float(pd.to_numeric(df_out[c], errors="coerce").min()) for c in OBJ_COLS + ["final_score"]},
        "objective_maxs": {c: float(pd.to_numeric(df_out[c], errors="coerce").max()) for c in OBJ_COLS + ["final_score"]},
    }
    if len(df_out):
        summary["top_peptide"] = str(df_out.iloc[0][args.peptide_col])
        summary["top_final_score"] = float(df_out.iloc[0]["final_score"])

    out_path = Path(args.summary_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON: {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score Cu corrected chain-mapped peptide data using blackbox_fc, matching the BO objective path."
    )
    p.add_argument("--input-csv", default="metalpdb_CU_chain_mapped_len10_high_confidence.csv")
    p.add_argument("--output-csv", default="metalpdb_CU_chain_mapped_len10_high_confidence_blackbox_scored_ranked.csv")
    p.add_argument("--summary-json", default="metalpdb_CU_chain_mapped_len10_high_confidence_blackbox_scored_ranked_summary.json")
    p.add_argument("--objective-module", default="black_box_fcn_mo_CU_f.py")
    p.add_argument("--ranges-json", default="cu_objective_fixed_ranges_training_CU_updated_margined.json")

    p.add_argument("--peptide-col", default="peptide_len10")
    p.add_argument("--labels-col", default="binding_site_labels_len10")
    p.add_argument("--dna-col", default="DNA_sequence")
    p.add_argument("--a3d-col", default="a3d_mean")
    p.add_argument("--seq-len", type=int, default=10)

    p.add_argument("--filter-high-confidence", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--deduplicate", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dedup-include-labels", action=argparse.BooleanOptionalAction, default=True)

    # Match BO blackbox behavior but let user avoid long ESMFold/A3D runs.
    p.add_argument("--compute-missing-a3d", action=argparse.BooleanOptionalAction, default=True)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    module = load_objective_module(args.objective_module)
    fixed_ranges = maybe_load_ranges(module, args.ranges_json)
    df_in = prepare_input_dataframe(args)
    df_out = evaluate_with_blackbox(df_in, module, args, fixed_ranges=fixed_ranges)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"Saved blackbox-scored ranked CSV: {out_path}")
    print(f"Rows scored: {len(df_out)}")
    if len(df_out):
        print("Top peptide:", df_out.iloc[0][args.peptide_col], "final_score=", float(df_out.iloc[0]["final_score"]))
    write_summary(df_out, args, fixed_ranges)


if __name__ == "__main__":
    main()
