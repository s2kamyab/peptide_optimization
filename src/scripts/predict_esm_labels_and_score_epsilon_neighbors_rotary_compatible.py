from __future__ import annotations

"""
Predict Cu residue-level binding labels for epsilon-neighbor peptides with the
fine-tuned ESM2-35M model, then compute the canonical Cu BO objectives.

Pipeline
--------
epsilon neighbor peptide
    -> ESM2-35M Cu residue classifier
    -> binding_site_labels_len10
    -> canonical fixed-range Cu objective scorer
    -> chelation_sub, solubility_sub, stability_sub, expression_sub, final_score

This script intentionally uses the SAME staged ESM Cu model architecture and
checkpoint format as:
    metal_binding_site_train_esm_cu_staged_ensemble.py

Checkpoint format expected:
    {
        "model_state": ...,
        "config": {...},
        "threshold": ...,
        ...
    }

The canonical Cu objective module is expected to expose:
    blackbox_fc(...)
    load_fixed_objective_ranges(...)
    initialize_fixed_objective_ranges_from_training_df(...)
    _active_ranges(...)
"""

import argparse
import importlib.util
import json
import os
import sys
import traceback
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean_peptide(x: object) -> Optional[str]:
    s = str(x).strip().upper()
    if len(s) != 10 or any(ch not in AA for ch in s):
        return None
    return s


def load_module_from_path(module_path: str, module_name: str):
    path = Path(module_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Module not found: {path}")

    module_dir = str(path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_peptide_col(df: pd.DataFrame, requested: str) -> str:
    if requested in df.columns:
        return requested
    for alt in ["peptide_len10", "sequence", "peptide"]:
        if alt in df.columns:
            return alt
    raise ValueError(
        f"Could not find peptide column. Requested={requested!r}; "
        f"available={list(df.columns)}"
    )


def validate_input(df: pd.DataFrame, peptide_col: str) -> pd.DataFrame:
    out = df.copy()
    out["_original_row"] = np.arange(len(out), dtype=int)
    out["peptide_len10"] = out[peptide_col].map(clean_peptide)

    bad = out["peptide_len10"].isna()
    if bad.any():
        examples = out.loc[bad, peptide_col].astype(str).head(10).tolist()
        raise ValueError(
            f"Found {int(bad.sum())} invalid peptides. "
            f"Expected exactly 10 canonical amino acids. Examples={examples}"
        )

    out = out.drop_duplicates("peptide_len10", keep="first").copy()
    return out


def load_existing_output(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def append_rows_atomic(output_csv: Path, new_rows: pd.DataFrame) -> None:
    if new_rows.empty:
        return

    if output_csv.exists() and output_csv.stat().st_size > 0:
        old = pd.read_csv(output_csv)
        combined = pd.concat([old, new_rows], ignore_index=True, sort=False)
    else:
        combined = new_rows.copy()

    if "peptide_len10" in combined.columns:
        combined = combined.drop_duplicates("peptide_len10", keep="last")
    if "_original_row" in combined.columns:
        combined = combined.sort_values("_original_row").reset_index(drop=True)

    tmp = output_csv.with_suffix(output_csv.suffix + ".tmp")
    combined.to_csv(tmp, index=False)
    os.replace(tmp, output_csv)


def append_error(error_csv: Path, row: Dict[str, object]) -> None:
    pd.DataFrame([row]).to_csv(
        error_csv,
        mode="a",
        header=not error_csv.exists(),
        index=False,
    )


# ============================================================================
# ESM Cu binding-label prediction
# ============================================================================

def build_esm_config(esm_module, ckpt: Dict[str, Any], device: str):
    """
    Reconstruct TrainConfig from the checkpoint, ignoring any future/unknown
    fields so older and newer compatible checkpoints remain loadable.
    """
    cfg_raw = dict(ckpt.get("config", {}))
    if not cfg_raw:
        raise KeyError("ESM checkpoint is missing 'config'.")

    allowed = {f.name for f in fields(esm_module.TrainConfig)}
    cfg_kwargs = {k: v for k, v in cfg_raw.items() if k in allowed}

    # The model only needs these paths for configuration metadata, but the
    # dataclass requires them.
    cfg_kwargs.setdefault("csv_path", "")
    cfg_kwargs.setdefault("output_dir", "")
    cfg_kwargs["device"] = device

    return esm_module.TrainConfig(**cfg_kwargs)


def threshold_from_metrics_json(path: str) -> Optional[float]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ESM metrics JSON not found: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))

    # Match the validated comparison workflow first.
    for key in [
        "ensemble_threshold",
        "threshold",
        "best_threshold",
    ]:
        value = raw.get(key)
        if value is not None:
            return float(value)
    return None


def resolve_esm_threshold(
    ckpt: Dict[str, Any],
    metrics_json: str,
    explicit_threshold: Optional[float],
) -> float:
    if explicit_threshold is not None:
        return float(explicit_threshold)

    metrics_threshold = threshold_from_metrics_json(metrics_json)
    if metrics_threshold is not None:
        return float(metrics_threshold)

    if ckpt.get("threshold") is not None:
        return float(ckpt["threshold"])

    return 0.5


@torch.no_grad()
def predict_esm_binding_labels(
    peptides: List[str],
    *,
    esm_script: str,
    esm_checkpoint: str,
    esm_metrics_json: str,
    esm_threshold: Optional[float],
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    """
    Predict one probability and one binary label per peptide residue.

    Output label convention:
        binding_site_labels_len10 = "0 1 0 0 ..."

    Also saves compact binary form:
        esm_binding_labels_binary = "0100..."
    """
    esm = load_module_from_path(
        esm_script,
        "esm_cu_inference_runtime",
    )

    ckpt_path = Path(esm_checkpoint).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ESM checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    if "model_state" not in ckpt:
        raise KeyError(
            f"{ckpt_path} does not contain 'model_state'. "
            "This script expects a checkpoint produced by "
            "metal_binding_site_train_esm_cu_staged_ensemble.py."
        )

    cfg = build_esm_config(esm, ckpt, str(device))
    threshold = resolve_esm_threshold(
        ckpt,
        esm_metrics_json,
        esm_threshold,
    )

    print("=" * 80)
    print("ESM Cu binding-label predictor")
    print("=" * 80)
    print(f"ESM script:      {Path(esm_script).resolve()}")
    print(f"ESM checkpoint:  {ckpt_path}")
    print(f"Model name:      {cfg.model_name}")
    print(f"Checkpoint seed: {ckpt.get('seed', 'unknown')}")
    print(f"Checkpoint stage:{ckpt.get('stage', 'unknown')}")
    print(f"Checkpoint epoch:{ckpt.get('epoch', 'unknown')}")
    print(f"Threshold used:  {threshold:.6f}")
    print(f"Device:          {device}")

    tokenizer = esm.AutoTokenizer.from_pretrained(cfg.model_name)
    model = esm.ESMCuResidueModel(cfg).to(device)

    # ------------------------------------------------------------------
    # Compatibility handling for Hugging Face ESM rotary-position buffers.
    #
    # Older Transformers versions saved one buffer:
    #   backbone.rotary_embeddings.inv_freq
    #
    # Newer versions expose one non-trainable buffer per encoder layer:
    #   backbone.encoder.layer.N.attention.self.rotary_embeddings.inv_freq
    #
    # These inv_freq tensors are deterministic rotary-position buffers, not
    # learned Cu-classifier parameters. We therefore load non-strictly, but
    # explicitly verify that EVERY mismatch is rotary-buffer-only. Any missing
    # or unexpected learned parameter still raises an error.
    # ------------------------------------------------------------------
    incompatible = model.load_state_dict(ckpt["model_state"], strict=False)

    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)

    def _is_rotary_inv_freq_key(k: str) -> bool:
        return (
            k.endswith("rotary_embeddings.inv_freq")
            or k.endswith("rotary_embedding.inv_freq")
        )

    bad_missing = [k for k in missing_keys if not _is_rotary_inv_freq_key(k)]
    bad_unexpected = [k for k in unexpected_keys if not _is_rotary_inv_freq_key(k)]

    if bad_missing or bad_unexpected:
        raise RuntimeError(
            "ESM checkpoint/model mismatch contains non-rotary parameters. "
            f"Non-rotary missing keys: {bad_missing[:20]}; "
            f"non-rotary unexpected keys: {bad_unexpected[:20]}"
        )

    if missing_keys or unexpected_keys:
        print("ESM checkpoint compatibility note:")
        print(
            f"  Ignoring {len(missing_keys)} missing and "
            f"{len(unexpected_keys)} unexpected rotary inv_freq buffer key(s)."
        )
        print(
            "  These are deterministic, non-trainable rotary-position buffers "
            "whose state-dict layout changed across Transformers versions."
        )

    model.eval()

    rows: List[Dict[str, object]] = []
    batch_size = max(1, int(batch_size))

    for start in range(0, len(peptides), batch_size):
        seqs = peptides[start:start + batch_size]

        encoded = tokenizer(
            seqs,
            padding=True,
            truncation=True,
            max_length=int(cfg.max_length),
            return_tensors="pt",
            return_special_tokens_mask=True,
        )

        special_tokens_mask = encoded.pop("special_tokens_mask")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        probs = torch.sigmoid(logits).detach().cpu()

        input_ids_cpu = encoded["input_ids"].cpu()
        attention_cpu = encoded["attention_mask"].cpu()
        special_cpu = special_tokens_mask.cpu()

        for i, seq in enumerate(seqs):
            active = attention_cpu[i].bool()
            residue_positions = (
                active & ~special_cpu[i].bool()
            ).nonzero(as_tuple=False).flatten()

            if len(residue_positions) != len(seq):
                tokens = tokenizer.convert_ids_to_tokens(
                    input_ids_cpu[i, active].tolist()
                )
                raise ValueError(
                    "ESM token/residue count mismatch: "
                    f"sequence={seq}, residues={len(seq)}, "
                    f"non_special_tokens={len(residue_positions)}, "
                    f"tokens={tokens}"
                )

            # Identity validation, same concept as the ESM training collator.
            residue_tokens = tokenizer.convert_ids_to_tokens(
                input_ids_cpu[i, residue_positions].tolist()
            )
            normalized = [
                str(t).replace("Ġ", "").replace("▁", "").upper()
                for t in residue_tokens
            ]
            mismatches = [
                (j, seq[j], normalized[j])
                for j in range(len(seq))
                if normalized[j] != seq[j]
            ]
            if mismatches:
                raise ValueError(
                    f"ESM residue identity mismatch for {seq}: "
                    f"{mismatches[:5]}"
                )

            residue_probs = [
                float(probs[i, int(pos)])
                for pos in residue_positions.tolist()
            ]
            labels = [1 if p >= threshold else 0 for p in residue_probs]

            rows.append(
                {
                    "peptide_len10": seq,
                    "esm_threshold": float(threshold),
                    "esm_binding_probabilities": json.dumps(residue_probs),
                    "esm_binding_labels_binary": "".join(str(v) for v in labels),
                    "binding_site_labels_len10": " ".join(str(v) for v in labels),
                    "esm_n_predicted_binding_residues": int(sum(labels)),
                    "esm_max_binding_probability": float(max(residue_probs)),
                    "esm_mean_binding_probability": float(np.mean(residue_probs)),
                }
            )

        print(
            f"ESM predictions: {min(start + batch_size, len(peptides))}/{len(peptides)}"
        )

    result = pd.DataFrame(rows)
    if len(result) != len(peptides):
        raise RuntimeError(
            f"ESM prediction count mismatch: input={len(peptides)}, "
            f"output={len(result)}"
        )
    return result


# ============================================================================
# Canonical Cu objective scoring
# ============================================================================

def active_fixed_ranges(
    bb,
    *,
    ranges_json: str,
    training_csv: str,
    save_ranges_json: str,
) -> Dict[str, Tuple[float, float]]:
    if ranges_json:
        ranges = bb.load_fixed_objective_ranges(ranges_json)
        print(f"Loaded fixed objective ranges from: {ranges_json}")
        return ranges

    if training_csv:
        train_df = pd.read_csv(training_csv)
        ranges = bb.initialize_fixed_objective_ranges_from_training_df(
            train_df,
            save_json_path=save_ranges_json or None,
        )
        print(f"Initialized fixed ranges from training CSV: {training_csv}")
        return ranges

    ranges = bb._active_ranges(None)
    print("Using fixed ranges resolved by the canonical Cu module.")
    return ranges


def score_batch(
    bb,
    batch: pd.DataFrame,
    *,
    fixed_ranges: Dict[str, Tuple[float, float]],
    compute_missing_a3d: bool,
) -> pd.DataFrame:
    peptides = batch["peptide_len10"].tolist()
    labels = batch["binding_site_labels_len10"].tolist()

    if "DNA_sequence" in batch.columns:
        dnas = batch["DNA_sequence"].fillna("").astype(str).tolist()
    else:
        dnas = [""] * len(batch)

    if "a3d_mean" in batch.columns:
        a3d = pd.to_numeric(batch["a3d_mean"], errors="coerce").tolist()
    else:
        a3d = [np.nan] * len(batch)

    scored = bb.blackbox_fc(
        peptides=peptides,
        binding_site_labels_len10=labels,
        dna_sequences=dnas,
        a3d_mean=a3d,
        fixed_ranges=fixed_ranges,
        compute_missing_a3d=compute_missing_a3d,
    )

    if len(scored) != len(batch):
        raise RuntimeError(
            f"Canonical scorer returned {len(scored)} rows for "
            f"{len(batch)} inputs."
        )

    scored = scored.copy()

    # Preserve the ESM audit columns in the final scored output.
    preserve_cols = [
        "_original_row",
        "esm_threshold",
        "esm_binding_probabilities",
        "esm_binding_labels_binary",
        "esm_n_predicted_binding_residues",
        "esm_max_binding_probability",
        "esm_mean_binding_probability",
    ]

    for col in reversed(preserve_cols):
        if col in batch.columns and col not in scored.columns:
            scored.insert(
                0,
                col,
                batch[col].to_numpy(),
            )

    return scored


def main(args: argparse.Namespace) -> None:
    input_csv = Path(args.input_csv).resolve()
    labels_csv = Path(args.labels_output_csv).resolve()
    output_csv = Path(args.output_csv).resolve()
    error_csv = Path(args.error_csv).resolve()
    work_dir = Path(args.work_dir).resolve()

    for p in [labels_csv.parent, output_csv.parent, error_csv.parent, work_dir]:
        p.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    peptide_col = resolve_peptide_col(df, args.peptide_col)
    df = validate_input(df, peptide_col)

    # ------------------------------------------------------------------
    # Phase 1: ESM residue-label prediction
    # ------------------------------------------------------------------
    if (
        args.resume
        and labels_csv.exists()
        and labels_csv.stat().st_size > 0
    ):
        labels_df = pd.read_csv(labels_csv)
        if (
            "peptide_len10" in labels_df.columns
            and "binding_site_labels_len10" in labels_df.columns
            and set(df["peptide_len10"]).issubset(
                set(labels_df["peptide_len10"].astype(str))
            )
        ):
            print(f"Reusing existing complete ESM label CSV: {labels_csv}")
        else:
            labels_df = pd.DataFrame()
    else:
        labels_df = pd.DataFrame()

    if labels_df.empty:
        device = torch.device(args.device)
        labels_pred = predict_esm_binding_labels(
            df["peptide_len10"].tolist(),
            esm_script=args.esm_script,
            esm_checkpoint=args.esm_checkpoint,
            esm_metrics_json=args.esm_metrics_json,
            esm_threshold=args.esm_threshold,
            batch_size=args.esm_batch_size,
            device=device,
        )

        labels_df = df[["_original_row", "peptide_len10"]].merge(
            labels_pred,
            on="peptide_len10",
            how="left",
            validate="one_to_one",
        )
        labels_df.to_csv(labels_csv, index=False)
        print(f"Saved ESM binding labels: {labels_csv}")

    # Merge optional original columns (e.g., precomputed DNA/A3D) into labels.
    optional_cols = [
        c for c in ["DNA_sequence", "a3d_mean"]
        if c in df.columns and c not in labels_df.columns
    ]
    if optional_cols:
        labels_df = labels_df.merge(
            df[["peptide_len10"] + optional_cols],
            on="peptide_len10",
            how="left",
            validate="one_to_one",
        )

    # ------------------------------------------------------------------
    # Phase 2: canonical fixed-range Cu objective scoring
    # ------------------------------------------------------------------
    bb = load_module_from_path(
        args.blackbox_module,
        "cu_blackbox_runtime",
    )

    ranges = active_fixed_ranges(
        bb,
        ranges_json=args.ranges_json,
        training_csv=args.training_csv,
        save_ranges_json=args.save_ranges_json,
    )

    ranges_used_path = output_csv.with_name(
        output_csv.stem + "_fixed_ranges_used.json"
    )
    ranges_used_path.write_text(
        json.dumps(
            {k: [float(v[0]), float(v[1])] for k, v in ranges.items()},
            indent=2,
        ),
        encoding="utf-8",
    )

    existing = load_existing_output(output_csv) if args.resume else pd.DataFrame()
    done = set()
    if (
        not existing.empty
        and "peptide_len10" in existing.columns
        and "chelation_sub" in existing.columns
    ):
        done = set(
            existing.loc[
                existing["chelation_sub"].notna(),
                "peptide_len10",
            ].astype(str)
        )

    pending = labels_df[
        ~labels_df["peptide_len10"].isin(done)
    ].copy()

    print("=" * 80)
    print("Canonical Cu objective scoring")
    print("=" * 80)
    print(f"Total peptides:  {len(labels_df)}")
    print(f"Already scored:  {len(done)}")
    print(f"Pending:         {len(pending)}")

    if pending.empty:
        print("Nothing left to score.")
        return

    original_cwd = Path.cwd()
    os.chdir(work_dir)

    try:
        batch_size = max(1, int(args.scoring_batch_size))
        total = len(pending)

        for start in range(0, total, batch_size):
            batch = pending.iloc[
                start:start + batch_size
            ].copy()
            end = min(start + batch_size, total)

            print()
            print(f"Scoring rows {start + 1}-{end} / {total}")

            try:
                scored = score_batch(
                    bb,
                    batch,
                    fixed_ranges=ranges,
                    compute_missing_a3d=args.compute_missing_a3d,
                )
                append_rows_atomic(output_csv, scored)

            except Exception as batch_exc:
                print(f"Batch failed: {batch_exc}")
                print("Retrying each peptide individually.")

                for _, one in batch.iterrows():
                    pep = str(one["peptide_len10"])
                    try:
                        scored_one = score_batch(
                            bb,
                            pd.DataFrame([one]),
                            fixed_ranges=ranges,
                            compute_missing_a3d=args.compute_missing_a3d,
                        )
                        append_rows_atomic(output_csv, scored_one)
                    except Exception as one_exc:
                        print(f"FAILED {pep}: {one_exc}")
                        append_error(
                            error_csv,
                            {
                                "peptide_len10": pep,
                                "error_type": type(one_exc).__name__,
                                "error": str(one_exc),
                                "traceback": traceback.format_exc(limit=4),
                            },
                        )

            current = load_existing_output(output_csv)
            n_done = (
                int(current["chelation_sub"].notna().sum())
                if "chelation_sub" in current.columns
                else len(current)
            )
            print(f"Saved completed objective rows: {n_done}")

    finally:
        os.chdir(original_cwd)

    result = load_existing_output(output_csv)
    required = [
        "chelation_sub",
        "solubility_sub",
        "stability_sub",
        "expression_sub",
    ]
    missing = [c for c in required if c not in result.columns]
    if missing:
        raise RuntimeError(f"Missing final objective columns: {missing}")

    complete = result[required].notna().all(axis=1)

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"ESM labels CSV: {labels_csv}")
    print(f"Scored CSV:     {output_csv}")
    print(f"Rows scored with all four objectives: {int(complete.sum())}/{len(result)}")
    print(f"Fixed ranges audit: {ranges_used_path}")
    if error_csv.exists():
        print(f"Errors CSV: {error_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Predict Cu binding residues using the fine-tuned ESM2-35M model "
            "and compute canonical fixed-range Cu BO objectives."
        )
    )

    p.add_argument(
        "--input-csv",
        default="epsilon_neighbors_needing_blackbox_scoring.csv",
    )
    p.add_argument(
        "--labels-output-csv",
        default="epsilon_neighbors_esm_binding_labels.csv",
    )
    p.add_argument(
        "--output-csv",
        default="epsilon_neighbors_blackbox_scored_with_esm_labels.csv",
    )
    p.add_argument(
        "--error-csv",
        default="epsilon_neighbors_esm_objective_scoring_errors.csv",
    )
    p.add_argument("--peptide-col", default="peptide_len10")

    # ESM predictor: paths match the user's validated Cu comparison workflow.
    p.add_argument(
        "--esm-script",
        default=(
            "metal_binding_prediction/"
            "metal_binding_site_train_esm_cu_staged_ensemble.py"
        ),
    )
    p.add_argument(
        "--esm-checkpoint",
        default=(
            "metal_binding_prediction/"
            "outputs_esm2_t12_cu_corrected_seed42/"
            "seed_42/best_model.pt"
        ),
    )
    p.add_argument(
        "--esm-metrics-json",
        default=(
            "metal_binding_prediction/"
            "outputs_esm2_t12_cu_corrected_seed42/"
            "esm_cu_ensemble_metrics.json"
        ),
        help=(
            "If present, ensemble_threshold is used first, matching the "
            "validated comparison workflow. Otherwise checkpoint threshold is used."
        ),
    )
    p.add_argument(
        "--esm-threshold",
        type=float,
        default=None,
        help="Optional explicit threshold override.",
    )
    p.add_argument("--esm-batch-size", type=int, default=32)

    # Canonical fixed-range Cu objective evaluator.
    p.add_argument(
        "--blackbox-module",
        default="black_box_fcn_mo_CU_f.py",
    )
    p.add_argument(
        "--ranges-json",
        default="",
        help="Preferred exact fixed-range JSON used by training/BO.",
    )
    p.add_argument(
        "--training-csv",
        default="",
        help="Alternative source for initializing fixed ranges once.",
    )
    p.add_argument("--save-ranges-json", default="")

    p.add_argument(
        "--work-dir",
        default="epsilon_neighbor_blackbox_cache",
        help="Persistent ESMFold/A3D cache working directory.",
    )
    p.add_argument("--scoring-batch-size", type=int, default=8)

    p.add_argument(
        "--compute-missing-a3d",
        dest="compute_missing_a3d",
        action="store_true",
        default=True,
    )
    p.add_argument(
        "--no-compute-missing-a3d",
        dest="compute_missing_a3d",
        action="store_false",
    )

    p.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
    )
    p.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
    )

    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
