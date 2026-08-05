#!/usr/bin/env python
from __future__ import annotations

"""Create fixed-length MetalPDB peptide windows with chain-mapped audited labels in chunked parts.

This script is the fixed-length-10 counterpart of
create_training_data_metalpdb_chain_mapped_corrected.py.

Corrections relative to the older extract_binding_labels_sequence_metaldb_peptides.py:
- MetalPDB residue numbers are mapped through RCSB mmCIF scheme tables.
- PDB residue numbers are never used directly as UniProt indices.
- Auth/label chains, insertion codes, and residue identity are retained and audited.
- Optional UniProt alignment maps accepted PDB-chain positions to UniProt positions.
- All accepted target-metal binding positions are aggregated before windows are cut.
- A fixed length-10 window is cut around each accepted binding position.
- Labels mark every accepted target-metal binding position that falls inside the window,
  not only the center residue.
- Ambiguous mappings are excluded from the training CSV and written to an audit CSV.
- Protein-aware splits can be created by UniProt, PDB-chain, or PDB.

Typical Cu command:
    python create_fixed_len10_metalpdb_chain_mapped_corrected_chunked.py ^
      --raw-csv metalpdb_Cu.csv ^
      --target-metal Cu ^
      --output-csv metalpdb_CU_chain_mapped_len10_high_confidence.csv ^
      --sequence-source uniprot ^
      --window-len 10 ^
      --negative-fraction 0.0

All-metal command using all records:
    python create_fixed_len10_metalpdb_chain_mapped_corrected_chunked.py ^
      --raw-csv metalpdb_all_metals_unique/metalpdb_all_metals_all_records.csv ^
      --target-metal ALL ^
      --output-dir metalpdb_all_metals_chain_mapped_len10_high_confidence_parts ^
      --sequence-source uniprot ^
      --window-len 10
"""

import argparse
import ast
import csv
import gzip
import hashlib
import json
import math
import random
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
import requests
from Bio.Align import PairwiseAligner
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Kept for mapping audit; exported fixed-length peptides are restricted to canonical 20 AA by default.
    "SEC": "U", "PYL": "O", "MSE": "M",
}
CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")
CHAIN_KEYS = (
    "chain", "protein_chain", "auth_asym_id", "label_asym_id",
    "protein_auth_asym_id", "protein_label_asym_id", "asym_id",
)
INS_KEYS = (
    "insertion_code", "pdb_insertion_code",
    "residue_pdb_insertion_code", "pdbx_PDB_ins_code",
)


@dataclass(frozen=True)
class Annotation:
    source_row: int
    pdb: str
    uniprot: str
    site: str
    metal: str
    residue_name_3: str
    residue_aa: str
    pdb_number: str
    insertion_code: str
    annotated_chain: str


@dataclass
class Audit:
    source_row: int
    pdb: str
    uniprot: str
    site: str
    metal: str
    residue_name_3: str
    residue_aa: str
    pdb_number: str
    insertion_code: str
    annotated_chain: str
    label_chain: str = ""
    auth_chain: str = ""
    chain_position_1based: Optional[int] = None
    uniprot_position_1based: Optional[int] = None
    chain_residue_aa: str = ""
    uniprot_residue_aa: str = ""
    alignment_identity: Optional[float] = None
    status: str = ""
    note: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create audited fixed-length MetalPDB peptide windows with least ambiguous chain-mapped labels."
    )
    p.add_argument("--raw-csv", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, default=None, help="Optional final combined CSV. Omit it for parts-only output.")
    p.add_argument("--output-dir", type=Path, default=None, help="Folder for chunked part CSVs, manifest, audit, and summary files.")
    p.add_argument("--output-prefix", default=None, help="Prefix for chunked part files. Default is derived from --output-csv or target metal.")
    p.add_argument("--save-every-rows", type=int, default=50000, help="Flush output peptide rows to a new part CSV after this many rows.")
    p.add_argument(
    "--raw-batch-rows",
    type=int,
    default=5000,
    help="Read and process this many raw MetalPDB rows per mapping batch.",
    )
    p.add_argument("--make-final-file", action="store_true", help="Also build a combined final CSV from the parts. For all-metal runs, usually leave this off to save memory/disk I/O.")
    p.add_argument("--target-metal", required=True, help="Metal symbol such as Cu, Zn, Mg, or ALL.")
    p.add_argument("--cache-dir", type=Path, default=Path("metalpdb_mapping_cache"))
    p.add_argument("--mapping-audit-csv", type=Path)
    p.add_argument("--sequence-source", choices=["uniprot", "pdb-chain"], default="uniprot")
    p.add_argument("--split-group", choices=["uniprot", "pdb_chain", "pdb"], default="uniprot")
    p.add_argument("--minimum-alignment-identity", type=float, default=0.90)
    p.add_argument("--window-len", type=int, default=10)
    p.add_argument("--pad-termini", action="store_true", help="Pad termini with X; default is to skip near-terminus windows.")
    p.add_argument("--negative-fraction", type=float, default=0.0, help="Optional negative fraction relative to total output. Default 0.")
    p.add_argument("--train-fraction", type=float, default=0.70)
    p.add_argument("--validation-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--dedup-level", choices=["none", "peptide_metal", "peptide_metal_labels", "full"], default="peptide_metal_labels")
    p.add_argument("--allow-noncanonical", action="store_true", help="Allow U/O/X in output peptides. Default keeps only canonical 20 AA.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--request-timeout", type=int, default=30)
    p.add_argument("--sleep", type=float, default=0.05)
    return p.parse_args()


def clean(value: Any) -> str:
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "?", ".", "null"} else text


def literal(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    try:
        return ast.literal_eval(str(value))
    except Exception:
        try:
            return json.loads(str(value))
        except Exception:
            return None


def first(mapping: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = clean(mapping.get(key, ""))
        if value:
            return value
    return ""


def normalize_metal(value: Any) -> str:
    text = clean(value)
    if not text:
        return ""
    match = re.search(r"[A-Za-z]{1,2}", text)
    return match.group(0).title() if match else ""


def number_part(value: Any) -> str:
    text = clean(value)
    match = re.fullmatch(r"(-?\d+)([A-Za-z]?)", text)
    if match:
        return match.group(1)
    try:
        return str(int(float(text)))
    except Exception:
        return text


def insertion_part(value: Any) -> str:
    match = re.fullmatch(r"-?\d+([A-Za-z])", clean(value))
    return match.group(1).upper() if match else ""


def row_value(row: pd.Series, names: Iterable[str]) -> str:
    lower = {str(c).lower(): c for c in row.index}
    for name in names:
        if name in row.index:
            return clean(row.get(name, ""))
        actual = lower.get(name.lower())
        if actual is not None:
            return clean(row.get(actual, ""))
    return ""


def ligands_from_row(row: pd.Series) -> list[dict[str, Any]]:
    """Support both old MetalPDB 'metals' rows and newer flat 'metal'/'ligands' rows."""
    ligands: list[dict[str, Any]] = []
    row_dict = dict(row)

    metals_obj = literal(row.get("metals", "")) if "metals" in row.index else None
    if isinstance(metals_obj, dict):
        metals_obj = [metals_obj]
    if isinstance(metals_obj, list):
        for metal_obj in metals_obj:
            if not isinstance(metal_obj, dict):
                continue
            metal_symbol = normalize_metal(
                metal_obj.get("symbol") or metal_obj.get("metal") or metal_obj.get("element")
            )
            metal_ligands = metal_obj.get("ligands", [])
            if isinstance(metal_ligands, dict):
                metal_ligands = [metal_ligands]
            if isinstance(metal_ligands, list):
                for lig in metal_ligands:
                    if isinstance(lig, dict):
                        item = dict(lig)
                        item.setdefault("_metal_symbol", metal_symbol)
                        ligands.append(item)

    lig_obj = literal(row.get("ligands", "")) if "ligands" in row.index else None
    if isinstance(lig_obj, dict):
        lig_obj = [lig_obj]
    if isinstance(lig_obj, list):
        fallback_metal = normalize_metal(row.get("query_metal") or row.get("metal"))
        for lig in lig_obj:
            if isinstance(lig, dict):
                item = dict(lig)
                item.setdefault("_metal_symbol", fallback_metal)
                ligands.append(item)

    # pd.json_normalize can create ligands.0.chain, ligands.0.residue_pdb_number, ...
    grouped: dict[int, dict[str, Any]] = defaultdict(dict)
    pattern = re.compile(r"^ligands\.(\d+)\.(.+)$")
    for col, value in row_dict.items():
        match = pattern.match(str(col))
        if match:
            grouped[int(match.group(1))][match.group(2)] = value
    if grouped:
        fallback_metal = normalize_metal(row.get("query_metal") or row.get("metal"))
        for idx in sorted(grouped):
            item = grouped[idx]
            item.setdefault("_metal_symbol", fallback_metal)
            ligands.append(item)

    return ligands


def parse_annotations(df: pd.DataFrame, target: str) -> list[Annotation]:
    target_upper = target.upper()
    output: list[Annotation] = []
    for index, row in df.iterrows():
        row_dict = dict(row)
        pdb = row_value(row, ["pdb", "pdb_code", "pdb_id", "PDB"])
        uniprot = row_value(row, ["uniprot", "uniprot_id", "uniprot.accession"])
        site = row_value(row, ["site", "site_id", "id"])
        fallback_chain = first(row_dict, CHAIN_KEYS)
        fallback_metal = normalize_metal(row.get("query_metal") or row.get("metal"))

        for ligand in ligands_from_row(row):
            metal = normalize_metal(ligand.get("_metal_symbol") or ligand.get("metal") or ligand.get("element") or fallback_metal)
            if target_upper != "ALL" and metal.upper() != target_upper:
                continue
            residue3 = clean(
                ligand.get("residue") or ligand.get("resname") or ligand.get("res_name")
                or ligand.get("comp_id") or ligand.get("residue_name") or ligand.get("ligand")
            ).upper()
            residue1 = AA3_TO_AA1.get(residue3)
            raw_number = (
                ligand.get("residue_pdb_number") or ligand.get("auth_seq_id")
                or ligand.get("residue_number") or ligand.get("resseq")
                or ligand.get("resnum") or ligand.get("residue_num")
            )
            pdb_number = number_part(raw_number)
            if not residue1 or not pdb_number or not metal:
                continue
            insertion = first(ligand, INS_KEYS) or insertion_part(raw_number)
            chain = first(ligand, CHAIN_KEYS) or fallback_chain
            output.append(
                Annotation(
                    source_row=int(index), pdb=clean(pdb).upper(), uniprot=clean(uniprot), site=clean(site),
                    metal=metal.upper(), residue_name_3=residue3, residue_aa=residue1,
                    pdb_number=pdb_number, insertion_code=insertion.upper(), annotated_chain=chain,
                )
            )
    return [item for item in output if item.pdb]


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    return [str(v) for v in value] if isinstance(value, list) else [str(value)]


def column(cif: dict[str, Any], key: str, n: int, default: str = "?") -> list[str]:
    values = as_list(cif.get(key))
    if not values:
        return [default] * n
    if len(values) == 1 and n > 1:
        return values * n
    if len(values) != n:
        raise ValueError(f"{key}: got {len(values)}, expected {n}")
    return values


def download_cif(session: requests.Session, pdb: str, args: argparse.Namespace) -> Path:
    folder = args.cache_dir / "rcsb_mmcif"
    folder.mkdir(parents=True, exist_ok=True)
    cif_path = folder / f"{pdb.lower()}.cif"
    if cif_path.exists() and cif_path.stat().st_size:
        return cif_path
    gz_path = folder / f"{pdb.lower()}.cif.gz"
    if not gz_path.exists() or not gz_path.stat().st_size:
        response = session.get(f"https://files.rcsb.org/download/{pdb}.cif.gz", timeout=args.request_timeout)
        response.raise_for_status()
        gz_path.write_bytes(response.content)
        time.sleep(args.sleep)
    with gzip.open(gz_path, "rb") as source, cif_path.open("wb") as target:
        target.write(source.read())
    return cif_path


def load_cif_map(path: Path) -> dict[str, Any]:
    cif = MMCIF2Dict(str(path))
    entity_ids = as_list(cif.get("_entity_poly.entity_id"))
    entity_seq = column(cif, "_entity_poly.pdbx_seq_one_letter_code_can", len(entity_ids), "")
    entity_type = column(cif, "_entity_poly.type", len(entity_ids), "")
    sequences = {
        eid: re.sub(r"\s+", "", seq).upper()
        for eid, seq, typ in zip(entity_ids, entity_seq, entity_type)
        if "polypeptide" in typ.lower()
    }

    atom_label = as_list(cif.get("_atom_site.label_asym_id"))
    n_atom = len(atom_label)
    atom_auth = column(cif, "_atom_site.auth_asym_id", n_atom, "")
    atom_entity = column(cif, "_atom_site.label_entity_id", n_atom, "")
    label_to_auth: dict[str, str] = {}
    label_to_seq: dict[str, str] = {}
    auth_to_label: dict[str, set[str]] = defaultdict(set)
    for label, auth, entity in zip(atom_label, atom_auth, atom_entity):
        seq = sequences.get(entity)
        if seq:
            label_to_auth.setdefault(label, auth)
            label_to_seq.setdefault(label, seq)
            auth_to_label[auth].add(label)

    labels = as_list(cif.get("_pdbx_poly_seq_scheme.asym_id"))
    n = len(labels)
    auths = column(cif, "_pdbx_poly_seq_scheme.pdb_strand_id", n, "")
    seq_ids = column(cif, "_pdbx_poly_seq_scheme.seq_id", n, "")
    auth_nums = column(cif, "_pdbx_poly_seq_scheme.auth_seq_num", n, "")
    pdb_nums = column(cif, "_pdbx_poly_seq_scheme.pdb_seq_num", n, "")
    insertions = column(cif, "_pdbx_poly_seq_scheme.pdb_ins_code", n, "")
    monomers = column(cif, "_pdbx_poly_seq_scheme.mon_id", n, "")
    residue_map: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for label, auth, seq_id, auth_num, pdb_num, ins, mon in zip(labels, auths, seq_ids, auth_nums, pdb_nums, insertions, monomers):
        try:
            chain_position = int(seq_id)
        except Exception:
            continue
        insertion = "" if ins in {"?", "."} else ins.upper()
        record = {
            "label_chain": label,
            "auth_chain": auth,
            "chain_position": chain_position,
            "residue_aa": AA3_TO_AA1.get(mon.upper(), "X"),
        }
        for chain_name in {label, auth}:
            for number in {number_part(auth_num), number_part(pdb_num)}:
                if chain_name and number:
                    residue_map[(chain_name, number, insertion)].append(record)
    return {"label_to_auth": label_to_auth, "label_to_seq": label_to_seq, "auth_to_label": auth_to_label, "residue_map": residue_map}


def chain_names(annotation: Annotation, cif_map: dict[str, Any]) -> list[str]:
    if annotation.annotated_chain:
        names = [annotation.annotated_chain]
        names += list(cif_map["auth_to_label"].get(annotation.annotated_chain, set()))
        auth = cif_map["label_to_auth"].get(annotation.annotated_chain)
        if auth:
            names.append(auth)
        return list(dict.fromkeys(names))
    names = list(cif_map["label_to_seq"]) + list(cif_map["auth_to_label"])
    return list(dict.fromkeys(names))


def map_chain_residue(annotation: Annotation, cif_map: dict[str, Any]) -> tuple[Optional[dict[str, Any]], str]:
    candidates: list[dict[str, Any]] = []
    for chain in chain_names(annotation, cif_map):
        candidates += cif_map["residue_map"].get((chain, annotation.pdb_number, annotation.insertion_code), [])
    unique = {(c["label_chain"], c["chain_position"]): c for c in candidates}
    candidates = list(unique.values())
    identity = [c for c in candidates if c["residue_aa"] == annotation.residue_aa]
    if len(identity) == 1:
        return identity[0], ""
    if len(identity) > 1:
        return None, f"{len(identity)} identity-matched candidates"
    if len(candidates) == 1:
        return None, f"identity mismatch: expected {annotation.residue_aa}, found {candidates[0]['residue_aa']}"
    return None, "no unique matching chain residue"


def fetch_uniprot(session: requests.Session, accession: str, cache: dict[str, Optional[str]], args: argparse.Namespace) -> Optional[str]:
    if not accession:
        return None
    if accession in cache:
        return cache[accession]
    try:
        response = session.get(f"https://rest.uniprot.org/uniprotkb/{accession}.fasta", timeout=args.request_timeout)
        if response.status_code != 200:
            cache[accession] = None
        else:
            sequence = "".join(line.strip() for line in response.text.splitlines() if line.strip() and not line.startswith(">"))
            cache[accession] = sequence.upper() or None
        time.sleep(args.sleep)
    except requests.RequestException:
        cache[accession] = None
    return cache[accession]


def align_chain_to_uniprot(chain: str, uniprot: str) -> tuple[dict[int, int], float]:
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score, aligner.mismatch_score = 2.0, -1.0
    aligner.open_gap_score, aligner.extend_gap_score = -5.0, -0.5
    alignment = aligner.align(uniprot, chain)[0]
    mapping: dict[int, int] = {}
    matches = aligned = 0
    for (u0, u1), (c0, c1) in zip(alignment.aligned[0], alignment.aligned[1]):
        length = min(int(u1 - u0), int(c1 - c0))
        for offset in range(length):
            ui, ci = int(u0) + offset, int(c0) + offset
            mapping[ci + 1] = ui + 1
            aligned += 1
            matches += int(uniprot[ui] == chain[ci])
    return mapping, matches / aligned if aligned else 0.0


def map_annotations(annotations: list[Annotation], args: argparse.Namespace):
    session = requests.Session()
    session.headers["User-Agent"] = "MetalPDB-fixed-len10-chain-mapped-builder/1.0"
    cif_cache: dict[str, dict[str, Any]] = {}
    uniprot_cache: dict[str, Optional[str]] = {}
    alignment_cache: dict[tuple[str, str, str], tuple[dict[int, int], float]] = {}
    sequence_cache: dict[tuple[str, str], str] = {}
    audits: list[Audit] = []

    for ann in annotations:
        audit = Audit(**asdict(ann))
        try:
            if ann.pdb not in cif_cache:
                cif_cache[ann.pdb] = load_cif_map(download_cif(session, ann.pdb, args))
            cif_map = cif_cache[ann.pdb]
            candidate, note = map_chain_residue(ann, cif_map)
            if candidate is None:
                audit.status, audit.note = "unresolved_pdb_mapping", note
                audits.append(audit)
                continue
            label = candidate["label_chain"]
            chain_seq = cif_map["label_to_seq"].get(label, "")
            pos = int(candidate["chain_position"])
            audit.label_chain = label
            audit.auth_chain = candidate["auth_chain"]
            audit.chain_position_1based = pos
            audit.chain_residue_aa = chain_seq[pos - 1] if 1 <= pos <= len(chain_seq) else ""
            if audit.chain_residue_aa != ann.residue_aa:
                audit.status = "chain_identity_mismatch"
                audits.append(audit)
                continue
            sequence_cache[(ann.pdb, label)] = chain_seq

            if args.sequence_source == "pdb-chain":
                audit.status = "accepted_pdb_chain"
                audits.append(audit)
                continue

            uni_seq = fetch_uniprot(session, ann.uniprot, uniprot_cache, args)
            if not uni_seq:
                audit.status = "uniprot_unavailable"
                audits.append(audit)
                continue
            key = (ann.pdb, label, ann.uniprot)
            if key not in alignment_cache:
                alignment_cache[key] = align_chain_to_uniprot(chain_seq, uni_seq)
            position_map, identity = alignment_cache[key]
            audit.alignment_identity = identity
            if identity < args.minimum_alignment_identity:
                audit.status = "low_alignment_identity"
                audits.append(audit)
                continue
            uni_pos = position_map.get(pos)
            if uni_pos is None:
                audit.status = "unaligned_chain_position"
                audits.append(audit)
                continue
            audit.uniprot_position_1based = uni_pos
            audit.uniprot_residue_aa = uni_seq[uni_pos - 1]
            if audit.uniprot_residue_aa != ann.residue_aa:
                audit.status = "uniprot_identity_mismatch"
                audits.append(audit)
                continue
            audit.status = "accepted_uniprot"
            sequence_cache[(ann.uniprot, "UNIPROT")] = uni_seq
        except Exception as exc:
            audit.status = "mapping_failed"
            audit.note = f"{type(exc).__name__}: {exc}"
        audits.append(audit)
    return audits, sequence_cache


def centered_window(sequence: str, center_pos_1based: int, window_len: int, pad_termini: bool) -> Optional[tuple[str, int, int, int]]:
    left = window_len // 2 - 1 if window_len % 2 == 0 else window_len // 2
    right = window_len - 1 - left
    start = center_pos_1based - left
    end = center_pos_1based + right
    if not pad_termini and (start < 1 or end > len(sequence)):
        return None
    chars: list[str] = []
    for pos in range(start, end + 1):
        chars.append(sequence[pos - 1] if 1 <= pos <= len(sequence) else "X")
    center_idx0 = center_pos_1based - start
    return "".join(chars), start, end, center_idx0


def build_labels_for_window(binding_positions: set[int], start: int, end: int) -> tuple[str, str, list[int]]:
    labels = ["0"] * (end - start + 1)
    inside: list[int] = []
    for pos in sorted(binding_positions):
        if start <= pos <= end:
            labels[pos - start] = "1"
            inside.append(pos)
    binary = "".join(labels)
    spaced = " ".join(labels)
    return binary, spaced, inside


def valid_output_peptide(peptide: str, allow_noncanonical: bool) -> bool:
    if allow_noncanonical:
        return bool(peptide) and all(ch.isalpha() for ch in peptide)
    return all(ch in CANONICAL_AA for ch in peptide)


def group_id_for(meta: dict[str, str], args: argparse.Namespace) -> str:
    if args.split_group == "uniprot":
        return meta.get("uniprot") or f"{meta.get('pdb')}|{meta.get('label_chain')}"
    if args.split_group == "pdb_chain":
        return f"{meta.get('pdb')}|{meta.get('label_chain')}"
    return meta.get("pdb", "")


def assign_splits(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if df.empty:
        return df
    if not math.isclose(args.train_fraction + args.validation_fraction + args.test_fraction, 1.0, abs_tol=1e-9):
        raise ValueError("Split fractions must sum to 1")
    summary = df.groupby("group_id").size().reset_index(name="n")
    records = summary.to_dict("records")
    rng = random.Random(args.seed)
    rng.shuffle(records)
    records.sort(key=lambda r: r["n"], reverse=True)
    targets = {
        "train": round(len(df) * args.train_fraction),
        "validation": round(len(df) * args.validation_fraction),
    }
    targets["test"] = len(df) - targets["train"] - targets["validation"]
    counts = {k: 0 for k in targets}
    mapping = {}
    for record in records:
        split = max(targets, key=lambda s: (targets[s] - counts[s], -counts[s]))
        mapping[record["group_id"]] = split
        counts[split] += int(record["n"])
    out = df.copy()
    out["split"] = out["group_id"].map(mapping)
    return out


def dedup_key(row: dict[str, Any], level: str) -> Optional[tuple[Any, ...]]:
    if level == "none":
        return None
    if level == "peptide_metal":
        return row["metal"], row["peptide_len10"]
    if level == "peptide_metal_labels":
        return row["metal"], row["peptide_len10"], row["labels"]
    return (
        row["metal"], row["sequence_source_id"], row["sequence_source_chain"],
        row["center_binding_position_1based"], row["peptide_len10"], row["labels"],
    )


def add_negative_windows(rows: list[dict[str, Any]], positions_by_key: dict[tuple[str, str, str], set[int]], metadata: dict[tuple[str, str, str], dict[str, str]], sequence_cache: dict[tuple[str, str], str], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.negative_fraction <= 0 or not rows:
        return []
    rng = random.Random(args.seed)
    target_neg = round(args.negative_fraction / max(1.0 - args.negative_fraction, 1e-12) * len(rows))
    negatives: list[dict[str, Any]] = []
    seen = {dedup_key(r, args.dedup_level) for r in rows if dedup_key(r, args.dedup_level) is not None}
    keys = list(positions_by_key.keys())
    attempts = 0
    while len(negatives) < target_neg and attempts < target_neg * 100:
        attempts += 1
        key = rng.choice(keys)
        meta = metadata[key]
        sequence = sequence_cache.get((meta["seq0"], meta["seq1"]), "")
        if len(sequence) < args.window_len:
            continue
        start = rng.randint(1, len(sequence) - args.window_len + 1)
        end = start + args.window_len - 1
        if any(start <= p <= end for p in positions_by_key[key]):
            continue
        peptide = sequence[start - 1:end]
        if not valid_output_peptide(peptide, args.allow_noncanonical):
            continue
        row = {
            "metal": meta["metal"], "group_id": group_id_for(meta, args), "site": "",
            "pdb": meta["pdb"], "uniprot": meta["uniprot"],
            "label_chain": meta["label_chain"], "auth_chain": meta["auth_chain"],
            "sequence_source": args.sequence_source, "sequence_source_id": meta["seq0"],
            "sequence_source_chain": meta["seq1"], "protein_length": len(sequence),
            "known_binding_positions": ";".join(map(str, sorted(positions_by_key[key]))),
            "center_binding_position_1based": "", "center_pos_in_window": "",
            "window_start_1based": start, "window_end_1based": end,
            "peptide_len10": peptide, "sequence": peptide,
            "labels": "0" * args.window_len,
            "binding_site_labels_len10": " ".join(["0"] * args.window_len),
            "binding_positions_in_window": "", "is_negative": 1,
            "label_confidence": "accepted_chain_mapped_negative",
            "mapping_status": "accepted_no_known_binding_position_in_window",
        }
        key2 = dedup_key(row, args.dedup_level)
        if key2 is not None and key2 in seen:
            continue
        if key2 is not None:
            seen.add(key2)
        negatives.append(row)
    return negatives




def output_base_paths(args: argparse.Namespace) -> tuple[Path, str]:
    """Resolve output directory and filename prefix for chunked output."""
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    elif args.output_csv is not None:
        out_dir = Path(args.output_csv).with_suffix("").parent / (Path(args.output_csv).stem + "_parts")
    else:
        out_dir = Path(f"metalpdb_{args.target_metal.upper()}_chain_mapped_len{args.window_len}_parts")

    if args.output_prefix:
        prefix = args.output_prefix
    elif args.output_csv is not None:
        prefix = Path(args.output_csv).stem
    else:
        prefix = f"metalpdb_{args.target_metal.upper()}_chain_mapped_len{args.window_len}_high_confidence"

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, prefix


def deterministic_split_for_group(group_id: str, args: argparse.Namespace) -> str:
    """Assign split by a stable group-level hash so parts can be written without holding all rows."""
    if not math.isclose(args.train_fraction + args.validation_fraction + args.test_fraction, 1.0, abs_tol=1e-9):
        raise ValueError("Split fractions must sum to 1")
    key = f"{args.seed}|{group_id}".encode("utf-8")
    value = int(hashlib.md5(key).hexdigest()[:12], 16) / float(16 ** 12)
    if value < args.train_fraction:
        return "train"
    if value < args.train_fraction + args.validation_fraction:
        return "validation"
    return "test"


def write_rows_part(rows: list[dict[str, Any]], parts_dir: Path, prefix: str, part_idx: int) -> tuple[Path, int]:
    """Write one peptide-window part CSV."""
    parts_dir.mkdir(parents=True, exist_ok=True)
    path = parts_dir / f"{prefix}_part_{part_idx:06d}.csv"
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    else:
        pd.DataFrame().to_csv(path, index=False)
    return path, len(rows)


def write_audit_parts(audits: list[Audit], out_dir: Path, prefix: str, rows_per_part: int = 100000) -> list[dict[str, Any]]:
    """Write mapping audit in parts to avoid one very large audit CSV."""
    audit_dir = out_dir / "mapping_audit_parts"
    audit_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for start in range(0, len(audits), max(1, rows_per_part)):
        part_no = len(manifest) + 1
        chunk = audits[start:start + rows_per_part]
        path = audit_dir / f"{prefix}_mapping_audit_part_{part_no:06d}.csv"
        pd.DataFrame([asdict(a) for a in chunk]).to_csv(path, index=False)
        manifest.append({"part": part_no, "audit_csv": str(path), "n_rows": len(chunk)})
    return manifest


def combine_part_csvs(part_files: list[Path], out_csv: Path) -> None:
    """Optional streaming combine of part CSVs. Avoid for memory-sensitive all-metal runs."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    first = True
    for part in part_files:
        for chunk in pd.read_csv(part, chunksize=100000):
            chunk.to_csv(out_csv, mode="w" if first else "a", header=first, index=False)
            first = False
    if first:
        pd.DataFrame().to_csv(out_csv, index=False)

def build_fixed_len10_dataset(args: argparse.Namespace) -> pd.DataFrame:
    """True batch-chunked version.

    Unlike the previous version, this does not wait for all MetalPDB annotations
    to be mapped before saving outputs. It processes raw CSV batches and writes
    audit/window parts during the run.
    """
    out_dir, prefix = output_base_paths(args)
    parts_dir = out_dir / "parts"
    audit_dir = out_dir / "mapping_audit_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    if args.output_csv is None:
        args.output_csv = out_dir / f"{prefix}.csv"

    # Shared caches across batches. These keep repeated PDB/UniProt work efficient.
    shared_session = requests.Session()
    shared_session.headers["User-Agent"] = "MetalPDB-fixed-len10-chain-mapped-builder/true-chunked/1.0"
    shared_cif_cache: dict[str, dict[str, Any]] = {}
    shared_uniprot_cache: dict[str, Optional[str]] = {}
    shared_alignment_cache: dict[tuple[str, str, str], tuple[dict[int, int], float]] = {}

    # Global dedup is still kept because it is much smaller than the full row table.
    seen: set[tuple[Any, ...]] = set()

    part_files: list[Path] = []
    manifest_rows: list[dict[str, Any]] = []
    audit_manifest_rows: list[dict[str, Any]] = []

    rows_buffer: list[dict[str, Any]] = []

    peptide_part_idx = 1
    audit_part_idx = 1
    raw_batch_idx = 0

    total_raw_rows = 0
    total_annotations = 0
    total_accepted = 0
    total_positive = 0

    skipped_near_terminus = 0
    skipped_noncanonical = 0

    split_counts: dict[str, int] = defaultdict(int)
    metal_counts: dict[str, int] = defaultdict(int)
    status_counts_total: dict[str, int] = defaultdict(int)
    unique_peptides: set[str] = set()

    accepted_status = "accepted_uniprot" if args.sequence_source == "uniprot" else "accepted_pdb_chain"

    def flush_peptide_part(force: bool = False) -> None:
        nonlocal rows_buffer, peptide_part_idx, total_positive
        if not rows_buffer and not force:
            return
        if not rows_buffer:
            return

        path, n_rows = write_rows_part(rows_buffer, parts_dir, prefix, peptide_part_idx)
        part_files.append(path)
        manifest_rows.append({
            "part": peptide_part_idx,
            "window_csv": str(path),
            "n_rows": n_rows,
            "cumulative_rows": total_positive,
        })
        print(
            f"[WINDOW-PART] saved part {peptide_part_idx:06d}: "
            f"rows={n_rows}, cumulative_positive_rows={total_positive}",
            flush=True,
        )
        rows_buffer = []
        peptide_part_idx += 1

    def write_one_audit_part(audits: list[Audit], batch_no: int) -> None:
        nonlocal audit_part_idx
        if not audits:
            return

        path = audit_dir / f"{prefix}_mapping_audit_part_{audit_part_idx:06d}.csv"
        pd.DataFrame([asdict(a) for a in audits]).to_csv(path, index=False)
        audit_manifest_rows.append({
            "part": audit_part_idx,
            "raw_batch": batch_no,
            "audit_csv": str(path),
            "n_rows": len(audits),
        })
        print(
            f"[AUDIT-PART] saved part {audit_part_idx:06d}: "
            f"audit_rows={len(audits)}, raw_batch={batch_no}",
            flush=True,
        )
        audit_part_idx += 1

    def map_annotations_streaming(
        annotations: list[Annotation],
    ) -> tuple[list[Audit], dict[tuple[str, str], str]]:
        """Map only one batch of annotations but reuse global PDB/UniProt/alignment caches."""
        sequence_cache: dict[tuple[str, str], str] = {}
        audits: list[Audit] = []

        for ann_i, ann in enumerate(annotations, start=1):
            audit = Audit(**asdict(ann))
            try:
                if ann.pdb not in shared_cif_cache:
                    shared_cif_cache[ann.pdb] = load_cif_map(
                        download_cif(shared_session, ann.pdb, args)
                    )

                cif_map = shared_cif_cache[ann.pdb]
                candidate, note = map_chain_residue(ann, cif_map)

                if candidate is None:
                    audit.status, audit.note = "unresolved_pdb_mapping", note
                    audits.append(audit)
                    continue

                label = candidate["label_chain"]
                chain_seq = cif_map["label_to_seq"].get(label, "")
                pos = int(candidate["chain_position"])

                audit.label_chain = label
                audit.auth_chain = candidate["auth_chain"]
                audit.chain_position_1based = pos
                audit.chain_residue_aa = chain_seq[pos - 1] if 1 <= pos <= len(chain_seq) else ""

                if audit.chain_residue_aa != ann.residue_aa:
                    audit.status = "chain_identity_mismatch"
                    audits.append(audit)
                    continue

                sequence_cache[(ann.pdb, label)] = chain_seq

                if args.sequence_source == "pdb-chain":
                    audit.status = "accepted_pdb_chain"
                    audits.append(audit)
                    continue

                uni_seq = fetch_uniprot(shared_session, ann.uniprot, shared_uniprot_cache, args)
                if not uni_seq:
                    audit.status = "uniprot_unavailable"
                    audits.append(audit)
                    continue

                align_key = (ann.pdb, label, ann.uniprot)
                if align_key not in shared_alignment_cache:
                    shared_alignment_cache[align_key] = align_chain_to_uniprot(chain_seq, uni_seq)

                position_map, identity = shared_alignment_cache[align_key]
                audit.alignment_identity = identity

                if identity < args.minimum_alignment_identity:
                    audit.status = "low_alignment_identity"
                    audits.append(audit)
                    continue

                uni_pos = position_map.get(pos)
                if uni_pos is None:
                    audit.status = "unaligned_chain_position"
                    audits.append(audit)
                    continue

                audit.uniprot_position_1based = uni_pos
                audit.uniprot_residue_aa = uni_seq[uni_pos - 1]

                if audit.uniprot_residue_aa != ann.residue_aa:
                    audit.status = "uniprot_identity_mismatch"
                    audits.append(audit)
                    continue

                audit.status = "accepted_uniprot"
                sequence_cache[(ann.uniprot, "UNIPROT")] = uni_seq

            except Exception as exc:
                audit.status = "mapping_failed"
                audit.note = f"{type(exc).__name__}: {exc}"

            audits.append(audit)

            if ann_i % 1000 == 0:
                print(
                    f"[MAPPING] current batch annotations mapped={ann_i}/{len(annotations)}",
                    flush=True,
                )

        return audits, sequence_cache

    def build_windows_from_batch(
        audits: list[Audit],
        sequence_cache: dict[tuple[str, str], str],
    ) -> None:
        nonlocal total_accepted, total_positive
        nonlocal skipped_near_terminus, skipped_noncanonical

        accepted = [a for a in audits if a.status == accepted_status]
        total_accepted += len(accepted)

        # Batch-local aggregation. This is the intended all-metal pretraining tradeoff:
        # it gives immediate outputs while still grouping accepted positions inside this batch.
        positions_by_key: dict[tuple[str, str, str], set[int]] = defaultdict(set)
        metadata: dict[tuple[str, str, str], dict[str, str]] = {}
        sites: dict[tuple[str, str, str], set[str]] = defaultdict(set)

        for a in accepted:
            if args.sequence_source == "uniprot":
                key = (a.uniprot, "UNIPROT", a.metal)
                position = a.uniprot_position_1based
                seq_key = (a.uniprot, "UNIPROT")
            else:
                key = (a.pdb, a.label_chain, a.metal)
                position = a.chain_position_1based
                seq_key = (a.pdb, a.label_chain)

            if position is None:
                continue

            positions_by_key[key].add(int(position))
            sites[key].add(a.site)
            metadata.setdefault(key, {
                "pdb": a.pdb,
                "uniprot": a.uniprot,
                "label_chain": a.label_chain,
                "auth_chain": a.auth_chain,
                "metal": a.metal,
                "seq0": seq_key[0],
                "seq1": seq_key[1],
            })

        for key, binding_positions in positions_by_key.items():
            meta = metadata[key]
            sequence = sequence_cache.get((meta["seq0"], meta["seq1"]), "")

            if len(sequence) < args.window_len:
                continue

            group_id = group_id_for(meta, args)
            split = deterministic_split_for_group(group_id, args)

            for center_pos in sorted(binding_positions):
                win = centered_window(sequence, center_pos, args.window_len, args.pad_termini)
                if win is None:
                    skipped_near_terminus += 1
                    continue

                peptide, start, end, center_idx0 = win

                if not valid_output_peptide(peptide, args.allow_noncanonical):
                    skipped_noncanonical += 1
                    continue

                labels_binary, labels_spaced, inside = build_labels_for_window(
                    binding_positions,
                    start,
                    end,
                )

                row = {
                    "metal": meta["metal"],
                    "group_id": group_id,
                    "site": ";".join(sorted(s for s in sites[key] if s)),
                    "pdb": meta["pdb"],
                    "uniprot": meta["uniprot"],
                    "label_chain": meta["label_chain"],
                    "auth_chain": meta["auth_chain"],
                    "sequence_source": args.sequence_source,
                    "sequence_source_id": meta["seq0"],
                    "sequence_source_chain": meta["seq1"],
                    "protein_length": len(sequence),
                    "known_binding_positions": ";".join(map(str, sorted(binding_positions))),
                    "center_binding_position_1based": center_pos,
                    "center_pos_in_window": center_idx0,
                    "window_start_1based": start,
                    "window_end_1based": end,
                    "peptide_len10": peptide,
                    "sequence": peptide,
                    "labels": labels_binary,
                    "binding_site_labels_len10": labels_spaced,
                    "binding_positions_in_window": ";".join(map(str, inside)),
                    "is_negative": 0,
                    "label_confidence": "high_chain_mapped_residue_identity_verified_batch_aggregated",
                    "mapping_status": accepted_status,
                    "split": split,
                    "seq_len": args.window_len,
                }

                key2 = dedup_key(row, args.dedup_level)
                if key2 is not None and key2 in seen:
                    continue
                if key2 is not None:
                    seen.add(key2)

                rows_buffer.append(row)
                total_positive += 1
                split_counts[split] += 1
                metal_counts[meta["metal"]] += 1
                unique_peptides.add(peptide)

                if len(rows_buffer) >= args.save_every_rows:
                    flush_peptide_part()

    print("[START] true batch-chunked MetalPDB fixed-length extraction", flush=True)
    print(f"[CONFIG] raw_batch_rows={args.raw_batch_rows}", flush=True)
    print(f"[CONFIG] save_every_rows={args.save_every_rows}", flush=True)
    print(f"[CONFIG] sequence_source={args.sequence_source}", flush=True)
    print(f"[CONFIG] target_metal={args.target_metal}", flush=True)

    for raw_batch in pd.read_csv(args.raw_csv, chunksize=args.raw_batch_rows):
        raw_batch_idx += 1
        total_raw_rows += len(raw_batch)

        print(
            f"\n[RAW-BATCH] {raw_batch_idx}: raw_rows={len(raw_batch)}, "
            f"total_raw_rows={total_raw_rows}",
            flush=True,
        )

        annotations = parse_annotations(raw_batch, args.target_metal)
        total_annotations += len(annotations)

        print(
            f"[RAW-BATCH] {raw_batch_idx}: parsed_annotations={len(annotations)}, "
            f"total_annotations={total_annotations}",
            flush=True,
        )

        if not annotations:
            continue

        audits, sequence_cache = map_annotations_streaming(annotations)

        for status, count in pd.Series([a.status for a in audits]).value_counts().to_dict().items():
            status_counts_total[status] += int(count)

        write_one_audit_part(audits, raw_batch_idx)
        build_windows_from_batch(audits, sequence_cache)

        # Write partial peptide output at the end of every raw batch, even if fewer
        # than --save-every-rows were generated. This guarantees visible progress.
        flush_peptide_part(force=True)

        # Explicitly release batch-local memory.
        del raw_batch, annotations, audits, sequence_cache

        print(
            f"[BATCH-DONE] {raw_batch_idx}: total_positive_windows={total_positive}, "
            f"seen_dedup_keys={len(seen)}",
            flush=True,
        )

    if not part_files:
        raise RuntimeError("No fixed-length windows generated.")

    manifest_path = out_dir / f"{prefix}_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    audit_manifest_path = out_dir / f"{prefix}_mapping_audit_manifest.csv"
    pd.DataFrame(audit_manifest_rows).to_csv(audit_manifest_path, index=False)

    if args.negative_fraction > 0:
        print(
            "[WARN] --negative-fraction > 0 was requested, but this true-chunked "
            "script skips negatives for memory safety. Use the non-chunked script "
            "for small single-metal datasets requiring negatives.",
            flush=True,
        )

    if args.make_final_file:
        combine_part_csvs(part_files, args.output_csv)
        combined_output = str(args.output_csv)
    else:
        combined_output = "not_created_parts_only"

    summary = {
        "target_metal": args.target_metal.upper(),
        "window_len": args.window_len,
        "sequence_source": args.sequence_source,
        "raw_batch_rows": args.raw_batch_rows,
        "save_every_rows": args.save_every_rows,
        "total_raw_rows": int(total_raw_rows),
        "parsed_annotations": int(total_annotations),
        "accepted_annotations": int(total_accepted),
        "mapping_status_counts": dict(sorted(status_counts_total.items())),
        "positive_windows": int(total_positive),
        "negative_windows": 0,
        "total_windows": int(total_positive),
        "unique_peptides": int(len(unique_peptides)),
        "split_counts": dict(split_counts),
        "metal_counts": dict(sorted(metal_counts.items())),
        "dedup_level": args.dedup_level,
        "skipped_near_terminus": int(skipped_near_terminus),
        "skipped_noncanonical": int(skipped_noncanonical),
        "output_dir": str(out_dir),
        "parts_dir": str(parts_dir),
        "manifest_csv": str(manifest_path),
        "combined_output_csv": combined_output,
        "mapping_audit_manifest_csv": str(audit_manifest_path),
        "split_assignment": "deterministic_group_hash",
        "label_aggregation_scope": "batch_local",
        "note": (
            "This true-chunked all-metal version writes outputs after each raw batch. "
            "Labels mark all accepted binding positions available within the current raw batch. "
            "For strict Cu validation, prefer the non-streaming/global aggregation version."
        ),
    }

    summary_path = out_dir / f"{prefix}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n[DONE]", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

    return pd.DataFrame(manifest_rows)


def main() -> int:
    args = parse_args()
    if args.window_len != 10:
        print(f"[INFO] This script is designed for fixed-length 10 peptides, but got --window-len={args.window_len}.")
    if not 0 <= args.negative_fraction < 1:
        raise ValueError("negative fraction must be in [0,1)")
    if args.output_csv is None and args.output_dir is None:
        raise ValueError("Provide --output-dir for chunked output or --output-csv for a final/derived output name.")
    if args.save_every_rows <= 0:
        raise ValueError("--save-every-rows must be positive.")
    build_fixed_len10_dataset(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
