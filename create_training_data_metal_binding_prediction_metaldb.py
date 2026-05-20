import ast
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import requests

# -----------------------------
# Configuration
# -----------------------------

RAW_CSV = "metalpdb_Cu.csv"
OUTPUT_CSV = "metal_binding_augmented_random_peptides.csv"

MIN_LEN = 4
MAX_LEN = 20

# Number of positive peptides sampled from each row/site
POSITIVES_PER_ROW = 2

# Target fraction of all peptides that should be all-zero negatives
NEGATIVE_FRACTION = 0.10

RANDOM_SEED = 42
REQUEST_TIMEOUT = 20
SLEEP_BETWEEN_REQUESTS = 0.1

# Standard amino acids only
AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # occasional alternates can be added if needed
    "SEC": "U", "PYL": "O",
}

rng = random.Random(RANDOM_SEED)


# -----------------------------
# Utilities
# -----------------------------

def safe_literal_eval(text: str):
    try:
        return ast.literal_eval(text)
    except Exception:
        return None


def fetch_uniprot_sequence(accession: str, cache: Dict[str, Optional[str]]) -> Optional[str]:
    """
    Fetch sequence in FASTA format from UniProt.
    """
    accession = str(accession).strip()
    if not accession or accession.lower() == "nan":
        return None

    if accession in cache:
        return cache[accession]

    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            cache[accession] = None
            return None

        lines = [line.strip() for line in resp.text.splitlines() if line.strip()]
        seq = "".join(line for line in lines if not line.startswith(">")).upper()

        if not seq:
            cache[accession] = None
            return None

        cache[accession] = seq
        time.sleep(SLEEP_BETWEEN_REQUESTS)
        return seq

    except Exception:
        cache[accession] = None
        return None


def parse_binding_positions_from_metals(
    metals_str: str,
    expected_symbol: Optional[str] = None,
) -> Set[int]:
    """
    Extract 1-based residue positions of amino-acid ligands from the metals field.
    Keeps only standard amino-acid residues.
    """
    out: Set[int] = set()
    data = safe_literal_eval(metals_str)
    if not isinstance(data, list):
        return out

    for metal_obj in data:
        if not isinstance(metal_obj, dict):
            continue

        symbol = str(metal_obj.get("symbol", "")).strip()
        if expected_symbol and symbol.upper() != expected_symbol.upper():
            continue

        ligands = metal_obj.get("ligands", [])
        if not isinstance(ligands, list):
            continue

        for lig in ligands:
            if not isinstance(lig, dict):
                continue

            residue_name = str(lig.get("residue", "")).strip().upper()
            residue_pos = lig.get("residue_pdb_number", None)

            if residue_name not in AA3_TO_AA1:
                continue

            try:
                residue_pos = int(residue_pos)
            except Exception:
                continue

            if residue_pos >= 1:
                out.add(residue_pos)

    return out


def random_positive_window(
    sequence: str,
    binding_positions_1based: Set[int],
    min_len: int,
    max_len: int,
    max_tries: int = 200,
) -> Optional[Tuple[str, str, int, int]]:
    """
    Sample a peptide that contains at least one binding residue.
    Returns: peptide, labels, start_1based, end_1based
    """
    n = len(sequence)
    valid_binding_positions = sorted(p for p in binding_positions_1based if 1 <= p <= n)
    if not valid_binding_positions:
        return None

    for _ in range(max_tries):
        anchor = rng.choice(valid_binding_positions)
        L = rng.randint(min_len, min(max_len, n))

        # Choose a random position inside the peptide for the anchor residue
        anchor_offset = rng.randint(0, L - 1)

        start = anchor - anchor_offset  # 1-based inclusive
        end = start + L - 1

        if start < 1:
            start = 1
            end = start + L - 1
        if end > n:
            end = n
            start = end - L + 1

        if start < 1 or end > n or start > end:
            continue

        inside = [p for p in valid_binding_positions if start <= p <= end]
        if not inside:
            continue

        peptide = sequence[start - 1:end]
        labels = ["0"] * len(peptide)
        for p in inside:
            labels[p - start] = "1"

        return peptide, "".join(labels), start, end

    return None


def random_negative_window(
    sequence: str,
    binding_positions_1based: Set[int],
    min_len: int,
    max_len: int,
    max_tries: int = 300,
) -> Optional[Tuple[str, str, int, int]]:
    """
    Sample a peptide that contains no binding residue.
    Returns: peptide, labels, start_1based, end_1based
    """
    n = len(sequence)
    if n < min_len:
        return None

    valid_binding_positions = sorted(p for p in binding_positions_1based if 1 <= p <= n)

    for _ in range(max_tries):
        L = rng.randint(min_len, min(max_len, n))
        start = rng.randint(1, n - L + 1)
        end = start + L - 1

        has_binding = any(start <= p <= end for p in valid_binding_positions)
        if has_binding:
            continue

        peptide = sequence[start - 1:end]
        labels = "0" * len(peptide)
        return peptide, labels, start, end

    return None


def infer_metal_symbol_from_row(row: pd.Series) -> Optional[str]:
    """
    Try to infer the metal symbol from the metals field.
    For your file this is expected to be Cu.
    """
    data = safe_literal_eval(row.get("metals", ""))
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        symbol = str(data[0].get("symbol", "")).strip()
        return symbol.upper() if symbol else None
    return None


# -----------------------------
# Main generation function
# -----------------------------

def generate_dataset(
    raw_csv: str,
    output_csv: str,
    positives_per_row: int = 2,
    negative_fraction: float = 0.10,
    min_len: int = 4,
    max_len: int = 20,
) -> pd.DataFrame:
    df = pd.read_csv(raw_csv)

    required_cols = {"site", "metals", "uniprot", "pdb"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    seq_cache: Dict[str, Optional[str]] = {}
    positive_rows: List[dict] = []
    negative_candidates: List[dict] = []

    seen = set()

    for idx, row in df.iterrows():
        uniprot = row.get("uniprot", None)
        pdb = row.get("pdb", None)
        site = row.get("site", None)

        seq = fetch_uniprot_sequence(uniprot, seq_cache)
        if not seq:
            continue

        metal_symbol = infer_metal_symbol_from_row(row) or "UNK"
        binding_positions = parse_binding_positions_from_metals(
            row.get("metals", ""),
            expected_symbol=metal_symbol,
        )

        # Skip rows with no amino-acid ligand residues
        if not binding_positions:
            continue

        # Keep only positions that fit in the fetched sequence
        binding_positions = {p for p in binding_positions if 1 <= p <= len(seq)}
        if not binding_positions:
            continue

        group_id = f"{pdb}|{uniprot}|{site}"

        # Positive peptides
        for rep in range(positives_per_row):
            out = random_positive_window(seq, binding_positions, min_len, max_len)
            if out is None:
                continue

            peptide, labels, start, end = out
            key = (group_id, peptide, labels, "positive")
            if key in seen:
                continue
            seen.add(key)

            positive_rows.append({
                "sequence": peptide,
                "metal": metal_symbol,
                "labels": labels,
                "is_negative": 0,
                "split": "",  # optional; fill later if needed
                "group_id": group_id,
                "site": site,
                "pdb": pdb,
                "uniprot": uniprot,
                "window_start_1based": start,
                "window_end_1based": end,
                "protein_length": len(seq),
            })

        # Negative sampling candidate pool from same protein/site
        neg = random_negative_window(seq, binding_positions, min_len, max_len)
        if neg is not None:
            peptide, labels, start, end = neg
            key = (group_id, peptide, labels, "negative")
            if key not in seen:
                seen.add(key)
                negative_candidates.append({
                    "sequence": peptide,
                    "metal": metal_symbol,
                    "labels": labels,
                    "is_negative": 1,
                    "split": "",
                    "group_id": group_id,
                    "site": site,
                    "pdb": pdb,
                    "uniprot": uniprot,
                    "window_start_1based": start,
                    "window_end_1based": end,
                    "protein_length": len(seq),
                })

    if not positive_rows:
        raise RuntimeError("No positive peptides were generated. Check sequence retrieval and residue mapping.")

    # Exact negative count to make ~10% of final dataset negative:
    # n_neg / (n_pos + n_neg) = negative_fraction
    n_pos = len(positive_rows)
    n_neg_target = int(round((negative_fraction / (1.0 - negative_fraction)) * n_pos))
    n_neg_target = min(n_neg_target, len(negative_candidates))

    rng.shuffle(negative_candidates)
    negative_rows = negative_candidates[:n_neg_target]

    final_rows = positive_rows + negative_rows
    rng.shuffle(final_rows)

    out_df = pd.DataFrame(final_rows)

    # Optional sanity checks
    out_df["seq_len"] = out_df["sequence"].str.len()
    assert (out_df["seq_len"] >= min_len).all()
    assert (out_df["seq_len"] <= max_len).all()
    assert (out_df["labels"].str.len() == out_df["sequence"].str.len()).all()

    out_df.to_csv(output_csv, index=False)

    print(f"Saved: {output_csv}")
    print(f"Total peptides: {len(out_df)}")
    print(f"Positive peptides: {(out_df['is_negative'] == 0).sum()}")
    print(f"Negative peptides: {(out_df['is_negative'] == 1).sum()}")
    print(f"Negative fraction: {(out_df['is_negative'] == 1).mean():.4f}")
    print("\nLength distribution:")
    print(out_df["seq_len"].value_counts().sort_index())

    return out_df


if __name__ == "__main__":
    generate_dataset(
        raw_csv=RAW_CSV,
        output_csv=OUTPUT_CSV,
        positives_per_row=POSITIVES_PER_ROW,
        negative_fraction=NEGATIVE_FRACTION,
        min_len=MIN_LEN,
        max_len=MAX_LEN,
    )