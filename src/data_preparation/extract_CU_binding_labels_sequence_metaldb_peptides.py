#! C:\Users\shima\OneDrive\Documentos\Leili\peptide_structure_optimization\peptide_optimization\peptide_opt_venv\Scripts\python.exe
import ast
from pathlib import Path

import pandas as pd
import requests
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1

# ---- Input MetalPDB CSV ----
CSV_PATH = "metalpdb_Cu.csv"   # change if needed
OUT_CSV  = "metalpdb_Cu_binding_windows_len10.csv"

# ---- RCSB structure download template ----
RCSB_CIF_URL = "https://files.rcsb.org/download/{pdb}.cif"

# Cache downloaded CIFs to avoid repeated downloads
CIF_CACHE_DIR = Path("cif_cache")
CIF_CACHE_DIR.mkdir(exist_ok=True)

WINDOW_LEN = 10  # peptide length to store


def safe_three_to_one(resname: str) -> str:
    # seq1 converts 3-letter AA to 1-letter; unknown -> 'X'
    return seq1(resname, custom_map={})


def extract_binding_resnums_per_chain(metals_str: str):
    """
    Parse MetalPDB 'metals' string and return:
      { chain_id: set(auth_residue_numbers_that_bind_metal) }
    Keeps only ligands that look like amino acids.
    """
    metals = ast.literal_eval(metals_str)
    out = {}
    for m in metals:
        for lig in m.get("ligands", []):
            chain = lig.get("chain")
            resnum = lig.get("residue_pdb_number")
            resname = lig.get("residue", "")
            if chain is None or resnum is None:
                continue

            aa1 = safe_three_to_one(str(resname).upper())
            if aa1 == "X":
                continue

            out.setdefault(chain, set()).add(int(resnum))
    return out


def download_cif(pdb_id: str) -> Path:
    pdb_id = pdb_id.lower()
    out_path = CIF_CACHE_DIR / f"{pdb_id}.cif"
    if out_path.exists():
        return out_path

    url = RCSB_CIF_URL.format(pdb=pdb_id)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    out_path.write_bytes(r.content)
    return out_path


def chain_sequence_and_resseqs_from_cif(cif_path: Path, chain_id: str):
    """
    Build:
      - sequence string
      - list of auth residue numbers (resseq) aligned with the sequence indices
    Only standard AAs are included.
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("S", str(cif_path))
    model = next(structure.get_models())

    if chain_id not in model:
        raise KeyError(f"Chain {chain_id} not found in structure")

    chain = model[chain_id]

    seq_chars = []
    resseqs = []

    for res in chain:
        if not is_aa(res, standard=True):
            continue

        aa1 = safe_three_to_one(res.get_resname().upper())
        if aa1 == "X":
            continue

        resseq = int(res.get_id()[1])  # (hetflag, resseq, icode)
        seq_chars.append(aa1)
        resseqs.append(resseq)

    return "".join(seq_chars), resseqs


def centered_window(sequence: str, center_idx: int, window_len: int):
    """
    Returns (pep, start, end, center_pos_in_window) where:
      pep = sequence[start:end] with length == window_len
      center_pos_in_window = index of center residue inside pep
    For even window_len (10), we place the center residue at position window_len//2 - 1 (i.e., 4),
    so you get 4 residues on the left, 5 on the right.
    """
    if window_len <= 0:
        raise ValueError("window_len must be > 0")

    left = window_len // 2 - 1 if window_len % 2 == 0 else window_len // 2
    right = window_len - 1 - left

    start = center_idx - left
    end = center_idx + right + 1  # exclusive

    if start < 0 or end > len(sequence):
        return None

    pep = sequence[start:end]
    center_pos = center_idx - start
    return pep, start, end, center_pos


def main():
    df = pd.read_csv(CSV_PATH)

    rows = []
    for i, r in df.iterrows():
        pdb_id = str(r["pdb"]).strip().lower()
        site_id = str(r["site"]).strip()
        print(f"Processing row {i+1}/{len(df)}: PDB={pdb_id} Site={site_id}")

        try:
            bind_map = extract_binding_resnums_per_chain(r["metals"])
        except Exception:
            continue

        try:
            cif_path = download_cif(pdb_id)
        except Exception:
            continue

        for chain_id, bind_resnums in bind_map.items():
            try:
                seq, resseqs = chain_sequence_and_resseqs_from_cif(cif_path, chain_id)
            except Exception:
                continue

            if not seq:
                continue

            # Map auth resseq -> sequence index
            resseq_to_idx = {resseq: idx for idx, resseq in enumerate(resseqs)}

            for bind_resseq in sorted(bind_resnums):
                if bind_resseq not in resseq_to_idx:
                    continue

                center_idx = resseq_to_idx[bind_resseq]
                win = centered_window(seq, center_idx, WINDOW_LEN)
                if win is None:
                    # skip if near termini; change to padding if you prefer
                    continue

                pep, start, end, center_pos = win

                labels = [0] * WINDOW_LEN
                labels[center_pos] = 1
                label_str = " ".join(map(str, labels))

                rows.append({
                    "site": site_id,
                    "pdb": pdb_id,
                    "chain": chain_id,
                    "bind_resseq": bind_resseq,         # PDB residue number
                    "center_idx": center_idx,           # index in filtered chain sequence
                    "window_start_idx": start,
                    "window_end_idx": end - 1,
                    "uniprot": r.get("uniprot", None),
                    "molecule": r.get("molecule", None),
                    "organism": r.get("organism", None),
                    "peptide_len10": pep,
                    "binding_site_labels_len10": label_str,
                })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}  (rows={len(out)})")


if __name__ == "__main__":
    main()
