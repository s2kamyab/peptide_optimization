#! C:\Users\shima\OneDrive\Documentos\Leili\peptide_structure_optimization\peptide_optimization\peptide_opt_venv\Scripts\python.exe
import ast
import gzip
from pathlib import Path

import pandas as pd
import requests
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1

# ---- Input MetalPDB CSV ----
CSV_PATH = "metalpdb_Mn.csv"   # change if needed
OUT_CSV  = "metalpdb_extracted_sequences_and_labels.csv"

# ---- RCSB structure download template ----
RCSB_CIF_URL = "https://files.rcsb.org/download/{pdb}.cif"

# Cache downloaded CIFs to avoid repeated downloads
CIF_CACHE_DIR = Path("cif_cache")
CIF_CACHE_DIR.mkdir(exist_ok=True)

# 3-letter amino acids we will treat as protein residues
# (BioPython's is_aa helps, but we also use three_to_one conversion)
def safe_three_to_one(resname: str) -> str:
    # seq1 converts 3-letter AA to 1-letter; unknown -> 'X'
    return seq1(resname, custom_map={})

def extract_binding_resnums_per_chain(metals_str: str):
    """
    Parse MetalPDB 'metals' string (python-literal-like) and return:
      { chain_id: set(auth_residue_numbers_that_bind_metal) }
    Filters out waters/ions by keeping only ligands that look like amino acids.
    """
    metals = ast.literal_eval(metals_str)  # string uses single quotes -> literal_eval works
    out = {}
    for m in metals:
        for lig in m.get("ligands", []):
            chain = lig.get("chain")
            resnum = lig.get("residue_pdb_number")
            resname = lig.get("residue", "")
            if chain is None or resnum is None:
                continue

            # Keep only protein-like residues (skip HOH, SO4, etc.)
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

def chain_sequence_and_labels_from_cif(cif_path: Path, chain_id: str, bind_resnums: set[int]):
    """
    Build sequence and a binary label vector aligned to the residue order
    in the structure for the given chain.
    Labels use PDB residue numbers (auth resseq) from the parsed structure.
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("S", str(cif_path))

    # Use first model
    model = next(structure.get_models())

    if chain_id not in model:
        raise KeyError(f"Chain {chain_id} not found in structure")

    chain = model[chain_id]

    seq_chars = []
    labels = []

    for res in chain:
        if not is_aa(res, standard=True):
            continue
        resname = res.get_resname().upper()
        aa1 = safe_three_to_one(resname)

        # res.get_id() -> (hetflag, resseq, icode)
        resseq = int(res.get_id()[1])

        seq_chars.append(aa1)
        labels.append(1 if resseq in bind_resnums else 0)

    sequence = "".join(seq_chars)
    label_str = " ".join(map(str, labels))  # "0 1 0 0 ..."

    return sequence, label_str

def main():
    df = pd.read_csv(CSV_PATH)

    rows = []
    for i, r in df.iterrows():
        print(f"Processing row {i+1}/{len(df)}: PDB={r['pdb']} Site={r['site']}")
        pdb_id = str(r["pdb"]).strip()
        site_id = str(r["site"]).strip()

        # Parse binding residues per chain from MetalPDB "metals" field
        try:
            bind_map = extract_binding_resnums_per_chain(r["metals"])
        except Exception:
            continue

        # Download structure once per PDB
        try:
            cif_path = download_cif(pdb_id)
        except Exception:
            continue

        # For each chain mentioned in the site, extract sequence+labels
        for chain_id, bind_resnums in bind_map.items():
            try:
                seq, labels = chain_sequence_and_labels_from_cif(cif_path, chain_id, bind_resnums)
            except Exception:
                continue

            if not seq:
                continue

            rows.append({
                "site": site_id,
                "pdb": pdb_id.lower(),
                "chain": chain_id,
                "uniprot": r.get("uniprot", None),
                "molecule": r.get("molecule", None),
                "organism": r.get("organism", None),
                "sequence": seq,
                "binding_site_labels": labels,
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}  (rows={len(out)})")

if __name__ == "__main__":
    main()
