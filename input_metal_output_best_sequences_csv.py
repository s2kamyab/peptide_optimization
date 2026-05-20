import os, gzip, io, re, urllib.request
from collections import defaultdict
import numpy as np
from typing import List, Tuple
import argparse
import pandas as pd



def make_header(row: pd.Series) -> str:
    """
    Build a compact header from common metadata if present.
    Falls back to a simple index-based header if none found.
    """
    parts = []
    for key in ("pdb", "chain", "uniprot", "ligand_id", "site_code"):
        if key in row and pd.notna(row[key]):
            parts.append(f"{key}={row[key]}")
    if parts:
        return "|".join(parts)
    return f"rec_{row.name}"

def build_records(
    df: pd.DataFrame,
    seq_col: str,
    rle_col: str,
    truncate: bool = True,
) :#-> List[Tuple[str, str, str]]:
    """
    Create FASTA records: (header, sequence, label_str).
    If lengths differ and truncate=True, truncate to min length; else skip the row.
    """
    recs = []
    skipped = 0
    # for idx, row in df.iterrows():
    seq = str(df[seq_col]).strip().replace(" ", "")
    lab = df[rle_col]
    # lab = decode_rle(rle)

    # if not seq or not lab:
    #     skipped += 1
        # continue

    if len(seq) != len(lab):
        if truncate:
            L = min(len(seq), len(lab))
            seq = seq[:L]
            lab = lab[:L]
        else:
            skipped += 1
            # continue

    header = make_header(df)
    recs.append((header, seq, lab))

    if skipped:
        print(f"[info] skipped {skipped} rows with missing/invalid data", file=sys.stderr)
    return recs

def write_fasta(path: str, recs: List[Tuple[str, str, str]]) :#-> None:
    # with open(path + '_sequences.fasta', "w") as f:
        for h, seq, lab in recs:
            bndst_no = [int(ch) for ch in lab.strip()]
            print(str(np.sum(bndst_no)))
            with open(path + h + str(np.sum(bndst_no)) + '_sequence.fasta', "w") as f:
                f.write(f">{h}\n")
                f.write(seq + "\n")

            with open(path + h + str(np.sum(bndst_no)) + '_label.fasta', "w") as f:
                # f.write(f">{h}\n")
                f.write(lab + "\n")

def decode_rle(rle_str: str):# -> str:
    """
    Decode strings like '0x8;1x1;0x2' -> '000000001001...'.
    Accepts separators ';' or whitespace; tokens '<bit>x<count>' or '<bit>*<count>'.
    """
    if pd.isna(rle_str):
        return ""
    rle_str = str(rle_str).strip()
    if not rle_str:
        return ""
    out = []
    for tok in re.split(r"[;\s]+", rle_str):
        if not tok:
            continue
        m = re.fullmatch(r"([01])\s*[*xX]\s*(\d+)", tok)
        if not m:
            raise ValueError(f"Bad RLE token: {tok!r} in {rle_str!r}")
        bit, n = m.group(1), int(m.group(2))
        out.append(bit * n)
    return "".join(out)


# -------- Build residue-level labels (using the renumbered residues) --------
# 'receptor_seq' is a continuous sequence for observed residues; 'bind_res_renum' contains positions starting at 1.
# We'll create a label vector of length len(receptor_seq) with 1 for binding residues.
def parse_renum(res_str):
    """Parse renumbered residue list like 'N180 L181 A182' or '73 74 75' -> set of ints (positions)."""
    if not res_str or res_str == "-":
        return set()
    toks = res_str.split()
    positions = set()
    for tok in toks:
        # Accept either pure numbers or residue+number (e.g., E219)
        m = re.search(r'(\d+)$', tok)
        if m:
            positions.add(int(m.group(1)))
    return positions


# -------- Convert to a flat table for quick training / CSV export --------
# One row per site with a compact run-length label encoding; for deep models keep as JSON/PKL instead.
def rle(bits):
    out = []
    cnt = 1
    for i in range(1, len(bits)+1):
        if i < len(bits) and bits[i] == bits[i-1]:
            cnt += 1
        else:
            out.append(f"{bits[i-1]}x{cnt}")
            cnt = 1
    return ";".join(out)

def main():
    ap = argparse.ArgumentParser(description="savr brst proteing to specified ligand metals.")
    ap.add_argument("--metal", type=str, default='DY', help="target metal")
    ap.add_argument("--out_dir", default="output", help="Output directory")
    # ap.add_argument("--csv", default= "biolip2_annotation/AU_sites_biolip2_peptide.csv",help="Input CSV path")
    ap.add_argument("--seq-col", default="sequence", help="Sequence column name (default: sequence)")
    ap.add_argument("--ann_path", default="Data/", help="Run-length encoding column name (default: rle)")
    ap.add_argument("--sample_no", default=-1, help="Number of top samples to process (default: 10)")
    ap.add_argument("--biolip_path", default="Data/", help="BioLiP annotation path (default: biolip2_annotation/)")
    args = ap.parse_args()
    # -------- Filter for Metal ions ------
    # CCD codes: CA (Co2+)
    metal_CODES = {args.metal}  # e.g., { 'AU' } for gold, {'CU'} for copper, {'FE'} for iron, {'ZN'} for zinc, {'CA'} for calcium
    pr_pep = "protein"#"protein" # "peptide" or "protein"
    keep_rows = args.sample_no
    # BASE = "https://zhanggroup.org/BioLiP/download/"
    FILES = {
        "ann": "BioLiP_nr.txt.gz",         # full set (use BioLiP_nr.txt.gz for non-redundant)
        # "ann": "BioLiP_nr.txt.gz",    # <- uncomment to use non-redundant annotations
        # "prot": "peptide_nr.fasta.gz"
        # "prot": f"{pr_pep}_nr.fasta.gz"
    }

    OUT_DIR = args.biolip_path

    ann_path = os.path.join(OUT_DIR, FILES["ann"])

    rows = []
    with gzip.open(ann_path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 21:
                continue
            rows.append(parts)

    cols = [
        "pdb","chain","resolution","site_code","ligand_id","ligand_chain","ligand_serial",
        "bind_res_pdb","bind_res_renum","cat_res_pdb","cat_res_renum",
        "ec","go","ki_manual","moad","pdbbind","bindingdb","uniprot","pubmed","ligand_auth_seq_id",
        "receptor_seq"]
    df1 = pd.DataFrame(rows, columns=cols)


    df_metal = df1[df1["ligand_id"].isin(metal_CODES)].copy()
    print(f"Found {len(df_metal)}  f{metal_CODES} sites (ligands in {metal_CODES})")
    dataset = []
    for _, r in df_metal.iterrows():
        seq = r["receptor_seq"].strip()
        if not seq or seq == "-":
            continue
        L = len(seq)
        pos_set = parse_renum(r["bind_res_renum"])
        labels = [1 if (i+1) in pos_set else 0 for i in range(L)]
        dataset.append({
            "pdb": r["pdb"],
            "chain": r["chain"],
            "uniprot": r["uniprot"],
            "ligand_id": r["ligand_id"],
            "site_code": r["site_code"],
            "sequence": seq,
            "labels": labels,             # 0/1 per residue for Co-binding at this site
            "L": L
        })

    print(f"Prepared {len(dataset)} (sequence, labels) {metal_CODES} samples")
    flat = []
    for item in dataset:
        flat.append({
            "pdb": item["pdb"],
            "chain": item["chain"],
            "uniprot": item["uniprot"],
            "ligand_id": item["ligand_id"],
            "site_code": item["site_code"],
            "length": item["L"],
            "sequence": item["sequence"],
            "labels": "".join(map(str, item["labels"]))
        })

    df = pd.DataFrame(flat)
    print(f"Final dataset: {len(df)} rows")
    rle_col = df.columns[-1]
    seq_col = "sequence"
    temp_count = []
    temp_labels = []
    for idx, row in df.iterrows():
        seq = str(row[seq_col]).strip().replace(" ", "")
        rle = row[rle_col]
        # lab = decode_rle(rle)

        y = np.fromiter((int(c) for c in rle), dtype=np.int64, count=len(rle))
        temp_count.append(np.sum(y))
        temp_labels.append(rle)
    df["num_binding_residues"] = temp_count
    df["labels"] = temp_labels
    df = df[["pdb","chain","uniprot","ligand_id","site_code","length","num_binding_residues","sequence","labels"]]

    df_sorted = df.sort_values("num_binding_residues", ascending=False)
    # df_sorted.to_csv(f"find_and_viz_protein_peptide/output/{metal_CODES}_sites_biolip2_peptide_sorted_desc.csv", index=False)
    df_sorted[:int(keep_rows)].to_csv(f"{args.out_dir}/{metal_CODES}_sites_biolip2_{pr_pep}_sorted_desc.csv", index=False)
    df_sorted[:int(keep_rows)].to_csv(f"{args.out_dir}/tmp_sorted_desc.csv", index=False)
    print(f"Wrote {args.out_dir}/{metal_CODES}_sites_biolip2_{pr_pep}_sorted_desc.csv with top {keep_rows} samples")

    # for i in range(min(int(keep_rows), len(df_sorted))):
    #   recs = build_records(df_sorted.iloc[i,:], 'sequence', rle_col, truncate= True)
    #   if len(recs) == 0:
    #       raise SystemExit("No valid records produced. Check your columns/format.")
    # #   write_fasta( f"/content/drive/MyDrive/Phonix_Metalix_for_AlphaFold/find_and_viz_protein_peptide/output/", recs)
    #   write_fasta( args.out_dir, recs)



if __name__ == "__main__":
    main()

