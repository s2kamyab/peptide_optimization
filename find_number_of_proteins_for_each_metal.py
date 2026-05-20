import gzip
from collections import Counter, defaultdict

path = "Data/BioLiP_nr.txt.gz"  # adjust path

# Column indices (0-based) from BioLiP readme:
# 01 PDB ID, 02 receptor chain, 05 ligand ID
PDB_COL = 0
CHAIN_COL = 1
LIG_COL = 4  # ligand ID (CCD)

# Put the metal CCD codes you care about (examples)
METALS = {
    "ZN", "CU", "CO", "NI", "MN", "MG", "FE", "CA",
    "NA", "K", "CD", "HG", "PB", "AG", "AU", "AL",
    "V", "CR", "MO", "W", "SE", "SR"
}

interaction_counts = Counter()
protein_sets = defaultdict(set)

with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
    for line in f:
        if not line.strip():
            continue
        cols = line.rstrip("\n").split("\t")
        if len(cols) <= LIG_COL:
            continue

        pdb = cols[PDB_COL].strip()
        chain = cols[CHAIN_COL].strip()
        lig = cols[LIG_COL].strip().upper()

        if lig in METALS:
            interaction_counts[lig] += 1
            protein_sets[lig].add((pdb, chain))

# Print results
print("Metal\t#Interactions\t#UniqueProteins(PDB+Chain)")
for lig, n in interaction_counts.most_common():
    print(f"{lig}\t{n}\t{len(protein_sets[lig])}")
