"""
Fetch all MetalPDB metal-binding protein records for all metals with non-zero sites,
then remove duplicates at two levels:

1) unique protein-metal pairs: one row per UniProt/accession + metal
2) unique proteins globally: one row per protein, with all associated metals grouped

Based on the user's original find_proteins_metaldb.py, extended from one metal to all metals.

MetalPDB REST API docs:
https://metalpdb.cerm.unifi.it/api_help
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import requests

BASE_URL = "https://metalpdb.cerm.unifi.it/api"

# Metals with non-zero sites in the MetalPDB summary table.
# Symbols are normalized to the usual capitalization expected by MetalPDB queries.
METALS_NONZERO = [
    "Mg", "Zn", "Ca", "Fe", "Na", "Mn", "K", "Ni", "Cu", "Co", "Cd", "Hg",
    "Be", "Pt", "Mo", "Al", "Ba", "V", "Ru", "Sr", "Cs", "W", "Au", "Gd",
    "Ag", "Yb", "Li", "Ir", "Rb", "Pb", "Y", "Tl", "U", "Pr", "Sm", "Tb",
    "Os", "Rh", "Pd", "Eu", "Re", "Ta", "Lu", "La", "Cr", "Ho", "Sb", "Ga",
    "Sn", "Ti", "Zr", "Ce", "Er", "In", "Th", "Nd", "Dy", "Hf", "Bi", "Sc",
    "Pa", "Pu", "Am", "Cm", "Cf",
]

# Useful output columns. You can add columns supported by MetalPDB here.
COLUMNS = [
    "pdb_code",
    "uniprot",
    "molecule",
    "organism",
    "ec_number",
    "pfam",
    "cath",
    "scop",
    "is_representative",
    "site_type",
    "geometry",
    "coordination",
    "pattern",
    "metal",
    "ligands",
    "donors",
]


@dataclass
class MetalPDBClient:
    timeout_sec: float = 90.0
    pause_sec: float = 0.25
    max_retries: int = 4

    def query(
        self,
        query: str,
        columns: Optional[List[str]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        params: Dict[str, Any] = {"query": query}
        if columns:
            params["columns"] = ",".join(columns)
        if extra_params:
            params.update(extra_params)

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = requests.get(BASE_URL, params=params, timeout=self.timeout_sec)
                r.raise_for_status()
                return r.json()
            except Exception as e:  # noqa: BLE001 - keep retry broad for network/API issues
                last_err = e
                if attempt < self.max_retries:
                    sleep_s = self.pause_sec * attempt
                    print(f"Retry {attempt}/{self.max_retries} after error: {e}; sleeping {sleep_s:.2f}s")
                    time.sleep(sleep_s)
                else:
                    raise RuntimeError(
                        f"MetalPDB request failed after {attempt} tries for {query}: {e}"
                    ) from last_err


def flatten_records(records: Any) -> pd.DataFrame:
    """Normalize MetalPDB JSON into a flat pandas table."""
    if records is None:
        return pd.DataFrame()
    if isinstance(records, list):
        items = records
    elif isinstance(records, dict):
        if isinstance(records.get("results"), list):
            items = records["results"]
        elif isinstance(records.get("data"), list):
            items = records["data"]
        else:
            items = [records]
    else:
        items = [{"value": records}]
    return pd.json_normalize(items)


def _as_clean_string(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    if isinstance(x, (list, tuple, set)):
        return ";".join(_as_clean_string(v) for v in x if _as_clean_string(v))
    if isinstance(x, dict):
        return json.dumps(x, sort_keys=True, ensure_ascii=False)
    return str(x).strip()


def split_uniprot_values(value: Any) -> List[str]:
    """
    MetalPDB may return uniprot as a string, list, or missing value.
    This function returns cleaned accession tokens.
    """
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, (list, tuple, set)):
        vals: List[str] = []
        for v in value:
            vals.extend(split_uniprot_values(v))
        return vals

    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null", "-"}:
        return []

    # Covers separators such as comma, semicolon, whitespace, and pipe.
    parts = re.split(r"[,;|\s]+", s)
    return sorted({p.strip() for p in parts if p.strip() and p.strip().lower() not in {"nan", "none", "null"}})


def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower_to_actual = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower_to_actual:
            return lower_to_actual[c.lower()]
    return None


def standardize_records(df: pd.DataFrame, query_metal: str) -> pd.DataFrame:
    """Add stable columns for deduplication: query_metal, uniprot_id, protein_key."""
    if df.empty:
        return df

    out = df.copy()
    out.insert(0, "query_metal", query_metal)

    uniprot_col = _first_existing_column(out, ["uniprot", "uniprot_id", "uniprot.accession"])
    pdb_col = _first_existing_column(out, ["pdb_code", "pdb", "pdb_id"])
    mol_col = _first_existing_column(out, ["molecule", "macromolecule", "protein_name"])
    org_col = _first_existing_column(out, ["organism"])

    if uniprot_col is None:
        out["uniprot_id"] = ""
    else:
        out["uniprot_id"] = out[uniprot_col].apply(lambda x: ";".join(split_uniprot_values(x)))

    # Fallback key when UniProt is missing: PDB + molecule + organism.
    pdb_s = out[pdb_col].apply(_as_clean_string) if pdb_col else ""
    mol_s = out[mol_col].apply(_as_clean_string) if mol_col else ""
    org_s = out[org_col].apply(_as_clean_string) if org_col else ""

    if isinstance(pdb_s, str):
        fallback = pd.Series([""] * len(out), index=out.index)
    else:
        fallback = pdb_s.str.upper() + "|" + mol_s.str.upper() + "|" + org_s.str.upper()

    out["protein_key"] = out["uniprot_id"].where(out["uniprot_id"].astype(str).str.len() > 0, fallback)
    out["has_uniprot"] = out["uniprot_id"].astype(str).str.len() > 0
    return out


def fetch_one_metal(client: MetalPDBClient, metal: str, raw_dir: Path) -> pd.DataFrame:
    print(f"\n=== Fetching {metal} ===")
    data = client.query(query=f"metal:{metal}", columns=COLUMNS)

    raw_json = raw_dir / f"metalpdb_{metal}_raw.json"
    raw_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    df = flatten_records(data)
    df = standardize_records(df, metal)
    print(f"{metal}: {len(df)} raw rows")
    return df


def collapse_values(series: pd.Series, max_items: int = 30) -> str:
    vals = []
    for x in series.dropna().tolist():
        sx = _as_clean_string(x)
        if sx:
            vals.extend([v for v in re.split(r"[;]+", sx) if v])
    uniq = sorted(set(vals))
    if len(uniq) > max_items:
        return ";".join(uniq[:max_items]) + f";...(+{len(uniq) - max_items} more)"
    return ";".join(uniq)


def build_deduplicated_outputs(all_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if all_df.empty:
        return {
            "all_records": all_df,
            "unique_protein_metal_pairs": all_df,
            "unique_proteins": all_df,
            "summary_by_metal": pd.DataFrame(),
        }

    # One row per protein_key + metal.
    pair_cols_preferred = [
        "protein_key", "uniprot_id", "has_uniprot", "query_metal",
        "pdb_code", "molecule", "organism", "ec_number", "pfam",
    ]
    existing_pair_cols = [c for c in pair_cols_preferred if c in all_df.columns]

    unique_pairs = (
        all_df.sort_values(["protein_key", "query_metal"])
        .drop_duplicates(subset=["protein_key", "query_metal"], keep="first")
        .loc[:, existing_pair_cols + [c for c in all_df.columns if c not in existing_pair_cols]]
    )

    # One row per protein_key globally, collecting metals and representative PDBs/names.
    group = all_df.groupby("protein_key", dropna=False)
    unique_proteins = group.agg(
        uniprot_id=("uniprot_id", "first"),
        has_uniprot=("has_uniprot", "first"),
        metals=("query_metal", collapse_values),
        n_metals=("query_metal", lambda s: len(set(s.dropna()))),
        n_site_records=("query_metal", "size"),
    ).reset_index()

    for optional_col in ["pdb_code", "molecule", "organism", "ec_number", "pfam"]:
        if optional_col in all_df.columns:
            extra = group[optional_col].apply(collapse_values).rename(optional_col).reset_index()
            unique_proteins = unique_proteins.merge(extra, on="protein_key", how="left")

    summary_by_metal = (
        all_df.groupby("query_metal")
        .agg(
            raw_site_records=("query_metal", "size"),
            unique_protein_keys=("protein_key", "nunique"),
            unique_uniprot_ids=("uniprot_id", lambda s: len({x for x in s if str(x).strip()})),
        )
        .reset_index()
        .sort_values("query_metal")
    )

    return {
        "all_records": all_df,
        "unique_protein_metal_pairs": unique_pairs,
        "unique_proteins": unique_proteins.sort_values(["n_metals", "protein_key"], ascending=[False, True]),
        "summary_by_metal": summary_by_metal,
    }


def fetch_all_metals_and_deduplicate(
    metals: Optional[List[str]] = None,
    out_dir: str | Path = "metalpdb_all_metals_unique",
) -> Dict[str, pd.DataFrame]:
    metals = metals or METALS_NONZERO
    out_path = Path(out_dir)
    raw_dir = out_path / "raw_json"
    out_path.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    client = MetalPDBClient()
    frames: List[pd.DataFrame] = []
    failures: List[Dict[str, str]] = []

    for metal in metals:
        try:
            df_metal = fetch_one_metal(client, metal, raw_dir)
            if not df_metal.empty:
                frames.append(df_metal)
            time.sleep(client.pause_sec)
        except Exception as e:  # noqa: BLE001
            print(f"FAILED {metal}: {e}")
            failures.append({"metal": metal, "error": str(e)})

    all_df = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    outputs = build_deduplicated_outputs(all_df)

    outputs["all_records"].to_csv(out_path / "metalpdb_all_metals_all_records.csv", index=False)
    outputs["unique_protein_metal_pairs"].to_csv(out_path / "metalpdb_unique_protein_metal_pairs.csv", index=False)
    outputs["unique_proteins"].to_csv(out_path / "metalpdb_unique_proteins_deduplicated.csv", index=False)
    outputs["summary_by_metal"].to_csv(out_path / "metalpdb_summary_by_metal_after_dedup.csv", index=False)
    pd.DataFrame(failures).to_csv(out_path / "metalpdb_failed_metals.csv", index=False)

    print("\nSaved outputs to:", out_path.resolve())
    print("All raw site records:", len(outputs["all_records"]))
    print("Unique protein-metal pairs:", len(outputs["unique_protein_metal_pairs"]))
    print("Unique proteins globally:", len(outputs["unique_proteins"]))
    if failures:
        print("Failures:", failures)
    return outputs


if __name__ == "__main__":
    # Run all metals:
    fetch_all_metals_and_deduplicate()

    # Or test a small subset first:
    # fetch_all_metals_and_deduplicate(metals=["Cu", "Zn", "Mn"], out_dir="metalpdb_test_subset")
