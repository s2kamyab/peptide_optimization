"""
Download MetalPDB entries for a specific metal ion using MetalPDB REST API.

Docs: https://metalpdb.cerm.unifi.it/api_help  (keys include metal, pdb, uniprot, etc.)
Example query: https://metalpdb.cerm.unifi.it/api?query=metal:Zn&columns=pdb_code,uniprot,molecule,metal,ligands,donors
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
import pandas as pd


BASE_URL = "https://metalpdb.cerm.unifi.it/api"  # programmatic access endpoint :contentReference[oaicite:1]{index=1}


@dataclass
class MetalPDBClient:
    timeout_sec: float = 60.0
    pause_sec: float = 0.2
    max_retries: int = 3

    def query(
        self,
        query: str,
        columns: Optional[List[str]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Runs a MetalPDB API query.
        - query format: "key:value,key:value" (comma-separated) :contentReference[oaicite:2]{index=2}
        - columns: list of columns you want back (optional) :contentReference[oaicite:3]{index=3}
        """
        params: Dict[str, Any] = {"query": query}
        if columns:
            params["columns"] = ",".join(columns)
        if extra_params:
            params.update(extra_params)

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = requests.get(BASE_URL, params=params, timeout=self.timeout_sec)
                r.raise_for_status()

                # API returns JSON
                return r.json()
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(self.pause_sec * attempt)
                else:
                    raise RuntimeError(f"MetalPDB request failed after {attempt} tries: {e}") from e


def flatten_records(records: Any) -> pd.DataFrame:
    """
    Converts MetalPDB JSON records into a flat table for CSV.
    The API may return either:
      - a list of dicts, or
      - a dict containing a list under some key (depends on endpoint behavior).
    """
    if records is None:
        return pd.DataFrame()

    if isinstance(records, list):
        items = records
    elif isinstance(records, dict):
        # Try common containers
        if "results" in records and isinstance(records["results"], list):
            items = records["results"]
        elif "data" in records and isinstance(records["data"], list):
            items = records["data"]
        else:
            # Fall back: treat the dict itself as one record
            items = [records]
    else:
        items = [{"value": records}]

    # json_normalize flattens nested objects like "metal", "ligands", "donors"
    return pd.json_normalize(items)


def fetch_by_metal(
    metal_symbol: str,
    out_prefix: str = "metalpdb",
) -> None:
    """
    Fetch entries that bind a given metal (e.g., 'Zn', 'Cu', 'Mn', 'Ni', 'Nd', 'Dy').
    """
    client = MetalPDBClient()

    # MetalPDB supports key 'metal' in queries :contentReference[oaicite:4]{index=4}
    q = f"metal:{metal_symbol}"

    # Choose useful columns (you can add/remove)
    cols = [
        "pdb_code",
        "uniprot",
        "molecule",
        "organism",
        "site_type",
        "geometry",
        "coordination",
        "pattern",
        "metal",
        "ligands",
        "donors",
    ]

    data = client.query(query=q, columns=cols)

    # Save raw JSON
    with open(f"{out_prefix}_{metal_symbol}_raw.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # Save flattened CSV
    df = flatten_records(data)
    df.to_csv(f"{out_prefix}_{metal_symbol}.csv", index=False)

    print(f"Saved: {out_prefix}_{metal_symbol}_raw.json")
    print(f"Saved: {out_prefix}_{metal_symbol}.csv")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    # Examples:
    # fetch_by_metal("Zn")
    # fetch_by_metal("Cu")
    # fetch_by_metal("Mn")
    # fetch_by_metal("Ni")
    # fetch_by_metal("Co")
    fetch_by_metal("Au")
