import subprocess
import os
import pathlib
import csv
from typing import Optional

# project_root = pathlib.Path(__file__).resolve().parent
a3d_path =  "../../py27venv/Scripts/aggrescan.exe"
print("Using A3D path:", a3d_path)

def _csv_has_numeric_score(csv_path: pathlib.Path) -> bool:
    try:
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return False

            score_cols = [
                c for c in ["score", "a3d_score", "Aggrescan3D_score", "value"]
                if c in reader.fieldnames
            ]
            if not score_cols:
                return False

            for row in reader:
                for col in score_cols:
                    try:
                        float(row.get(col, ""))
                        return True
                    except (TypeError, ValueError):
                        continue
    except OSError:
        return False

    return False

def run_a3d_on_pdb(pdb_path: pathlib.Path, out_dir: pathlib.Path, verbosity: int = 3) -> Optional[pathlib.Path]:
    print(f"Running A3D on PDB: {pdb_path} -> {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    cmd = [str(a3d_path), "-i", str(pdb_path), "-w", str(out_dir), "-v", str(verbosity)]
    p = subprocess.run(cmd, capture_output=True, text=True)

    (out_dir / "a3d_stdout.txt").write_text(p.stdout or "", encoding="utf-8")
    (out_dir / "a3d_stderr.txt").write_text(p.stderr or "", encoding="utf-8")

    if p.returncode != 0:
        print(f"A3D failed with return code {p.returncode}; see logs in {out_dir}")
        return None

    # Success case:
    # If you know the exact output filename produced by aggrescan.exe, return that path.
    # Otherwise you can search for the newest .csv:
    csvs = sorted(out_dir.glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
    for csv_path in csvs:
        if _csv_has_numeric_score(csv_path):
            return csv_path

    print(f"A3D did not produce a CSV with numeric scores; see logs in {out_dir}")
    return None
