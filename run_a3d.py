import subprocess
import os
import pathlib
import csv

project_root = pathlib.Path(__file__).resolve().parent
a3d_path = project_root / "py27venv/Scripts/aggrescan.exe"
print("Using A3D path:", a3d_path)

def _write_empty_csv(out_dir: pathlib.Path, filename: str = "aggrescan3d_scores.csv") -> pathlib.Path:
    """
    Create an empty CSV (header-only) so pipelines don't crash when A3D fails.
    Adjust headers to match what your downstream expects.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / filename

    # If your downstream expects specific columns, keep them here.
    headers = ["residue_index", "residue", "score"]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

    return out_csv

def run_a3d_on_pdb(pdb_path: pathlib.Path, out_dir: pathlib.Path, verbosity: int = 3) -> pathlib.Path:
    print(f"Running A3D on PDB: {pdb_path} -> {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    cmd = [str(a3d_path), "-i", str(pdb_path), "-w", str(out_dir), "-v", str(verbosity)]
    p = subprocess.run(cmd, capture_output=True, text=True)

    # If it failed: write empty CSV and return it (no exception)
    if p.returncode != 0:
        empty_csv = _write_empty_csv(out_dir, filename="aggrescan3d_scores.csv")
        # Optional: write logs for debugging
        (out_dir / "a3d_error_stderr.txt").write_text(p.stderr or "", encoding="utf-8")
        (out_dir / "a3d_error_stdout.txt").write_text(p.stdout or "", encoding="utf-8")
        return empty_csv

    # Success case:
    # If you know the exact output filename produced by aggrescan.exe, return that path.
    # Otherwise you can search for the newest .csv:
    csvs = sorted(out_dir.glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not csvs:
        # A3D returned success but produced no CSV: still create empty for safety
        return _write_empty_csv(out_dir, filename="aggrescan3d_scores.csv")

    return csvs[0]
