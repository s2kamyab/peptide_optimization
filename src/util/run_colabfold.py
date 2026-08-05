import os
import re
import subprocess
from pathlib import Path
from typing import Optional

def win_to_wsl(p: str) -> str:
    p = str(p).strip().strip('"')
    if p.startswith("/"):
        return p
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", p)
    if not m:
        raise ValueError(f"Not a Windows absolute path: {p}")
    drive = m.group(1).lower()
    rest = m.group(2).replace("\\", "/")
    return f"/mnt/{drive}/{rest}"

def run_colabfold_wsl(
    seq: str,
    out_dir_win: str,
    *,
    conda_env: str = "colabfold",
    miniforge_home_wsl: str = "~/miniforge3",
    threads: int = 8,
    num_models: int = 1,
    num_recycle: int = 1,
    msa_mode: str = "single_sequence",
    model_type: str = "alphafold2",
    cache_dir_win: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    # -------------------------
    # Normalize output directory
    # -------------------------
    out_path = Path(out_dir_win)

    # If caller accidentally passed a fasta path, use its parent directory
    if out_path.suffix.lower() in {".fasta", ".fa", ".faa"}:
        out_path = out_path.parent

    # If output exists and is a file -> error
    if out_path.exists() and out_path.is_file():
        raise ValueError(f"out_dir_win must be a directory, got file: {out_path}")

    out_path.mkdir(parents=True, exist_ok=True)

    # Always write input.fasta inside out_dir
    fasta_win = out_path / "input.fasta"
    fasta_win.write_text(">query\n" + seq.strip().upper() + "\n", encoding="utf-8")

    # Cache dir (keep it stable + inside project by default)
    if cache_dir_win is None:
        cache_dir_win = str((Path.cwd() / ".cache").resolve())
    cache_path = Path(cache_dir_win)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Convert to WSL paths
    fasta_wsl  = win_to_wsl(str(fasta_win.resolve()))
    outdir_wsl = win_to_wsl(str(out_path.resolve()))
    cache_wsl  = win_to_wsl(str(cache_path.resolve()))

    overwrite_flag = "--overwrite-existing-results" if overwrite else ""

    bash_cmd = f"""
        set -euo pipefail

        source {miniforge_home_wsl}/etc/profile.d/conda.sh
        conda activate {conda_env}

        mkdir -p "{cache_wsl}" "{outdir_wsl}"

        export CUDA_VISIBLE_DEVICES=""
        export JAX_PLATFORM_NAME=cpu
        export OMP_NUM_THREADS="{threads}"
        export MKL_NUM_THREADS="{threads}"
        export XDG_CACHE_HOME="{cache_wsl}"
        export HF_HOME="{cache_wsl}/hf"

        colabfold_batch \\
        --msa-mode {msa_mode} \\
        --num-models {num_models} \\
        --model-type {model_type} \\
        --num-recycle {num_recycle} \\
        {overwrite_flag} \\
        "{fasta_wsl}" \\
        "{outdir_wsl}"
        """

    print("WSL cmd:\n", bash_cmd)
    subprocess.run(["wsl.exe", "bash", "-lc", bash_cmd], check=True)

    # Return any produced PDB
    for p in out_path.glob("*.pdb"):
        return str(p)

    # ColabFold typically writes ranked_*.pdb; if not found, raise
    raise RuntimeError(f"No PDB produced in: {out_path}")


# import os
# import subprocess
# from pathlib import Path
# from typing import Optional

def run_colabfold_gpu(
    seq: str,
    out_dir: str,
    *,
    conda_env: str = "colabfold",
    conda_path: Optional[str] = None,
    num_models: int = 1,
    num_recycle: int = 1,
    msa_mode: str = "single_sequence",
    model_type: str = "alphafold2",
    cache_dir: Optional[str] = None,
    overwrite: bool = False,
    gpu_id: int = 0,
) -> str:
    """
    Run ColabFold with GPU acceleration on Windows with NVIDIA GPU.
    
    Args:
        seq: Protein sequence (amino acids)
        out_dir: Output directory (Windows path)
        conda_env: Name of conda environment with ColabFold installed
        conda_path: Path to conda installation (auto-detected if None)
        num_models: Number of models to run (1-5)
        num_recycle: Number of recycling iterations
        msa_mode: MSA mode ('single_sequence' for fast, 'MMseqs2' for full)
        model_type: 'alphafold2' or 'alphafold2_ptm'
        cache_dir: Cache directory for models/params
        overwrite: Overwrite existing results
        gpu_id: GPU device ID (0 for single GPU)
    
    Returns:
        Path to first generated PDB file
    """
    
    # -------------------------
    # Normalize output directory
    # -------------------------
    out_path = Path(out_dir)

    # If caller accidentally passed a fasta path, use its parent directory
    if out_path.suffix.lower() in {".fasta", ".fa", ".faa"}:
        out_path = out_path.parent

    # If output exists and is a file -> error
    if out_path.exists() and out_path.is_file():
        raise ValueError(f"out_dir must be a directory, got file: {out_path}")

    out_path.mkdir(parents=True, exist_ok=True)

    # Always write input.fasta inside out_dir
    fasta_path = out_path / "input.fasta"
    fasta_path.write_text(f">query\n{seq.strip().upper()}\n", encoding="utf-8")

    # Cache dir (keep it stable + inside project by default)
    if cache_dir is None:
        cache_dir = str((Path.cwd() / ".cache").resolve())
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Auto-detect conda path if not provided
    if conda_path is None:
        # Try common locations
        possible_paths = [
            Path.home() / "miniconda3",
            Path.home() / "anaconda3",
            Path.home() / "miniforge3",
            Path("C:/ProgramData/miniconda3"),
            Path("C:/ProgramData/anaconda3"),
        ]
        for p in possible_paths:
            if (p / "Scripts" / "conda.exe").exists():
                conda_path = str(p)
                break
        
        if conda_path is None:
            raise RuntimeError(
                "Could not find conda installation. Please specify conda_path explicitly."
            )
    
    conda_path = Path(conda_path)
    conda_exe = conda_path / "Scripts" / "conda.exe"
    
    if not conda_exe.exists():
        raise FileNotFoundError(f"conda.exe not found at: {conda_exe}")

    # Build command
    cmd = [
        str(conda_exe), "run",
        "-n", conda_env,
        "--no-capture-output",  # Show output in real-time
        "colabfold_batch",
        "--msa-mode", msa_mode,
        "--num-models", str(num_models),
        "--model-type", model_type,
        "--num-recycle", str(num_recycle),
    ]
    
    if overwrite:
        cmd.append("--overwrite-existing-results")
    
    cmd.extend([
        str(fasta_path),
        str(out_path),
    ])

    # Environment variables for GPU
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
        "XDG_CACHE_HOME": str(cache_path),
        "HF_HOME": str(cache_path / "hf"),
        # Force GPU usage
        "JAX_PLATFORM_NAME": "gpu",
        # Enable mixed precision for faster inference on modern GPUs
        "TF_FORCE_GPU_ALLOW_GROWTH": "true",
    })

    print(f"Running ColabFold on GPU {gpu_id}...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Output directory: {out_path}")
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ColabFold failed with exit code {e.returncode}") from e

    # Return first produced PDB (ColabFold typically writes ranked_0.pdb)
    pdb_files = sorted(out_path.glob("*.pdb"))
    if not pdb_files:
        raise RuntimeError(f"No PDB files produced in: {out_path}")
    
    # Prefer ranked files
    ranked = [p for p in pdb_files if "ranked" in p.name]
    if ranked:
        return str(ranked[0])
    
    return str(pdb_files[0])


# ============================================================
# Convenience wrapper for batch processing
# ============================================================

def run_colabfold_batch_gpu(
    sequences: list[str],
    out_base_dir: str,
    **kwargs
) -> list[str]:
    """
    Run ColabFold on multiple sequences.
    
    Args:
        sequences: List of protein sequences
        out_base_dir: Base output directory
        **kwargs: Additional arguments passed to run_colabfold_gpu
    
    Returns:
        List of paths to generated PDB files
    """
    out_base = Path(out_base_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    
    pdb_files = []
    for i, seq in enumerate(sequences):
        out_dir = out_base / f"seq_{i:04d}"
        try:
            pdb = run_colabfold_gpu(seq, str(out_dir), **kwargs)
            pdb_files.append(pdb)
            print(f"✓ Sequence {i+1}/{len(sequences)}: {pdb}")
        except Exception as e:
            print(f"✗ Sequence {i+1}/{len(sequences)} failed: {e}")
            pdb_files.append(None)
    
    return pdb_files


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    # Single sequence
    seq = "WAGAKRLVLRRE"
    pdb = run_colabfold_gpu(
        seq=seq,
        out_dir="./colabfold_output/test",
        num_models=1,
        num_recycle=3,
        msa_mode="single_sequence",  # Fast mode (no MSA search)
        overwrite=True,
        gpu_id=0,  # Use first GPU
    )
    print(f"Generated PDB: {pdb}")
    
    # Batch processing
    sequences = [
        "WAGAKRLVLRRE",
        "MHGKTQATSGTIQS",
        "ACDEFGHIKLMNPQ",
    ]
    
    pdbs = run_colabfold_batch_gpu(
        sequences=sequences,
        out_base_dir="./colabfold_output/batch",
        num_models=1,
        num_recycle=1,
        msa_mode="single_sequence",
    )
    
    for i, pdb in enumerate(pdbs):
        if pdb:
            print(f"Sequence {i}: {pdb}")


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    # Single sequence
    seq = "WAGAKRLVLRRE"
    pdb = run_colabfold_wsl(
        seq=seq,
        out_dir="./colabfold_output/test",
        num_models=1,
        num_recycle=3,
        msa_mode="single_sequence",  # Fast mode (no MSA search)
        overwrite=True,
        gpu_id=0,  # Use first GPU
    )
    print(f"Generated PDB: {pdb}")
    
    # Batch processing
    sequences = [
        "WAGAKRLVLRRE",
        "MHGKTQATSGTIQS",
        "ACDEFGHIKLMNPQ",
    ]
    
    pdbs = run_colabfold_batch_gpu(
        sequences=sequences,
        out_base_dir="./colabfold_output/batch",
        num_models=1,
        num_recycle=1,
        msa_mode="single_sequence",
    )
    
    for i, pdb in enumerate(pdbs):
        if pdb:
            print(f"Sequence {i}: {pdb}")