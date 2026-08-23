import re
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from transformers.models.esm.openfold_utils.protein import Protein as OFProtein, to_pdb


_ESMFOLD_MODEL = None
_ESMFOLD_TOKENIZER = None
_ESMFOLD_MODEL_NAME = None


def _sanitize_sequence(seq: str) -> str:
    seq = re.sub(r"\s+", "", seq).upper()
    if not seq:
        raise ValueError("Sequence is empty.")

    # Standard protein alphabet + X for unknown
    if not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWYX]+", seq):
        raise ValueError(
            "Sequence contains invalid characters. "
            "Allowed amino acids: ACDEFGHIKLMNPQRSTVWYX"
        )
    return seq


def _load_esmfold_model(
    model_name: str = "facebook/esmfold_v1",
    cache_dir: Optional[str] = None,
    device: Optional[str] = None,
):
    global _ESMFOLD_MODEL, _ESMFOLD_TOKENIZER, _ESMFOLD_MODEL_NAME

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if (
        _ESMFOLD_MODEL is not None
        and _ESMFOLD_TOKENIZER is not None
        and _ESMFOLD_MODEL_NAME == model_name
    ):
        return _ESMFOLD_TOKENIZER, _ESMFOLD_MODEL, device

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    model = EsmForProteinFolding.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
    )

    model = model.to(device)
    model.eval()

    # Optional GPU speed/memory optimization commonly used for ESMFold
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            model.esm = model.esm.half()
        except Exception:
            pass

    _ESMFOLD_TOKENIZER = tokenizer
    _ESMFOLD_MODEL = model
    _ESMFOLD_MODEL_NAME = model_name

    return tokenizer, model, device

def _convert_outputs_to_pdb(outputs) -> str:
    """
    Convert Hugging Face ESMFold output to a PDB string.
    """
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)

    outputs_np = {}
    for k, v in outputs.items():
        if torch.is_tensor(v):
            outputs_np[k] = v.detach().cpu().numpy()
        else:
            outputs_np[k] = v

    final_atom_positions = final_atom_positions.detach().cpu().numpy()
    final_atom_mask = outputs_np["atom37_atom_exists"]

    pdb_strings = []
    batch_size = outputs_np["aatype"].shape[0]

    for i in range(batch_size):
        aa = outputs_np["aatype"][i]                       # (L,)
        pred_pos = final_atom_positions[i]                # (L, 37, 3)
        mask = final_atom_mask[i]                         # (L, 37)
        resid = outputs_np["residue_index"][i] + 1        # (L,)

        # pLDDT is usually per-residue: shape (L,)
        plddt = outputs_np["plddt"][i]

        # If extra trailing dimension exists, squeeze it safely
        if getattr(plddt, "ndim", 0) > 1:
            plddt = plddt.squeeze()

        # Expand per-residue pLDDT to per-atom b-factors: (L, 37)
        if plddt.ndim == 1:
            b_factors = plddt[:, None] * mask
        elif plddt.ndim == 2 and plddt.shape == mask.shape:
            b_factors = plddt
        else:
            raise ValueError(
                f"Unexpected plddt shape {plddt.shape}; expected (L,) or (L, 37)"
            )

        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=b_factors,
            chain_index=outputs_np["chain_index"][i] if "chain_index" in outputs_np else None,
        )
        pdb_strings.append(to_pdb(pred))

    return "".join(pdb_strings)
# def _convert_outputs_to_pdb(outputs) -> str:
#     """
#     Convert Hugging Face ESMFold output to a PDB string.
#     """
#     final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)

#     outputs_np = {}
#     for k, v in outputs.items():
#         if torch.is_tensor(v):
#             outputs_np[k] = v.detach().cpu().numpy()
#         else:
#             outputs_np[k] = v

#     final_atom_positions = final_atom_positions.detach().cpu().numpy()
#     final_atom_mask = outputs_np["atom37_atom_exists"]

#     pdb_strings = []
#     batch_size = outputs_np["aatype"].shape[0]

#     for i in range(batch_size):
#         aa = outputs_np["aatype"][i]
#         pred_pos = final_atom_positions[i]
#         mask = final_atom_mask[i]
#         resid = outputs_np["residue_index"][i] + 1

#         plddt = outputs_np["plddt"][i]
#         if plddt.ndim > 1:
#             plddt = plddt[..., 0]

#         pred = OFProtein(
#             aatype=aa,
#             atom_positions=pred_pos,
#             atom_mask=mask,
#             residue_index=resid,
#             b_factors=plddt,
#             chain_index=outputs_np["chain_index"][i] if "chain_index" in outputs_np else None,
#         )
#         pdb_strings.append(to_pdb(pred))

#     return "".join(pdb_strings)


def run_esmfold_hf(
    seq: str,
    out_dir: str,
    *,
    model_name: str = "facebook/esmfold_v1",
    cache_dir: Optional[str] = None,
    overwrite: bool = False,
    num_recycles: Optional[int] = 3,
    chunk_size: Optional[int] = None,
    device: Optional[str] = None,
    pdb_filename: str = "ranked_0.pdb",
) -> str:
    """
    Predict a protein structure with Hugging Face ESMFold and save a PDB file.

    Args:
        seq: Protein sequence
        out_dir: Output directory
        model_name: HF model name, typically 'facebook/esmfold_v1'
        cache_dir: HF cache directory
        overwrite: Whether to overwrite existing PDB
        num_recycles: Number of ESMFold recycle iterations
        chunk_size: Optional trunk chunk size for lower memory usage
        device: 'cuda', 'cpu', or None for auto
        pdb_filename: Name of output PDB file

    Returns:
        Path to generated PDB file
    """
    seq = _sanitize_sequence(seq)

    out_path = Path(out_dir)
    if out_path.suffix.lower() in {".fasta", ".fa", ".faa", ".pdb"}:
        out_path = out_path.parent

    if out_path.exists() and out_path.is_file():
        raise ValueError(f"out_dir must be a directory, got file: {out_path}")

    out_path.mkdir(parents=True, exist_ok=True)
    pdb_path = out_path / pdb_filename

    if pdb_path.exists() and not overwrite:
        return str(pdb_path)

    tokenizer, model, device = _load_esmfold_model(
        model_name=model_name,
        cache_dir=cache_dir,
        device=device,
    )

    if chunk_size is not None:
        try:
            model.trunk.set_chunk_size(chunk_size)
        except Exception:
            pass

    inputs = tokenizer(
        [seq],
        return_tensors="pt",
        add_special_tokens=False,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs, num_recycles=num_recycles)

    pdb_str = _convert_outputs_to_pdb(outputs)

    if not pdb_str.strip():
        raise RuntimeError("ESMFold ran but produced an empty PDB string.")

    pdb_path.write_text(pdb_str, encoding="utf-8")
    return str(pdb_path)