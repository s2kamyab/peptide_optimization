from __future__ import annotations

"""
BO-ready GRU-VAE pretraining for downstream RealNVP or latent diffusion.

This is a corrected replacement for the earlier H64/Z64 latent-conditioned GRU-VAE
pretraining script.  It keeps the same encoder/decoder parameter names and shapes
so the existing RealNVP and latent-diffusion fine-tuning scripts can still load the
checkpoint, but changes the *training objective* to make the latent mu space more
useful for Bayesian optimization.

Main fixes versus the earlier script:
  1. Decode from encoder mu by default, not noisy z, because downstream RealNVP and
     latent diffusion both operate on encoder mu.
  2. Apply decoder-input dropout during teacher forcing so the decoder cannot ignore
     the latent vector and rely only on previous ground-truth tokens.
  3. Add a free-running autoregressive reconstruction loss so training better matches
     generation/BO-time decoding.
  4. Add minimum-information and latent-spread regularizers to reduce KL/mu collapse
     and make the latent space smoother for RealNVP/diffusion.
  5. Use PDB/exact-peptide connected-component splitting plus a strict Hamming-distance holdout filter to prevent structural/near-neighbor leakage.
  6. Add latent covariance decorrelation and global participation-ratio/PCA diagnostics to prevent correlated effective-dimensionality collapse.
  7. Save a BO-aware checkpoint that balances reconstruction with latent geometry.

Checkpoint compatibility:
  - checkpoint_type is still a pretrained GRU-VAE checkpoint.
  - model_state_dict keys for enc.* and dec.* are compatible with your current
    RealNVP and latent-diffusion fine-tuning scripts.
"""

import argparse
import glob
import hashlib
import json
import math
import os
import random
import itertools
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple, Sequence, Set

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_I = {a: i for i, a in enumerate(AA)}
I_TO_AA = {i: a for i, a in enumerate(AA)}
SEQ_LEN = 10
VOCAB = 20


@dataclass
class ModelConfig:
    hidden_size: int = 64
    latent_dim: int = 64
    n_layers: int = 2
    dropout: float = 0.0
    decoder_conditioning: str = "concat_z_at_every_decoder_step"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_peptide(x: object) -> Optional[str]:
    pep = str(x).strip().upper()
    if len(pep) != SEQ_LEN:
        return None
    if any(ch not in AA_TO_I for ch in pep):
        return None
    return pep


def onehot_encode_peptides(peptides: List[str]) -> torch.Tensor:
    x = torch.zeros((len(peptides), SEQ_LEN, VOCAB), dtype=torch.float32)
    for n, pep in enumerate(peptides):
        pep2 = clean_peptide(pep)
        if pep2 is None:
            raise ValueError(f"Invalid peptide: {pep!r}")
        for t, ch in enumerate(pep2):
            x[n, t, AA_TO_I[ch]] = 1.0
    return x


def levenshtein_edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + int(ca != cb)))
        previous = current
    return int(previous[-1])


class _DisjointSet:
    """Small union-find used to keep exact peptides and PDB groups in one split."""

    def __init__(self):
        self.parent: List[int] = []
        self.rank: List[int] = []

    def add(self) -> int:
        i = len(self.parent)
        self.parent.append(i)
        self.rank.append(0)
        return i

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return ra


def _stable_hash_fraction(token: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{token}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) / float(2**64)


def _detect_group_column(files: Sequence[str], peptide_col: str, requested: str) -> str:
    header = pd.read_csv(files[0], nrows=0).columns.tolist()
    if requested and requested.lower() != "auto":
        if requested not in header:
            raise KeyError(
                f"Requested --group-col={requested!r} is not in shard columns. "
                f"Available columns include: {header[:60]}"
            )
        return requested
    candidates = [
        "pdb_id", "pdb", "pdb_code", "pdbid", "pdb_entry", "entry_id",
        "structure_id", "rcsb_id", "source_pdb", "pdb_identifier",
    ]
    for c in candidates:
        if c in header:
            return c
    raise KeyError(
        "Could not auto-detect a PDB/group column. Pass --group-col explicitly. "
        f"Available columns include: {header[:80]}"
    )


def _peptide_code(pep: str) -> int:
    """Compact base-21 encoding; 0 is reserved for a masked residue."""
    code = 0
    for aa in pep:
        code = code * 21 + (AA_TO_I[aa] + 1)
    return code


def _masked_signature_array(peptides: Sequence[str], max_hamming: int) -> np.ndarray:
    """
    Return uint64 signatures sufficient to detect Hamming distance <= max_hamming.
    For max_hamming=2, masking every pair of positions is sufficient to detect
    distances 0, 1, or 2. The mask identity is packed into the high bits.
    """
    if max_hamming <= 0 or len(peptides) == 0:
        return np.asarray([], dtype=np.uint64)
    if max_hamming > 2:
        raise ValueError("Strict Hamming filtering currently supports max distance <= 2.")

    k = int(max_hamming)
    masks = list(itertools.combinations(range(SEQ_LEN), k))
    pow21 = [21 ** (SEQ_LEN - 1 - i) for i in range(SEQ_LEN)]
    base_span = 21 ** SEQ_LEN
    out = np.empty(len(peptides) * len(masks), dtype=np.uint64)
    q = 0
    for pep in peptides:
        vals = [AA_TO_I[a] + 1 for a in pep]
        full = sum(v * pow21[i] for i, v in enumerate(vals))
        for mask_id, pos_tuple in enumerate(masks):
            masked = full
            for pos in pos_tuple:
                masked -= vals[pos] * pow21[pos]
            out[q] = np.uint64(masked + mask_id * base_span)
            q += 1
    return out


def _filter_near_neighbors(
    reference_peptides: Set[str],
    query_peptides: Set[str],
    max_hamming: int,
) -> Tuple[Set[str], Set[str]]:
    """Return (kept, excluded) query peptides based on Hamming proximity."""
    if max_hamming < 0 or not reference_peptides or not query_peptides:
        return set(query_peptides), set()
    if max_hamming == 0:
        excluded = set(query_peptides).intersection(reference_peptides)
        return set(query_peptides) - excluded, excluded

    ref_sig = _masked_signature_array(sorted(reference_peptides), max_hamming)
    ref_sig = np.unique(ref_sig)
    ref_sig.sort()

    kept: Set[str] = set()
    excluded: Set[str] = set()
    masks_per_pep = math.comb(SEQ_LEN, max_hamming)
    q_peps = sorted(query_peptides)
    q_sig = _masked_signature_array(q_peps, max_hamming)
    q_sig = q_sig.reshape(len(q_peps), masks_per_pep)

    for pep, sigs in zip(q_peps, q_sig):
        idx = np.searchsorted(ref_sig, sigs)
        hit = False
        for sig, j in zip(sigs, idx):
            if j < len(ref_sig) and ref_sig[j] == sig:
                hit = True
                break
        (excluded if hit else kept).add(pep)
    return kept, excluded


def prepare_leakage_safe_split(files: Sequence[str], args: argparse.Namespace) -> Dict[str, Set[str]]:
    """
    Build a split once before training.

    Core guarantee:
      * all rows from a PDB/group stay together;
      * identical peptide sequences stay together even if observed in multiple PDBs;
      * therefore exact peptide overlap and PDB-group overlap are both zero.

    This is implemented by assigning connected components of the bipartite
    peptide<->PDB graph to train/validation/test.

    Optional strict holdout filtering then removes validation/test peptides that
    are within --holdout-min-hamming substitutions of the development data.
    With the default 3, holdout peptides at Hamming distance 0, 1, or 2 are
    excluded from evaluation rather than silently leaking near-neighbors.
    """
    group_col = _detect_group_column(files, args.peptide_col, args.group_col)
    print(f"Leakage-safe splitting group column: {group_col}")

    dsu = _DisjointSet()
    node_id: Dict[str, int] = {}
    peptide_nodes: Dict[str, int] = {}
    group_nodes: Dict[str, int] = {}
    group_to_peptides: Dict[str, Set[str]] = {}

    def get_node(token: str) -> int:
        if token not in node_id:
            node_id[token] = dsu.add()
        return node_id[token]

    n_rows = 0
    for csv_path in files:
        for chunk in pd.read_csv(
            csv_path,
            usecols=[args.peptide_col, group_col],
            chunksize=args.chunksize,
        ):
            for pep_raw, grp_raw in zip(chunk[args.peptide_col].tolist(), chunk[group_col].tolist()):
                pep = clean_peptide(pep_raw)
                if pep is None or pd.isna(grp_raw):
                    continue
                grp = str(grp_raw).strip().upper()
                if not grp:
                    continue
                ptoken = f"P:{pep}"
                gtoken = f"G:{grp}"
                pnode = get_node(ptoken)
                gnode = get_node(gtoken)
                dsu.union(pnode, gnode)
                peptide_nodes[pep] = pnode
                group_nodes[grp] = gnode
                group_to_peptides.setdefault(grp, set()).add(pep)
                n_rows += 1

    if not peptide_nodes:
        raise RuntimeError("No valid peptide/group records found while preparing split.")

    # Stable canonical token per connected component.
    canonical: Dict[int, str] = {}
    for token, nid in node_id.items():
        root = dsu.find(nid)
        if root not in canonical or token < canonical[root]:
            canonical[root] = token

    component_peptides: Dict[int, Set[str]] = {}
    component_groups: Dict[int, Set[str]] = {}
    for pep, nid in peptide_nodes.items():
        component_peptides.setdefault(dsu.find(nid), set()).add(pep)
    for grp, nid in group_nodes.items():
        component_groups.setdefault(dsu.find(nid), set()).add(grp)

    split_sets: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}
    split_groups: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}

    val_frac = float(args.validation_fraction)
    test_frac = float(args.test_fraction)
    if val_frac <= 0 or test_frac < 0 or val_frac + test_frac >= 1:
        raise ValueError("Require validation_fraction > 0, test_fraction >= 0, and sum < 1.")

    for root, peps in component_peptides.items():
        u = _stable_hash_fraction(canonical[root], args.validation_split_seed)
        if u < test_frac:
            split = "test"
        elif u < test_frac + val_frac:
            split = "val"
        else:
            split = "train"
        split_sets[split].update(peps)
        split_groups[split].update(component_groups.get(root, set()))

    # Strict near-neighbor exclusion for validation and test.
    max_leaky_distance = int(args.holdout_min_hamming) - 1
    excluded_val: Set[str] = set()
    excluded_test: Set[str] = set()
    if args.holdout_min_hamming > 0:
        val_kept, excluded_val = _filter_near_neighbors(
            split_sets["train"], split_sets["val"], max_leaky_distance
        )
        split_sets["val"] = val_kept

        # Test must be separated from everything used to fit/select the model.
        dev_reference = set(split_sets["train"]) | set(split_sets["val"])
        test_kept, excluded_test = _filter_near_neighbors(
            dev_reference, split_sets["test"], max_leaky_distance
        )
        split_sets["test"] = test_kept

    # Audit guarantees.
    exact_tv = len(split_sets["train"] & split_sets["val"])
    exact_tt = len(split_sets["train"] & split_sets["test"])
    group_tv = len(split_groups["train"] & split_groups["val"])
    group_tt = len(split_groups["train"] & split_groups["test"])
    if exact_tv or exact_tt or group_tv or group_tt:
        raise RuntimeError(
            "Leakage-safe split invariant failed: "
            f"train/val exact={exact_tv}, train/test exact={exact_tt}, "
            f"train/val groups={group_tv}, train/test groups={group_tt}"
        )

    # Save a compact split manifest and audit report.
    os.makedirs(args.out_dir, exist_ok=True)
    manifest_rows = []
    for split, peps in split_sets.items():
        manifest_rows.extend({"peptide": p, "split": split} for p in sorted(peps))
    manifest_rows.extend({"peptide": p, "split": "excluded_near_train"} for p in sorted(excluded_val))
    manifest_rows.extend({"peptide": p, "split": "excluded_near_development"} for p in sorted(excluded_test))
    pd.DataFrame(manifest_rows).to_csv(
        os.path.join(args.out_dir, "pretraining_leakage_safe_split_manifest.csv"), index=False
    )

    component_sizes = [len(v) for v in component_peptides.values()]
    audit = {
        "split_method": "connected_component_hash_of_peptide_PDB_bipartite_graph",
        "group_col": group_col,
        "rows_scanned": int(n_rows),
        "unique_peptides": int(len(peptide_nodes)),
        "unique_groups": int(len(group_nodes)),
        "connected_components": int(len(component_peptides)),
        "largest_component_peptides": int(max(component_sizes)),
        "validation_fraction_target": val_frac,
        "test_fraction_target": test_frac,
        "holdout_min_hamming": int(args.holdout_min_hamming),
        "train_peptides": len(split_sets["train"]),
        "validation_peptides": len(split_sets["val"]),
        "test_peptides": len(split_sets["test"]),
        "excluded_validation_near_train": len(excluded_val),
        "excluded_test_near_development": len(excluded_test),
        "exact_train_validation_overlap": exact_tv,
        "exact_train_test_overlap": exact_tt,
        "PDB_group_train_validation_overlap": group_tv,
        "PDB_group_train_test_overlap": group_tt,
    }
    with open(os.path.join(args.out_dir, "pretraining_split_audit.json"), "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    print("Leakage-safe split audit:")
    for k, v in audit.items():
        print(f"  {k}: {v}")

    return split_sets


class PeptideCSVIterable(IterableDataset):
    """Stream peptides using a precomputed leakage-safe split membership set."""

    def __init__(
        self,
        files: List[str],
        allowed_peptides: Set[str],
        peptide_col: str = "peptide_len10",
        chunksize: int = 8192,
        shuffle_files: bool = True,
        deduplicate_within_epoch: bool = True,
    ):
        self.files = list(files)
        self.allowed_peptides = set(allowed_peptides)
        self.peptide_col = peptide_col
        self.chunksize = int(chunksize)
        self.shuffle_files = bool(shuffle_files)
        self.deduplicate_within_epoch = bool(deduplicate_within_epoch)

    def __iter__(self) -> Iterable[torch.Tensor]:
        files = list(self.files)
        if self.shuffle_files:
            random.shuffle(files)

        seen: Set[str] = set()
        for csv_path in files:
            for chunk in pd.read_csv(csv_path, usecols=[self.peptide_col], chunksize=self.chunksize):
                peptides: List[str] = []
                for value in chunk[self.peptide_col].tolist():
                    pep = clean_peptide(value)
                    if pep is None or pep not in self.allowed_peptides:
                        continue
                    if self.deduplicate_within_epoch and pep in seen:
                        continue
                    seen.add(pep)
                    peptides.append(pep)
                if peptides:
                    yield onehot_encode_peptides(peptides)


class GRUEncoder(nn.Module):
    def __init__(self, hidden_size: int, latent_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Linear(VOCAB, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.to_mu = nn.Linear(hidden_size, latent_dim)
        self.to_logvar = nn.Linear(hidden_size, latent_dim)

    def forward(self, x_onehot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.in_proj(x_onehot)
        out_top, h_n = self.gru(h)
        h_last = h_n[-1]
        mu = self.to_mu(h_last)
        # Keep the clamp for checkpoint compatibility, but the new loss discourages
        # pathological collapse to the lower bound.
        logvar = self.to_logvar(h_last).clamp(min=-8.0, max=8.0)
        return mu, logvar, out_top


class LatentConditionedGRUDecoder(nn.Module):
    """GRU decoder where z is concatenated to the token embedding at every step."""

    def __init__(self, hidden_size: int, latent_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.token_embed = nn.Linear(VOCAB, hidden_size)
        self.z_to_h = nn.Linear(latent_dim, n_layers * hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size + latent_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.to_logits = nn.Linear(hidden_size, VOCAB)

    def forward(self, z: torch.Tensor, x_onehot: torch.Tensor, input_dropout: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = z.size(0)
        x_shift = torch.zeros_like(x_onehot)
        x_shift[:, 1:, :] = x_onehot[:, :-1, :]
        if self.training and input_dropout > 0.0:
            keep = (torch.rand(x_shift.shape[:2], device=x_shift.device) > float(input_dropout)).float().unsqueeze(-1)
            # Never drop the artificial BOS zero vector at position 0; it is already zero.
            keep[:, 0, :] = 1.0
            x_shift = x_shift * keep
        emb = self.token_embed(x_shift)
        z_repeat = z.unsqueeze(1).expand(-1, x_onehot.size(1), -1)
        decoder_input = torch.cat([emb, z_repeat], dim=-1)
        h0 = self.z_to_h(z).view(self.gru.num_layers, batch_size, self.gru.hidden_size)
        out_top, _ = self.gru(decoder_input, h0)
        logits = self.to_logits(out_top)
        return logits, out_top

    def initial_hidden(self, z: torch.Tensor) -> torch.Tensor:
        return self.z_to_h(z).view(self.gru.num_layers, z.size(0), self.gru.hidden_size)

    def step(self, z: torch.Tensor, current_token: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.token_embed(current_token)
        z_step = z.unsqueeze(1)
        dec_input = torch.cat([emb, z_step], dim=-1)
        out_step, h = self.gru(dec_input, h)
        logits = self.to_logits(out_step[:, -1, :])
        return logits, h


class GRUVAE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.enc = GRUEncoder(cfg.hidden_size, cfg.latent_dim, cfg.n_layers, cfg.dropout)
        self.dec = LatentConditionedGRUDecoder(cfg.hidden_size, cfg.latent_dim, cfg.n_layers, cfg.dropout)

    def reparam(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x_onehot: torch.Tensor, use_mu_for_recon: bool = True, input_dropout: float = 0.0) -> Dict[str, torch.Tensor]:
        mu, logvar, enc_out = self.enc(x_onehot)
        z = mu if use_mu_for_recon else self.reparam(mu, logvar)
        logits, dec_out = self.dec(z, x_onehot, input_dropout=input_dropout)
        return {"mu": mu, "logvar": logvar, "z": z, "logits": logits, "enc_out": enc_out, "dec_out": dec_out}


@torch.no_grad()
def decode_argmax_from_logits(logits: torch.Tensor) -> List[str]:
    idx = logits.argmax(dim=-1).detach().cpu().tolist()
    return ["".join(I_TO_AA[int(i)] for i in row) for row in idx]


def straight_through_argmax_onehot(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    tau = max(float(temperature), 1e-6)
    probs = torch.softmax(logits / tau, dim=-1)
    hard_idx = probs.argmax(dim=-1)
    hard = F.one_hot(hard_idx, num_classes=VOCAB).to(dtype=probs.dtype)
    return hard + probs - probs.detach()


def autoregressive_free_decode_st(model: GRUVAE, z: torch.Tensor, temperature: float) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = z.size(0)
    h = model.dec.initial_hidden(z)
    current_token = torch.zeros(batch_size, 1, VOCAB, dtype=z.dtype, device=z.device)
    logits_steps = []
    hard_steps = []
    for _ in range(SEQ_LEN):
        logits, h = model.dec.step(z, current_token, h)
        x_st = straight_through_argmax_onehot(logits, temperature=temperature)
        logits_steps.append(logits)
        hard_steps.append(x_st)
        current_token = x_st.unsqueeze(1)
    return torch.stack(logits_steps, dim=1), torch.stack(hard_steps, dim=1)


@torch.no_grad()
def autoregressive_free_decode_argmax(
    model: GRUVAE,
    z: torch.Tensor,
    temperature: float = 1.0,
) -> List[str]:
    """
    Free-running argmax decode for diagnostics.

    Preserve and restore the model's train/eval state. The previous version
    called model.eval() and left the model in evaluation mode. On CUDA/cuDNN,
    the next training forward through nn.GRU was then executed in eval mode,
    causing:
        RuntimeError: cudnn RNN backward can only be called in training mode
    when loss.backward() was called.
    """
    was_training = model.training
    model.eval()
    try:
        batch_size = z.size(0)
        h = model.dec.initial_hidden(z)
        current_token = torch.zeros(
            batch_size, 1, VOCAB, dtype=z.dtype, device=z.device
        )
        out_tokens = [[] for _ in range(batch_size)]

        for _ in range(SEQ_LEN):
            logits, h = model.dec.step(z, current_token, h)
            idx = torch.softmax(
                logits / max(float(temperature), 1e-6), dim=-1
            ).argmax(dim=-1)
            current_token = (
                F.one_hot(idx, num_classes=VOCAB)
                .to(dtype=z.dtype)
                .unsqueeze(1)
            )
            for b in range(batch_size):
                out_tokens[b].append(I_TO_AA[int(idx[b].item())])

        return ["".join(row) for row in out_tokens]
    finally:
        model.train(was_training)


def kl_per_dim(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # [B, D]
    return -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())


def latent_covariance_regularizer(mu: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    VICReg-style decorrelation penalty on standardized encoder means.

    The previous spread loss only forced each coordinate to have non-zero variance.
    It did not stop all 64 coordinates from becoming highly correlated.  This loss
    directly penalizes off-diagonal covariance/correlation and therefore attacks the
    observed participation-ratio collapse.
    """
    if mu.size(0) < 2:
        zero = mu.new_zeros(())
        return zero, zero
    z = mu - mu.mean(dim=0, keepdim=True)
    z = z / z.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    cov = (z.T @ z) / float(mu.size(0))
    offdiag = cov - torch.diag(torch.diagonal(cov))
    cov_loss = offdiag.pow(2).sum() / max(1, mu.size(1) * (mu.size(1) - 1))
    mean_abs_corr = offdiag.abs().sum() / max(1, mu.size(1) * (mu.size(1) - 1))
    return cov_loss, mean_abs_corr


def latent_geometry_diagnostics(mu: torch.Tensor, eps: float = 1e-8) -> Dict[str, float]:
    """Global covariance-spectrum diagnostics for validation/test encoder means."""
    if mu.ndim != 2 or mu.size(0) < 2:
        return {
            "effective_dim_pr": float("nan"),
            "effective_dim_pr_standardized": float("nan"),
            "pc1_variance_fraction": float("nan"),
            "pcs90": float("nan"),
            "pcs95": float("nan"),
            "mean_abs_offdiag_corr": float("nan"),
            "max_abs_offdiag_corr": float("nan"),
        }

    def spectrum_stats(x: torch.Tensor):
        x = x.double()
        x = x - x.mean(dim=0, keepdim=True)
        cov = (x.T @ x) / max(1, x.size(0) - 1)
        eig = torch.linalg.eigvalsh(cov).clamp_min(0.0)
        total = eig.sum().clamp_min(eps)
        pr = (total * total / eig.pow(2).sum().clamp_min(eps)).item()
        eig_desc = torch.flip(eig, dims=[0])
        frac = eig_desc / total
        cum = torch.cumsum(frac, dim=0)
        pc1 = float(frac[0].item())
        pcs90 = int(torch.searchsorted(cum, torch.tensor(0.90, dtype=cum.dtype, device=cum.device)).item() + 1)
        pcs95 = int(torch.searchsorted(cum, torch.tensor(0.95, dtype=cum.dtype, device=cum.device)).item() + 1)
        return pr, pc1, pcs90, pcs95

    raw_pr, pc1, pcs90, pcs95 = spectrum_stats(mu)

    z = mu.double() - mu.double().mean(dim=0, keepdim=True)
    z = z / z.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    std_pr, _, _, _ = spectrum_stats(z)
    corr = (z.T @ z) / float(z.size(0))
    off = corr - torch.diag(torch.diagonal(corr))
    denom = max(1, z.size(1) * (z.size(1) - 1))
    return {
        "effective_dim_pr": float(raw_pr),
        "effective_dim_pr_standardized": float(std_pr),
        "pc1_variance_fraction": float(pc1),
        "pcs90": float(pcs90),
        "pcs95": float(pcs95),
        "mean_abs_offdiag_corr": float(off.abs().sum().item() / denom),
        "max_abs_offdiag_corr": float(off.abs().max().item()),
    }


def bo_ready_vae_loss(x_onehot: torch.Tensor, out: Dict[str, torch.Tensor], model: GRUVAE, args: argparse.Namespace, beta: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    target = x_onehot.argmax(dim=-1)

    teacher_recon = F.cross_entropy(out["logits"].reshape(-1, VOCAB), target.reshape(-1), reduction="mean")
    pred_teacher = out["logits"].argmax(dim=-1)
    teacher_token_acc = (pred_teacher == target).float().mean()

    free_logits, _ = autoregressive_free_decode_st(model, out["mu"], temperature=args.free_decode_temperature)
    free_recon = F.cross_entropy(free_logits.reshape(-1, VOCAB), target.reshape(-1), reduction="mean")
    pred_free = free_logits.argmax(dim=-1)
    free_token_acc = (pred_free == target).float().mean()

    kld = kl_per_dim(out["mu"], out["logvar"])
    kl_total = kld.sum(dim=-1).mean().clamp(min=0.0)
    kl_dim_mean = kld.mean(dim=0)
    kl_per_dim_mean = kl_dim_mean.mean().clamp(min=0.0)

    # Encourage a non-degenerate latent code without making KL dominate reconstruction.
    min_info_loss = F.relu(float(args.min_kl_per_dim) - kl_dim_mean).mean()

    # Encourage active latent dimensions and a diffusion/flow-friendly scale.
    mu_batch_std = out["mu"].std(dim=0, unbiased=False)
    latent_spread_loss = ((mu_batch_std - float(args.target_mu_std)) ** 2).mean()
    mu_mean_loss = out["mu"].mean(dim=0).pow(2).mean()
    latent_cov_loss, mean_abs_latent_corr = latent_covariance_regularizer(out["mu"])

    # Discourage logvar from sticking at the lower clamp, but keep this weak.
    logvar_floor_loss = F.relu(float(args.logvar_floor_target) - out["logvar"]).mean()

    loss = (
        teacher_recon
        + float(args.free_decode_loss_weight) * free_recon
        + float(beta) * kl_total
        + float(args.min_info_loss_weight) * min_info_loss
        + float(args.latent_spread_loss_weight) * latent_spread_loss
        + float(args.mu_mean_loss_weight) * mu_mean_loss
        + float(args.latent_cov_loss_weight) * latent_cov_loss
        + float(args.logvar_floor_loss_weight) * logvar_floor_loss
    )

    return loss, {
        "loss": float(loss.detach().cpu()),
        "teacher_recon": float(teacher_recon.detach().cpu()),
        "free_recon": float(free_recon.detach().cpu()),
        "kl": float(kl_total.detach().cpu()),
        "kl_per_dim_mean": float(kl_per_dim_mean.detach().cpu()),
        "min_info_loss": float(min_info_loss.detach().cpu()),
        "latent_spread_loss": float(latent_spread_loss.detach().cpu()),
        "mu_mean_loss": float(mu_mean_loss.detach().cpu()),
        "latent_cov_loss": float(latent_cov_loss.detach().cpu()),
        "mean_abs_latent_corr_batch": float(mean_abs_latent_corr.detach().cpu()),
        "logvar_floor_loss": float(logvar_floor_loss.detach().cpu()),
        "teacher_token_acc": float(teacher_token_acc.detach().cpu()),
        "free_token_acc": float(free_token_acc.detach().cpu()),
        "mu_abs_mean": float(out["mu"].abs().mean().detach().cpu()),
        "mu_std": float(out["mu"].std().detach().cpu()),
        "logvar_mean": float(out["logvar"].mean().detach().cpu()),
        "logvar_std": float(out["logvar"].std().detach().cpu()),
    }


@torch.no_grad()
def free_decode_diagnostics(
    model: GRUVAE,
    x_onehot: torch.Tensor,
    out: Dict[str, torch.Tensor],
    temperature: float,
) -> Dict[str, float]:
    was_training = model.training
    try:
        source = decode_argmax_from_logits(x_onehot)
        decoded = autoregressive_free_decode_argmax(
            model,
            out["mu"],
            temperature=temperature,
        )
    finally:
        model.train(was_training)

    edits = [
        levenshtein_edit_distance(a, b)
        for a, b in zip(source, decoded)
    ]
    counts = (
        pd.Series(decoded).value_counts()
        if decoded
        else pd.Series(dtype=int)
    )
    return {
        "free_decode_edit_mean": float(np.mean(edits)) if edits else float("nan"),
        "free_decode_edit_median": float(np.median(edits)) if edits else float("nan"),
        "free_decode_unique_fraction": float(len(set(decoded)) / max(1, len(decoded))),
        "free_decode_dominant_fraction": float(counts.iloc[0] / max(1, len(decoded))) if len(counts) else float("nan"),
    }



def init_metric_accumulators(metric_keys: List[str]):
    return ({k: 0.0 for k in metric_keys}, {k: 0 for k in metric_keys})


def accumulate_metrics(
    sums: Dict[str, float],
    counts: Dict[str, int],
    metrics: Dict[str, float],
    metric_keys: List[str],
) -> None:
    for k in metric_keys:
        try:
            v = float(metrics.get(k, float("nan")))
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            sums[k] += v
            counts[k] += 1


def finalize_metrics(
    sums: Dict[str, float],
    counts: Dict[str, int],
    metric_keys: List[str],
) -> Dict[str, float]:
    return {
        k: (sums[k] / counts[k] if counts[k] > 0 else float("nan"))
        for k in metric_keys
    }


@torch.no_grad()
def evaluate_validation(
    model: GRUVAE,
    loader: DataLoader,
    args: argparse.Namespace,
    beta: float,
    metric_keys: List[str],
) -> Tuple[Dict[str, float], int]:
    """Evaluate held-out peptides and compute global latent-geometry diagnostics."""
    was_training = model.training
    model.eval()
    sums, counts = init_metric_accumulators(metric_keys)
    n_batches = 0
    mu_samples: List[torch.Tensor] = []
    n_geometry = 0

    try:
        for x in loader:
            if args.validation_max_batches > 0 and n_batches >= args.validation_max_batches:
                break

            x = x.to(next(model.parameters()).device)
            out = model(x, use_mu_for_recon=args.use_mu_for_recon, input_dropout=0.0)
            _, metrics = bo_ready_vae_loss(x, out, model, args, beta=beta)

            if n_geometry < args.geometry_max_samples:
                take = min(out["mu"].size(0), args.geometry_max_samples - n_geometry)
                mu_samples.append(out["mu"][:take].detach().cpu())
                n_geometry += take

            if (
                args.validation_diagnostics_every_batches > 0
                and n_batches % args.validation_diagnostics_every_batches == 0
            ):
                diag = free_decode_diagnostics(model, x, out, temperature=args.free_decode_temperature)
            else:
                diag = {
                    "free_decode_edit_mean": float("nan"),
                    "free_decode_edit_median": float("nan"),
                    "free_decode_unique_fraction": float("nan"),
                    "free_decode_dominant_fraction": float("nan"),
                }

            metrics.update(diag)
            accumulate_metrics(sums, counts, metrics, metric_keys)
            n_batches += 1
    finally:
        model.train(was_training)

    result = finalize_metrics(sums, counts, metric_keys)
    if mu_samples:
        geom = latent_geometry_diagnostics(torch.cat(mu_samples, dim=0))
        result.update(geom)
        result["geometry_n_samples"] = float(n_geometry)
    return result, n_batches

def kl_beta_for_epoch(epoch: int, kl_beta: float, warmup_epochs: int) -> float:
    return min(float(kl_beta), float(kl_beta) * float(epoch) / max(1, int(warmup_epochs)))


def append_history(path: str, row: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, cfg: ModelConfig, args: argparse.Namespace, epoch: int, metrics: Dict[str, float]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "checkpoint_type": "pretrained_gru_vae_latent_conditioned_bo_ready_no_flow",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": asdict(cfg),
        "args": vars(args),
        "epoch": int(epoch),
        "metrics": metrics,
        "aa": AA,
        "seq_len": SEQ_LEN,
        "decoder_latent_conditioning": cfg.decoder_conditioning,
        "validation": {
            "split_method": "connected_component_hash_of_peptide_PDB_bipartite_graph",
            "group_col": args.group_col,
            "validation_fraction": float(args.validation_fraction),
            "test_fraction": float(args.test_fraction),
            "validation_split_seed": int(args.validation_split_seed),
            "holdout_min_hamming": int(args.holdout_min_hamming),
        },
        "bo_ready_training_fixes": [
            "mu_deterministic_reconstruction",
            "decoder_input_dropout",
            "free_running_autoregressive_reconstruction_loss",
            "minimum_information_regularization",
            "latent_spread_regularization",
            "latent_covariance_decorrelation_regularization",
            "PDB_group_and_exact_sequence_disjoint_splitting",
            "strict_holdout_hamming_filter",
            "global_latent_effective_dimension_monitoring",
            "logvar_floor_regularization",
        ],
        "downstream_compatible_with": [
            "finetune_gru_vae_cu_h64_z64_latent_conditioned_realnvp_roundtrip_high_confidence_data.py",
            "finetune_gru_vae_cu_h64_z64_latent_conditioned_latent_diffusion_high_confidence_data_fix_json.py",
        ],
    }
    torch.save(payload, path)
    json_payload = {k: v for k, v in payload.items() if k not in {"model_state_dict", "optimizer_state_dict"}}
    with open(path + ".json", "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)


def load_training_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, cfg: ModelConfig, device: torch.device, load_optimizer: bool = True) -> int:
    ckpt = torch.load(path, map_location=device)
    saved_cfg = ckpt.get("model_config", {})
    current_cfg = asdict(cfg)
    mismatches = {
        k: (saved_cfg.get(k), current_cfg.get(k))
        for k in ["hidden_size", "latent_dim", "n_layers", "dropout"]
        if saved_cfg.get(k) != current_cfg.get(k)
    }
    if mismatches:
        raise ValueError(f"Checkpoint architecture mismatch: {mismatches}")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    if load_optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
    epoch = int(ckpt.get("epoch", 0))
    print(f"Resumed {path} from epoch {epoch}")
    return epoch + 1


def find_pretrain_files(parts_dir: str, file_pattern: str) -> List[str]:
    if file_pattern == "auto":
        patterns = [
            "metalpdb_ALL_chain_mapped_len10_high_confidence_part_*.csv",
            "metalpdb_binding_windows_len10_part_*.csv",
            "*chain_mapped*len10*part_*.csv",
            "*len10*part_*.csv",
            "*.csv",
        ]
    else:
        patterns = [file_pattern]
    for pat in patterns:
        full = os.path.join(parts_dir, pat)
        files = sorted(glob.glob(full))
        files = [p for p in files if "failures" not in os.path.basename(p).lower()]
        if files:
            print(f"Found {len(files)} files using pattern: {full}")
            return files
    raise FileNotFoundError(f"No pretraining CSV shards found in {parts_dir!r} with pattern={file_pattern!r}")


def train(args: argparse.Namespace) -> str:
    set_seed(args.seed)
    device = torch.device(args.device)
    cfg = ModelConfig(args.hidden_size, args.latent_dim, args.n_layers, args.dropout)

    files = find_pretrain_files(args.parts_dir, args.file_pattern)
    split_sets = prepare_leakage_safe_split(files, args)
    train_dataset = PeptideCSVIterable(
        files=files,
        allowed_peptides=split_sets["train"],
        peptide_col=args.peptide_col,
        chunksize=args.chunksize,
        shuffle_files=True,
        deduplicate_within_epoch=args.deduplicate,
    )
    val_dataset = PeptideCSVIterable(
        files=files,
        allowed_peptides=split_sets["val"],
        peptide_col=args.peptide_col,
        chunksize=args.chunksize,
        shuffle_files=False,
        deduplicate_within_epoch=True,
    )
    train_loader = DataLoader(train_dataset, batch_size=None, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=None, num_workers=0)

    model = GRUVAE(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ckpt_name = f"pretrained_gru_vae_bo_ready_latent_conditioned_h{cfg.hidden_size}_z{cfg.latent_dim}.pt"
    out_path = os.path.join(args.out_dir, ckpt_name)
    best_val_path = os.path.join(
        args.out_dir,
        f"best_val_free_recon_gru_vae_bo_ready_latent_conditioned_h{cfg.hidden_size}_z{cfg.latent_dim}.pt",
    )
    history_path = args.history_csv or os.path.join(args.out_dir, "pretraining_history_bo_ready.csv")
    best_val_free_recon = float("inf")
    best_bo_ready_score = float("inf")
    best_bo_ready_path = os.path.join(
        args.out_dir,
        f"best_bo_ready_gru_vae_latent_conditioned_h{cfg.hidden_size}_z{cfg.latent_dim}.pt",
    )

    start_epoch = 1
    resume_path = args.resume_checkpoint or out_path
    if not args.no_resume and os.path.exists(resume_path):
        start_epoch = load_training_checkpoint(resume_path, model, optimizer, cfg, device, load_optimizer=not args.no_resume_optimizer)
    elif args.resume_checkpoint and not os.path.exists(args.resume_checkpoint):
        raise FileNotFoundError(args.resume_checkpoint)

    if args.no_resume and os.path.exists(history_path) and not args.append_history:
        os.remove(history_path)

    print(f"Pretraining BO-ready latent-conditioned GRU-VAE H{cfg.hidden_size}/Z{cfg.latent_dim} on {len(files)} all-metal shards.")
    print("Training objective: deterministic mu decode + input dropout + free-running decode loss + latent anti-collapse regularizers.")
    print(f"Output checkpoint: {out_path}")
    print(f"History CSV: {history_path}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        beta = kl_beta_for_epoch(epoch, args.kl_beta, args.kl_warmup_epochs)
        metric_keys = [
            "loss", "teacher_recon", "free_recon", "kl", "kl_per_dim_mean", "min_info_loss",
            "latent_spread_loss", "mu_mean_loss", "latent_cov_loss", "mean_abs_latent_corr_batch", "logvar_floor_loss", "teacher_token_acc",
            "free_token_acc", "mu_abs_mean", "mu_std", "logvar_mean", "logvar_std",
            "free_decode_edit_mean", "free_decode_edit_median", "free_decode_unique_fraction",
            "free_decode_dominant_fraction",
        ]
        sums, counts = init_metric_accumulators(metric_keys)
        n_batches = 0

        for x in train_loader:
            model.train()
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)

            out = model(
                x,
                use_mu_for_recon=args.use_mu_for_recon,
                input_dropout=args.decoder_input_dropout,
            )
            loss, metrics = bo_ready_vae_loss(x, out, model, args, beta=beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            if (
                args.diagnostics_every_batches > 0
                and n_batches % args.diagnostics_every_batches == 0
            ):
                diag = free_decode_diagnostics(
                    model, x, out, temperature=args.free_decode_temperature
                )
            else:
                diag = {
                    "free_decode_edit_mean": float("nan"),
                    "free_decode_edit_median": float("nan"),
                    "free_decode_unique_fraction": float("nan"),
                    "free_decode_dominant_fraction": float("nan"),
                }

            metrics.update(diag)
            accumulate_metrics(sums, counts, metrics, metric_keys)
            n_batches += 1

            if args.log_every and n_batches % args.log_every == 0:
                avg = finalize_metrics(sums, counts, metric_keys)
                print(
                    f"epoch={epoch} batch={n_batches} beta={beta:.6g} "
                    f"train_loss={avg['loss']:.4f} "
                    f"train_teacher_ce={avg['teacher_recon']:.4f} "
                    f"train_free_ce={avg['free_recon']:.4f} "
                    f"train_teacher_acc={avg['teacher_token_acc']:.4f} "
                    f"train_free_acc={avg['free_token_acc']:.4f} "
                    f"train_mu_std={avg['mu_std']:.4f}"
                )

        train_metrics = finalize_metrics(sums, counts, metric_keys)

        val_metrics = {k: float("nan") for k in metric_keys}
        n_val_batches = 0
        if epoch % args.validation_every == 0:
            val_metrics, n_val_batches = evaluate_validation(
                model=model,
                loader=val_loader,
                args=args,
                beta=beta,
                metric_keys=metric_keys,
            )
            print(
                f"epoch={epoch} VALIDATION "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_teacher_ce={val_metrics['teacher_recon']:.4f} "
                f"val_free_ce={val_metrics['free_recon']:.4f} "
                f"val_teacher_acc={val_metrics['teacher_token_acc']:.4f} "
                f"val_free_acc={val_metrics['free_token_acc']:.4f} "
                f"val_kl_per_dim={val_metrics['kl_per_dim_mean']:.4f} "
                f"val_mu_std={val_metrics['mu_std']:.4f} "
                f"val_edit={val_metrics['free_decode_edit_mean']:.4f} "
                f"val_unique={val_metrics['free_decode_unique_fraction']:.4f} "
                f"val_effdim={val_metrics.get('effective_dim_pr_standardized', float('nan')):.3f} "
                f"val_abs_corr={val_metrics.get('mean_abs_offdiag_corr', float('nan')):.3f}"
            )

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "epoch": epoch,
            "n_batches": n_batches,
            "n_val_batches": n_val_batches,
            "beta": beta,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **train_metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "hidden_size": cfg.hidden_size,
            "latent_dim": cfg.latent_dim,
            "n_layers": cfg.n_layers,
            "dropout": cfg.dropout,
            "decoder_conditioning": cfg.decoder_conditioning,
            "decoder_input_dropout": args.decoder_input_dropout,
            "free_decode_loss_weight": args.free_decode_loss_weight,
            "min_kl_per_dim": args.min_kl_per_dim,
            "min_info_loss_weight": args.min_info_loss_weight,
            "latent_spread_loss_weight": args.latent_spread_loss_weight,
            "latent_cov_loss_weight": args.latent_cov_loss_weight,
            "target_mu_std": args.target_mu_std,
            "holdout_min_hamming": args.holdout_min_hamming,
            "use_mu_for_recon": bool(args.use_mu_for_recon),
        }
        append_history(history_path, row)

        combined_metrics = {
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        save_checkpoint(out_path, model, optimizer, cfg, args, epoch, combined_metrics)

        current_val_free = float(val_metrics.get("free_recon", float("nan")))
        if math.isfinite(current_val_free) and current_val_free < best_val_free_recon:
            best_val_free_recon = current_val_free
            best_metrics = dict(combined_metrics)
            best_metrics["selection_metric"] = "val_free_recon"
            best_metrics["selection_value"] = current_val_free
            save_checkpoint(
                best_val_path,
                model,
                optimizer,
                cfg,
                args,
                epoch,
                best_metrics,
            )
            print(
                f"NEW BEST VALIDATION: epoch={epoch} "
                f"val_free_recon={current_val_free:.6f}"
            )

        # BO-aware checkpoint selection: preserve reconstruction quality while
        # explicitly penalizing low participation-ratio dimensionality and high
        # latent correlation. This prevents selecting a nearly 1-D latent solely
        # because its reconstruction CE is marginally smaller.
        val_effdim = float(val_metrics.get("effective_dim_pr_standardized", float("nan")))
        val_corr = float(val_metrics.get("mean_abs_offdiag_corr", float("nan")))
        if math.isfinite(current_val_free) and math.isfinite(val_effdim) and math.isfinite(val_corr):
            eff_deficit = max(0.0, float(args.selection_min_effective_dim) - val_effdim) / max(1e-8, float(args.selection_min_effective_dim))
            bo_ready_score = (
                current_val_free
                + float(args.selection_effdim_penalty_weight) * eff_deficit
                + float(args.selection_corr_penalty_weight) * val_corr
            )
            if bo_ready_score < best_bo_ready_score:
                best_bo_ready_score = bo_ready_score
                best_metrics = dict(combined_metrics)
                best_metrics.update({
                    "selection_metric": "bo_ready_score",
                    "selection_value": bo_ready_score,
                    "selection_val_free_recon": current_val_free,
                    "selection_effective_dim_pr_standardized": val_effdim,
                    "selection_mean_abs_offdiag_corr": val_corr,
                })
                save_checkpoint(
                    best_bo_ready_path, model, optimizer, cfg, args, epoch, best_metrics
                )
                print(
                    f"NEW BEST BO-READY: epoch={epoch} score={bo_ready_score:.6f} "
                    f"val_free={current_val_free:.6f} effdim={val_effdim:.3f} corr={val_corr:.3f}"
                )

        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"train_free_acc={train_metrics['free_token_acc']:.6f} "
            f"val_free_acc={val_metrics['free_token_acc']:.6f}"
        )

    print(f"Saved latest checkpoint: {out_path}")
    if os.path.exists(best_val_path):
        print(
            f"Saved best-validation checkpoint: {best_val_path} "
            f"(best val_free_recon={best_val_free_recon:.6f})"
        )
    if os.path.exists(best_bo_ready_path):
        print(f"Saved best BO-ready checkpoint: {best_bo_ready_path} (score={best_bo_ready_score:.6f})")
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pretrain BO-ready H64/Z64 latent-conditioned GRU-VAE for downstream RealNVP or latent diffusion.")
    p.add_argument("--parts-dir", default="metalpdb_all_metals_chain_mapped_len10_high_confidence_parts/parts")
    p.add_argument("--file-pattern", default="auto", help="CSV shard glob under --parts-dir. Use 'auto' for common shard names.")
    p.add_argument("--peptide-col", default="peptide_len10")
    p.add_argument("--group-col", default="auto", help="PDB/structure group column. 'auto' detects common PDB column names.")
    p.add_argument("--test-fraction", type=float, default=0.10, help="Leakage-safe group-held-out test fraction prepared but not used for checkpoint selection.")
    p.add_argument("--holdout-min-hamming", type=int, default=3, choices=[0, 1, 2, 3], help="Minimum Hamming distance from training/development for holdout peptides. Default 3 excludes validation/test peptides within <=2 substitutions.")
    p.add_argument("--out-dir", default="transfer_gru_vae_checkpoints_h64_z64_latent_conditioned_bo_ready_dataset")
    p.add_argument("--history-csv", default=None)
    p.add_argument("--append-history", action="store_true")
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--chunksize", type=int, default=8192)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--kl-beta", type=float, default=5e-5, help="Small KL weight; anti-collapse is handled mostly by min-info/spread losses.")
    p.add_argument("--kl-warmup-epochs", type=int, default=50)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--resume-checkpoint", default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--no-resume-optimizer", action="store_true")
    p.add_argument("--deduplicate", action=argparse.BooleanOptionalAction, default=True, help="Deduplicate exact peptide sequences within each epoch; recommended for sequence-only pretraining.")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # BO-ready objective controls.
    p.add_argument("--use-mu-for-recon", action="store_true", default=True)
    p.add_argument("--sample-z-for-recon", dest="use_mu_for_recon", action="store_false")
    p.add_argument("--decoder-input-dropout", type=float, default=0.20)
    p.add_argument("--free-decode-loss-weight", type=float, default=0.30)
    p.add_argument("--free-decode-temperature", type=float, default=1.0)
    p.add_argument("--min-kl-per-dim", type=float, default=0.02)
    p.add_argument("--min-info-loss-weight", type=float, default=0.05)
    p.add_argument("--latent-spread-loss-weight", type=float, default=0.05)
    p.add_argument("--latent-cov-loss-weight", type=float, default=0.05, help="VICReg-style off-diagonal covariance penalty to prevent correlated effective-dimensionality collapse.")
    p.add_argument("--target-mu-std", type=float, default=1.0)
    p.add_argument("--geometry-max-samples", type=int, default=20000, help="Maximum validation mu samples used for covariance-spectrum/effective-dimension diagnostics.")
    p.add_argument("--selection-min-effective-dim", type=float, default=8.0, help="Soft target used by BO-aware checkpoint selection; not a hard constraint.")
    p.add_argument("--selection-effdim-penalty-weight", type=float, default=0.02)
    p.add_argument("--selection-corr-penalty-weight", type=float, default=0.02)
    p.add_argument("--mu-mean-loss-weight", type=float, default=0.01)
    p.add_argument("--logvar-floor-target", type=float, default=-4.0)
    p.add_argument("--logvar-floor-loss-weight", type=float, default=0.005)
    p.add_argument("--diagnostics-every-batches", type=int, default=10)

    p.add_argument(
        "--validation-fraction",
        type=float,
        default=0.10,
        help="Held-out validation fraction assigned at leakage-safe peptide/PDB connected-component level.",
    )
    p.add_argument(
        "--validation-split-seed",
        type=int,
        default=12345,
    )
    p.add_argument(
        "--validation-every",
        type=int,
        default=1,
        help="Run validation every N epochs.",
    )
    p.add_argument(
        "--validation-max-batches",
        type=int,
        default=0,
        help="0 means evaluate all validation batches.",
    )
    p.add_argument(
        "--validation-diagnostics-every-batches",
        type=int,
        default=1,
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train(args)
