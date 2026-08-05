from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Configuration
# ============================================================

OPTIMIZED_CSV = "bo_results_transfer_h32_z32_latent_conditioned_epoch199_balanced/bo_pareto_dominates_train_CU_flow_space_gp.csv"

# Optional. The scrambling analysis itself does not require the train-Pareto
# file. Set this to a CSV path only when you also want generated scrambles to
# avoid sequences present in the training Pareto set.
TRAIN_PARETO_CSV = None

OUTPUT_DIR = (
    "scrambling_control_epoch199_best_roundtrip_"
    "latent_conditioned_flow_space_gp"
)
N_SCRAMBLES_PER_PEPTIDE = 100
RANDOM_SEED = 42

# ============================================================
# Latent round-trip / peptide-space round-trip configuration
# ============================================================
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_I = {a: i for i, a in enumerate(AA)}
I_TO_AA = {i: a for a, i in AA_TO_I.items()}
SEQ_LEN = 10
VOCAB = 20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Updated transfer-learning framework.
HIDDEN_SIZE = 32
LATENT_DIM = 32
N_GRU_LAYERS = 2
N_FLOWS = 2
DROPOUT = 0.0

TRANSFER_CHECKPOINT = (
    "transfer_gru_vae_flow_checkpoints_h32_z32_latent_conditioned_autoreg_rt001/"
    "best_roundtrip_h32_z32_latent_conditioned_autoreg_rt.pt"
)

EXPECTED_CHECKPOINT_EPOCH = 199
EXPECTED_CHECKPOINT_SELECTION = "minimum_validation_roundtrip_l2"
EXPECTED_DECODER_CONDITIONING = "concat_z_at_every_decoder_step"

# Local latent-space smoothness diagnostic.
N_LATENT_NEIGHBORS = 3
LATENT_PERTURBATION_STD = 0.05
LATENT_NEIGHBOR_SEED = 12345

# If True, avoid generating scrambled peptides that already exist in the
# optimized or training Pareto files.
REJECT_EXISTING_SEQUENCES = True

# Your objective function file must be in the same folder or importable.
# It should contain blackbox_fc.
from backup_code_result_folders.black_box_fcn_mo_CU_f_updated_batch_normalized import blackbox_fc


OBJECTIVE_COLS = [
    "chelation_sub",
    "solubility_sub",
    "stability_sub",
    "expression_sub",
]

FINAL_SCORE_COL = "final_score"


# ============================================================
# Updated H32/Z32 GRU-VAE + normalizing flow
# ============================================================

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

    def forward(self, x_onehot: torch.Tensor):
        x = self.in_proj(x_onehot)
        out_top, h_n = self.gru(x)
        h_last = h_n[-1]
        mu = self.to_mu(h_last)
        logvar = self.to_logvar(h_last).clamp(min=-8.0, max=8.0)
        return mu, logvar, out_top


class GRUDecoder(nn.Module):
    """Latent-conditioned decoder matching the epoch-199 checkpoint.

    Every GRU step consumes [token_embedding ; z].
    """

    def __init__(
        self,
        hidden_size: int,
        latent_dim: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
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

    def forward(
        self,
        z: torch.Tensor,
        x_onehot: torch.Tensor,
    ):
        batch_size = z.size(0)

        x_shift = torch.zeros_like(x_onehot)
        x_shift[:, 1:, :] = x_onehot[:, :-1, :]

        emb = self.token_embed(x_shift)
        z_repeat = z.unsqueeze(1).expand(
            -1,
            x_onehot.size(1),
            -1,
        )
        decoder_input = torch.cat(
            [emb, z_repeat],
            dim=-1,
        )

        h0 = self.z_to_h(z).view(
            self.gru.num_layers,
            batch_size,
            self.gru.hidden_size,
        )

        out_top, _ = self.gru(
            decoder_input,
            h0,
        )
        logits = self.to_logits(out_top)
        return logits, out_top


class PlanarFlow(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim) * 0.02)
        self.u = nn.Parameter(torch.randn(dim) * 0.02)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, z: torch.Tensor):
        a = z @ self.w + self.b
        h = torch.tanh(a)
        z_new = z + h.unsqueeze(-1) * self.u
        psi = (1.0 - h.pow(2)).unsqueeze(-1) * self.w.unsqueeze(0)
        det_term = 1.0 + (psi * self.u.unsqueeze(0)).sum(dim=-1)
        logabsdet = torch.log(torch.abs(det_term) + 1e-8)
        return z_new, logabsdet


class FlowSequence(nn.Module):
    def __init__(self, dim: int, n_flows: int):
        super().__init__()
        self.flows = nn.ModuleList([PlanarFlow(dim) for _ in range(n_flows)])

    def forward(self, z0: torch.Tensor):
        z = z0
        sum_logdet = torch.zeros(z0.size(0), device=z0.device)
        for flow in self.flows:
            z, logdet = flow(z)
            sum_logdet = sum_logdet + logdet
        return z, sum_logdet


class FlowFineTuneModel(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        latent_dim: int,
        n_layers: int,
        dropout: float,
        n_flows: int,
    ):
        super().__init__()
        self.enc = GRUEncoder(hidden_size, latent_dim, n_layers, dropout)
        self.dec = GRUDecoder(hidden_size, latent_dim, n_layers, dropout)
        self.flow = FlowSequence(latent_dim, n_flows)
        self.score_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )


def onehot_encode_peptides(peptides: List[str]) -> torch.Tensor:
    x = torch.zeros((len(peptides), SEQ_LEN, VOCAB), dtype=torch.float32)
    for n, peptide in enumerate(peptides):
        peptide = str(peptide).strip().upper()
        if len(peptide) != SEQ_LEN:
            raise ValueError(
                f"Expected peptide length {SEQ_LEN}, got {len(peptide)} for {peptide!r}"
            )
        for t, aa in enumerate(peptide):
            if aa not in AA_TO_I:
                raise ValueError(f"Unsupported amino acid {aa!r} in {peptide!r}")
            x[n, t, AA_TO_I[aa]] = 1.0
    return x


def decode_logits_to_peptide(logits: torch.Tensor) -> str:
    idx = logits.argmax(dim=-1).detach().cpu().tolist()
    return "".join(I_TO_AA[int(i)] for i in idx)


def levenshtein_edit_distance(a: str, b: str) -> int:
    """Unit-cost Levenshtein insertion/deletion/substitution distance."""
    a = str(a)
    b = str(b)
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + int(char_a != char_b)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return int(previous[-1])


def load_roundtrip_model(checkpoint_path: str) -> FlowFineTuneModel:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\\n"
            "Update TRANSFER_CHECKPOINT in the configuration section."
        )

    model = FlowFineTuneModel(
        hidden_size=HIDDEN_SIZE,
        latent_dim=LATENT_DIM,
        n_layers=N_GRU_LAYERS,
        dropout=DROPOUT,
        n_flows=N_FLOWS,
    ).to(DEVICE)

    ckpt = torch.load(checkpoint, map_location=DEVICE)
    state = ckpt.get("model_state_dict", ckpt)

    cfg = ckpt.get("model_config", {})
    for key, expected_value in {
        "hidden_size": HIDDEN_SIZE,
        "latent_dim": LATENT_DIM,
        "n_layers": N_GRU_LAYERS,
    }.items():
        found_value = cfg.get(key, expected_value)
        if int(found_value) != int(expected_value):
            raise ValueError(
                f"Checkpoint {key}={found_value} does not match configured "
                f"{key}={expected_value}"
            )

    checkpoint_n_flows = ckpt.get("n_flows")
    if checkpoint_n_flows is None:
        checkpoint_n_flows = ckpt.get("args", {}).get("n_flows")
    if checkpoint_n_flows is not None and int(checkpoint_n_flows) != N_FLOWS:
        raise ValueError(
            f"Checkpoint n_flows={checkpoint_n_flows} does not match "
            f"N_FLOWS={N_FLOWS}"
        )

    checkpoint_epoch = int(ckpt.get("epoch", -1))
    if checkpoint_epoch != EXPECTED_CHECKPOINT_EPOCH:
        raise ValueError(
            f"Expected epoch {EXPECTED_CHECKPOINT_EPOCH}, "
            f"but checkpoint metadata reports epoch {checkpoint_epoch}."
        )

    checkpoint_selection = ckpt.get("checkpoint_selection")
    if checkpoint_selection != EXPECTED_CHECKPOINT_SELECTION:
        raise ValueError(
            f"Expected checkpoint selection "
            f"{EXPECTED_CHECKPOINT_SELECTION!r}, "
            f"but found {checkpoint_selection!r}."
        )

    decoder_conditioning = ckpt.get("decoder_latent_conditioning")
    if decoder_conditioning != EXPECTED_DECODER_CONDITIONING:
        raise ValueError(
            f"Expected decoder conditioning "
            f"{EXPECTED_DECODER_CONDITIONING!r}, "
            f"but found {decoder_conditioning!r}."
        )

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"Checkpoint missing model tensors: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint tensors: {unexpected}")

    model.eval()
    print(f"Loaded latent diagnostic checkpoint: {checkpoint}")
    print(f"Device: {DEVICE}")
    print(f"Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"Checkpoint selection: {ckpt.get('checkpoint_selection', 'unknown')}")
    metrics = ckpt.get("metrics", {})
    if metrics:
        print("Checkpoint val roundtrip_l2:", metrics.get("roundtrip_l2", "unknown"))
        print("Checkpoint val roundtrip_cosine:", metrics.get("roundtrip_cosine", "unknown"))
    return model


@torch.no_grad()
def encode_peptide_to_after_flow_latent(
    model: FlowFineTuneModel,
    peptide: str,
) -> torch.Tensor:
    x = onehot_encode_peptides([peptide]).to(DEVICE)
    mu, _, _ = model.enc(x)
    z, _ = model.flow(mu)
    return z

def decoder_step(
    model: FlowFineTuneModel,
    z: torch.Tensor,
    current_token: torch.Tensor,
    h: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run one autoregressive step with z concatenated to the token embedding."""
    emb = model.dec.token_embed(current_token)
    z_step = z.unsqueeze(1)
    decoder_input = torch.cat(
        [emb, z_step],
        dim=-1,
    )
    out_step, h = model.dec.gru(
        decoder_input,
        h,
    )
    logits = model.dec.to_logits(
        out_step[:, -1, :]
    )
    return logits, h


@torch.no_grad()
def autoregressive_decode(
    model,
    z: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, list[str]]:
    """
    True free-running autoregressive decoding.

    z:
        [B, latent_dim]

    Returns:
        logits_all: [B, SEQ_LEN, VOCAB]
        peptides: list[str]
    """
    model.eval()

    batch_size = z.size(0)
    device = z.device

    # Initialize decoder hidden state from latent z.
    h = model.dec.z_to_h(z).view(
        model.dec.gru.num_layers,
        batch_size,
        model.dec.gru.hidden_size,
    )

    # BOS representation = all-zero token.
    current_token = torch.zeros(
        batch_size,
        1,
        VOCAB,
        dtype=z.dtype,
        device=device,
    )

    logits_steps = []
    generated_indices = []

    for _ in range(SEQ_LEN):
        logits, h = decoder_step(
            model=model,
            z=z,
            current_token=current_token,
            h=h,
        )

        logits = logits / max(
            float(temperature),
            1e-6,
        )

        token_idx = logits.argmax(dim=-1)

        logits_steps.append(
            logits.unsqueeze(1)
        )

        generated_indices.append(
            token_idx.unsqueeze(1)
        )

        # Feed the generated token into the next step.
        current_token = F.one_hot(
            token_idx,
            num_classes=VOCAB,
        ).to(dtype=z.dtype).unsqueeze(1)

    logits_all = torch.cat(
        logits_steps,
        dim=1,
    )

    generated_indices = torch.cat(
        generated_indices,
        dim=1,
    )

    peptides = [
        "".join(
            I_TO_AA[int(i)]
            for i in row
        )
        for row in generated_indices.detach().cpu().tolist()
    ]

    return logits_all, peptides
@torch.no_grad()
def decode_after_flow_latent(
    model: FlowFineTuneModel,
    z: torch.Tensor,
) -> str:
    z = z.reshape(1, -1).to(DEVICE)
    _, peptides = autoregressive_decode(
        model,
        z,
    )

    return peptides[0]


@torch.no_grad()
def compute_sequence_roundtrip_diagnostics(
    model: FlowFineTuneModel,
    source_peptide: str,
) -> Dict[str, object]:
    """
    Deterministic path:
        source -> z1=flow(mu(source))
        z1 -> p1
        p1 -> z2=flow(mu(p1))
        z2 -> p2

    Peptide round trip = Levenshtein(p1, p2).
    Latent round trip = L2/cosine(z1, z2).
    """
    z1 = encode_peptide_to_after_flow_latent(model, source_peptide)
    p1 = decode_after_flow_latent(model, z1)
    z2 = encode_peptide_to_after_flow_latent(model, p1)
    p2 = decode_after_flow_latent(model, z2)

    latent_l2 = torch.linalg.norm(z1 - z2, dim=-1).item()
    latent_cosine = F.cosine_similarity(z1, z2, dim=-1, eps=1e-8).item()
    peptide_edit_distance = levenshtein_edit_distance(p1, p2)

    return {
        "source_peptide": source_peptide,
        "roundtrip_p1": p1,
        "roundtrip_p2": p2,
        "peptide_roundtrip_edit_distance": int(peptide_edit_distance),
        "peptide_roundtrip_edit_distance_normalized": (
            float(peptide_edit_distance) / max(len(p1), len(p2), 1)
        ),
        "latent_roundtrip_l2": float(latent_l2),
        "latent_roundtrip_cosine": float(latent_cosine),
        "z1_norm": float(torch.linalg.norm(z1, dim=-1).item()),
        "z2_norm": float(torch.linalg.norm(z2, dim=-1).item()),
    }


@torch.no_grad()
def compute_local_latent_smoothness(
    model: FlowFineTuneModel,
    source_peptide: str,
    sequence_id: int,
    generator: torch.Generator,
    n_neighbors: int,
    perturbation_std: float,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """
    For z1, decode center p1. Create n_neighbors by z_neighbor=z1+epsilon,
    decode each neighbor, and measure Levenshtein(neighbor_peptide, p1).
    """
    z1 = encode_peptide_to_after_flow_latent(model, source_peptide)
    p1 = decode_after_flow_latent(model, z1)

    detail_rows = []
    edit_distances = []
    normalized_edit_distances = []

    for neighbor_index in range(n_neighbors):
        noise_cpu = torch.randn(
            z1.shape,
            generator=generator,
            dtype=z1.dtype,
            device="cpu",
        )
        noise = noise_cpu.to(z1.device) * float(perturbation_std)
        z_neighbor = z1 + noise
        neighbor_peptide = decode_after_flow_latent(model, z_neighbor)

        edit_distance = levenshtein_edit_distance(p1, neighbor_peptide)
        normalized_edit = float(edit_distance) / max(
            len(p1),
            len(neighbor_peptide),
            1,
        )

        edit_distances.append(edit_distance)
        normalized_edit_distances.append(normalized_edit)

        detail_rows.append(
            {
                "sequence_id": int(sequence_id),
                "source_peptide": source_peptide,
                "center_p1": p1,
                "neighbor_index": int(neighbor_index),
                "perturbation_std": float(perturbation_std),
                "perturbation_l2": float(
                    torch.linalg.norm(noise, dim=-1).item()
                ),
                "neighbor_peptide": neighbor_peptide,
                "neighbor_edit_distance_to_p1": int(edit_distance),
                "neighbor_edit_distance_to_p1_normalized": float(normalized_edit),
            }
        )

    summary = {
        "local_smoothness_center_p1": p1,
        "local_smoothness_n_neighbors": int(n_neighbors),
        "local_smoothness_perturbation_std": float(perturbation_std),
        "local_neighbor_edit_distance_mean": float(np.mean(edit_distances)),
        "local_neighbor_edit_distance_std": float(np.std(edit_distances, ddof=0)),
        "local_neighbor_edit_distance_min": int(np.min(edit_distances)),
        "local_neighbor_edit_distance_max": int(np.max(edit_distances)),
        "local_neighbor_normalized_edit_distance_mean": float(
            np.mean(normalized_edit_distances)
        ),
        "local_neighbor_identical_fraction": float(
            np.mean(np.asarray(edit_distances) == 0)
        ),
        "local_neighbor_unique_peptides": int(
            len({row["neighbor_peptide"] for row in detail_rows})
        ),
    }
    return summary, detail_rows


def append_overall_average_row(
    df: pd.DataFrame,
    metric_columns: List[str],
    label_column: str = "source_peptide",
) -> pd.DataFrame:
    average_row = {col: np.nan for col in df.columns}
    if label_column in average_row:
        average_row[label_column] = "__AVERAGE__"

    for col in metric_columns:
        if col in df.columns:
            average_row[col] = float(
                pd.to_numeric(df[col], errors="coerce").mean()
            )

    return pd.concat(
        [df, pd.DataFrame([average_row])],
        ignore_index=True,
    )


# ============================================================
# Helper functions
# ============================================================

def get_peptide_column(df: pd.DataFrame) -> str:
    """
    Detect peptide column name.
    """
    candidates = ["peptide", "peptide_len10", "sequence"]
    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        f"Could not find peptide column. Available columns: {list(df.columns)}"
    )


def scramble_sequence(seq: str, rng: random.Random) -> str:
    """
    Return one random permutation of the amino-acid sequence.
    """
    chars = list(seq)
    rng.shuffle(chars)
    return "".join(chars)


def generate_unique_scrambles(
    seq: str,
    n: int,
    rng: random.Random,
    forbidden: set[str] | None = None,
    max_attempts: int = 10000,
) -> List[str]:
    """
    Generate unique scrambled versions of a sequence.

    Preserves amino-acid composition exactly.
    Rejects:
      - the original sequence
      - duplicates
      - forbidden sequences, if provided
    """
    seq = seq.strip().upper()
    forbidden = forbidden or set()

    scrambles = set()
    attempts = 0

    while len(scrambles) < n and attempts < max_attempts:
        attempts += 1
        s = scramble_sequence(seq, rng)

        if s == seq:
            continue

        if s in scrambles:
            continue

        if s in forbidden:
            continue

        scrambles.add(s)

    if len(scrambles) < n:
        print(
            f"[WARNING] Only generated {len(scrambles)} unique scrambles "
            f"for {seq}. Requested {n}."
        )

    return sorted(scrambles)


def dominates_maximize(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Pareto dominance for maximization.
    a dominates b if:
      all a_i >= b_i and at least one a_i > b_i
    """
    return bool(np.all(a >= b) and np.any(a > b))


def safe_zscore(x: float, values: np.ndarray) -> float:
    """
    z-score of x relative to values.
    """
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values, ddof=1))

    if not np.isfinite(std) or std == 0:
        return np.nan

    return float((x - mean) / std)


def empirical_p_value_greater_equal(original_value: float, scrambled_values: np.ndarray) -> float:
    """
    One-sided empirical p-value:
    probability that scrambled score is >= original score.

    Smaller is better for showing optimized peptide is better than scrambles.
    Uses +1 smoothing.
    """
    scrambled_values = np.asarray(scrambled_values, dtype=float)
    n = np.sum(np.isfinite(scrambled_values))

    if n == 0:
        return np.nan

    count_ge = np.sum(scrambled_values >= original_value)
    return float((count_ge + 1) / (n + 1))


# ============================================================
# Main analysis
# ============================================================

def main():
    rng = random.Random(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    latent_neighbor_generator = torch.Generator(device="cpu")
    latent_neighbor_generator.manual_seed(LATENT_NEIGHBOR_SEED)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    optimized_df = pd.read_csv(OPTIMIZED_CSV)

    if TRAIN_PARETO_CSV is not None:
        train_path = Path(TRAIN_PARETO_CSV)
        if not train_path.exists():
            raise FileNotFoundError(
                f"Configured TRAIN_PARETO_CSV does not exist: {train_path}"
            )
        train_df = pd.read_csv(train_path)
    else:
        train_df = None

    roundtrip_model = load_roundtrip_model(TRANSFER_CHECKPOINT)

    opt_pep_col = get_peptide_column(optimized_df)

    optimized_peptides = (
        optimized_df[opt_pep_col]
        .astype(str)
        .str.strip()
        .str.upper()
        .dropna()
        .unique()
        .tolist()
    )

    if train_df is not None:
        train_pep_col = get_peptide_column(train_df)
        train_peptides = (
            train_df[train_pep_col]
            .astype(str)
            .str.strip()
            .str.upper()
            .dropna()
            .unique()
            .tolist()
        )
    else:
        train_peptides = []

    forbidden = set()
    if REJECT_EXISTING_SEQUENCES:
        forbidden.update(optimized_peptides)
        forbidden.update(train_peptides)

    print(f"Optimized peptides: {len(optimized_peptides)}")
    print(f"Optional training Pareto peptides used for scramble rejection: {len(train_peptides)}")
    print(f"Optimized input CSV: {OPTIMIZED_CSV}")
    print(f"Latent checkpoint: {TRANSFER_CHECKPOINT}")
    print(f"Scrambles per optimized peptide: {N_SCRAMBLES_PER_PEPTIDE}")

    all_scored_rows = []
    summary_rows = []

    # Diagnostics for every scored sequence: original + all scrambles.
    roundtrip_rows = []
    local_smoothness_detail_rows = []

    for peptide_index, original_peptide in enumerate(optimized_peptides, start=1):
        print(f"\n[{peptide_index}/{len(optimized_peptides)}] Original: {original_peptide}")

        scrambles = generate_unique_scrambles(
            seq=original_peptide,
            n=N_SCRAMBLES_PER_PEPTIDE,
            rng=rng,
            forbidden=forbidden,
        )

        candidate_peptides = [original_peptide] + scrambles

        # ------------------------------------------------------------
        # Peptide-space round trip, latent-space round trip, and
        # local latent smoothness for every sequence in this group.
        # ------------------------------------------------------------
        for sequence in candidate_peptides:
            sequence_id = len(roundtrip_rows)

            rt_row = compute_sequence_roundtrip_diagnostics(
                roundtrip_model,
                sequence,
            )
            rt_row["sequence_id"] = int(sequence_id)
            rt_row["scramble_group_id"] = int(peptide_index)
            rt_row["original_peptide"] = original_peptide
            rt_row["control_type"] = (
                "optimized_original"
                if sequence == original_peptide
                else "scrambled_control"
            )

            smooth_summary, smooth_details = compute_local_latent_smoothness(
                model=roundtrip_model,
                source_peptide=sequence,
                sequence_id=sequence_id,
                generator=latent_neighbor_generator,
                n_neighbors=N_LATENT_NEIGHBORS,
                perturbation_std=LATENT_PERTURBATION_STD,
            )
            rt_row.update(smooth_summary)

            roundtrip_rows.append(rt_row)
            local_smoothness_detail_rows.extend(smooth_details)

        # ------------------------------------------------------------
        # Re-score original + scrambles together.
        #
        # Important:
        # Your blackbox_fc uses min-max normalization inside the batch.
        # Therefore, the fairest control is to score the original peptide
        # and its own scrambled controls in the same call.
        # ------------------------------------------------------------
        scored_df = blackbox_fc(candidate_peptides)

        scored_pep_col = get_peptide_column(scored_df)
        scored_df[scored_pep_col] = scored_df[scored_pep_col].astype(str).str.upper()

        scored_df["original_peptide"] = original_peptide
        scored_df["control_type"] = np.where(
            scored_df[scored_pep_col] == original_peptide,
            "optimized_original",
            "scrambled_control",
        )

        scored_df["scramble_group_id"] = peptide_index

        all_scored_rows.append(scored_df)

        original_rows = scored_df[scored_df["control_type"] == "optimized_original"].copy()
        scrambled_rows = scored_df[scored_df["control_type"] == "scrambled_control"].copy()

        if len(original_rows) != 1:
            print(
                f"[WARNING] Expected exactly one original row for {original_peptide}, "
                f"found {len(original_rows)}."
            )
            continue

        original_row = original_rows.iloc[0]

        original_obj = original_row[OBJECTIVE_COLS].to_numpy(dtype=float)
        scrambled_obj = scrambled_rows[OBJECTIVE_COLS].to_numpy(dtype=float)

        n_scrambles = len(scrambled_rows)

        n_scrambled_dominating_original = 0
        n_scrambled_dominated_by_original = 0

        for i in range(n_scrambles):
            s_obj = scrambled_obj[i]

            if dominates_maximize(s_obj, original_obj):
                n_scrambled_dominating_original += 1

            if dominates_maximize(original_obj, s_obj):
                n_scrambled_dominated_by_original += 1

        summary = {
            "original_peptide": original_peptide,
            "n_scrambles_scored": n_scrambles,

            "original_chelation_sub": float(original_row["chelation_sub"]),
            "original_solubility_sub": float(original_row["solubility_sub"]),
            "original_stability_sub": float(original_row["stability_sub"]),
            "original_expression_sub": float(original_row["expression_sub"]),
            "original_final_score": float(original_row["final_score"]),

            "scrambled_mean_final_score": float(scrambled_rows["final_score"].mean()),
            "scrambled_std_final_score": float(scrambled_rows["final_score"].std(ddof=1)),
            "scrambled_max_final_score": float(scrambled_rows["final_score"].max()),
            "scrambled_95pct_final_score": float(scrambled_rows["final_score"].quantile(0.95)),

            "original_final_score_z_vs_scrambled": safe_zscore(
                float(original_row["final_score"]),
                scrambled_rows["final_score"].to_numpy(dtype=float),
            ),
            "empirical_p_scrambled_ge_original_final_score": empirical_p_value_greater_equal(
                float(original_row["final_score"]),
                scrambled_rows["final_score"].to_numpy(dtype=float),
            ),

            "n_scrambled_dominating_original": int(n_scrambled_dominating_original),
            "n_scrambled_dominated_by_original": int(n_scrambled_dominated_by_original),
            "fraction_scrambled_dominated_by_original": (
                n_scrambled_dominated_by_original / n_scrambles if n_scrambles > 0 else np.nan
            ),
            "fraction_scrambled_dominating_original": (
                n_scrambled_dominating_original / n_scrambles if n_scrambles > 0 else np.nan
            ),
        }

        # Per-objective scrambled summary
        for col in OBJECTIVE_COLS:
            original_value = float(original_row[col])
            scrambled_values = scrambled_rows[col].to_numpy(dtype=float)

            summary[f"scrambled_mean_{col}"] = float(np.nanmean(scrambled_values))
            summary[f"scrambled_std_{col}"] = float(np.nanstd(scrambled_values, ddof=1))
            summary[f"scrambled_max_{col}"] = float(np.nanmax(scrambled_values))
            summary[f"empirical_p_scrambled_ge_original_{col}"] = empirical_p_value_greater_equal(
                original_value,
                scrambled_values,
            )

        summary_rows.append(summary)

    # ============================================================
    # Save detailed and summary outputs
    # ============================================================

    if len(all_scored_rows) == 0:
        raise RuntimeError("No scored rows were generated.")

    all_scores_df = pd.concat(all_scored_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    all_scores_path = out_dir / "scrambling_control_all_scores.csv"
    summary_path = out_dir / "scrambling_control_summary.csv"

    all_scores_df.to_csv(all_scores_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    # ------------------------------------------------------------
    # Save round-trip and latent smoothness diagnostics.
    # ------------------------------------------------------------
    roundtrip_df = pd.DataFrame(roundtrip_rows)
    local_smoothness_detail_df = pd.DataFrame(
        local_smoothness_detail_rows
    )

    roundtrip_metric_columns = [
        "peptide_roundtrip_edit_distance",
        "peptide_roundtrip_edit_distance_normalized",
        "latent_roundtrip_l2",
        "latent_roundtrip_cosine",
        "z1_norm",
        "z2_norm",
        "local_neighbor_edit_distance_mean",
        "local_neighbor_edit_distance_std",
        "local_neighbor_edit_distance_min",
        "local_neighbor_edit_distance_max",
        "local_neighbor_normalized_edit_distance_mean",
        "local_neighbor_identical_fraction",
        "local_neighbor_unique_peptides",
    ]

    roundtrip_with_average_df = append_overall_average_row(
        roundtrip_df,
        metric_columns=roundtrip_metric_columns,
        label_column="source_peptide",
    )

    roundtrip_path = out_dir / "roundtrip_sequence_diagnostics.csv"
    local_smoothness_detail_path = (
        out_dir / "latent_local_smoothness_neighbors.csv"
    )
    roundtrip_average_path = (
        out_dir / "roundtrip_and_smoothness_overall_averages.csv"
    )

    roundtrip_with_average_df.to_csv(roundtrip_path, index=False)
    local_smoothness_detail_df.to_csv(
        local_smoothness_detail_path,
        index=False,
    )

    overall_average_row = {
        "n_sequences_evaluated": int(len(roundtrip_df)),
        "mean_peptide_roundtrip_edit_distance": float(
            roundtrip_df["peptide_roundtrip_edit_distance"].mean()
        ),
        "mean_peptide_roundtrip_edit_distance_normalized": float(
            roundtrip_df[
                "peptide_roundtrip_edit_distance_normalized"
            ].mean()
        ),
        "mean_latent_roundtrip_l2": float(
            roundtrip_df["latent_roundtrip_l2"].mean()
        ),
        "mean_latent_roundtrip_cosine": float(
            roundtrip_df["latent_roundtrip_cosine"].mean()
        ),
        "mean_local_neighbor_edit_distance_to_p1": float(
            roundtrip_df["local_neighbor_edit_distance_mean"].mean()
        ),
        "mean_local_neighbor_normalized_edit_distance_to_p1": float(
            roundtrip_df[
                "local_neighbor_normalized_edit_distance_mean"
            ].mean()
        ),
        "mean_local_neighbor_identical_fraction": float(
            roundtrip_df["local_neighbor_identical_fraction"].mean()
        ),
        "mean_local_neighbor_unique_peptides": float(
            roundtrip_df["local_neighbor_unique_peptides"].mean()
        ),
        "n_latent_neighbors_per_sequence": int(N_LATENT_NEIGHBORS),
        "latent_perturbation_std": float(LATENT_PERTURBATION_STD),
        "checkpoint": TRANSFER_CHECKPOINT,
    }
    pd.DataFrame([overall_average_row]).to_csv(
        roundtrip_average_path,
        index=False,
    )

    print(f"\nSaved detailed scores: {all_scores_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved per-sequence round-trip diagnostics: {roundtrip_path}")
    print(
        "Saved local latent-neighbor diagnostics: "
        f"{local_smoothness_detail_path}"
    )
    print(
        "Saved overall round-trip/smoothness averages: "
        f"{roundtrip_average_path}"
    )

    print("\n=== Round-trip / latent smoothness averages ===")
    for key, value in overall_average_row.items():
        if key != "checkpoint":
            print(f"{key}: {value}")

    # ============================================================
    # Plot 1: final score boxplot by original peptide
    # ============================================================

    plt.figure(figsize=(12, 6))

    # peptides_order = optimized_peptides
    peptides_order = (
    all_scores_df["original_peptide"]
    .astype(str)
    .drop_duplicates()
    .tolist()
    )
    box_data = []

    for pep in peptides_order:
        vals = all_scores_df[
            (all_scores_df["original_peptide"] == pep)
            & (all_scores_df["control_type"] == "scrambled_control")
        ]["final_score"].to_numpy(dtype=float)

        box_data.append(vals)

    plt.boxplot(box_data, labels=peptides_order, showfliers=True)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Final score")
    plt.title("Scrambling control: final-score distribution of scrambled peptides")

    # Add original final scores as red dots
    for i, pep in enumerate(peptides_order, start=1):
        orig_rows_for_plot = all_scores_df[
            (all_scores_df["original_peptide"] == pep)
            & (
                all_scores_df["control_type"]
                == "optimized_original"
            )
        ]["final_score"]

        if orig_rows_for_plot.empty:
            print(
                f"[WARNING] Skipping original-score marker for {pep}: "
                "no optimized_original row was found."
            )
            continue

        orig_val = float(orig_rows_for_plot.iloc[0])
        # orig_val = all_scores_df[
        #     (all_scores_df["original_peptide"] == pep)
        #     & (all_scores_df["control_type"] == "optimized_original")
        # ]["final_score"].iloc[0]

        plt.scatter(i, orig_val, marker="D", s=70, label="Original" if i == 1 else None)

    plt.legend()
    plt.tight_layout()

    plot_path = out_dir / "scrambling_control_final_score_boxplot.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Saved plot: {plot_path}")

    # ============================================================
    # Plot 2: original vs scrambled mean final score
    # ============================================================

    plt.figure(figsize=(12, 6))

    x = np.arange(len(summary_df))
    width = 0.35

    plt.bar(
        x - width / 2,
        summary_df["original_final_score"],
        width,
        label="Original optimized peptide",
    )

    plt.bar(
        x + width / 2,
        summary_df["scrambled_mean_final_score"],
        width,
        yerr=summary_df["scrambled_std_final_score"],
        capsize=3,
        label="Scrambled controls mean ± SD",
    )

    plt.xticks(x, summary_df["original_peptide"], rotation=45, ha="right")
    plt.ylabel("Final score")
    plt.title("Original optimized peptides vs scrambled controls")
    plt.legend()
    plt.tight_layout()

    plot_path = out_dir / "original_vs_scrambled_mean_final_score.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Saved plot: {plot_path}")

    # ============================================================
    # Plot 3: empirical p-values
    # ============================================================

    plt.figure(figsize=(12, 5))

    plt.bar(
        summary_df["original_peptide"],
        summary_df["empirical_p_scrambled_ge_original_final_score"],
    )

    plt.axhline(0.05, linestyle="--", linewidth=1, label="p = 0.05")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Empirical p-value")
    plt.title("Probability that scrambled controls score at least as high as original")
    plt.legend()
    plt.tight_layout()

    plot_path = out_dir / "empirical_p_values_final_score.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Saved plot: {plot_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()