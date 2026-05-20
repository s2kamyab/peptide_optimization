# peptide_seq_vae_with_test.py
# Sequence-VAE (GRU) for peptide strings length <= 10 (variable supported)
#
# Adds:
# - train/val/test split
# - explicit test evaluation (reconstruction error, KL, accuracy)
# - mapping latent z -> peptide strings (decode from z)
# - learning curve: logs + plot (train/val recon per epoch)
#
# Run:
#   pip install torch pandas numpy scikit-learn matplotlib
#   python peptide_seq_vae_with_test.py

import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

CSV_PATH = "metalpdb_binding_windows_len10_MN.csv"
# --- ADD near the top (after CSV_PATH / SEQ_COL defs) ---
CKPT_PATH = os.path.join("peptide_vae_out", "seqvae.pt")  # load from here if exists
SAVE_EVERY_EPOCH = True  # keep simple; you can also save every N epochs if you want
SEQ_COL = "peptide_len10"
LABEL_COL = "binding_site_labels_len10"  # optional; used to derive a toy objective
def save_ckpt(ckpt_path: str, model: nn.Module, opt: torch.optim.Optimizer, *, epoch: int, step: int, tok_vocab: list):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "opt_state": opt.state_dict(),
            "epoch": int(epoch),
            "step": int(step),
            "vocab": tok_vocab,
        },
        ckpt_path,
    )
    print(f"[ckpt] Saved -> {ckpt_path} (epoch={epoch}, step={step})")


def load_ckpt(ckpt_path: str, model: nn.Module, opt: torch.optim.Optimizer):
    """
    True resume: model + optimizer + epoch + step.
    Returns: (start_epoch, step)
    """
    if not os.path.isfile(ckpt_path):
        print(f"[ckpt] No checkpoint at {ckpt_path}; starting fresh.")
        return 1, 0

    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    model.load_state_dict(ckpt["state_dict"], strict=True)
    opt.load_state_dict(ckpt["opt_state"])

    last_epoch = int(ckpt.get("epoch", 0))
    step = int(ckpt.get("step", 0))

    print(f"[ckpt] Loaded <- {ckpt_path} (last_epoch={last_epoch}, step={step})")
    return last_epoch + 1, step
# --- ADD this helper somewhere above main() ---
def maybe_resume_from_ckpt(model: nn.Module, ckpt_path: str) -> int:
    """
    Load model weights (and optionally optimizer/step/epoch if present) and return start_epoch.
    Your existing ckpt only stores {"state_dict","vocab"} so we resume model weights only.
    """
    if not os.path.isfile(ckpt_path):
        print(f"[resume] No checkpoint found at: {ckpt_path} (starting from scratch)")
        return 1

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)

    print(f"[resume] Loaded weights from: {ckpt_path}")
    if missing:
        print(f"[resume] Missing keys (ok if architecture changed): {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"[resume] Unexpected keys: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")

    # If you later save epoch in ckpt, you can resume it here:
    last_epoch = int(ckpt.get("epoch", 0))
    return last_epoch + 1
# -----------------------------
# Reproducibility
# -----------------------------
def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_all(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Tokenizer
# -----------------------------
class PeptideTokenizer:
    def __init__(self):
        aa = list("ACDEFGHIKLMNPQRSTVWY")
        self.specials = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
        self.vocab = self.specials + aa
        self.stoi = {t: i for i, t in enumerate(self.vocab)}
        self.itos = {i: t for t, i in self.stoi.items()}

        self.PAD = self.stoi["<PAD>"]
        self.BOS = self.stoi["<BOS>"]
        self.EOS = self.stoi["<EOS>"]
        self.UNK = self.stoi["<UNK>"]

    def encode(self, seq: str, add_bos_eos: bool = True) -> List[int]:
        seq = (seq or "").strip().upper()
        toks = []
        if add_bos_eos:
            toks.append(self.BOS)
        for ch in seq:
            toks.append(self.stoi.get(ch, self.UNK))
        if add_bos_eos:
            toks.append(self.EOS)
        return toks

    def decode(self, ids: List[int], stop_at_eos: bool = True) -> str:
        out = []
        for i in ids:
            if stop_at_eos and i == self.EOS:
                break
            if i in (self.BOS, self.PAD):
                continue
            tok = self.itos.get(int(i), "")
            if tok.startswith("<"):
                continue
            out.append(tok)
        return "".join(out)


# -----------------------------
# Dataset & collate
# -----------------------------
@dataclass
class Batch:
    x_in: torch.Tensor      # (B, T) decoder input tokens
    x_tgt: torch.Tensor     # (B, T) decoder targets
    lengths: torch.Tensor   # (B,) true lengths for x_in (incl BOS, excl last token)
    y: Optional[torch.Tensor] = None


class PeptideDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: PeptideTokenizer, max_len: int = 10):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _binding_count(label_str: str) -> float:
        if not isinstance(label_str, str):
            return 0.0
        return float(sum(1 for t in label_str.strip().split() if t == "1"))

    def __getitem__(self, idx: int):
        seq = str(self.df.loc[idx, SEQ_COL]).strip()
        ids = self.tok.encode(seq, add_bos_eos=True)

        # Max token length = max_len + 2 (BOS + EOS)
        max_tok_len = self.max_len + 2
        ids = ids[:max_tok_len]

        y = None
        if LABEL_COL in self.df.columns:
            y = self._binding_count(self.df.loc[idx, LABEL_COL])

        return ids, y


def collate_fn(batch, pad_id: int) -> Batch:
    ids_list, y_list = zip(*batch)
    lengths_full = torch.tensor([len(ids) for ids in ids_list], dtype=torch.long)

    T = int(max(lengths_full).item())
    x = torch.full((len(ids_list), T), pad_id, dtype=torch.long)
    for i, ids in enumerate(ids_list):
        x[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

    # Teacher forcing
    x_in = x[:, :-1].contiguous()
    x_tgt = x[:, 1:].contiguous()
    lengths_in = (lengths_full - 1).clamp(min=1)

    y = None
    if y_list[0] is not None:
        y = torch.tensor(y_list, dtype=torch.float32)

    return Batch(x_in=x_in, x_tgt=x_tgt, lengths=lengths_in, y=y)


# -----------------------------
# VAE Model (GRU)
# -----------------------------
class SeqVAE(nn.Module):
    """
    More expressive SeqVAE:
      - bidirectional encoder GRU
      - multi-layer decoder GRU
      - dropout
      - condition decoder on z via:
          (a) initializing ALL decoder layers from z
          (b) adding z->embedding bias at every time step
    """
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        emb_dim: int = 128,
        hid_dim: int = 256,
        z_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidir_encoder: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.z_dim = z_dim
        self.num_layers = num_layers
        self.bidir_encoder = bidir_encoder

        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.emb_drop = nn.Dropout(dropout)

        # -------- Encoder --------
        self.enc_gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidir_encoder,
        )

        enc_out_dim = hid_dim * (2 if bidir_encoder else 1)

        # Small MLP before mu/logvar (helps)
        self.enc_mlp = nn.Sequential(
            nn.Linear(enc_out_dim, enc_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mu = nn.Linear(enc_out_dim, z_dim)
        self.logvar = nn.Linear(enc_out_dim, z_dim)

        # -------- Decoder --------
        # Initialize ALL decoder layers from z
        self.z_to_h = nn.Linear(z_dim, num_layers * hid_dim)

        # Add z-conditioned bias to token embedding at each step
        self.z_to_emb = nn.Linear(z_dim, emb_dim)

        self.dec_gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        # Output projection + dropout
        self.dec_drop = nn.Dropout(dropout)
        self.out = nn.Linear(hid_dim, vocab_size)

    def encode(self, x_in: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x_in: (B, T)
        emb = self.emb_drop(self.emb(x_in))  # (B, T, E)

        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.enc_gru(packed)

        if self.bidir_encoder:
            # h_n shape: (num_layers*2, B, H)
            # Take last layer forward/back and concat
            h_fwd = h_n[-2]  # (B, H)
            h_bwd = h_n[-1]  # (B, H)
            h_last = torch.cat([h_fwd, h_bwd], dim=-1)  # (B, 2H)
        else:
            # h_n shape: (num_layers, B, H)
            h_last = h_n[-1]  # (B, H)

        h_last = self.enc_mlp(h_last)
        mu = self.mu(h_last)
        logvar = self.logvar(h_last)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _init_dec_hidden(self, z: torch.Tensor) -> torch.Tensor:
        # Return h0 for all layers: (num_layers, B, H)
        B = z.size(0)
        h0 = torch.tanh(self.z_to_h(z))  # (B, L*H)
        h0 = h0.view(B, self.num_layers, self.hid_dim).transpose(0, 1).contiguous()
        return h0

    def decode_logits(self, x_in: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # x_in: (B, T), z: (B, Z)
        B, T = x_in.shape
        emb = self.emb(x_in)  # (B, T, E)
        z_bias = self.z_to_emb(z).unsqueeze(1)  # (B, 1, E)
        emb = self.emb_drop(emb + z_bias)       # (B, T, E)

        h0 = self._init_dec_hidden(z)           # (L, B, H)
        out_seq, _ = self.dec_gru(emb, h0)      # (B, T, H)
        out_seq = self.dec_drop(out_seq)
        logits = self.out(out_seq)              # (B, T, V)
        return logits

    def forward(self, x_in: torch.Tensor, x_tgt: torch.Tensor, lengths: torch.Tensor):
        mu, logvar = self.encode(x_in, lengths)
        z = self.reparameterize(mu, logvar)
        logits = self.decode_logits(x_in, z)
        return logits, mu, logvar, z


# -----------------------------
# Losses & metrics
# -----------------------------
def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

def kl_divergence_free_bits(mu, logvar, free_bits=0.1):
    # kl per sample per dim: (B, Z)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    # enforce minimum per-dimension KL
    kl = torch.clamp(kl, min=free_bits)
    return kl.sum(dim=-1).mean()


def recon_loss(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> torch.Tensor:
    B, T, V = logits.shape
    return F.cross_entropy(
        logits.view(B * T, V),
        targets.view(B * T),
        ignore_index=pad_id,
        reduction="mean",
    )

@torch.no_grad()
def token_accuracy(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> float:
    pred = torch.argmax(logits, dim=-1)
    mask = targets.ne(pad_id)
    correct = (pred.eq(targets) & mask).sum().item()
    total = mask.sum().item()
    return float(correct) / float(total) if total > 0 else 0.0


# -----------------------------
# z -> peptide mapping (decode)
# -----------------------------
@torch.no_grad()
@torch.no_grad()
def z_to_peptides(
    model: SeqVAE,
    tok: PeptideTokenizer,
    z: torch.Tensor,
    max_pep_len: int = 10,
    greedy: bool = True,
    temperature: float = 1.0,
) -> List[str]:
    model.eval()
    z = z.to(next(model.parameters()).device)
    B = z.size(0)

    x = torch.full((B, 1), tok.BOS, dtype=torch.long, device=z.device)

    # init all decoder layers from z
    h = model._init_dec_hidden(z)  # (L, B, H)
    z_bias = model.z_to_emb(z)     # (B, E) add to token embeddings

    out_ids = []
    max_steps = max_pep_len + 1
    for _ in range(max_steps):
        emb_tok = model.emb(x[:, -1:])                 # (B, 1, E)
        emb_tok = model.emb_drop(emb_tok + z_bias.unsqueeze(1))  # (B, 1, E)

        o, h = model.dec_gru(emb_tok, h)               # (B, 1, H)
        logits = model.out(model.dec_drop(o[:, 0, :])) # (B, V)

        if temperature != 1.0:
            logits = logits / max(1e-6, float(temperature))

        if greedy:
            nxt = torch.argmax(logits, dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1).squeeze(1)

        out_ids.append(nxt)
        x = torch.cat([x, nxt.unsqueeze(1)], dim=1)

    peptides = []
    out_mat = torch.stack(out_ids, dim=1)  # (B, steps)
    for b in range(B):
        peptides.append(tok.decode(out_mat[b].tolist(), stop_at_eos=True))
    return peptides


# -----------------------------
# Encode utilities
# -----------------------------
@torch.no_grad()
def encode_to_z(model: SeqVAE, loader: DataLoader, use_mean: bool = True) -> np.ndarray:
    model.eval()
    Zs = []
    for batch in loader:
        x_in = batch.x_in.to(DEVICE)
        lengths = batch.lengths.to(DEVICE)
        mu, logvar = model.encode(x_in, lengths)
        z = mu if use_mean else model.reparameterize(mu, logvar)
        Zs.append(z.detach().cpu().numpy())
    return np.vstack(Zs) if len(Zs) else np.zeros((0,))


@torch.no_grad()
def reconstruct(model: SeqVAE, tok: PeptideTokenizer, batch: Batch, max_pep_len: int = 10) -> Tuple[List[str], List[str]]:
    model.eval()
    x_in = batch.x_in.to(DEVICE)
    lengths = batch.lengths.to(DEVICE)
    mu, _ = model.encode(x_in, lengths)
    recons = z_to_peptides(model, tok, mu, max_pep_len=max_pep_len, greedy=True)

    originals = []
    x_tgt = batch.x_tgt.cpu()
    for i in range(x_tgt.size(0)):
        originals.append(tok.decode(x_tgt[i].tolist(), stop_at_eos=True))
    return originals, recons


# -----------------------------
# Eval loop (val/test)
# -----------------------------
@torch.no_grad()
def evaluate(model: SeqVAE, loader: DataLoader, pad_id: int, beta: float = 1.0) -> Dict[str, float]:
    model.eval()
    recon_sum, kl_sum, acc_sum, n_batches = 0.0, 0.0, 0.0, 0
    for batch in loader:
        x_in = batch.x_in.to(DEVICE)
        x_tgt = batch.x_tgt.to(DEVICE)
        lengths = batch.lengths.to(DEVICE)

        logits, mu, logvar, _ = model(x_in, x_tgt, lengths)
        r = recon_loss(logits, x_tgt, pad_id).item()
        k = kl_divergence(mu, logvar).item()
        a = token_accuracy(logits, x_tgt, pad_id)

        recon_sum += r
        kl_sum += k
        acc_sum += a
        n_batches += 1

    if n_batches == 0:
        return {"recon": 0.0, "kl": 0.0, "acc": 0.0, "elbo": 0.0}

    recon = recon_sum / n_batches
    kl = kl_sum / n_batches
    acc = acc_sum / n_batches
    elbo = recon + beta * kl
    return {"recon": recon, "kl": kl, "acc": acc, "elbo": elbo}


def save_learning_curve(history: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "learning_curve.csv")
    png_path = os.path.join(out_dir, "learning_curve.png")

    history.to_csv(csv_path, index=False)

    # Plot: recon loss (train vs val)
    plt.figure()
    plt.plot(history["epoch"], history["train_recon"], label="train_recon")
    plt.plot(history["epoch"], history["val_recon"], label="val_recon")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction loss (CE)")
    plt.title("Learning Curve: Reconstruction Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    print(f"Saved learning curve CSV to {csv_path}")
    print(f"Saved learning curve plot to {png_path}")

def word_dropout(x_in: torch.Tensor, unk_id: int, bos_id: int, drop_prob: float) -> torch.Tensor:
    """
    Randomly replace tokens in x_in with UNK (except BOS and PAD).
    x_in: (B, T)
    """
    tok = PeptideTokenizer()
    if drop_prob <= 0:
        return x_in
    x = x_in.clone()
    # mask tokens eligible for dropout (not BOS, not PAD)
    eligible = (x != bos_id) & (x != tok.PAD)
    drop = (torch.rand_like(x.float()) < drop_prob) & eligible
    x[drop] = unk_id
    return x

# -----------------------------
# Main
# -----------------------------
def main():
    df = pd.read_csv(CSV_PATH)
    df = df.drop_duplicates(subset=[SEQ_COL]).reset_index(drop=True) # At minimum, deduplicate sequences for VAE training:

    # df = df[df[SEQ_COL].notna()].copy()
    df[SEQ_COL] = df[SEQ_COL].astype(str).str.strip().str.upper()

    tok = PeptideTokenizer()
    max_pep_len = 10

    # Split: train/val/test
    n = len(df)
    idx = np.random.permutation(n)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train_df = df.iloc[idx[:n_train]].copy()
    val_df = df.iloc[idx[n_train:n_train + n_val]].copy()
    test_df = df.iloc[idx[n_train + n_val:]].copy()

    train_ds = PeptideDataset(train_df, tok, max_len=max_pep_len)
    val_ds = PeptideDataset(val_df, tok, max_len=max_pep_len)
    test_ds = PeptideDataset(test_df, tok, max_len=max_pep_len)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, tok.PAD), num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, tok.PAD), num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             collate_fn=lambda b: collate_fn(b, tok.PAD), num_workers=0)

    # # model = SeqVAE(
    # # vocab_size=len(tok.vocab),
    # # pad_id=tok.PAD,
    # # emb_dim=128,
    # # hid_dim=256,
    # # z_dim=64,
    # # num_layers=2,
    # # dropout=0.2,
    # # bidir_encoder=True,
    # # ).to(DEVICE)

    # model = SeqVAE(
    # vocab_size=len(tok.vocab),
    # pad_id=tok.PAD,
    # emb_dim=128,
    # hid_dim=256,
    # z_dim=64,
    # num_layers=2,
    # dropout=0.2,
    # bidir_encoder=True,
    # ).to(DEVICE)

    # start_epoch = maybe_resume_from_ckpt(model, CKPT_PATH)

    # opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    model = SeqVAE(
    vocab_size=len(tok.vocab),
    pad_id=tok.PAD,
    emb_dim=128,
    hid_dim=256,
    z_dim=64,
    num_layers=2,
    dropout=0.2,
    bidir_encoder=True,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total params:", total_params)
    print("Trainable params:", trainable_params)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    # --- TRUE RESUME HERE ---
    start_epoch, step = load_ckpt(CKPT_PATH, model, opt)

    # KL annealing
    epochs = 60
    beta_max = 0.02
    warmup_steps = 20000
    # step = 0


    # opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    # KL annealing
    # epochs = 20
    # beta_max = 0.02
    # warmup_steps = 20000 #This lets the model behave more like an autoencoder early, so it learns to encode information in z before KL becomes meaningful.
    # step = 0

    def beta_schedule(step_: int) -> float:
        return min(beta_max, beta_max * (step_ / float(warmup_steps)))

    history_rows = []

    # Train
    # for ep in range(1, epochs + 1):
    best_val = float("inf")
    BEST_CKPT_PATH = os.path.join("peptide_vae_out", "seqvae_best.pt")
    for ep in range(start_epoch, epochs + 1):
        # --- OPTIONAL BUT RECOMMENDED: when saving, include epoch so resume can continue epoch count ---
        
        model.train()
        tr_recon, tr_kl, tr_acc, tr_batches = 0.0, 0.0, 0.0, 0

        for batch in train_loader:
            step += 1
            beta = beta_schedule(step)

            x_in = batch.x_in.to(DEVICE)
            x_tgt = batch.x_tgt.to(DEVICE)
            lengths = batch.lengths.to(DEVICE)
            # FORCE reliance on z
            x_in_noisy = word_dropout(x_in, tok.UNK, tok.BOS, drop_prob=0.3) # drop_prob=0.3 (try 0.2–0.5). #Add word dropout (a.k.a. token dropout) on decoder inputs during training
            # This is the standard fix for sequence VAEs. You randomly replace some tokens in x_in with <UNK> so the decoder cannot rely purely on teacher forcing and is forced to use z.

            logits, mu, logvar, _ = model(x_in_noisy, x_tgt, lengths)
            rloss = recon_loss(logits, x_tgt, tok.PAD)
            # kll = kl_divergence(mu, logvar)
            # loss = rloss + beta * kll
            kll = kl_divergence_free_bits(mu, logvar, free_bits=0.05)  # tune 0.02–0.2 # This prevents the optimizer from driving KL to ~0 for “free.”

            #Replace your KL computation with a “free bits” version:
            loss = rloss + beta * kll
            #Print average ||mu|| and mean(logvar) on val; if mu is near 0 and logvar near 0, you’re collapsed.

            #Track KL over time; if it stays near 0, z is dead.

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_recon += float(rloss.item())
            tr_kl += float(kll.item())
            tr_acc += token_accuracy(logits.detach(), x_tgt, tok.PAD)
            tr_batches += 1
            

        tr_recon /= max(1, tr_batches)
        tr_kl /= max(1, tr_batches)
        tr_acc /= max(1, tr_batches)

        # Validation metrics
        val_metrics = evaluate(model, val_loader, pad_id=tok.PAD, beta=beta_max)
        if val_metrics["recon"] < best_val:
            best_val = val_metrics["recon"]
            save_ckpt(BEST_CKPT_PATH, model, opt, epoch=ep, step=step, tok_vocab=tok.vocab)

        # Save history row (learning curve)
        history_rows.append({
            "epoch": ep,
            "train_recon": tr_recon,
            "train_kl": tr_kl,
            "train_acc": tr_acc,
            "val_recon": val_metrics["recon"],
            "val_kl": val_metrics["kl"],
            "val_acc": val_metrics["acc"],
        })

        print(
            f"Epoch {ep:02d} | "
            f"train recon {tr_recon:.4f} kl {tr_kl:.4f} acc {tr_acc:.4f} | "
            f"val recon {val_metrics['recon']:.4f} kl {val_metrics['kl']:.4f} "
            f"acc {val_metrics['acc']:.4f}"
        )

        # Quick qualitative: decode from random z
        with torch.no_grad():
            z = torch.randn(8, model.z_dim, device=DEVICE)  # 8x64
            samples = z_to_peptides(model, tok, z, max_pep_len=max_pep_len, greedy=True)
            print("  z samples:", samples[:5])

        # Quick qualitative: recon a batch from val
        with torch.no_grad():
            b = next(iter(val_loader))
            orig, rec = reconstruct(model, tok, b, max_pep_len=max_pep_len)
            for i in range(min(3, len(orig))):
                print(f"  recon {i}: {orig[i]}  ->  {rec[i]}")
        if SAVE_EVERY_EPOCH:
            save_ckpt(CKPT_PATH, model, opt, epoch=ep, step=step, tok_vocab=tok.vocab)

    # Save learning curve artifacts
    out_dir = "peptide_vae_out"
    history = pd.DataFrame(history_rows)
    if len(history_rows) > 0:
        save_learning_curve(history, out_dir)
    else:
        print("[warn] No epochs ran; skipping learning curve save.")

    # Test evaluation
    test_metrics = evaluate(model, test_loader, pad_id=tok.PAD, beta=beta_max)
    print("\n=== Test set ===")
    print(
        f"test recon {test_metrics['recon']:.4f} | "
        f"test kl {test_metrics['kl']:.4f} | "
        f"test acc {test_metrics['acc']:.4f} | "
        f"test elbo {test_metrics['elbo']:.4f}"
    )

    # # Save model
    # ckpt_path = os.path.join(out_dir, "seqvae.pt")
    # torch.save(
    #     {
    #         "state_dict": model.state_dict(),
    #         "vocab": tok.vocab,
    #         "epoch": ep,  # last finished epoch
    #     },
    #     ckpt_path,
    # )
    # print(f"\nSaved model to {ckpt_path}")
    # ckpt_path = os.path.join(out_dir, "seqvae.pt")
    # torch.save({"state_dict": model.state_dict(), "vocab": tok.vocab}, ckpt_path)
    # print(f"\nSaved model to {ckpt_path}")

    # Example: encode test -> z and decode
    with torch.no_grad():
        Z_test = encode_to_z(model, test_loader, use_mean=True)
        z0 = torch.tensor(Z_test[:8], dtype=torch.float32, device=DEVICE)
        decoded = z_to_peptides(model, tok, z0, max_pep_len=max_pep_len, greedy=True)
        print("\nDecoded from test mu (first 5):", decoded[:5])


if __name__ == "__main__":
    main()
