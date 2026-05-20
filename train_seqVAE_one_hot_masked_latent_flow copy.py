# flow_gru_vae_fixedlen10.py
# Fixed-length (10) one-hot (20) latent-flow GRU VAE with per-layer monitoring
# Reads: metalpdb_binding_windows_len10_MN.csv column "peptide_len10"

import math
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# -----------------------
# Config
# -----------------------
CSV_PATH = "metalpdb_binding_windows_len10_MN.csv"   # your uploaded file path
SEQ_COL = "peptide_len10"

AA = "ACDEFGHIKLMNPQRSTVWY"  # 20
AA_TO_I = {a: i for i, a in enumerate(AA)}
VOCAB = len(AA)              # 20
T = 10                       # fixed length

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = "flow_gru_vae_monitoring"
os.makedirs(OUT_DIR, exist_ok=True)


# -----------------------
# Helpers: log probs
# -----------------------
def log_normal_diag(z, mu, logvar):
    # (B,D)
    return -0.5 * (math.log(2 * math.pi) + logvar + ((z - mu) ** 2) / logvar.exp()).sum(dim=1)

def log_standard_normal(z):
    # (B,D)
    return -0.5 * (math.log(2 * math.pi) + z**2).sum(dim=1)


# -----------------------
# Planar Flow
# z' = z + u * tanh(w^T z + b)
# enforce invertibility via u_hat
# -----------------------
class PlanarFlow(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim) * 0.01)
        self.u = nn.Parameter(torch.randn(dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(()))

    @staticmethod
    def h(x):
        return torch.tanh(x)

    @staticmethod
    def h_prime(x):
        t = torch.tanh(x)
        return 1.0 - t * t

    def _u_hat(self):
        wu = torch.dot(self.w, self.u)
        m = -1.0 + F.softplus(wu)
        w_norm_sq = (self.w * self.w).sum() + 1e-8
        return self.u + (m - wu) * self.w / w_norm_sq

    def forward(self, z):
        u_hat = self._u_hat()
        a = (z * self.w).sum(dim=1) + self.b                 # (B,)  <-- pre-activation
        h = self.h(a)
        z_new = z + u_hat.unsqueeze(0) * h.unsqueeze(1)

        psi = self.h_prime(a).unsqueeze(1) * self.w.unsqueeze(0)
        det_term = 1.0 + (psi * u_hat.unsqueeze(0)).sum(dim=1)
        log_abs_det = torch.log(det_term.abs() + 1e-8)
        return z_new, log_abs_det, a


class Flow(nn.Module):
    def __init__(self, dim: int, n_flows: int):
        super().__init__()
        self.flows = nn.ModuleList([PlanarFlow(dim) for _ in range(n_flows)])

    def forward(self, z0):
        logdet_sum = torch.zeros(z0.size(0), device=z0.device)
        z = z0
        preacts = []  # list of a tensors
        for f in self.flows:
            z, ld, a = f(z)
            logdet_sum = logdet_sum + ld
            preacts.append(a)
        return z, logdet_sum, preacts


# @dataclass
# class VAEOutputs:
#     logits: torch.Tensor
#     mu: torch.Tensor
#     logvar: torch.Tensor
#     z0: torch.Tensor
#     zK: torch.Tensor
#     logdet: torch.Tensor
#     flow_preacts: list                 # <-- NEW: list[(B,)] length n_flows
#     enc_cache: Dict[str, torch.Tensor]
#     dec_cache: Dict[str, torch.Tensor]


# -----------------------
# Stacked GRUCells with per-layer monitoring
# (no mask, fixed length)
# -----------------------
class StackedGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.cells = nn.ModuleList([
            nn.GRUCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                 # (B,T,Din)
        h0: Optional[torch.Tensor] = None,  # (L,B,H)
        return_layer_outputs: bool = True,
    ):
        B, T_, Din = x.shape
        device = x.device

        if h0 is None:
            h = [torch.zeros(B, self.hidden_size, device=device) for _ in range(self.num_layers)]
        else:
            h = [h0[i] for i in range(self.num_layers)]

        layer_outs = {f"layer_{i}": [] for i in range(self.num_layers)} if return_layer_outputs else None
        final_outs = []

        for t in range(T_):
            xt = x[:, t, :]

            for i, cell in enumerate(self.cells):
                inp = xt if i == 0 else h[i - 1]
                h[i] = cell(inp, h[i])

                if i < self.num_layers - 1 and self.dropout > 0:
                    h[i] = self.drop(h[i])

                if return_layer_outputs:
                    layer_outs[f"layer_{i}"].append(h[i])

            final_outs.append(h[-1])

        out = torch.stack(final_outs, dim=1)   # (B,T,H)
        hN = torch.stack(h, dim=0)             # (L,B,H)

        cache = None
        if return_layer_outputs:
            cache = {k: torch.stack(v, dim=1) for k, v in layer_outs.items()}  # each (B,T,H)

        return out, hN, cache


# -----------------------
# Outputs container
# -----------------------
@dataclass
class VAEOutputs:
    logits: torch.Tensor               # (B,10,20)
    mu: torch.Tensor                   # (B,D)
    logvar: torch.Tensor               # (B,D)
    z0: torch.Tensor                   # (B,D)
    zK: torch.Tensor                   # (B,D)
    logdet: torch.Tensor               # (B,)
    flow_preacts: list                 # <-- NEW: list[(B,)] length n_flows
    enc_cache: Dict[str, torch.Tensor] # per-layer (B,10,D)
    dec_cache: Dict[str, torch.Tensor] # per-layer (B,10,D)

# -----------------------
# FixedLen10 Flow-GRU VAE
# - input onehot (B,10,20), no mask
# - internal dim D everywhere (encoder layers, decoder layers, latent)
# -----------------------
class FixedLenFlowGRUVAE(nn.Module):
    def __init__(
        self,
        d_model: int = 256,          # internal dimension (all GRU layers)
        num_layers: int = 2,
        dropout: float = 0.1,
        n_flows: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        # Project one-hot (20) -> D for equal internal dims
        self.in_proj = nn.Linear(VOCAB, d_model)
        self.in_drop = nn.Dropout(dropout)

        # Encoder and decoder stacks
        self.encoder = StackedGRU(input_size=d_model, hidden_size=d_model, num_layers=num_layers, dropout=dropout)
        self.decoder = StackedGRU(input_size=d_model, hidden_size=d_model, num_layers=num_layers, dropout=dropout)

        # mu/logvar are D->D (no shrinking)
        self.to_mu = nn.Linear(d_model, d_model)
        self.to_logvar = nn.Linear(d_model, d_model)

        # Flow in latent space (dim = D)
        self.flow = Flow(dim=d_model, n_flows=n_flows)

        # Decoder init from zK for ALL layers (D -> L*D)
        self.z_to_h = nn.Linear(d_model, num_layers * d_model)

        # Decoder input conditioning: add z bias into inputs (optional but helpful)
        self.z_to_inbias = nn.Linear(d_model, d_model)

        # Output projection: D -> 20
        self.out = nn.Linear(d_model, VOCAB)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def init_dec_hidden(self, zK):
        # (B,D) -> (L,B,D)
        B = zK.size(0)
        h0 = torch.tanh(self.z_to_h(zK)).view(B, self.num_layers, self.d_model).transpose(0, 1).contiguous()
        return h0

    def forward(self, x_onehot: torch.Tensor) -> VAEOutputs:
        """
        x_onehot: (B,10,20)
        """
        # project to D (keep dims equal internally)
        x = self.in_drop(self.in_proj(x_onehot))  # (B,10,D)

        # Encoder
        enc_out, enc_hN, enc_cache = self.encoder(x, h0=None, return_layer_outputs=True)
        h_last = enc_hN[-1]  # (B,D) last layer final hidden

        mu = self.to_mu(h_last)
        logvar = self.to_logvar(h_last)

        z0 = self.reparameterize(mu, logvar)
        zK, logdet, preacts = self.flow(z0)


        # Decoder teacher forcing: feed same x, but bias with zK
        h0_dec = self.init_dec_hidden(zK)                 # (L,B,D)
        z_bias = self.z_to_inbias(zK).unsqueeze(1)        # (B,1,D)
        x_dec = x + z_bias                                # (B,10,D)

        dec_out, dec_hN, dec_cache = self.decoder(x_dec, h0=h0_dec, return_layer_outputs=True)

        logits = self.out(dec_out)  # (B,10,20)
        return VAEOutputs(
            logits=logits,
            mu=mu, logvar=logvar,
            z0=z0, zK=zK, logdet=logdet,
            flow_preacts=preacts,
            enc_cache=enc_cache, dec_cache=dec_cache,
        )



# -----------------------
# Loss: recon + flow KL
# recon: predict token at each of 10 positions
# -----------------------
def flow_vae_loss(outputs: VAEOutputs, targets_ids: torch.Tensor, beta: float = 0.01):
    """
    targets_ids: (B,10) values in [0..19]
    """
    B, T_, V = outputs.logits.shape

    recon = F.cross_entropy(
        outputs.logits.reshape(B * T_, V),
        targets_ids.reshape(B * T_),
        reduction="mean",
    )

    log_q0 = log_normal_diag(outputs.z0, outputs.mu, outputs.logvar)  # (B,)
    log_qK = log_q0 - outputs.logdet                                  # (B,)
    log_p = log_standard_normal(outputs.zK)                           # (B,)

    kl = (log_qK - log_p).mean()

    loss = recon + beta * kl
    return loss, {"recon": float(recon.item()), "kl": float(kl.item()), "loss": float(loss.item())}


# -----------------------
# Dataset: fixed length 10 peptides in column peptide_len10
# -----------------------
class PeptideLen10Dataset(Dataset):
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        seqs = df[SEQ_COL].astype(str).str.strip().str.upper()

        # keep only exact length 10 and only valid AA
        ok = seqs.str.len().eq(T) & seqs.str.match(r"^[ACDEFGHIKLMNPQRSTVWY]{10}$", na=False)
        self.seqs = seqs[ok].drop_duplicates().tolist()

        if len(self.seqs) == 0:
            raise ValueError("No valid length-10 peptide sequences found in column peptide_len10.")

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        s = self.seqs[idx]
        ids = torch.tensor([AA_TO_I[ch] for ch in s], dtype=torch.long)    # (10,)
        onehot = F.one_hot(ids, num_classes=VOCAB).float()                # (10,20)
        return onehot, ids, s


def collate(batch):
    onehot, ids, seqs = zip(*batch)
    return torch.stack(onehot, dim=0), torch.stack(ids, dim=0), list(seqs)


# -----------------------
# Monitoring utilities
# -----------------------
@torch.no_grad()
def layer_stats(cache: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
    """
    cache[layer_i] = (B,10,D)
    returns mean/std/norm for quick monitoring
    """
    out = {}
    for k, v in cache.items():
        # v: (B,T,D)
        out[k] = {
            "mean": float(v.mean().item()),
            "std": float(v.std().item()),
            "l2_mean": float(v.norm(dim=-1).mean().item()),  # mean over (B,T)
        }
    return out
def _mkdir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def plot_learning_curves(history: pd.DataFrame, out_dir: str):
    # Recon
    plt.figure()
    plt.plot(history["epoch"], history["train_recon"], label="train_recon")
    plt.plot(history["epoch"], history["val_recon"], label="val_recon")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction CE")
    plt.title("Learning Curve: Reconstruction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "learning_curve_recon.png"), dpi=200)
    plt.close()

    # KL
    plt.figure()
    plt.plot(history["epoch"], history["train_kl"], label="train_kl")
    plt.plot(history["epoch"], history["val_kl"], label="val_kl")
    plt.xlabel("Epoch")
    plt.ylabel("KL (flow-adjusted)")
    plt.title("Learning Curve: KL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "learning_curve_kl.png"), dpi=200)
    plt.close()

    # Total
    plt.figure()
    plt.plot(history["epoch"], history["train_loss"], label="train_loss")
    plt.plot(history["epoch"], history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve: Total Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "learning_curve_total.png"), dpi=200)
    plt.close()

def plot_token_accuracy_per_position(logits: torch.Tensor, targets: torch.Tensor, out_dir: str, name: str):
    pred = logits.argmax(dim=-1)  # (B,10)
    accs = [(pred[:, t] == targets[:, t]).float().mean().item() for t in range(T)]
    plt.figure()
    plt.plot(range(1, T+1), accs)
    plt.xlabel("Position (1-10)")
    plt.ylabel("Token accuracy")
    plt.title(f"{name}: Token Accuracy per Position")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_token_acc_per_pos.png"), dpi=200)
    plt.close()
    return accs

def plot_recon_confidence(logits: torch.Tensor, targets: torch.Tensor, out_dir: str, name: str):
    probs = logits.softmax(dim=-1)
    conf = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B,10)
    conf_np = conf.detach().cpu().numpy().ravel()
    plt.figure()
    plt.hist(conf_np, bins=60)
    plt.xlabel("P(true token)")
    plt.ylabel("Count")
    plt.title(f"{name}: Reconstruction Confidence")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_recon_confidence_hist.png"), dpi=200)
    plt.close()

def plot_hidden_norms_over_time(cache: Dict[str, torch.Tensor], out_dir: str, name: str):
    # cache[layer_i] = (B,10,D)
    plt.figure()
    for k, v in cache.items():
        norms = v.norm(dim=-1).mean(dim=0).detach().cpu().numpy()  # (10,)
        plt.plot(range(1, T+1), norms, label=k)
    plt.xlabel("Time step (1-10)")
    plt.ylabel("Mean hidden L2 norm")
    plt.title(f"{name}: Hidden Norms over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_hidden_norms_over_time.png"), dpi=200)
    plt.close()

def plot_hidden_mean_std_over_time(cache: Dict[str, torch.Tensor], out_dir: str, name: str):
    # mean/std of activations per time step
    for k, v in cache.items():
        v_cpu = v.detach().cpu()  # (B,10,D)
        mean_t = v_cpu.mean(dim=(0,2)).numpy()  # (10,)
        std_t  = v_cpu.std(dim=(0,2)).numpy()   # (10,)

        plt.figure()
        plt.plot(range(1, T+1), mean_t, label="mean")
        plt.plot(range(1, T+1), std_t, label="std")
        plt.xlabel("Time step (1-10)")
        plt.ylabel("Activation")
        plt.title(f"{name}: {k} mean/std over time")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}_{k}_mean_std_over_time.png"), dpi=200)
        plt.close()

def plot_representation_spectrum(cache: Dict[str, torch.Tensor], out_dir: str, name: str, max_svs: int = 30):
    # Flatten (B,10,D) -> (B*10,D), compute top singular values
    for k, v in cache.items():
        x = v.detach().cpu().reshape(-1, v.shape[-1]).numpy()
        x = x - x.mean(axis=0, keepdims=True)
        try:
            s = np.linalg.svd(x, full_matrices=False, compute_uv=False)
        except Exception:
            continue
        s = s[:max_svs]
        plt.figure()
        plt.plot(range(1, len(s)+1), s)
        plt.xlabel("Component")
        plt.ylabel("Singular value")
        plt.title(f"{name}: {k} representation spectrum")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}_{k}_spectrum.png"), dpi=200)
        plt.close()

def plot_latent_distributions(mu, logvar, z0, zK, out_dir: str, name: str):
    def _hist(arr, title, fname):
        plt.figure()
        plt.hist(arr, bins=60)
        plt.xlabel(title)
        plt.ylabel("Count")
        plt.title(f"{name}: {title}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=200)
        plt.close()

    _hist(mu.detach().cpu().numpy().ravel(), "mu", f"{name}_mu_hist.png")
    _hist(logvar.detach().cpu().numpy().ravel(), "logvar", f"{name}_logvar_hist.png")
    _hist(z0.detach().cpu().numpy().ravel(), "z0", f"{name}_z0_hist.png")
    _hist(zK.detach().cpu().numpy().ravel(), "zK", f"{name}_zK_hist.png")

    # norms
    z0n = z0.norm(dim=1).detach().cpu().numpy()
    zKn = zK.norm(dim=1).detach().cpu().numpy()
    _hist(z0n, "||z0||", f"{name}_z0_norm_hist.png")
    _hist(zKn, "||zK||", f"{name}_zK_norm_hist.png")

def plot_flow_diagnostics(flow_preacts: list, logdet: torch.Tensor, out_dir: str, name: str):
    # Planar flow saturation diagnostics
    for i, a in enumerate(flow_preacts):
        a_np = a.detach().cpu().numpy()
        t = np.tanh(a_np)
        tp = 1.0 - t*t  # tanh'(a)

        plt.figure()
        plt.hist(a_np, bins=60)
        plt.xlabel("a = w^T z + b")
        plt.ylabel("Count")
        plt.title(f"{name}: Flow {i} pre-activation a")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}_flow{i}_a_hist.png"), dpi=200)
        plt.close()

        plt.figure()
        plt.hist(tp, bins=60)
        plt.xlabel("tanh'(a)")
        plt.ylabel("Count")
        plt.title(f"{name}: Flow {i} tanh'(a) (saturation proxy)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}_flow{i}_tanhprime_hist.png"), dpi=200)
        plt.close()

    ld = logdet.detach().cpu().numpy()
    plt.figure()
    plt.hist(ld, bins=60)
    plt.xlabel("sum log|det J|")
    plt.ylabel("Count")
    plt.title(f"{name}: logdet distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_logdet_hist.png"), dpi=200)
    plt.close()

def _aa_labels():
    # labels in the same order as AA and AA_TO_I
    return list(AA)

@torch.no_grad()
def compute_confusion_20x10(logits: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
    """
    Returns confusion counts as (T, 20, 20) where:
      confusion[t, true, pred] = count
    """
    pred = logits.argmax(dim=-1)  # (B,10)
    B = targets.size(0)
    conf = np.zeros((T, VOCAB, VOCAB), dtype=np.int64)
    pred_np = pred.detach().cpu().numpy()
    true_np = targets.detach().cpu().numpy()

    for t in range(T):
        for b in range(B):
            conf[t, true_np[b, t], pred_np[b, t]] += 1
    return conf


def plot_confusion_heatmap_20x10(conf_t_20x20: np.ndarray, out_dir: str, name: str):
    """
    conf_t_20x20: (20,20) for one position t
    Saves a heatmap image. (No seaborn)
    """
    labels = _aa_labels()
    plt.figure(figsize=(7, 6))
    plt.imshow(conf_t_20x20, aspect="auto")
    plt.colorbar(label="Count")
    plt.xticks(range(VOCAB), labels, rotation=90)
    plt.yticks(range(VOCAB), labels)
    plt.xlabel("Predicted AA")
    plt.ylabel("True AA")
    plt.title(name)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}.png"), dpi=200)
    plt.close()


def plot_confusion_per_position(logits: torch.Tensor, targets: torch.Tensor, out_dir: str, name: str):
    """
    Saves 10 confusion heatmaps (one per position) and also stores the raw confusion tensor as .npy.
    """
    conf = compute_confusion_20x10(logits, targets)  # (10,20,20)
    np.save(os.path.join(out_dir, f"{name}_confusion_Tx20x20.npy"), conf)

    # One heatmap per position
    for t in range(T):
        plot_confusion_heatmap_20x10(
            conf[t],
            out_dir=out_dir,
            name=f"{name}_confusion_pos{t+1:02d}_20x20"
        )


@torch.no_grad()
def plot_most_frequent_error_token_per_position(
    logits: torch.Tensor,
    targets: torch.Tensor,
    out_dir: str,
    name: str,
):
    """
    For each position, find the most frequent *wrong* predicted token.
    Saves:
      - PNG bar plot
      - CSV summary (position, most_freq_error, count, error_rate)
    """
    pred = logits.argmax(dim=-1)  # (B,10)
    pred_np = pred.detach().cpu().numpy()
    true_np = targets.detach().cpu().numpy()

    labels = _aa_labels()
    rows = []
    err_tokens = []
    err_counts = []
    err_rates = []

    for t in range(T):
        mask_err = (pred_np[:, t] != true_np[:, t])
        n_err = int(mask_err.sum())
        n_all = pred_np.shape[0]
        err_rate = n_err / max(1, n_all)

        if n_err == 0:
            # no errors: encode as "-"
            rows.append({
                "position": t + 1,
                "most_freq_error_token": "-",
                "count": 0,
                "error_rate": err_rate
            })
            err_tokens.append("-")
            err_counts.append(0)
            err_rates.append(err_rate)
            continue

        wrong_preds = pred_np[mask_err, t]
        counts = np.bincount(wrong_preds, minlength=VOCAB)
        j = int(counts.argmax())
        rows.append({
            "position": t + 1,
            "most_freq_error_token": labels[j],
            "count": int(counts[j]),
            "error_rate": err_rate
        })
        err_tokens.append(labels[j])
        err_counts.append(int(counts[j]))
        err_rates.append(err_rate)

    # Save CSV
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f"{name}_most_freq_error_per_pos.csv"), index=False)

    # Plot (counts) with token labels annotated
    plt.figure(figsize=(9, 4))
    xs = np.arange(1, T + 1)
    plt.bar(xs, err_counts)
    plt.xticks(xs)
    plt.xlabel("Position (1-10)")
    plt.ylabel("Count of most frequent wrong token")
    plt.title(f"{name}: Most Frequent Error Token per Position")

    # annotate token + error rate
    for x, c, tok, er in zip(xs, err_counts, err_tokens, err_rates):
        plt.text(x, c + 0.5, f"{tok}\n{er:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_most_freq_error_per_pos.png"), dpi=200)
    plt.close()


@torch.no_grad()
def plot_logdet_vs_znorm_scatter(
    z0: torch.Tensor,
    zK: torch.Tensor,
    logdet: torch.Tensor,
    out_dir: str,
    name: str,
):
    """
    Saves:
      - scatter plot logdet vs ||z0|| and logdet vs ||zK||
      - CSV of points for later analysis
    """
    z0n = z0.norm(dim=1).detach().cpu().numpy()
    zKn = zK.norm(dim=1).detach().cpu().numpy()
    ld  = logdet.detach().cpu().numpy()

    # Save CSV
    df = pd.DataFrame({"z0_norm": z0n, "zK_norm": zKn, "logdet": ld})
    df.to_csv(os.path.join(out_dir, f"{name}_logdet_vs_znorm.csv"), index=False)

    # Scatter: logdet vs z0 norm
    plt.figure()
    plt.scatter(z0n, ld, s=8)
    plt.xlabel("||z0||")
    plt.ylabel("sum log|det J|")
    plt.title(f"{name}: logdet vs ||z0||")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_logdet_vs_z0norm.png"), dpi=200)
    plt.close()

    # Scatter: logdet vs zK norm
    plt.figure()
    plt.scatter(zKn, ld, s=8)
    plt.xlabel("||zK||")
    plt.ylabel("sum log|det J|")
    plt.title(f"{name}: logdet vs ||zK||")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_logdet_vs_zKnorm.png"), dpi=200)
    plt.close()

# -----------------------
# Train (fixed + plotting)
# -----------------------
def train():
    ds = PeptideLen10Dataset(CSV_PATH)

    # reproducible split (optional)
    rng = np.random.default_rng(0)
    n = len(ds)
    idx = rng.permutation(n)

    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    train_loader = DataLoader(
        torch.utils.data.Subset(ds, train_idx),
        batch_size=256, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(ds, val_idx),
        batch_size=256, shuffle=False, collate_fn=collate
    )
    test_loader = DataLoader(
        torch.utils.data.Subset(ds, test_idx),
        batch_size=256, shuffle=False, collate_fn=collate
    )

    model = FixedLenFlowGRUVAE(
        d_model=256,
        num_layers=2,
        dropout=0.1,
        n_flows=4,
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    epochs = 10
    beta = 0.01  # tune: 0.001–0.05 typically
    best_val = float("inf")

    history_rows = []

    for ep in range(1, epochs + 1):
        # -----------------------
        # Train
        # -----------------------
        model.train()
        tr = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
        nb = 0

        for x_onehot, y_ids, _ in train_loader:
            x_onehot = x_onehot.to(DEVICE)  # (B,10,20)
            y_ids    = y_ids.to(DEVICE)     # (B,10)

            out = model(x_onehot)
            loss, m = flow_vae_loss(out, y_ids, beta=beta)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            for k in tr:
                tr[k] += m[k]
            nb += 1

        for k in tr:
            tr[k] /= max(1, nb)

        # -----------------------
        # Validation
        # -----------------------
        model.eval()
        va = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
        nb = 0
        first_val_batch = None  # (out, y_ids)

        with torch.no_grad():
            for x_onehot, y_ids, _ in val_loader:
                x_onehot = x_onehot.to(DEVICE)
                y_ids    = y_ids.to(DEVICE)

                out = model(x_onehot)
                loss, m = flow_vae_loss(out, y_ids, beta=beta)

                for k in va:
                    va[k] += m[k]
                nb += 1

                if first_val_batch is None:
                    # keep only the first batch to plot (descriptive snapshot)
                    first_val_batch = (out, y_ids)

        for k in va:
            va[k] /= max(1, nb)

        # -----------------------
        # Record history + print
        # -----------------------
        history_rows.append({
            "epoch": ep,
            "train_loss": tr["loss"], "train_recon": tr["recon"], "train_kl": tr["kl"],
            "val_loss": va["loss"],   "val_recon": va["recon"],   "val_kl": va["kl"],
        })

        print(
            f"Epoch {ep:02d} | "
            f"train loss {tr['loss']:.4f} recon {tr['recon']:.4f} kl {tr['kl']:.4f} | "
            f"val loss {va['loss']:.4f} recon {va['recon']:.4f} kl {va['kl']:.4f}"
        )

        # -----------------------
        # Save best model
        # -----------------------
        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "flow_gru_vae_best.pt"))

        # -----------------------
        # Per-epoch snapshot plots (first val batch)
        # -----------------------
        if ep == 1 or ep % 1 == 0 or ep == epochs:
            if first_val_batch is not None:
                out_b, y_b = first_val_batch
                ep_dir = _mkdir(os.path.join(OUT_DIR, f"epoch_{ep}"))

                # Reconstruction metrics
                plot_token_accuracy_per_position(out_b.logits, y_b, ep_dir, name=f"val_")
                plot_recon_confidence(out_b.logits, y_b, ep_dir, name=f"val_")

                # NEW: confusion + error token diagnostics
                plot_confusion_per_position(out_b.logits, y_b, ep_dir, name=f"val_")
                plot_most_frequent_error_token_per_position(out_b.logits, y_b, ep_dir, name=f"val_")

                # NEW: flow degeneracy scatter
                plot_logdet_vs_znorm_scatter(out_b.z0, out_b.zK, out_b.logdet, ep_dir, name=f"val_")


                # Hidden monitoring
                plot_hidden_norms_over_time(out_b.enc_cache, ep_dir, name=f"enc_")
                plot_hidden_norms_over_time(out_b.dec_cache, ep_dir, name=f"dec_")

                plot_hidden_mean_std_over_time(out_b.enc_cache, ep_dir, name=f"enc_")
                plot_hidden_mean_std_over_time(out_b.dec_cache, ep_dir, name=f"dec_")

                plot_representation_spectrum(out_b.enc_cache, ep_dir, name=f"enc_")
                plot_representation_spectrum(out_b.dec_cache, ep_dir, name=f"dec_")

                # Latent + flow monitoring
                plot_latent_distributions(out_b.mu, out_b.logvar, out_b.z0, out_b.zK, ep_dir, name=f"val_")
                plot_flow_diagnostics(out_b.flow_preacts, out_b.logdet, ep_dir, name=f"val_")

        # -----------------------
        # Update global history plots each epoch (optional)
        # -----------------------
        history = pd.DataFrame(history_rows)
        history.to_csv(os.path.join(OUT_DIR, "training_history.csv"), index=False)
        plot_learning_curves(history, OUT_DIR)

    # -----------------------
    # Test (using best model)
    # -----------------------
    best_path = os.path.join(OUT_DIR, "flow_gru_vae_best.pt")
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    model.eval()

    te = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
    nb = 0
    with torch.no_grad():
        for x_onehot, y_ids, _ in test_loader:
            x_onehot = x_onehot.to(DEVICE)
            y_ids    = y_ids.to(DEVICE)

            out = model(x_onehot)
            _, m = flow_vae_loss(out, y_ids, beta=beta)

            for k in te:
                te[k] += m[k]
            nb += 1

    for k in te:
        te[k] /= max(1, nb)

    print(f"\nTEST | loss {te['loss']:.4f} recon {te['recon']:.4f} kl {te['kl']:.4f}")
    print(f"Saved monitoring plots to: {OUT_DIR}/")



if __name__ == "__main__":
    train()
