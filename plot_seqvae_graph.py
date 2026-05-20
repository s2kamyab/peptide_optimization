import os
import torch
from torchviz import make_dot
from train_VAE_GRU_MN_peptide_len10 import SeqVAE, PeptideTokenizer

# ---- import your model definition ----
# from peptide_seq_vae_with_test import SeqVAE, PeptideTokenizer
# If it's in the same file, you can just import it.

def plot_seqvae_computation_graph(out_path="seqvae_computation_graph", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Create tokenizer for vocab sizes / pad id
    tok = PeptideTokenizer()

    # Build model (match your hyperparams)
    model = SeqVAE(
        vocab_size=len(tok.vocab),
        pad_id=tok.PAD,
        emb_dim=128,
        hid_dim=256,
        z_dim=64,
        num_layers=2,
        dropout=0.2,
        bidir_encoder=True,
    ).to(device)
    model.eval()

    # Dummy batch: (B,T)
    B, T = 4, 12  # 10 + BOS/EOS => up to 12 tokens, then decoder uses T-1
    x = torch.randint(low=0, high=len(tok.vocab), size=(B, T), device=device)
    x[:, 0] = tok.BOS
    x[:, -1] = tok.EOS

    x_in = x[:, :-1].contiguous()
    x_tgt = x[:, 1:].contiguous()
    lengths = torch.full((B,), T - 1, dtype=torch.long, device=device)

    # Forward pass
    logits, mu, logvar, z = model(x_in, x_tgt, lengths)

    # Build computation graph from logits
    dot = make_dot(
        logits,
        params=dict(model.named_parameters()),
        show_attrs=False,
        show_saved=False,
    )

    # Save (creates .png by default if format='png')
    dot.format = "png"
    dot.render(out_path, cleanup=True)
    print(f"Saved computation graph -> {out_path}.png")

if __name__ == "__main__":
    plot_seqvae_computation_graph()
