import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def box(ax, xy, w, h, text, fontsize=10):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.5,
        facecolor="white"
    )
    ax.add_patch(patch)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize)
    return patch

def arrow(ax, p1, p2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="->", mutation_scale=12, linewidth=1.5))

def plot_seqvae_architecture_png(out_path="seqvae_architecture.png"):
    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Encoder side
    b_in   = box(ax, (0.05, 0.70), 0.18, 0.12, "Input tokens\nx_in (B,T)", 10)
    b_emb  = box(ax, (0.27, 0.70), 0.18, 0.12, "Embedding + Dropout\nE=128", 10)
    b_enc  = box(ax, (0.49, 0.70), 0.20, 0.12, "Encoder GRU\nBiGRU, L=2\nH=256", 10)
    b_mlp  = box(ax, (0.73, 0.70), 0.20, 0.12, "Enc MLP\nLinear+ReLU+Drop", 10)

    b_mu   = box(ax, (0.73, 0.52), 0.09, 0.10, "μ\n(B,Z)", 10)
    b_lv   = box(ax, (0.84, 0.52), 0.09, 0.10, "logσ²\n(B,Z)", 10)

    b_z    = box(ax, (0.73, 0.34), 0.20, 0.10, "z = μ + ε·σ\nReparameterize\nZ=64", 10)

    # Decoder side
    b_z2h  = box(ax, (0.05, 0.34), 0.18, 0.10, "z → h0\ninit all GRU layers\n(L,B,H)", 9)
    b_z2e  = box(ax, (0.27, 0.34), 0.18, 0.10, "z → emb bias\n(B,E)", 9)
    b_demb = box(ax, (0.49, 0.34), 0.20, 0.10, "Decoder input emb\n(emb + z_bias)\nDropout", 9)
    b_dec  = box(ax, (0.73, 0.34), 0.20, 0.10, "Decoder GRU\nL=2, H=256", 9)
    b_out  = box(ax, (0.73, 0.16), 0.20, 0.10, "Linear → logits\n(B,T,V)", 10)

    # Arrows encoder
    arrow(ax, (0.23, 0.76), (0.27, 0.76))
    arrow(ax, (0.45, 0.76), (0.49, 0.76))
    arrow(ax, (0.69, 0.76), (0.73, 0.76))
    arrow(ax, (0.83, 0.70), (0.83, 0.62))
    arrow(ax, (0.78, 0.70), (0.78, 0.62))

    arrow(ax, (0.78, 0.52), (0.78, 0.44))
    arrow(ax, (0.89, 0.52), (0.89, 0.44))

    # z to decoder components
    arrow(ax, (0.73, 0.39), (0.23, 0.39))
    arrow(ax, (0.73, 0.39), (0.45, 0.39))

    arrow(ax, (0.23, 0.39), (0.27, 0.39))
    arrow(ax, (0.45, 0.39), (0.49, 0.39))
    arrow(ax, (0.69, 0.39), (0.73, 0.39))

    # decoder output
    arrow(ax, (0.83, 0.34), (0.83, 0.26))

    ax.text(0.05, 0.93, "SeqVAE Architecture (GRU) — Encoder → Latent z → Decoder", fontsize=14)

    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()
    print(f"Saved architecture diagram -> {out_path}")

if __name__ == "__main__":
    plot_seqvae_architecture_png()
