from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


HISTORY_CSV = Path("transfer_gru_vae_flow_checkpoints_h32_z32_latent_conditioned_autoreg_rt001/training_history_latent_conditioned_autoreg_rt.csv")
OUTPUT_DIR = Path("transfer_gru_vae_flow_checkpoints_h32_z32_latent_conditioned_autoreg_rt001/history_plots_150")
FONT_SIZE = 24


def style_axes(ax, title, xlabel="Epoch", ylabel=None):
    ax.set_title(title, fontsize=FONT_SIZE)
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.grid(True, alpha=0.3)
    legend = ax.legend(fontsize=FONT_SIZE)
    if legend:
        legend.get_frame().set_alpha(0.9)


def save_figure(fig, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    history = pd.read_csv(HISTORY_CSV)
    history = history.sort_values("epoch")

    epochs = history["epoch"]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(epochs, history["loss"], label="Loss", linewidth=2)
    ax.plot(epochs, history["recon"], label="Reconstruction", linewidth=2)
    style_axes(ax, "Training Loss History", ylabel="Loss")
    save_figure(fig, "loss_history.png")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(epochs, history["kl"], label="KL", linewidth=2, color="tab:orange")
    style_axes(ax, "KL Divergence History", ylabel="KL")
    save_figure(fig, "kl_history.png")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(epochs, history["token_acc"], label="Token Accuracy", linewidth=2, color="tab:green")
    style_axes(ax, "Token Accuracy History", ylabel="Accuracy")
    save_figure(fig, "token_accuracy_history.png")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(epochs, history["beta"], label="Beta", linewidth=2)
    ax.plot(epochs, history["lr"], label="Learning Rate", linewidth=2)
    style_axes(ax, "Beta and Learning Rate History", ylabel="Value")
    save_figure(fig, "beta_lr_history.png")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(epochs, history["mu_abs_mean"], label="Mean |mu|", linewidth=2)
    ax.plot(epochs, history["mu_std"], label="mu std", linewidth=2)
    ax.plot(epochs, history["logvar_mean"], label="logvar mean", linewidth=2)
    ax.plot(epochs, history["logvar_std"], label="logvar std", linewidth=2)
    style_axes(ax, "Latent Statistics History", ylabel="Value")
    save_figure(fig, "latent_stats_history.png")

    print(f"Saved plots to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
