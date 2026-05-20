import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


EPOCHS = [1, 5, 10, 15, 20, 25, 30, 35, 40]

# Root directory that contains epoch_*/latent_quality
ROOT = Path("flow_gru_vae_monitoring")   # <- change if needed

# Folder inside each epoch dir that contains the PNGs
SUBDIR = Path("latent_quality")

# Where to save the single mega-collage figure
OUT_DIR = ROOT / "latent_quality_collages"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_png_names(root: Path, epochs: list[int], subdir: Path) -> list[str]:
    """Infer the shared PNG filenames from the first epoch folder."""
    first = root / f"epoch_{epochs[0]}" / subdir
    if not first.exists():
        raise FileNotFoundError(f"Missing folder: {first}")

    pngs = sorted([p.name for p in first.glob("*.png")])
    if not pngs:
        raise FileNotFoundError(f"No PNGs found in: {first}")

    return pngs


def plot_4x9_mega_collage(
    root: Path,
    epochs: list[int],
    png_names: list[str],
    subdir: Path,
    out_dir: Path,
    out_name: str = "latent_quality_4x9_collage.png",
):
    """
    Creates ONE figure with 4 rows (png files) x 9 cols (epochs).
    Row i corresponds to png_names[i], col j corresponds to epochs[j].
    """
    if len(png_names) < 4:
        raise ValueError(f"Expected at least 4 PNG names, got {len(png_names)}.")
    png_names = png_names[:4]

    nrows = 4
    ncols = len(epochs)  # 9

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows))

    for r, png_name in enumerate(png_names):
        for c, ep in enumerate(epochs):
            ax = axes[r, c]
            img_path = root / f"epoch_{ep}" / subdir / png_name

            if img_path.exists():
                img = mpimg.imread(str(img_path))
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "MISSING", ha="center", va="center", fontsize=10)

            ax.axis("off")

            # Column titles: epochs (top row only)
            if r == 0:
                ax.set_title(f"ep {ep}", fontsize=10)

            # Row labels: png filename (first column only)
            if c == 0:
                # put label on the left side
                ax.text(
                    -0.02, 0.5, png_name,
                    transform=ax.transAxes,
                    ha="right", va="center",
                    rotation=90,
                    fontsize=10
                )

    fig.suptitle("Latent Quality Diagnostics Across Epochs", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved: {out_path}")


def main():
    png_names = find_png_names(ROOT, EPOCHS, SUBDIR)
    plot_4x9_mega_collage(
        root=ROOT,
        epochs=EPOCHS,
        png_names=png_names,
        subdir=SUBDIR,
        out_dir=OUT_DIR,
        out_name="latent_quality_4x9_collage.png",
    )


if __name__ == "__main__":
    main()
