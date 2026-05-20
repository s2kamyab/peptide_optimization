import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


def collect_pngs_by_epoch(root_dir: str, epochs):
    root = Path(root_dir)
    epoch_dirs = {e: root / f"epoch_{e}" for e in epochs}

    for e, d in epoch_dirs.items():
        if not d.exists():
            raise FileNotFoundError(f"Missing folder: {d}")

    epoch_files = {}
    for e, d in epoch_dirs.items():
        files = sorted([p.name for p in d.glob("*.png")])
        epoch_files[e] = set(files)

    # common names across ALL epochs
    common = set.intersection(*epoch_files.values()) if epoch_files else set()
    if not common:
        raise ValueError(
            "No common PNG filenames found across all epoch folders.\n"
            "Tip: ensure each epoch folder has the same filenames (e.g., val_ep05_*.png etc.)."
        )

    return epoch_dirs, sorted(common)


def plot_monitor_grids(
    root_dir: str,
    epochs=(1, 5, 10, 15, 20, 25, 30, 35, 40),
    rows_per_page=5,
    out_dir=None,
    dpi=200,
    font_size=12,
):
    epochs = list(epochs)
    epoch_dirs, names = collect_pngs_by_epoch(root_dir, epochs)

    out_path = Path(out_dir) if out_dir else (Path(root_dir) / "monitor_grids")
    out_path.mkdir(parents=True, exist_ok=True)

    total = len(names)
    pages = (total + rows_per_page - 1) // rows_per_page  # e.g., 40 -> 8 pages

    print(f"Found {total} common PNGs across {len(epochs)} epochs.")
    print(f"Creating {pages} pages with {rows_per_page} rows × {len(epochs)} cols each.")
    print(f"Saving to: {out_path.resolve()}")

    for page in range(pages):
        start = page * rows_per_page
        end = min(start + rows_per_page, total)
        page_names = names[start:end]

        n_rows = len(page_names)
        n_cols = len(epochs)

        # Figure size heuristic: widen for 8 columns, height for up to 5 rows
        fig_w = 2.2 * n_cols
        fig_h = 2.1 * n_rows
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))

        # Normalize axes indexing for n_rows=1 case
        if n_rows == 1:
            axes = [axes]

        for r, fname in enumerate(page_names):
            for c, e in enumerate(epochs):
                ax = axes[r][c] if n_rows > 1 else axes[c]
                img_path = epoch_dirs[e] / fname

                ax.axis("off")

                if not img_path.exists():
                    ax.text(0.5, 0.5, "MISSING", ha="center", va="center")
                else:
                    try:
                        img = Image.open(img_path).convert("RGB")
                        ax.imshow(img)
                    except Exception as ex:
                        ax.text(0.5, 0.5, f"ERROR\n{ex}", ha="center", va="center")

                # column titles (epochs) on top row
                if r == 0:
                    ax.set_title(f"epoch {e}", fontsize=font_size)

            # row label (filename) on left-most subplot
            left_ax = axes[r][0] if n_rows > 1 else axes[0]
            left_ax.text(
                -0.02, 0.5, fname,
                transform=left_ax.transAxes,
                ha="right", va="center",
                fontsize=font_size,
                rotation=0
            )

        fig.suptitle(
            f"Monitoring across epochs (page {page+1}/{pages})",
            fontsize=font_size + 2
        )
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        out_file = out_path / f"monitor_page_{page+1:02d}.png"
        fig.savefig(out_file, dpi=dpi)
        plt.close(fig)

    print("Done.")


if __name__ == "__main__":
    # Change this to your root folder that contains epoch_5, epoch_10, ..., epoch_40
    ROOT = "flow_gru_vae_monitoring"  # <-- edit me if needed
    plot_monitor_grids(
        root_dir=ROOT,
        # epochs=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        epochs=(1, 5, 10, 15, 20, 25, 30, 35, 40),
        rows_per_page=5,
        out_dir=None,   # default: ROOT/monitor_grids
        dpi=200,
        font_size=12,
    )
