"""Figures for the README and the app's Experiments tab (matplotlib only)."""
import matplotlib

matplotlib.use("Agg")  # draw to files, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


def plot_confusion_matrix(cm: np.ndarray, path) -> None:
    """Row-normalised so each row shows what happened to images of that true class."""
    rows = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, rows, out=np.zeros(cm.shape, dtype=float), where=rows > 0)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xlabel("Predicted class index")
    ax.set_ylabel("True class index")
    ax.set_title("Confusion matrix (each row sums to 1)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_coverage_curve(table, path) -> None:
    """table: [(threshold, coverage, accuracy), ...] from metrics.coverage_table."""
    t, coverage, acc = zip(*table)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t, [a * 100 for a in acc], marker="o", label="Accuracy on kept images")
    ax.plot(t, [c * 100 for c in coverage], marker="s", label="Images kept")
    ax.set_xlabel("Minimum confidence threshold")
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 102)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Higher threshold: more accurate, fewer predictions")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_worst_errors(samples, path, cols: int = 4) -> None:
    """samples: [(image_path, true_name, predicted_name, confidence), ...]"""
    if not samples:
        return
    rows = int(np.ceil(len(samples) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (img_path, true_name, pred_name, conf) in zip(axes, samples):
        ax.imshow(Image.open(img_path).convert("RGB"))
        ax.set_title(f"true: {true_name}\npred: {pred_name} ({conf:.0%})", fontsize=7)
        ax.axis("off")
    for ax in axes[len(samples):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)