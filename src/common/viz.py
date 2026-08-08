"""Plot helpers shared by the EDA and evaluation notebooks.

Every function returns the matplotlib Figure so the caller decides whether to
show it, save it via ``io_utils.save_figure``, or both.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def set_plot_style() -> None:
    """One consistent look for all report figures."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "figure.autolayout": False,
        }
    )


def plot_class_distribution(counts: dict[str, int], title: str = "Class distribution",
                            ylabel: str = "Number of images"):
    """Bar chart of label counts with the count printed above each bar."""
    fig, ax = plt.subplots(figsize=(max(5, 1.4 * len(counts)), 4))
    labels = list(counts.keys())
    values = [counts[k] for k in labels]
    bars = ax.bar(labels, values, color="#4C72B0", edgecolor="black", linewidth=0.5)
    total = sum(values) or 1
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value}\n({value / total:.1%})",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.margins(y=0.18)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return fig


def plot_image_grid(images: Sequence[np.ndarray], titles: Sequence[str] | None = None,
                    ncols: int = 4, figsize_per: float = 3.0, cmap: str = "gray",
                    suptitle: str | None = None):
    """Grid of images (H,W) or (H,W,3), used for sample/EDA/prediction panels."""
    images = list(images)
    n = len(images)
    if n == 0:
        raise ValueError("plot_image_grid received an empty image list.")
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per * ncols, figsize_per * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            img = images[i]
            ax.imshow(img, cmap=cmap if img.ndim == 2 else None)
            if titles is not None and i < len(titles):
                ax.set_title(str(titles[i]), fontsize=10)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_training_curves(history, metrics: Sequence[str] = ("loss",), title: str = "Training curves"):
    """Plot train/val curves. ``history`` is a list of per-epoch dicts or a DataFrame."""
    import pandas as pd

    df = history if isinstance(history, pd.DataFrame) else pd.DataFrame(list(history))
    metrics = [m for m in metrics if f"train_{m}" in df.columns or f"val_{m}" in df.columns]
    if not metrics:
        raise ValueError(f"No train_/val_ columns found for metrics={metrics}. Columns: {list(df.columns)}")

    fig, axes = plt.subplots(1, len(metrics), figsize=(5.5 * len(metrics), 4), squeeze=False)
    x = df["epoch"] if "epoch" in df.columns else np.arange(1, len(df) + 1)
    for ax, metric in zip(axes[0], metrics):
        if f"train_{metric}" in df.columns:
            ax.plot(x, df[f"train_{metric}"], marker="o", ms=3, label=f"train {metric}")
        if f"val_{metric}" in df.columns:
            ax.plot(x, df[f"val_{metric}"], marker="s", ms=3, label=f"val {metric}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.legend()
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(1.0, 0.0, 0.0), alpha: float = 0.4) -> np.ndarray:
    """Blend a binary mask over a grayscale/RGB image and return an RGB float array."""
    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.max() > 1.0:
        img = img / 255.0
    img = np.clip(img, 0.0, 1.0)

    m = np.asarray(mask).astype(bool)
    if m.ndim == 3:
        m = m.squeeze()
    out = img.copy()
    for c in range(3):
        out[..., c] = np.where(m, (1 - alpha) * out[..., c] + alpha * color[c], out[..., c])
    return out


def denormalize(tensor, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    """Undo Normalize() so a tensor image can be displayed. Returns HWC in [0,1]."""
    arr = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW -> HWC
        arr = np.transpose(arr, (1, 2, 0))
    mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, -1)
    std = np.asarray(std, dtype=np.float32).reshape(1, 1, -1)
    arr = arr * std + mean
    return np.clip(arr, 0.0, 1.0).squeeze()


def savefig_or_show(fig, path: str | Path | None):
    """Save when a path is given, otherwise just display."""
    if path is None:
        plt.show()
        return None
    from .io_utils import save_figure

    return save_figure(fig, path)
