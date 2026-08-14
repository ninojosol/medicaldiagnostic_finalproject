"""Rendering helpers for MRI slices, masks and overlays.

Used by the visual-validation gate, the saved run figures and the Streamlit UI,
so an overlay looks the same wherever it appears.
"""

from __future__ import annotations

import numpy as np

from .constants import MODALITY_NAMES

# Ground truth = amber, prediction = cyan. Distinct in both light and dark UI and
# distinguishable without relying on a red/green contrast.
GT_RGB = (255, 176, 32)
PRED_RGB = (0, 208, 255)
OVERLAP_RGB = (124, 240, 168)


def window_slice(slice_2d: np.ndarray, *, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    """Percentile-window a z-scored slice into 0..1 for display.

    Windowing uses brain (nonzero) voxels only; the z-scored background is exactly
    0 and would otherwise dominate the percentiles and wash the brain out.
    """
    arr = np.asarray(slice_2d, dtype=np.float32)
    brain = arr != 0
    if not brain.any():
        return np.zeros_like(arr)
    lo, hi = np.percentile(arr[brain], [p_lo, p_hi])
    if hi <= lo:
        hi = lo + 1e-6
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return np.where(brain, out, 0.0).astype(np.float32)


def to_uint8_rgb(slice_2d: np.ndarray) -> np.ndarray:
    """Windowed grayscale slice -> (H,W,3) uint8."""
    g = (window_slice(slice_2d) * 255).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def mask_to_rgba(mask: np.ndarray, rgb: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    """Binary mask -> (H,W,4) uint8 RGBA tint."""
    m = np.asarray(mask).astype(bool)
    out = np.zeros(m.shape + (4,), dtype=np.uint8)
    out[m, 0], out[m, 1], out[m, 2] = rgb
    out[m, 3] = int(round(alpha * 255))
    return out


def overlay_masks(
    slice_2d: np.ndarray,
    gt: np.ndarray | None = None,
    pred: np.ndarray | None = None,
    *,
    alpha: float = 0.45,
    outline_only: bool = False,
) -> np.ndarray:
    """Composite ground truth and/or prediction onto a grayscale slice.

    Colour key: amber = ground truth only, cyan = prediction only,
    green = agreement.
    """
    base = to_uint8_rgb(slice_2d).astype(np.float32)

    g = np.asarray(gt).astype(bool) if gt is not None else None
    p = np.asarray(pred).astype(bool) if pred is not None else None

    if outline_only:
        if g is not None:
            g = _boundary(g)
        if p is not None:
            p = _boundary(p)

    layers: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    if g is not None and p is not None:
        layers = [
            (g & ~p, GT_RGB),
            (p & ~g, PRED_RGB),
            (g & p, OVERLAP_RGB),
        ]
    elif g is not None:
        layers = [(g, GT_RGB)]
    elif p is not None:
        layers = [(p, PRED_RGB)]

    for mask, rgb in layers:
        if not mask.any():
            continue
        tint = np.array(rgb, dtype=np.float32)
        base[mask] = (1 - alpha) * base[mask] + alpha * tint

    return np.clip(base, 0, 255).astype(np.uint8)


def _boundary(mask: np.ndarray) -> np.ndarray:
    """1-pixel outline of a binary mask (no scipy dependency)."""
    m = mask.astype(bool)
    eroded = m.copy()
    eroded[1:, :] &= m[:-1, :]
    eroded[:-1, :] &= m[1:, :]
    eroded[:, 1:] &= m[:, :-1]
    eroded[:, :-1] &= m[:, 1:]
    return m & ~eroded


def modality_panel(image_chw: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """(name, uint8 RGB) for each of the four sequences of one slice."""
    return [
        (MODALITY_NAMES[c] if c < len(MODALITY_NAMES) else f"ch{c}", to_uint8_rgb(image_chw[c]))
        for c in range(image_chw.shape[0])
    ]


def save_overlay_grid(
    samples: list[dict],
    out_path,
    *,
    title: str,
    columns: tuple[str, ...] = ("FLAIR", "Ground truth", "Prediction", "Overlay"),
    dpi: int = 130,
) -> None:
    """Grid figure: one row per sample, matplotlib Agg backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(samples)
    if n == 0:
        return
    ncols = len(columns)
    fig, axes = plt.subplots(n, ncols, figsize=(3.0 * ncols, 3.0 * n), squeeze=False)
    fig.patch.set_facecolor("white")

    for r, s in enumerate(samples):
        panels = s["panels"]
        for c in range(ncols):
            ax = axes[r][c]
            ax.imshow(panels[c], interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#d7dce5")
            if r == 0:
                ax.set_title(columns[c], fontsize=11, color="#1f2733", pad=8)
        axes[r][0].set_ylabel(s["label"], fontsize=9, color="#41505f")

    fig.suptitle(title, fontsize=13, color="#1f2733", y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def save_training_history_figure(history_csv, out_path, *, title: str, dpi: int = 130) -> None:
    """White-background training curves: losses on the left axis, Dice on the right."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(history_csv)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(df["epoch"], df["train_loss"], color="#2f6fed", lw=2, label="Train loss")
    ax.plot(df["epoch"], df["valid_loss"], color="#8b5cf6", lw=2, ls="--", label="Validation loss")
    ax.set_xlabel("Epoch", color="#1f2733")
    ax.set_ylabel("Dice + BCE loss", color="#1f2733")
    ax.tick_params(colors="#41505f")
    ax.grid(True, color="#eef1f6", lw=1)
    for spine in ax.spines.values():
        spine.set_color("#d7dce5")

    ax2 = ax.twinx()
    ax2.plot(df["epoch"], df["valid_micro_dice"], color="#12a594", lw=2.4, label="Validation Dice")
    ax2.set_ylabel("Validation Dice", color="#12a594")
    ax2.tick_params(colors="#12a594")
    ax2.set_ylim(0, 1)
    for spine in ax2.spines.values():
        spine.set_color("#d7dce5")

    if (df["is_best"] == 1).any():
        best = int(df.loc[df["is_best"] == 1, "epoch"].iloc[-1])
        ax.axvline(best, color="#f59e0b", lw=1.5, ls=":", label=f"Best epoch ({best})")

    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(handles, labels, loc="center right", frameon=True, facecolor="white",
              edgecolor="#d7dce5", fontsize=9)

    ax.set_title(title, fontsize=12, color="#1f2733", pad=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
