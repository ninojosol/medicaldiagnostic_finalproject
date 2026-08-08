"""Evaluation and qualitative visualisation for MRI tumour segmentation.

Protocol:

1. sweep the binarisation threshold on **validation**, pick the best mean Dice;
2. apply that frozen threshold to **test** and compute per-slice metrics;
3. bootstrap a CI for mean Dice, resampling by patient;
4. save per-slice CSV, summary JSON, overlay figures and predicted mask PNGs
   under ``outputs/segmentation/<run>/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..common.io_utils import save_csv, save_figure, save_json
from ..common.paths import run_dirs
from ..common.viz import overlay_mask
from .metrics import (
    bootstrap_dice_ci,
    dice_threshold_sweep,
    per_slice_metrics,
    summarize_segmentation,
)


@torch.no_grad()
def predict_masks(model, loader, device, max_batches: int | None = None):
    """Return ``(probabilities, true_masks, images, paths)`` as lists of numpy arrays.

    Kept on CPU as float32 arrays so the notebook can plot them directly. Use
    ``max_batches`` on very large test sets to bound memory.
    """
    model = model.to(device).eval()
    probabilities, truths, images, paths = [], [], [], []

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        image_batch = batch[0].to(device, non_blocking=True)
        mask_batch = batch[1]
        probs = torch.sigmoid(model(image_batch)).cpu().numpy()

        batch_paths = list(batch[2]) if len(batch) > 2 else [""] * len(probs)
        for j in range(len(probs)):
            probabilities.append(probs[j, 0])
            truths.append(mask_batch[j, 0].numpy())
            images.append(image_batch[j].cpu().numpy())
            paths.append(batch_paths[j])

    return probabilities, truths, images, paths


def evaluate_segmentation(model, loaders: dict, cfg, device, run_name: str | None = None,
                          pair_table: pd.DataFrame | None = None) -> dict:
    """Full evaluation protocol; saves every artefact and returns the results."""
    run_name = run_name or str(cfg.get("run_name", "unet"))
    dirs = run_dirs(cfg.get("output.root", "outputs/segmentation"), run_name)
    n_boot = int(cfg.get("eval.n_bootstrap", 1000))
    seed = int(cfg.get("seed", 42))
    results: dict = {"run_name": run_name, "dirs": dirs}

    # -- 1. threshold from validation -------------------------------------
    threshold = float(cfg.get("eval.threshold", 0.5))
    if "val" in loaders and bool(cfg.get("eval.tune_threshold", True)):
        val_probs, val_truths, _, _ = predict_masks(model, loaders["val"], device)
        sweep = dice_threshold_sweep(val_probs, val_truths)
        threshold = float(sweep.loc[sweep["mean_dice"].idxmax(), "threshold"])
        save_csv(sweep, dirs["metrics"] / "threshold_sweep_validation.csv")
        print(f"[eval] threshold selected on VALIDATION: {threshold:.2f} "
              f"(mean Dice {sweep['mean_dice'].max():.4f})")
        results["threshold_sweep"] = sweep
    results["threshold"] = threshold

    if "test" not in loaders:
        print("[warn] no test loader - stopping after the validation stage.")
        return results

    # -- 2. per-slice test metrics ----------------------------------------
    probabilities, truths, images, paths = predict_masks(model, loaders["test"], device)
    rows = []
    for prob, truth, path in zip(probabilities, truths, paths):
        row = {"image_path": path, **per_slice_metrics(prob >= threshold, truth > 0.5)}
        rows.append(row)
    per_slice = pd.DataFrame(rows)

    # Attach patient IDs so the bootstrap can resample by patient.
    if pair_table is not None and "image_path" in pair_table.columns:
        lookup = dict(zip(pair_table["image_path"], pair_table.get("patient_id", pd.Series(dtype=str))))
        per_slice["patient_id"] = per_slice["image_path"].map(lookup)

    save_csv(per_slice, dirs["metrics"] / "per_slice_metrics_test.csv")

    summary = summarize_segmentation(per_slice)
    summary["threshold"] = threshold
    summary["run_name"] = run_name

    # -- 3. bootstrap CIs --------------------------------------------------
    if n_boot > 0:
        summary["dice_ci_tumour_slices"] = bootstrap_dice_ci(
            per_slice, "dice", n_boot=n_boot, seed=seed, tumour_only=True)
        summary["iou_ci_tumour_slices"] = bootstrap_dice_ci(
            per_slice, "iou", n_boot=n_boot, seed=seed, tumour_only=True)
    save_json(summary, dirs["metrics"] / "metrics_test.json")
    save_csv(pd.DataFrame([{k: v for k, v in summary.items() if not isinstance(v, dict)}]),
             dirs["metrics"] / "metrics_test_summary.csv")

    # -- 4. figures --------------------------------------------------------
    save_figure(plot_dice_distribution(per_slice), dirs["figures"] / "dice_distribution_test.png")

    n_examples = int(cfg.get("eval.n_example_overlays", 6))
    best_worst = pick_best_worst(per_slice, n=max(n_examples // 2, 1))
    for name, indices in best_worst.items():
        if not indices:
            continue
        fig = plot_prediction_overlays(
            [images[i] for i in indices], [truths[i] for i in indices],
            [probabilities[i] for i in indices], threshold=threshold,
            titles=[f"Dice={per_slice.iloc[i]['dice']:.3f}" for i in indices],
            suptitle=f"{name.replace('_', ' ').title()} test predictions (threshold {threshold:.2f})",
        )
        save_figure(fig, dirs["figures"] / f"overlays_{name}.png")

    if bool(cfg.get("eval.save_predicted_masks", True)):
        saved = save_predicted_masks(probabilities, paths, dirs["predictions"], threshold,
                                     limit=int(cfg.get("eval.max_saved_masks", 50)))
        results["saved_masks"] = saved

    results.update({"per_slice": per_slice, "summary": summary, "probabilities": probabilities,
                    "truths": truths, "images": images, "paths": paths})

    print(f"\n[eval] test summary for run '{run_name}':")
    for key in ["n_slices", "n_slices_with_tumour", "mean_dice_all_slices",
                "mean_dice_tumour_slices", "mean_iou_tumour_slices", "global_dice"]:
        print(f"    {key:<28s} {summary[key]}")
    return results


def pick_best_worst(per_slice: pd.DataFrame, n: int = 3) -> dict[str, list[int]]:
    """Indices of the best and worst tumour-containing slices.

    Showing the worst cases is not optional - a figure of only good results
    misrepresents the model.
    """
    tumour = per_slice[per_slice["has_tumour"] == 1] if "has_tumour" in per_slice else per_slice
    if len(tumour) == 0:
        return {"best": [], "worst": []}
    ordered = tumour.sort_values("dice")
    return {
        "worst": ordered.head(n).index.tolist(),
        "best": ordered.tail(n).iloc[::-1].index.tolist(),
    }


def plot_prediction_overlays(images, true_masks, probabilities, threshold: float = 0.5,
                             titles=None, suptitle: str | None = None):
    """Four columns per slice: MRI | ground truth | prediction | overlay comparison.

    In the last column green = correct tumour pixels, red = false positive,
    blue = missed tumour - the fastest way to read a failure mode.
    """
    import matplotlib.pyplot as plt

    n = len(images)
    fig, axes = plt.subplots(n, 4, figsize=(13, 3.2 * n), squeeze=False)

    for row in range(n):
        image = np.asarray(images[row], dtype=np.float32)
        if image.ndim == 3:                       # CHW -> HWC
            image = np.transpose(image, (1, 2, 0))
        image = _to_unit_range(image)
        display = image if image.ndim == 2 else image[..., :3]

        truth = np.asarray(true_masks[row]).squeeze() > 0.5
        pred = np.asarray(probabilities[row]).squeeze() >= threshold

        axes[row][0].imshow(display, cmap="gray" if display.ndim == 2 else None)
        axes[row][0].set_title("MRI slice", fontsize=10)
        axes[row][1].imshow(overlay_mask(display, truth, color=(0, 1, 0), alpha=0.45))
        axes[row][1].set_title("Ground truth", fontsize=10)
        axes[row][2].imshow(overlay_mask(display, pred, color=(1, 0, 0), alpha=0.45))
        axes[row][2].set_title(f"Prediction (t={threshold:.2f})", fontsize=10)

        comparison = display if display.ndim == 3 else np.stack([display] * 3, axis=-1)
        comparison = overlay_mask(comparison, truth & pred, color=(0, 1, 0), alpha=0.5)
        comparison = overlay_mask(comparison, pred & ~truth, color=(1, 0, 0), alpha=0.5)
        comparison = overlay_mask(comparison, truth & ~pred, color=(0, 0.4, 1), alpha=0.5)
        axes[row][3].imshow(comparison)
        extra = f" - {titles[row]}" if titles is not None and row < len(titles) else ""
        axes[row][3].set_title(f"green=TP red=FP blue=FN{extra}", fontsize=9)

        for ax in axes[row]:
            ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def _to_unit_range(image: np.ndarray) -> np.ndarray:
    """Bring an image array into [0,1] for display.

    Handles all three cases the pipeline can produce: already in [0,1], 0-255
    integers, and per-slice z-scored values (which are centred on zero and go
    negative, so a plain /255 would render them almost black).
    """
    low, high = float(image.min()), float(image.max())
    if 0.0 <= low and high <= 1.0:
        return image
    if 0.0 <= low and high <= 255.0 and high > 1.0:
        return image / 255.0
    span = high - low
    return (image - low) / span if span > 1e-8 else np.zeros_like(image)


def plot_dice_distribution(per_slice: pd.DataFrame):
    """Histogram of per-slice Dice - shows the spread behind the headline mean."""
    import matplotlib.pyplot as plt

    tumour = per_slice[per_slice["has_tumour"] == 1] if "has_tumour" in per_slice else per_slice
    fig, ax = plt.subplots(figsize=(7, 4.2))
    if len(tumour):
        ax.hist(tumour["dice"], bins=25, range=(0, 1), color="#4C72B0", edgecolor="black", linewidth=0.5)
        mean = float(tumour["dice"].mean())
        ax.axvline(mean, color="red", ls="--", lw=2, label=f"mean = {mean:.3f}")
        ax.legend()
    ax.set_xlabel("Per-slice Dice coefficient")
    ax.set_ylabel("Number of slices")
    ax.set_title(f"Dice distribution on test slices containing tumour (n={len(tumour)})")
    fig.tight_layout()
    return fig


def save_predicted_masks(probabilities, paths, out_dir: Path, threshold: float = 0.5,
                         limit: int = 50) -> list[str]:
    """Write binarised predicted masks as PNGs (capped by ``limit``)."""
    from PIL import Image

    out_dir = Path(out_dir) / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for prob, path in list(zip(probabilities, paths))[:limit]:
        mask = ((np.asarray(prob) >= threshold).astype(np.uint8) * 255)
        stem = Path(path).stem if path else f"slice_{len(saved):04d}"
        out = out_dir / f"{stem}_pred.png"
        Image.fromarray(mask).save(out)
        saved.append(str(out))
    print(f"[saved] {len(saved)} predicted masks -> {out_dir}")
    return saved
