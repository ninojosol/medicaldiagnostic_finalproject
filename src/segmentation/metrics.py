"""Segmentation metrics: Dice, IoU/Jaccard, pixel sensitivity and specificity.

Reporting conventions that matter for a fair result
---------------------------------------------------
* **Per-image Dice averaged over images** is reported, not "global Dice over all
  pixels pooled". The pooled version is dominated by large tumours and hides
  failures on small ones.
* **Empty masks** need an explicit rule. A slice with no tumour where the model
  predicts nothing has an undefined Dice (0/0). Here it counts as 1.0 when
  ``empty_score=1.0`` (correct behaviour rewarded), and the tumour-only mean is
  reported alongside so the number cannot be inflated by a dataset full of empty
  slices. **Always report both**, and say which one you are quoting.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# tensor metrics (used inside the training loop)
# ---------------------------------------------------------------------------
@torch.no_grad()
def dice_coefficient(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5,
                     smooth: float = 1e-7, empty_score: float = 1.0) -> torch.Tensor:
    """Per-image Dice for a batch. Returns a 1-D tensor of length ``B``."""
    preds = (torch.sigmoid(logits) > threshold).float()
    targets = (targets > 0.5).float()
    dims = tuple(range(1, preds.dim()))

    intersection = (preds * targets).sum(dim=dims)
    cardinality = preds.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2 * intersection + smooth) / (cardinality + smooth)
    # Both empty -> perfect by convention.
    both_empty = cardinality == 0
    return torch.where(both_empty, torch.full_like(dice, empty_score), dice)


@torch.no_grad()
def iou_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5,
              smooth: float = 1e-7, empty_score: float = 1.0) -> torch.Tensor:
    """Per-image IoU (Jaccard index) for a batch."""
    preds = (torch.sigmoid(logits) > threshold).float()
    targets = (targets > 0.5).float()
    dims = tuple(range(1, preds.dim()))

    intersection = (preds * targets).sum(dim=dims)
    union = preds.sum(dim=dims) + targets.sum(dim=dims) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return torch.where(union == 0, torch.full_like(iou, empty_score), iou)


# ---------------------------------------------------------------------------
# numpy metrics (used for per-slice reporting)
# ---------------------------------------------------------------------------
def per_slice_metrics(pred_mask: np.ndarray, true_mask: np.ndarray,
                      empty_score: float = 1.0) -> dict:
    """Dice, IoU, sensitivity, specificity and pixel counts for one slice."""
    pred = np.asarray(pred_mask).astype(bool).ravel()
    true = np.asarray(true_mask).astype(bool).ravel()

    tp = int(np.logical_and(pred, true).sum())
    fp = int(np.logical_and(pred, ~true).sum())
    fn = int(np.logical_and(~pred, true).sum())
    tn = int(np.logical_and(~pred, ~true).sum())

    both_empty = (tp + fp + fn) == 0
    dice = empty_score if both_empty else 2 * tp / (2 * tp + fp + fn)
    iou = empty_score if both_empty else tp / (tp + fp + fn)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "precision": float(tp / (tp + fp)) if (tp + fp) else float("nan"),
        "true_pixels": int(true.sum()),
        "pred_pixels": int(pred.sum()),
        "has_tumour": int(true.sum() > 0),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def summarize_segmentation(per_slice: pd.DataFrame) -> dict:
    """Aggregate a per-slice table into the headline numbers for the report."""
    tumour_only = per_slice[per_slice["has_tumour"] == 1] if "has_tumour" in per_slice else per_slice

    def _mean(frame: pd.DataFrame, column: str) -> float:
        return float(frame[column].mean()) if len(frame) and column in frame else float("nan")

    tp = int(per_slice["tp"].sum())
    fp = int(per_slice["fp"].sum())
    fn = int(per_slice["fn"].sum())

    return {
        "n_slices": int(len(per_slice)),
        "n_slices_with_tumour": int(len(tumour_only)),
        "mean_dice_all_slices": round(_mean(per_slice, "dice"), 4),
        "mean_iou_all_slices": round(_mean(per_slice, "iou"), 4),
        "mean_dice_tumour_slices": round(_mean(tumour_only, "dice"), 4),
        "mean_iou_tumour_slices": round(_mean(tumour_only, "iou"), 4),
        "median_dice_tumour_slices": round(float(tumour_only["dice"].median()), 4) if len(tumour_only) else float("nan"),
        "mean_sensitivity_tumour_slices": round(_mean(tumour_only, "sensitivity"), 4),
        "mean_specificity_all_slices": round(_mean(per_slice, "specificity"), 4),
        "global_dice": round(2 * tp / max(2 * tp + fp + fn, 1), 4),
        "global_iou": round(tp / max(tp + fp + fn, 1), 4),
    }


def bootstrap_dice_ci(per_slice: pd.DataFrame, column: str = "dice", n_boot: int = 1000,
                      alpha: float = 0.05, seed: int = 42, tumour_only: bool = True) -> dict:
    """Percentile bootstrap CI for the mean per-slice Dice/IoU.

    Resampling is done over **patients** when a ``patient_id`` column exists,
    because slices from one volume are not independent; falling back to slice
    resampling would produce intervals that are too narrow.
    """
    frame = per_slice[per_slice["has_tumour"] == 1] if (tumour_only and "has_tumour" in per_slice) else per_slice
    if len(frame) == 0:
        return {"metric": column, "point_estimate": float("nan"), "note": "no rows to bootstrap"}

    rng = np.random.default_rng(seed)
    point = float(frame[column].mean())

    if "patient_id" in frame.columns and frame["patient_id"].nunique() > 2:
        groups = [g[column].to_numpy() for _, g in frame.groupby("patient_id")]
        unit = "patient"
        samples = [float(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean())
                   for _ in range(n_boot)]
    else:
        values = frame[column].to_numpy()
        unit = "slice"
        samples = [float(values[rng.integers(0, len(values), len(values))].mean()) for _ in range(n_boot)]

    samples = np.asarray(samples)
    return {
        "metric": column,
        "resampling_unit": unit,
        "tumour_slices_only": bool(tumour_only),
        "point_estimate": round(point, 4),
        "ci_low": round(float(np.percentile(samples, 100 * alpha / 2)), 4),
        "ci_high": round(float(np.percentile(samples, 100 * (1 - alpha / 2))), 4),
        "ci_level": 1 - alpha,
        "n_boot": n_boot,
    }


def dice_threshold_sweep(probabilities: Sequence[np.ndarray], masks: Sequence[np.ndarray],
                         grid: Sequence[float] = tuple(np.arange(0.1, 0.95, 0.05))) -> pd.DataFrame:
    """Mean Dice as a function of the binarisation threshold.

    Run this on the **validation** set to pick the threshold, then apply it
    unchanged to test - same discipline as the classification module.
    """
    rows = []
    for threshold in grid:
        scores = [per_slice_metrics(prob >= threshold, mask)["dice"]
                  for prob, mask in zip(probabilities, masks)]
        rows.append({"threshold": round(float(threshold), 3),
                     "mean_dice": round(float(np.mean(scores)), 4)})
    return pd.DataFrame(rows)
