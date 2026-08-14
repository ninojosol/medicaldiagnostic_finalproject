"""Dice / IoU for binary whole-tumour segmentation.

Two levels are reported and must not be confused:

  * **slice level** — used inside the training loop for a cheap running signal.
  * **per-case (volumetric) level** — the headline number. Confusion-matrix
    counts are accumulated across every slice of a case and Dice/IoU are computed
    once from the case totals. This is the honest measure: averaging per-slice
    Dice would let hundreds of trivially-empty slices inflate the score.

Empty-target convention: when a case has no ground-truth tumour AND the model
predicts none, Dice and IoU are defined as 1.0. If the model predicts tumour
where there is none, both are 0.0. Cases with no ground-truth tumour are counted
and reported separately so they can never quietly carry the aggregate.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch


@dataclass
class ConfusionCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def __add__(self, other: "ConfusionCounts") -> "ConfusionCounts":
        return ConfusionCounts(
            self.tp + other.tp, self.fp + other.fp, self.fn + other.fn, self.tn + other.tn
        )

    @property
    def dice(self) -> float:
        denom = 2 * self.tp + self.fp + self.fn
        if denom == 0:
            return 1.0  # no ground truth and no prediction
        return 2.0 * self.tp / denom

    @property
    def iou(self) -> float:
        denom = self.tp + self.fp + self.fn
        if denom == 0:
            return 1.0
        return self.tp / denom

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else (1.0 if self.fn == 0 else 0.0)

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else (1.0 if self.fp == 0 else 0.0)

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(
            {
                "dice": self.dice,
                "iou": self.iou,
                "precision": self.precision,
                "recall": self.recall,
            }
        )
        return d


def confusion_from_arrays(pred: np.ndarray, target: np.ndarray) -> ConfusionCounts:
    """Counts from two boolean arrays of identical shape."""
    p = pred.astype(bool)
    t = target.astype(bool)
    tp = int(np.count_nonzero(p & t))
    fp = int(np.count_nonzero(p & ~t))
    fn = int(np.count_nonzero(~p & t))
    tn = int(p.size - tp - fp - fn)
    return ConfusionCounts(tp, fp, fn, tn)


@torch.no_grad()
def batch_confusion(
    logits: torch.Tensor, targets: torch.Tensor, *, threshold: float = 0.5
) -> ConfusionCounts:
    pred = (torch.sigmoid(logits.float()) > threshold)
    tgt = targets > 0.5
    tp = int((pred & tgt).sum().item())
    fp = int((pred & ~tgt).sum().item())
    fn = int((~pred & tgt).sum().item())
    tn = int(pred.numel() - tp - fp - fn)
    return ConfusionCounts(tp, fp, fn, tn)


@torch.no_grad()
def batch_mean_slice_dice(
    logits: torch.Tensor, targets: torch.Tensor, *, threshold: float = 0.5, eps: float = 1e-7
) -> float:
    """Mean hard Dice over the slices in a batch (running training signal only)."""
    pred = (torch.sigmoid(logits.float()) > threshold).float()
    tgt = (targets > 0.5).float()
    dims = tuple(range(1, pred.ndim))
    inter = (pred * tgt).sum(dim=dims)
    denom = pred.sum(dim=dims) + tgt.sum(dim=dims)
    dice = torch.where(denom > 0, (2 * inter) / (denom + eps), torch.ones_like(denom))
    return float(dice.mean().item())


def aggregate_case_metrics(per_case: list[dict]) -> dict:
    """Aggregate per-case rows into the reported summary."""
    if not per_case:
        return {}
    dice = np.array([r["dice"] for r in per_case], dtype=float)
    iou = np.array([r["iou"] for r in per_case], dtype=float)
    with_tumour = [r for r in per_case if r["gt_tumour_voxels"] > 0]
    d_wt = np.array([r["dice"] for r in with_tumour], dtype=float)
    i_wt = np.array([r["iou"] for r in with_tumour], dtype=float)

    total = ConfusionCounts()
    for r in per_case:
        total = total + ConfusionCounts(r["tp"], r["fp"], r["fn"], r["tn"])

    def _pack(arr: np.ndarray) -> dict:
        if arr.size == 0:
            return {"mean": None, "std": None, "median": None, "p25": None, "p75": None,
                    "min": None, "max": None}
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    return {
        "n_cases": len(per_case),
        "n_cases_with_tumour": len(with_tumour),
        "n_cases_without_tumour": len(per_case) - len(with_tumour),
        "dice": _pack(dice),
        "iou": _pack(iou),
        "dice_cases_with_tumour": _pack(d_wt),
        "iou_cases_with_tumour": _pack(i_wt),
        "micro": {
            "dice": total.dice,
            "iou": total.iou,
            "precision": total.precision,
            "recall": total.recall,
            "tp": total.tp,
            "fp": total.fp,
            "fn": total.fn,
            "tn": total.tn,
        },
        "mean_dice": float(dice.mean()),
        "mean_iou": float(iou.mean()),
    }


def bootstrap_ci(
    values: list[float], *, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42
) -> dict:
    """Percentile bootstrap CI over per-case values."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": None, "lo": None, "hi": None, "n_boot": n_boot}
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "lo": float(np.percentile(means, 100 * alpha / 2)),
        "hi": float(np.percentile(means, 100 * (1 - alpha / 2))),
        "n_boot": n_boot,
        "alpha": alpha,
    }
