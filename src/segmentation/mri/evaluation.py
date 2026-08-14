"""Per-case volumetric evaluation and single-slice inference.

`evaluate_case()` is the one code path used by:
  * the final **reported** validation metrics,
  * the one-time held-out internal **test** evaluation,
  * the Streamlit **inference** demo.

All three read the original NIfTI in float32 and call
`preprocess.prepare_case` / `preprocess.prepare_slice`, so preprocessing, tensor
shape and dtype are identical in every context. Nothing here reads the training
slice cache and nothing here is stochastic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .constants import PROJECT_ROOT
from .metrics import ConfusionCounts, confusion_from_arrays
from .preprocess import (
    PreprocessConfig,
    prepare_case,
    prepare_slice,
    resize_mask_slice,
    take_slice,
)


@torch.no_grad()
def predict_volume_masks(
    model: torch.nn.Module,
    norm_volume: np.ndarray,
    cfg: PreprocessConfig,
    *,
    device: torch.device,
    threshold: float = 0.5,
    batch_size: int = 32,
    return_probs: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Run the 2D U-Net slice-by-slice over a whole volume.

    Returns predicted binary masks (D, S, S) uint8 and, optionally, the sigmoid
    probabilities (D, S, S) float32 at the model's working resolution.
    """
    model.eval()
    n_slices = norm_volume.shape[cfg.slice_axis]

    preds = np.zeros((n_slices, cfg.input_size, cfg.input_size), dtype=np.uint8)
    probs = np.zeros((n_slices, cfg.input_size, cfg.input_size), dtype=np.float32) if return_probs else None

    for start in range(0, n_slices, batch_size):
        stop = min(start + batch_size, n_slices)
        tensors = [prepare_slice(norm_volume, None, i, cfg)[0] for i in range(start, stop)]
        batch = torch.stack(tensors).to(device, non_blocking=True)
        logits = model(batch)
        p = torch.sigmoid(logits.float()).squeeze(1).cpu().numpy()
        preds[start:stop] = (p > threshold).astype(np.uint8)
        if probs is not None:
            probs[start:stop] = p
    return preds, probs


def ground_truth_masks(
    binary_label: np.ndarray, cfg: PreprocessConfig
) -> np.ndarray:
    """Resample the binary label volume to model resolution, nearest-neighbour.

    Uses the same `take_slice` + `resize_mask_slice` + `>0.5` steps as
    `prepare_slice`, so ground truth is built exactly as it is during training.
    """
    n_slices = binary_label.shape[cfg.slice_axis]
    out = np.zeros((n_slices, cfg.input_size, cfg.input_size), dtype=np.uint8)
    for i in range(n_slices):
        m_np = take_slice(binary_label, i, slice_axis=cfg.slice_axis)
        mask = torch.from_numpy(m_np.astype(np.float32)).unsqueeze(0)
        mask = resize_mask_slice(mask, cfg.input_size)
        out[i] = (mask.squeeze(0) > 0.5).numpy().astype(np.uint8)
    return out


@torch.no_grad()
def evaluate_case(
    model: torch.nn.Module,
    image_path: str | Path,
    label_path: str | Path | None,
    cfg: PreprocessConfig,
    *,
    device: torch.device,
    threshold: float = 0.5,
    batch_size: int = 32,
    return_arrays: bool = False,
) -> dict:
    """Full per-case evaluation from the original NIfTI.

    When `label_path` is None (external upload) only the prediction and its area
    are returned — no metrics, because there is no ground truth.
    """
    image_path = Path(image_path)
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path
    if label_path is not None:
        label_path = Path(label_path)
        if not label_path.is_absolute():
            label_path = PROJECT_ROOT / label_path

    norm, binary = prepare_case(image_path, label_path, cfg)
    preds, probs = predict_volume_masks(
        model, norm, cfg, device=device, threshold=threshold,
        batch_size=batch_size, return_probs=return_arrays,
    )

    result: dict = {
        "case_id": image_path.name.replace(".nii.gz", ""),
        "n_slices": int(preds.shape[0]),
        "input_size": cfg.input_size,
        "threshold": float(threshold),
        "pred_tumour_voxels": int(preds.sum()),
        "pred_tumour_fraction": float(preds.sum() / preds.size),
        "pred_tumour_slices": int((preds.reshape(preds.shape[0], -1).sum(axis=1) > 0).sum()),
        "has_ground_truth": binary is not None,
    }

    if binary is None:
        if return_arrays:
            result["norm_volume"] = norm
            result["pred_masks"] = preds
            result["probs"] = probs
        return result

    gt = ground_truth_masks(binary, cfg)
    counts = confusion_from_arrays(preds, gt)
    result.update(
        {
            "tp": counts.tp,
            "fp": counts.fp,
            "fn": counts.fn,
            "tn": counts.tn,
            "dice": counts.dice,
            "iou": counts.iou,
            "precision": counts.precision,
            "recall": counts.recall,
            "gt_tumour_voxels": int(gt.sum()),
            "gt_tumour_fraction": float(gt.sum() / gt.size),
            "gt_tumour_slices": int((gt.reshape(gt.shape[0], -1).sum(axis=1) > 0).sum()),
        }
    )
    if return_arrays:
        result["norm_volume"] = norm
        result["binary_label"] = binary
        result["pred_masks"] = preds
        result["gt_masks"] = gt
        result["probs"] = probs
    return result


@torch.no_grad()
def infer_single_slice(
    model: torch.nn.Module,
    norm_volume: np.ndarray,
    slice_index: int,
    cfg: PreprocessConfig,
    *,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """Predict one slice — the exact path used by the Streamlit demo."""
    model.eval()
    image, _ = prepare_slice(norm_volume, None, slice_index, cfg)
    batch = image.unsqueeze(0).to(device)
    logits = model(batch)
    prob = torch.sigmoid(logits.float()).squeeze().cpu().numpy()
    pred = (prob > threshold).astype(np.uint8)
    return {
        "image": image.numpy(),          # (4, S, S) float32
        "prob": prob,                    # (S, S) float32
        "pred": pred,                    # (S, S) uint8
        "slice_index": int(slice_index),
        "pred_pixels": int(pred.sum()),
        "pred_fraction": float(pred.sum() / pred.size),
        "threshold": float(threshold),
        "input_size": cfg.input_size,
    }


def slice_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Dice/IoU for a single slice (presentation samples with known ground truth)."""
    counts = confusion_from_arrays(pred, gt)
    return {
        "dice": counts.dice,
        "iou": counts.iou,
        "precision": counts.precision,
        "recall": counts.recall,
        "tp": counts.tp,
        "fp": counts.fp,
        "fn": counts.fn,
        "gt_pixels": int(np.count_nonzero(gt)),
        "pred_pixels": int(np.count_nonzero(pred)),
    }


def evaluate_split(
    model: torch.nn.Module,
    manifest,
    cfg: PreprocessConfig,
    *,
    device: torch.device,
    threshold: float = 0.5,
    batch_size: int = 32,
    progress: bool = True,
    label: str = "split",
) -> list[dict]:
    """Per-case evaluation over every case in a manifest frame."""
    rows: list[dict] = []
    n = len(manifest)
    for i, row in manifest.reset_index(drop=True).iterrows():
        res = evaluate_case(
            model,
            row["image_path"],
            row["label_path"],
            cfg,
            device=device,
            threshold=threshold,
            batch_size=batch_size,
        )
        res["case_id"] = row["case_id"]
        res["patient_id"] = row["patient_id"]
        res["split"] = row["split"]
        rows.append(res)
        if progress and ((i + 1) % 10 == 0 or i + 1 == n):
            mean_d = float(np.mean([r["dice"] for r in rows]))
            print(f"  [{label}] {i + 1}/{n} cases — running mean Dice {mean_d:.4f}", flush=True)
    return rows
