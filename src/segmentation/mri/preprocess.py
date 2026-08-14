"""Phase 3 — deterministic MRI preprocessing.

This module is the **single source of truth** for how a volume becomes model
input. Training, validation, the held-out test evaluation and the Streamlit
inference demo all call the same functions here, so the tensor the model sees at
demo time is byte-for-byte comparable to the one it saw at validation time.

Pipeline (per volume)::

    load NIfTI (nibabel)
      -> verify image/label spatial alignment
      -> per-volume, per-modality z-score over NONZERO brain voxels only
      -> background voxels forced back to exactly 0.0
      -> binary whole-tumour mask: (label > 0)
      -> slice along the axial axis
      -> resize slice to the configured square input size
         (bilinear for image, nearest-neighbour for mask)

Augmentation is deliberately *not* in this module — it is train-only and lives in
`dataset.py`, so the eval/inference path stays free of any stochastic step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from .constants import MODALITY_NAMES, N_MODALITIES, SLICE_AXIS


class AlignmentError(ValueError):
    """Raised when an image and its label do not describe the same volume."""


@dataclass(frozen=True)
class PreprocessConfig:
    """Everything that affects the numbers fed to the network."""

    input_size: int = 192
    slice_axis: int = SLICE_AXIS
    normalization: str = "nonzero_zscore_per_modality"
    image_interpolation: str = "bilinear"
    mask_interpolation: str = "nearest"
    binary_rule: str = "label > 0"

    def as_dict(self) -> dict:
        return {
            "input_size": self.input_size,
            "slice_axis": self.slice_axis,
            "normalization": self.normalization,
            "image_interpolation": self.image_interpolation,
            "mask_interpolation": self.mask_interpolation,
            "binary_rule": self.binary_rule,
            "modalities": list(MODALITY_NAMES),
            "in_channels": N_MODALITIES,
            "out_channels": 1,
        }


# --------------------------------------------------------------------------
# volume level
# --------------------------------------------------------------------------
def load_volume(image_path: str | Path, label_path: str | Path | None = None):
    """Load an image volume (H,W,D,C) and optional label volume (H,W,D)."""
    img_nii = nib.load(Path(image_path))
    image = np.asanyarray(img_nii.dataobj).astype(np.float32)

    label = None
    if label_path is not None:
        lab_nii = nib.load(Path(label_path))
        label = np.asanyarray(lab_nii.dataobj).astype(np.int16)
        verify_alignment(img_nii, lab_nii, image, label)
    return image, label


def verify_alignment(img_nii, lab_nii, image: np.ndarray, label: np.ndarray) -> None:
    """Fail loudly if image and label are not the same volume in the same frame."""
    if image.ndim != 4:
        raise AlignmentError(f"image must be 4D (H,W,D,C), got shape {image.shape}")
    if label.ndim != 3:
        raise AlignmentError(f"label must be 3D (H,W,D), got shape {label.shape}")
    if image.shape[:3] != label.shape:
        raise AlignmentError(
            f"spatial mismatch: image {image.shape[:3]} vs label {label.shape}"
        )
    if not np.allclose(img_nii.affine, lab_nii.affine, atol=1e-4):
        raise AlignmentError("image and label affines differ — volumes are not aligned")


def normalize_volume(image: np.ndarray) -> np.ndarray:
    """Per-modality z-score using only nonzero (brain) voxels; background stays 0.

    Air outside the skull is exactly 0 in this dataset. Including it in the
    statistics would drag the mean toward zero and shrink the dynamic range of
    the brain itself, so the mean/std are computed over the nonzero mask only and
    the background is written back as exact 0.0 afterwards.
    """
    if image.ndim != 4:
        raise ValueError(f"expected 4D (H,W,D,C) volume, got {image.shape}")

    out = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[3]):
        channel = image[..., c]
        brain = channel != 0
        if not brain.any():
            continue  # empty modality — leave as all zeros
        vals = channel[brain]
        mean = float(vals.mean())
        std = float(vals.std())
        if std < 1e-8:
            std = 1.0
        normed = (channel - mean) / std
        out[..., c] = np.where(brain, normed, 0.0).astype(np.float32)
    return out


def binarize_label(label: np.ndarray) -> np.ndarray:
    """Whole tumour: merge every documented non-background source label."""
    return (label > 0).astype(np.uint8)


def volume_normalization_stats(image: np.ndarray) -> list[dict]:
    """Per-modality nonzero statistics — recorded for documentation/audit."""
    stats = []
    for c in range(image.shape[3]):
        channel = image[..., c]
        brain = channel != 0
        vals = channel[brain]
        stats.append(
            {
                "modality": MODALITY_NAMES[c] if c < len(MODALITY_NAMES) else f"ch{c}",
                "nonzero_voxels": int(brain.sum()),
                "mean": float(vals.mean()) if vals.size else 0.0,
                "std": float(vals.std()) if vals.size else 0.0,
            }
        )
    return stats


# --------------------------------------------------------------------------
# slice level
# --------------------------------------------------------------------------
def take_slice(volume: np.ndarray, index: int, *, slice_axis: int = SLICE_AXIS) -> np.ndarray:
    """Extract one 2D slice, returning (C,H,W) for images or (H,W) for labels.

    Basic slicing, not ``np.take``: on a (240,240,155,4) volume ``np.take`` costs
    ~200 ms per slice because it falls back to a full fancy-index gather, while
    basic slicing returns a view and copies only the 2D plane (~0.4 ms). The two
    were verified to produce identical arrays.
    """
    sl = volume[(slice(None),) * slice_axis + (index,)]
    if volume.ndim == 4:  # (H,W,D,C) -> (H,W,C) -> (C,H,W)
        return np.ascontiguousarray(np.transpose(sl, (2, 0, 1)))
    return np.ascontiguousarray(sl)


def resize_image_slice(image: torch.Tensor, size: int) -> torch.Tensor:
    """Bilinear resize for the image only. Input/output (C,H,W)."""
    if image.shape[-2:] == (size, size):
        return image
    return F.interpolate(
        image.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False
    ).squeeze(0)


def resize_mask_slice(mask: torch.Tensor, size: int) -> torch.Tensor:
    """Nearest-neighbour resize for the mask only — never interpolate labels.

    Input/output (1,H,W). Bilinear here would invent fractional label values and
    smear the tumour boundary, so nearest is the only correct choice.
    """
    if mask.shape[-2:] == (size, size):
        return mask
    return F.interpolate(mask.unsqueeze(0).float(), size=(size, size), mode="nearest").squeeze(0)


def prepare_slice(
    norm_volume: np.ndarray,
    binary_label: np.ndarray | None,
    index: int,
    cfg: PreprocessConfig,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Deterministic slice -> model-ready tensors.

    Returns image (C,size,size) float32 and mask (1,size,size) float32 in {0,1}.
    This exact function is used by training, validation, test and inference.
    """
    img_np = take_slice(norm_volume, index, slice_axis=cfg.slice_axis)  # (C,H,W)
    image = torch.from_numpy(img_np.astype(np.float32))
    image = resize_image_slice(image, cfg.input_size)

    mask = None
    if binary_label is not None:
        m_np = take_slice(binary_label, index, slice_axis=cfg.slice_axis)  # (H,W)
        mask = torch.from_numpy(m_np.astype(np.float32)).unsqueeze(0)
        mask = resize_mask_slice(mask, cfg.input_size)
        mask = (mask > 0.5).float()
    return image, mask


def prepare_case(
    image_path: str | Path,
    label_path: str | Path | None,
    cfg: PreprocessConfig,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load + normalize + binarize one case. Returns (norm_volume, binary_label)."""
    image, label = load_volume(image_path, label_path)
    norm = normalize_volume(image)
    binary = binarize_label(label) if label is not None else None
    return norm, binary


def tumour_slice_indices(binary_label: np.ndarray, *, slice_axis: int = SLICE_AXIS) -> np.ndarray:
    """Indices of slices containing at least one tumour voxel."""
    other = tuple(a for a in range(binary_label.ndim) if a != slice_axis)
    return np.flatnonzero(binary_label.any(axis=other))


def brain_slice_indices(
    norm_volume: np.ndarray, *, slice_axis: int = SLICE_AXIS, min_fraction: float = 0.02
) -> np.ndarray:
    """Indices of slices with a meaningful amount of brain tissue.

    Used to keep the deterministic negative sample on real anatomy rather than
    the pure-air slices at the top and bottom of the volume, which are trivial.
    """
    nonzero = norm_volume != 0
    axes = tuple(a for a in range(nonzero.ndim) if a != slice_axis)
    frac = nonzero.mean(axis=axes)
    return np.flatnonzero(frac >= min_fraction)
