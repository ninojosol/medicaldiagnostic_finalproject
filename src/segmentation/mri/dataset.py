"""2D slice Dataset / DataLoader for the MRI whole-tumour U-Net.

Each sample is one axial slice:

    image : (4, S, S) float32  — FLAIR, T1w, T1gd, T2w, nonzero-voxel z-scored
    mask  : (1, S, S) float32  — binary whole tumour, values in {0, 1}

Augmentation is train-only and always applies the *same* geometric operation to
the image and the mask, with bilinear resampling reserved for the image and
nearest-neighbour for the mask, so alignment can never drift.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .cache import cache_dir
from .preprocess import PreprocessConfig


class SliceAugment:
    """Deterministic-per-seed geometric + intensity augmentation for training.

    Geometry is applied identically to image and mask. Intensity jitter touches
    the image only and is masked to the brain, so background stays exactly zero.
    """

    def __init__(
        self,
        *,
        hflip_p: float = 0.5,
        vflip_p: float = 0.0,
        rot90_p: float = 0.0,
        max_shift_frac: float = 0.0625,
        intensity_scale: float = 0.1,
        intensity_shift: float = 0.1,
        p_intensity: float = 0.5,
    ) -> None:
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.rot90_p = rot90_p
        self.max_shift_frac = max_shift_frac
        self.intensity_scale = intensity_scale
        self.intensity_shift = intensity_shift
        self.p_intensity = p_intensity

    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor, rng: np.random.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.hflip_p and rng.random() < self.hflip_p:
            image = torch.flip(image, dims=[-1])
            mask = torch.flip(mask, dims=[-1])
        if self.vflip_p and rng.random() < self.vflip_p:
            image = torch.flip(image, dims=[-2])
            mask = torch.flip(mask, dims=[-2])
        if self.rot90_p and rng.random() < self.rot90_p:
            k = int(rng.integers(1, 4))
            image = torch.rot90(image, k, dims=(-2, -1))
            mask = torch.rot90(mask, k, dims=(-2, -1))

        if self.max_shift_frac > 0:
            size = image.shape[-1]
            max_px = int(round(self.max_shift_frac * size))
            if max_px > 0:
                dy = int(rng.integers(-max_px, max_px + 1))
                dx = int(rng.integers(-max_px, max_px + 1))
                if dy or dx:
                    # Integer roll + zero-fill: exact for both tensors, no resampling,
                    # so image and mask stay pixel-locked.
                    image = torch.roll(image, shifts=(dy, dx), dims=(-2, -1))
                    mask = torch.roll(mask, shifts=(dy, dx), dims=(-2, -1))
                    if dy > 0:
                        image[..., :dy, :] = 0
                        mask[..., :dy, :] = 0
                    elif dy < 0:
                        image[..., dy:, :] = 0
                        mask[..., dy:, :] = 0
                    if dx > 0:
                        image[..., :, :dx] = 0
                        mask[..., :, :dx] = 0
                    elif dx < 0:
                        image[..., :, dx:] = 0
                        mask[..., :, dx:] = 0

        if self.p_intensity and rng.random() < self.p_intensity:
            brain = image != 0
            scale = 1.0 + float(rng.uniform(-self.intensity_scale, self.intensity_scale))
            shift = float(rng.uniform(-self.intensity_shift, self.intensity_shift))
            image = torch.where(brain, image * scale + shift, image)

        return image, mask


class MRISliceCacheDataset(Dataset):
    """Reads preprocessed slices from the memory-mapped split cache."""

    def __init__(
        self,
        split: str,
        cfg: PreprocessConfig,
        *,
        augment: SliceAugment | None = None,
        seed: int = 42,
        cache_root: Path | None = None,
    ) -> None:
        self.split = split
        self.cfg = cfg
        self.augment = augment
        self.seed = seed
        self.dir = cache_root or cache_dir(split, cfg.input_size)
        if not (self.dir / "index.csv").exists():
            raise FileNotFoundError(
                f"slice cache for '{split}' not found at {self.dir}. "
                "Run scripts/mri_build_slice_cache.py first."
            )
        self.index = pd.read_csv(self.dir / "index.csv")
        self._images: np.ndarray | None = None
        self._masks: np.ndarray | None = None
        self.epoch = 0

    # memmaps are opened lazily so the dataset survives DataLoader worker forks
    def _ensure_open(self) -> None:
        if self._images is None:
            self._images = np.load(self.dir / "images.npy", mmap_mode="r")
            self._masks = np.load(self.dir / "masks.npy", mmap_mode="r")

    def set_epoch(self, epoch: int) -> None:
        """Vary augmentation across epochs while staying fully reproducible."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        self._ensure_open()
        image = torch.from_numpy(np.asarray(self._images[i], dtype=np.float32))
        mask = torch.from_numpy(np.asarray(self._masks[i], dtype=np.float32))

        if self.augment is not None:
            rng = np.random.default_rng((self.seed, self.epoch, int(i)))
            image, mask = self.augment(image, mask, rng)

        row = self.index.iloc[i]
        return {
            "image": image,
            "mask": mask,
            "case_id": str(row["case_id"]),
            "slice_index": int(row["slice_index"]),
            "has_tumour": int(row["has_tumour"]),
        }


def build_dataloaders(
    cfg: PreprocessConfig,
    *,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 42,
    augment: SliceAugment | None = None,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, MRISliceCacheDataset, MRISliceCacheDataset]:
    """Train (shuffled, augmented) and validation (ordered, clean) loaders."""
    train_ds = MRISliceCacheDataset("train", cfg, augment=augment, seed=seed)
    valid_ds = MRISliceCacheDataset("valid", cfg, augment=None, seed=seed)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        generator=generator,
        persistent_workers=num_workers > 0,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    return train_loader, valid_loader, train_ds, valid_ds
