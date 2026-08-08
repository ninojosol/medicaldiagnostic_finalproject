"""PyTorch Dataset / DataLoader construction for chest X-ray classification.

Labels are always returned as a float vector of length ``len(labels)``, so the
exact same code path serves binary (1 column), multi-class-as-multi-label, and
true multi-label problems. The loss (``BCEWithLogitsLoss``) matches that shape.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..common.errors import DataLayoutError
from ..common.seed import make_generator, seed_worker
from .transforms import build_transforms


class XrayDataset(Dataset):
    """Chest X-ray dataset backed by a manifest DataFrame.

    Parameters
    ----------
    df:
        Manifest rows for ONE split (see :mod:`src.classification.manifest`).
    labels:
        Ordered label names; also the order of the returned target vector.
    transform:
        A torchvision v2 transform applied to the PIL image.
    grayscale_to_rgb:
        Replicate the single X-ray channel to 3 channels. Required for
        ImageNet-pretrained backbones, harmless for the baseline CNN.
    return_path:
        Also return the file path - handy for Grad-CAM and error analysis.
    """

    def __init__(self, df: pd.DataFrame, labels: list[str], transform=None,
                 grayscale_to_rgb: bool = True, return_path: bool = False):
        missing = [c for c in ["image_path", *labels] if c not in df.columns]
        if missing:
            raise DataLayoutError(f"Manifest is missing required columns: {missing}")
        if len(df) == 0:
            raise DataLayoutError("XrayDataset received an empty DataFrame (check your split sizes).")

        self.df = df.reset_index(drop=True)
        self.labels = list(labels)
        self.transform = transform
        self.grayscale_to_rgb = grayscale_to_rgb
        self.return_path = return_path
        self.paths = self.df["image_path"].tolist()
        # An explicit copy is required: pandas can return a read-only array, and
        # torch.from_numpy then warns about non-writable tensors on every sample.
        self.targets = np.array(self.df[self.labels].to_numpy(dtype=np.float32), copy=True)

    def __len__(self) -> int:
        return len(self.df)

    def load_image(self, idx: int) -> Image.Image:
        path = self.paths[idx]
        try:
            img = Image.open(path)
            img.load()
        except FileNotFoundError as exc:
            raise DataLayoutError(
                f"Image listed in the manifest is missing on disk: {path}\n"
                "  Fix: re-run the EDA notebook's missing-file check and drop bad rows before training."
            ) from exc
        except Exception as exc:
            raise DataLayoutError(f"Could not read image {path}: {exc}") from exc
        return img.convert("RGB" if self.grayscale_to_rgb else "L")

    def __getitem__(self, idx: int):
        img = self.load_image(idx)
        if self.transform is not None:
            img = self.transform(img)
        target = torch.from_numpy(self.targets[idx])
        if self.return_path:
            return img, target, self.paths[idx]
        return img, target


def build_dataloaders(df: pd.DataFrame, cfg, splits=("train", "val", "test")) -> dict[str, DataLoader]:
    """Create one DataLoader per split using values from the config.

    Reproducibility: shuffling uses a seeded generator and each worker is seeded
    via ``seed_worker``, so epoch order and augmentations repeat run to run.
    """
    labels = list(cfg.data.labels)
    image_size = int(cfg.get("data.image_size", 224))
    batch_size = int(cfg.get("train.batch_size", 32))
    num_workers = int(cfg.get("train.num_workers", 0))
    pretrained = bool(cfg.get("model.pretrained", True))
    aug = cfg.get("augmentation", {})
    aug = aug.to_dict() if hasattr(aug, "to_dict") else dict(aug or {})
    seed = int(cfg.get("seed", 42))

    if "split" not in df.columns:
        raise DataLayoutError("DataFrame has no 'split' column - run the splitting step first.")

    loaders: dict[str, DataLoader] = {}
    for split in splits:
        subset = df[df["split"] == split]
        if len(subset) == 0:
            print(f"[warn] split '{split}' is empty - skipping.")
            continue
        is_train = split == "train"
        dataset = XrayDataset(
            subset,
            labels=labels,
            transform=build_transforms(image_size, train=is_train, pretrained=pretrained, aug=aug),
            grayscale_to_rgb=True,
            return_path=not is_train,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=is_train and len(dataset) > batch_size,
            persistent_workers=num_workers > 0,
            worker_init_fn=seed_worker if num_workers > 0 else None,
            generator=make_generator(seed) if is_train else None,
        )
        print(f"[data] {split:<6s} {len(dataset):>7d} images -> {len(loaders[split])} batches "
              f"of {batch_size}")
    return loaders


def compute_pos_weights(df: pd.DataFrame, labels: list[str], split: str = "train",
                        cap: float = 20.0) -> torch.Tensor:
    """``pos_weight`` for ``BCEWithLogitsLoss``, computed on the TRAIN split only.

    For each label the weight is ``n_negative / n_positive``, which raises the
    loss contribution of the rare positive class. Weights are capped (default
    20x) because extremely rare findings otherwise produce unstable gradients.

    Computing this on train only is deliberate: touching val/test statistics
    would leak information into model selection.
    """
    train_df = df[df["split"] == split] if "split" in df.columns else df
    if len(train_df) == 0:
        raise DataLayoutError(f"No rows with split=='{split}' - cannot compute class weights.")

    weights = []
    print(f"pos_weight (computed on '{split}', capped at {cap}x):")
    for lab in labels:
        n_pos = int(train_df[lab].sum())
        n_neg = int(len(train_df) - n_pos)
        if n_pos == 0:
            print(f"  [warn] label '{lab}' has NO positive examples in {split}; weight set to 1.0")
            weights.append(1.0)
            continue
        if n_neg == 0:
            # n_neg/n_pos would be 0.0, which zeroes out the positive class in the
            # loss - far worse than no weighting at all.
            print(f"  [warn] label '{lab}' has NO negative examples in {split}: every image is "
                  f"positive, so the task is degenerate. Weight set to 1.0.")
            print(f"         Check data.labels and data.unlisted_folders_as_negative in the config.")
            weights.append(1.0)
            continue
        w = min(n_neg / n_pos, cap)
        weights.append(w)
        print(f"  {lab:<24s} pos={n_pos:<7d} neg={n_neg:<7d} -> pos_weight={w:.3f}")
    return torch.tensor(weights, dtype=torch.float32)


def peek_batch(loader: DataLoader) -> dict:
    """Fetch one batch and report shapes/ranges - a cheap sanity check for notebooks."""
    batch = next(iter(loader))
    images, targets = batch[0], batch[1]
    info = {
        "images_shape": tuple(images.shape),
        "images_dtype": str(images.dtype),
        "images_min": round(float(images.min()), 4),
        "images_max": round(float(images.max()), 4),
        "targets_shape": tuple(targets.shape),
        "targets_positive_rate": round(float(targets.mean()), 4),
    }
    for key, value in info.items():
        print(f"{key:<24s} {value}")
    return info
