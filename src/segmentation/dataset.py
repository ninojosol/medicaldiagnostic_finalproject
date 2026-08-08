"""Dataset, joint image/mask augmentation and DataLoaders for MRI segmentation.

Augmentation must transform the image and its mask **identically** - rotate the
image without rotating the mask and the labels become noise. This is handled by
``torchvision.transforms.v2`` together with ``tv_tensors.Mask``, which tags the
mask so geometric transforms apply to it while photometric ones (brightness,
contrast) do not.

Masks are resized with nearest-neighbour interpolation to keep them binary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from ..common.errors import DataLayoutError
from ..common.seed import make_generator, seed_worker


def build_seg_transforms(image_size: int = 256, train: bool = False, aug: dict | None = None):
    """Joint transforms for (image, mask).

    Only mild geometric augmentation is used: brain MRI slices are already
    roughly registered, so large deformations create anatomically implausible
    training examples. Flips are acceptable here (unlike chest X-rays) because
    the brain is broadly left-right symmetric and tumours occur on either side.
    """
    aug = dict(aug or {})
    steps: list = [
        v2.Resize((image_size, image_size), antialias=True),
    ]

    if train:
        if float(aug.get("hflip", 0.5)) > 0:
            steps.append(v2.RandomHorizontalFlip(p=float(aug.get("hflip", 0.5))))
        if float(aug.get("vflip", 0.0)) > 0:
            steps.append(v2.RandomVerticalFlip(p=float(aug.get("vflip", 0.0))))
        degrees = float(aug.get("rotation_degrees", 15))
        translate = float(aug.get("translate", 0.05))
        scale = tuple(aug.get("scale", (0.9, 1.1)))
        if degrees or translate or scale != (1.0, 1.0):
            steps.append(v2.RandomAffine(degrees=degrees, translate=(translate, translate), scale=scale))
        brightness = float(aug.get("brightness", 0.15))
        contrast = float(aug.get("contrast", 0.15))
        if brightness or contrast:
            steps.append(v2.ColorJitter(brightness=brightness, contrast=contrast))

    steps += [
        v2.ToImage(),
        # Scale ONLY the image to [0,1]. The dict form is deliberate: a plain
        # ToDtype(float32, scale=True) risks dividing the 0/1 mask by 255 as
        # well, which silently turns every mask into zeros.
        v2.ToDtype({tv_tensors.Image: torch.float32, "others": None}, scale=True),
    ]
    return v2.Compose(steps)


class MRISegmentationDataset(Dataset):
    """Paired MRI slice + binary tumour mask.

    Returns ``(image, mask)`` where image is ``(C,H,W)`` float in [0,1] and mask
    is ``(1,H,W)`` float with values in {0,1} - the shape ``BCEWithLogitsLoss``
    and the Dice loss both expect.
    """

    def __init__(self, df: pd.DataFrame, image_size: int = 256, transform=None,
                 in_channels: int = 3, return_path: bool = False,
                 normalize_per_image: bool = False):
        for column in ("image_path", "mask_path"):
            if column not in df.columns:
                raise DataLayoutError(f"Pair table is missing the '{column}' column.")
        if len(df) == 0:
            raise DataLayoutError("MRISegmentationDataset received an empty DataFrame.")

        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.transform = transform if transform is not None else build_seg_transforms(image_size, train=False)
        self.in_channels = in_channels
        self.return_path = return_path
        self.normalize_per_image = normalize_per_image
        self.image_paths = self.df["image_path"].tolist()
        self.mask_paths = self.df["mask_path"].tolist()

    def __len__(self) -> int:
        return len(self.df)

    def _read(self, idx: int):
        try:
            with Image.open(self.image_paths[idx]) as im:
                image = im.convert("RGB" if self.in_channels == 3 else "L")
                image.load()
            with Image.open(self.mask_paths[idx]) as mm:
                mask = np.array(mm.convert("L"))
        except FileNotFoundError as exc:
            raise DataLayoutError(
                f"Missing file for pair {idx}: {exc}\n"
                "  Fix: re-run the pair validation step and drop invalid rows."
            ) from exc
        return image, mask

    def __getitem__(self, idx: int):
        image, mask = self._read(idx)
        binary = (mask > (127 if mask.max() > 1 else 0)).astype(np.uint8)

        mask_tv = tv_tensors.Mask(torch.from_numpy(binary)[None])  # (1,H,W) uint8 {0,1}
        image_t, mask_t = self.transform(image, mask_tv)

        # The mask stays uint8 {0,1} through the pipeline (see build_seg_transforms),
        # so > 0 is the correct binarisation here.
        mask_t = (mask_t > 0).float()
        if self.normalize_per_image:
            # Per-slice z-scoring: helps when MRI intensity scaling varies between scanners.
            std = image_t.std()
            image_t = (image_t - image_t.mean()) / (std + 1e-6) if std > 0 else image_t

        if self.return_path:
            return image_t, mask_t, self.image_paths[idx]
        return image_t, mask_t


def build_seg_dataloaders(df: pd.DataFrame, cfg, splits=("train", "val", "test")) -> dict[str, DataLoader]:
    """One DataLoader per split, configured from ``cfg``."""
    if "split" not in df.columns:
        raise DataLayoutError("Pair table has no 'split' column - run the splitting step first.")

    image_size = int(cfg.get("data.image_size", 256))
    batch_size = int(cfg.get("train.batch_size", 16))
    num_workers = int(cfg.get("train.num_workers", 0))
    in_channels = int(cfg.get("model.in_channels", 3))
    normalize_per_image = bool(cfg.get("data.normalize_per_image", False))
    aug = cfg.get("augmentation", {})
    aug = aug.to_dict() if hasattr(aug, "to_dict") else dict(aug or {})
    seed = int(cfg.get("seed", 42))

    loaders: dict[str, DataLoader] = {}
    for split in splits:
        subset = df[df["split"] == split]
        if len(subset) == 0:
            print(f"[warn] split '{split}' is empty - skipping.")
            continue
        is_train = split == "train"
        dataset = MRISegmentationDataset(
            subset,
            image_size=image_size,
            transform=build_seg_transforms(image_size, train=is_train, aug=aug),
            in_channels=in_channels,
            return_path=not is_train,
            normalize_per_image=normalize_per_image,
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
        print(f"[data] {split:<6s} {len(dataset):>7d} slices -> {len(loaders[split])} batches of {batch_size}")
    return loaders


def peek_seg_batch(loader: DataLoader) -> dict:
    """Shape/range sanity check for a segmentation batch."""
    batch = next(iter(loader))
    images, masks = batch[0], batch[1]
    info = {
        "images_shape": tuple(images.shape),
        "images_min": round(float(images.min()), 4),
        "images_max": round(float(images.max()), 4),
        "masks_shape": tuple(masks.shape),
        "masks_unique": sorted(float(v) for v in torch.unique(masks))[:5],
        "foreground_fraction": round(float(masks.mean()), 6),
    }
    for key, value in info.items():
        print(f"{key:<22s} {value}")
    return info
