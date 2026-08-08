"""Image transforms for chest X-ray classification.

Built on ``torchvision.transforms.v2``. Augmentations are deliberately mild and
anatomy-preserving:

* horizontal flip is **off by default** - a flipped chest X-ray swaps the heart
  to the right side, which is anatomically wrong (and is genuine dextrocardia in
  reality). Enable it only if you justify it in the report.
* no vertical flip or large rotations - orientation is clinically meaningful.
* small rotation/translation/scale mimics realistic patient positioning
  variation, and mild brightness/contrast jitter mimics exposure differences.
"""

from __future__ import annotations

from torchvision.transforms import v2
import torch

# ImageNet statistics - correct choice when using ImageNet-pretrained backbones.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Neutral statistics for a from-scratch model on grayscale-as-RGB input.
NEUTRAL_MEAN = (0.5, 0.5, 0.5)
NEUTRAL_STD = (0.5, 0.5, 0.5)


def get_norm_stats(pretrained: bool) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """ImageNet stats for pretrained backbones, 0.5/0.5 otherwise."""
    return (IMAGENET_MEAN, IMAGENET_STD) if pretrained else (NEUTRAL_MEAN, NEUTRAL_STD)


def build_transforms(image_size: int = 224, train: bool = False, pretrained: bool = True,
                     aug: dict | None = None) -> v2.Compose:
    """Compose the transform pipeline for one split.

    Parameters
    ----------
    image_size:
        Output side length in pixels (square).
    train:
        When False, only resize + normalize (no randomness), so validation and
        test scores are deterministic.
    aug:
        Optional dict from ``config.augmentation``. Recognised keys:
        ``hflip`` (probability, default 0.0), ``rotation_degrees`` (default 7),
        ``translate`` (fraction, default 0.05), ``scale`` (tuple, default
        (0.92, 1.08)), ``brightness`` / ``contrast`` (default 0.10),
        ``random_erasing`` (probability, default 0.0).
    """
    aug = dict(aug or {})
    mean, std = get_norm_stats(pretrained)

    steps: list = [v2.Resize((image_size, image_size), antialias=True)]

    if train:
        hflip_p = float(aug.get("hflip", 0.0))
        if hflip_p > 0:
            steps.append(v2.RandomHorizontalFlip(p=hflip_p))

        degrees = float(aug.get("rotation_degrees", 7))
        translate = float(aug.get("translate", 0.05))
        scale = tuple(aug.get("scale", (0.92, 1.08)))
        if degrees or translate or scale != (1.0, 1.0):
            steps.append(
                v2.RandomAffine(degrees=degrees, translate=(translate, translate), scale=scale)
            )

        brightness = float(aug.get("brightness", 0.10))
        contrast = float(aug.get("contrast", 0.10))
        if brightness or contrast:
            steps.append(v2.ColorJitter(brightness=brightness, contrast=contrast))

    steps += [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ]

    if train and float(aug.get("random_erasing", 0.0)) > 0:
        steps.append(v2.RandomErasing(p=float(aug["random_erasing"]), scale=(0.02, 0.12)))

    return v2.Compose(steps)
