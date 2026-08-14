"""Binary segmentation loss for the whole-tumour U-Net.

Documented formula (the exact objective minimised in this project)::

    L = w_bce * BCEWithLogits(z, y) + w_dice * (1 - SoftDice(sigmoid(z), y))

    SoftDice = (2 * sum(p*y) + eps) / (sum(p) + sum(y) + eps)

with, as configured in `configs/mri_unet_whole_tumour.yaml`:

    w_bce  = 0.5
    w_dice = 0.5
    eps    = 1.0            (Dice smoothing, applied to numerator and denominator)

Why this combination: whole-tumour voxels are roughly 1-2% of a brain volume.
Plain BCE is dominated by the easy background and converges to near-empty masks;
plain soft Dice gives a strong overlap signal but is noisy on slices with no
tumour at all (the gradient is ill-conditioned when the target is empty). The
equal-weight sum keeps a stable pixel-wise term and a region-overlap term.

Soft Dice is computed **per sample and then averaged over the batch**, not over
the flattened batch, so a single large tumour cannot dominate the batch term.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """1 - soft Dice, averaged per sample. Expects raw logits."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits.float())
        targets = targets.float()
        dims = tuple(range(1, probs.ndim))
        intersection = (probs * targets).sum(dim=dims)
        denom = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    """w_bce * BCEWithLogits + w_dice * (1 - soft Dice)."""

    def __init__(
        self,
        *,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)
        self.dice = SoftDiceLoss(smooth=smooth)
        self.register_buffer(
            "pos_weight",
            torch.tensor([pos_weight], dtype=torch.float32) if pos_weight else None,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # float32 for numerical stability under AMP
        logits32 = logits.float()
        targets32 = targets.float()
        bce = F.binary_cross_entropy_with_logits(
            logits32, targets32, pos_weight=self.pos_weight
        )
        dice = self.dice(logits32, targets32)
        return self.bce_weight * bce + self.dice_weight * dice

    def components(self, logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        with torch.no_grad():
            logits32 = logits.float()
            targets32 = targets.float()
            bce = F.binary_cross_entropy_with_logits(
                logits32, targets32, pos_weight=self.pos_weight
            )
            dice = self.dice(logits32, targets32)
        return {"bce": float(bce), "dice_loss": float(dice)}

    def describe(self) -> str:
        return (
            f"L = {self.bce_weight} * BCEWithLogits + {self.dice_weight} * "
            f"(1 - SoftDice), Dice smoothing eps={self.smooth}, "
            f"per-sample Dice averaged over batch"
        )


def build_loss(cfg: dict) -> DiceBCELoss:
    train_cfg = cfg.get("train", {})
    return DiceBCELoss(
        bce_weight=train_cfg.get("bce_weight", 0.5),
        dice_weight=train_cfg.get("dice_weight", 0.5),
        smooth=train_cfg.get("dice_smooth", 1.0),
        pos_weight=train_cfg.get("pos_weight"),
    )
