"""Segmentation losses: Dice, BCE and the standard Dice + BCE combination.

Why not plain BCE
-----------------
Tumours typically occupy well under 2% of a brain MRI slice. Pixel-wise BCE is
dominated by the easy background pixels, so a model that predicts "background
everywhere" reaches a low loss while segmenting nothing.

Why Dice + BCE
--------------
Dice loss optimises region overlap directly, which is exactly the reported
metric, and is insensitive to the background majority. But Dice alone gives weak
gradients when a prediction is empty (nothing overlaps yet) and is unstable on
slices with no tumour at all. BCE supplies dense, stable per-pixel gradients.
Summing the two is the standard, well-justified default in medical segmentation:

    loss = bce_weight * BCE + dice_weight * (1 - Dice)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss on logits.

    ``smooth`` appears in numerator and denominator so an empty prediction on an
    empty mask scores a perfect 1.0 instead of dividing by zero.
    """

    def __init__(self, smooth: float = 1.0, per_image: bool = True):
        super().__init__()
        self.smooth = smooth
        self.per_image = per_image

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        if self.per_image:
            dims = tuple(range(1, probs.dim()))          # keep the batch dimension
        else:
            dims = tuple(range(probs.dim()))             # one Dice over the whole batch

        intersection = (probs * targets).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class TverskyLoss(nn.Module):
    """Generalised Dice where ``alpha``/``beta`` trade false positives against false negatives.

    ``beta > alpha`` penalises missed tumour pixels harder - the usual preference
    in a clinical context. ``alpha = beta = 0.5`` is exactly Dice.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = tuple(range(1, probs.dim()))
        tp = (probs * targets).sum(dim=dims)
        fp = (probs * (1 - targets)).sum(dim=dims)
        fn = ((1 - probs) * targets).sum(dim=dims)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky.mean()


class DiceBCELoss(nn.Module):
    """``bce_weight * BCE + dice_weight * DiceLoss`` - the recommended default.

    ``pos_weight`` optionally re-weights the BCE term by the background/foreground
    ratio; leave it None when Dice already handles the imbalance, which it
    usually does.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5,
                 smooth: float = 1.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss(smooth=smooth)
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight if self.pos_weight is not None else None
        )
        return self.bce_weight * bce + self.dice_weight * self.dice(logits, targets)


def compute_foreground_pos_weight(df, column: str = "tumour_pixel_fraction",
                                  split: str = "train", cap: float = 50.0) -> torch.Tensor | None:
    """``(1 - fg) / fg`` from the TRAIN split, for the BCE term of DiceBCELoss.

    Returns None when the fraction column is unavailable.
    """
    if column not in getattr(df, "columns", []):
        print(f"[info] no '{column}' column - BCE term will be unweighted.")
        return None
    subset = df[df["split"] == split] if "split" in df.columns else df
    fg = float(subset[column].mean())
    if not 0 < fg < 1:
        print(f"[warn] foreground fraction is {fg}; skipping pos_weight.")
        return None
    weight = min((1 - fg) / fg, cap)
    print(f"[loss] foreground fraction={fg:.5f} -> BCE pos_weight={weight:.2f} (cap {cap})")
    return torch.tensor([weight], dtype=torch.float32)


def build_seg_loss(cfg, pos_weight: torch.Tensor | None = None, device=None) -> nn.Module:
    """Build the loss described by ``cfg.train.loss``."""
    name = str(cfg.get("train.loss", "dice_bce")).lower()
    smooth = float(cfg.get("train.dice_smooth", 1.0))

    if name in {"dice_bce", "bce_dice", "combo"}:
        bce_weight = float(cfg.get("train.bce_weight", 0.5))
        dice_weight = float(cfg.get("train.dice_weight", 0.5))
        weight = pos_weight.to(device) if (pos_weight is not None and
                                           bool(cfg.get("train.use_pos_weight", False))) else None
        print(f"[loss] DiceBCELoss(bce={bce_weight}, dice={dice_weight}, "
              f"pos_weight={'set' if weight is not None else 'None'})")
        loss = DiceBCELoss(bce_weight, dice_weight, smooth=smooth, pos_weight=weight)
        return loss.to(device) if device is not None else loss

    if name == "dice":
        print("[loss] DiceLoss")
        return DiceLoss(smooth=smooth)

    if name == "bce":
        print("[loss] BCEWithLogitsLoss")
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None)

    if name == "tversky":
        alpha = float(cfg.get("train.tversky_alpha", 0.3))
        beta = float(cfg.get("train.tversky_beta", 0.7))
        print(f"[loss] TverskyLoss(alpha={alpha}, beta={beta})")
        return TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)

    raise ValueError(f"Unknown train.loss={name!r}. Use 'dice_bce', 'dice', 'bce' or 'tversky'.")
