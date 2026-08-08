"""Loss functions for (multi-)label chest X-ray classification.

Disease labels are almost always heavily imbalanced - a finding may be present in
1-5% of images. An unweighted loss then converges to "predict negative for
everything", which scores high accuracy and is clinically useless. Two remedies
are provided:

``weighted_bce``
    ``BCEWithLogitsLoss(pos_weight=n_neg/n_pos)`` - the standard, well-understood
    choice. Recommended default.
``focal``
    Focal loss down-weights easy examples; useful with extreme imbalance, but it
    has an extra hyper-parameter and is harder to justify in a short report.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossWithLogits(nn.Module):
    """Binary focal loss (Lin et al., 2017) operating on logits.

    ``gamma`` controls how strongly well-classified examples are down-weighted
    (0 reduces to plain BCE); ``alpha`` optionally re-balances positives.
    """

    def __init__(self, gamma: float = 2.0, alpha: float | None = 0.25, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = bce * (1 - p_t).pow(self.gamma)
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(cfg, pos_weight: torch.Tensor | None = None, device=None) -> nn.Module:
    """Build the loss described by ``cfg.train.loss``.

    ``pos_weight`` must be computed on the training split only
    (see :func:`src.classification.dataset.compute_pos_weights`).
    """
    name = str(cfg.get("train.loss", "weighted_bce")).lower()

    if name in {"weighted_bce", "bce", "bce_with_logits"}:
        use_weight = name != "bce" and bool(cfg.get("train.use_class_weights", True))
        weight = pos_weight.to(device) if (use_weight and pos_weight is not None) else None
        if use_weight and weight is None:
            print("[warn] train.use_class_weights is true but no pos_weight was supplied; "
                  "falling back to unweighted BCE.")
        print(f"[loss] BCEWithLogitsLoss(pos_weight={'set' if weight is not None else 'None'})")
        return nn.BCEWithLogitsLoss(pos_weight=weight)

    if name == "focal":
        gamma = float(cfg.get("train.focal_gamma", 2.0))
        alpha = cfg.get("train.focal_alpha", 0.25)
        alpha = None if alpha in (None, "none") else float(alpha)
        print(f"[loss] FocalLossWithLogits(gamma={gamma}, alpha={alpha})")
        return FocalLossWithLogits(gamma=gamma, alpha=alpha)

    raise ValueError(f"Unknown train.loss={name!r}. Use 'weighted_bce', 'bce' or 'focal'.")
