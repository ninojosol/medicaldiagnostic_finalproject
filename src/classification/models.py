"""Classification models: a from-scratch CNN baseline and transfer-learning backbones.

All models output **raw logits** of shape ``(B, num_labels)``. Sigmoid is applied
in the metric/inference code, never inside the model, because
``BCEWithLogitsLoss`` needs logits for numerical stability.

Having a simple baseline alongside the pretrained model is the point of the
comparison in the report: it shows how much of the performance comes from
transfer learning rather than from the architecture being "deep".
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models as tvm

SUPPORTED_BACKBONES = ["densenet121", "efficientnet_b0", "efficientnet_b3", "resnet18", "resnet50"]


class SimpleCNN(nn.Module):
    """Four-block VGG-style CNN trained from scratch - the baseline.

    Each block is Conv-BN-ReLU x2 then MaxPool, so the spatial size halves four
    times (224 -> 14). Global average pooling replaces a large flatten layer,
    which keeps the parameter count near 1.5M and reduces overfitting on the
    small datasets typical of a course project.
    """

    def __init__(self, num_labels: int, in_channels: int = 3, base_width: int = 32,
                 dropout: float = 0.3):
        super().__init__()
        widths = [base_width, base_width * 2, base_width * 4, base_width * 8]
        blocks: list[nn.Module] = []
        prev = in_channels
        for width in widths:
            blocks.append(self._block(prev, width))
            prev = width
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(prev, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_labels),
        )

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


def build_transfer_model(name: str = "densenet121", num_labels: int = 1, pretrained: bool = True,
                         freeze_backbone: bool = False, dropout: float = 0.2) -> nn.Module:
    """Load a torchvision backbone and replace its classifier head.

    Parameters
    ----------
    name:
        One of :data:`SUPPORTED_BACKBONES`.
    pretrained:
        Download ImageNet weights. Requires internet on first use; weights are
        then cached in ``~/.cache/torch/hub``.
    freeze_backbone:
        Train only the new head first (fast, useful when data is small). Unfreeze
        later with :func:`unfreeze_backbone` for fine-tuning.
    """
    name = name.lower()
    if name not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unknown model.name={name!r}. Supported: {SUPPORTED_BACKBONES}")

    weights = "DEFAULT" if pretrained else None
    try:
        model = getattr(tvm, name)(weights=weights)
    except Exception as exc:
        raise RuntimeError(
            f"Could not create backbone '{name}' with pretrained={pretrained}.\n"
            f"  Underlying error: {exc}\n"
            "  If this is a download failure, connect to the internet once, or set "
            "model.pretrained: false in the config to train from scratch."
        ) from exc

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Swap the head; the new layer is always trainable.
    if name.startswith("densenet"):
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_labels))
    elif name.startswith("efficientnet"):
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_labels))
    elif name.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_labels))
    else:  # pragma: no cover - guarded by SUPPORTED_BACKBONES
        raise ValueError(f"No head-replacement rule for {name}")

    return model


def build_model(cfg, num_labels: int | None = None) -> nn.Module:
    """Build the model described by ``cfg.model``."""
    num_labels = num_labels if num_labels is not None else len(cfg.data.labels)
    name = str(cfg.get("model.name", "densenet121")).lower()
    dropout = float(cfg.get("model.dropout", 0.2))

    if name in {"simple_cnn", "baseline", "simplecnn"}:
        model = SimpleCNN(
            num_labels=num_labels,
            base_width=int(cfg.get("model.base_width", 32)),
            dropout=float(cfg.get("model.dropout", 0.3)),
        )
    else:
        model = build_transfer_model(
            name=name,
            num_labels=num_labels,
            pretrained=bool(cfg.get("model.pretrained", True)),
            freeze_backbone=bool(cfg.get("model.freeze_backbone", False)),
            dropout=dropout,
        )

    total, trainable = count_parameters(model)
    print(f"[model] {name}: {total:,} parameters ({trainable:,} trainable) -> {num_labels} output(s)")
    return model


def unfreeze_backbone(model: nn.Module) -> nn.Module:
    """Make every parameter trainable (stage 2 of a freeze-then-fine-tune schedule)."""
    for param in model.parameters():
        param.requires_grad = True
    total, trainable = count_parameters(model)
    print(f"[model] backbone unfrozen: {trainable:,}/{total:,} parameters now trainable")
    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return ``(total, trainable)`` parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_gradcam_target_layer(model: nn.Module, name: str) -> nn.Module:
    """Return the last convolutional feature block, which is what Grad-CAM hooks.

    Grad-CAM needs the deepest layer that still has spatial structure: earlier
    layers give noisy, low-level maps; after global pooling there is no spatial
    information left.
    """
    name = name.lower()
    if name in {"simple_cnn", "baseline", "simplecnn"}:
        return model.features[-1]
    if name.startswith("densenet"):
        return model.features[-1]           # final BatchNorm of the dense stack
    if name.startswith("efficientnet"):
        return model.features[-1]           # final Conv-BN-SiLU
    if name.startswith("resnet"):
        return model.layer4[-1]
    raise ValueError(f"No Grad-CAM target layer defined for model '{name}'")
