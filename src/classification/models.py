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

SUPPORTED_BACKBONES = [
    "densenet121", "efficientnet_b0", "efficientnet_b3", "resnet18", "resnet50",
    # Vision Transformers. Unlike the CNNs above these have a FIXED input
    # resolution (224 for vit_b_16/vit_b_32): the learned positional embeddings
    # are tied to the 14x14 / 7x7 patch grid, so data.image_size must be 224.
    "vit_b_16", "vit_b_32",
]


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
    elif name.startswith("vit_"):
        # torchvision ViT keeps its classifier in `heads`, an OrderedDict Sequential
        # whose only entry is `head`. Replacing the whole `heads` module (rather than
        # `heads.head`) keeps the Dropout/Linear pattern identical to the CNNs above,
        # and `forward` calls `self.heads(x)` so the swap is transparent.
        in_features = model.heads.head.in_features
        model.heads = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_labels))
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


def unfreeze_efficientnet_backbone_blocks(model: nn.Module,
                                          unfreeze_last_n_blocks: int = 2) -> nn.Module:
    """Selectively unfreeze the last N blocks of a torchvision EfficientNet backbone.

    This is stage 2 of the conservative fine-tuning schedule used by
    ``run_xray_finetune_efficientnet_b0.py``: stage 1 trains the new head with the
    whole backbone frozen, then stage 2 additionally re-opens only the deepest
    backbone blocks. Early blocks keep their ImageNet edge/texture filters, which
    is what makes fine-tuning viable on a few hundred training images.

    ``model.features`` of ``efficientnet_b0`` is a ``Sequential`` of 9 entries:
    ``features[0]`` is the stem convolution, ``features[1:8]`` are the MBConv
    stages, and ``features[8]`` is the final Conv-BN-SiLU. "Last N blocks"
    therefore means the last N entries of that Sequential, i.e. ``features[7]``
    and ``features[8]`` for the default ``unfreeze_last_n_blocks=2``.

    The classifier head is left untouched: :func:`build_transfer_model` replaces it
    *after* freezing, so its parameters are already trainable.

    Note
    ----
    Freezing controls gradients only. BatchNorm running statistics in the frozen
    blocks still update while the model is in ``train()`` mode - this matches the
    behaviour of the saved runs and is reflected in their checkpoints.
    """
    features = getattr(model, "features", None)
    if features is None or not hasattr(features, "__getitem__") or not hasattr(features, "__len__"):
        raise ValueError(
            "unfreeze_efficientnet_backbone_blocks expects a torchvision EfficientNet "
            f"with a 'features' Sequential; got {type(model).__name__}."
        )

    n_blocks = len(features)
    n = int(unfreeze_last_n_blocks)
    if n < 0:
        raise ValueError(f"unfreeze_last_n_blocks must be >= 0, got {unfreeze_last_n_blocks}")
    n = min(n, n_blocks)

    unfrozen_indices = list(range(n_blocks - n, n_blocks)) if n else []
    for index in unfrozen_indices:
        for param in features[index].parameters():
            param.requires_grad = True

    total, trainable = count_parameters(model)
    pretty = ", ".join(f"features[{i}]" for i in unfrozen_indices) or "none"
    print(f"[model] efficientnet: unfroze last {n} backbone block(s) -> {pretty} "
          f"(of {n_blocks}); {trainable:,}/{total:,} parameters now trainable")
    return model


def unfreeze_vit_encoder_blocks(model: nn.Module, unfreeze_last_n_blocks: int = 2,
                                unfreeze_final_norm: bool = True) -> nn.Module:
    """Selectively unfreeze the last N transformer blocks of a torchvision ViT.

    Stage 2 of the conservative schedule used by ``scripts/run_xray_finetune_vit.py``,
    and the direct analogue of :func:`unfreeze_efficientnet_backbone_blocks`: stage 1
    trains the new 14-output head with the whole backbone frozen, stage 2 additionally
    re-opens only the deepest encoder blocks. The patch embedding, class token,
    positional embeddings and the early blocks keep their ImageNet weights, which is
    what makes fine-tuning an 86M-parameter model viable on a few hundred X-rays.

    ``model.encoder.layers`` is a ``Sequential`` of 12 ``EncoderBlock`` modules for
    ``vit_b_16``/``vit_b_32``, so "last N blocks" means ``layers[12-N:]``.

    ``unfreeze_final_norm`` also re-opens ``model.encoder.ln``, the LayerNorm applied
    to the encoder output before the head. It is 2 x hidden_dim parameters and sits
    directly in front of the newly initialised head, so leaving it frozen would force
    the head to absorb a shift it cannot represent.

    The classifier head is left untouched: :func:`build_transfer_model` replaces it
    *after* freezing, so its parameters are already trainable.

    Note
    ----
    Freezing controls gradients only. ViT uses LayerNorm, which has no running
    statistics, so unlike the BatchNorm backbones nothing in the frozen blocks
    changes during ``train()`` mode.
    """
    encoder = getattr(model, "encoder", None)
    layers = getattr(encoder, "layers", None)
    if layers is None or not hasattr(layers, "__getitem__") or not hasattr(layers, "__len__"):
        raise ValueError(
            "unfreeze_vit_encoder_blocks expects a torchvision VisionTransformer with an "
            f"'encoder.layers' Sequential; got {type(model).__name__}."
        )

    n_blocks = len(layers)
    n = int(unfreeze_last_n_blocks)
    if n < 0:
        raise ValueError(f"unfreeze_last_n_blocks must be >= 0, got {unfreeze_last_n_blocks}")
    n = min(n, n_blocks)

    unfrozen_indices = list(range(n_blocks - n, n_blocks)) if n else []
    for index in unfrozen_indices:
        for param in layers[index].parameters():
            param.requires_grad = True

    if unfreeze_final_norm and hasattr(encoder, "ln"):
        for param in encoder.ln.parameters():
            param.requires_grad = True

    total, trainable = count_parameters(model)
    pretty = ", ".join(f"encoder.layers[{i}]" for i in unfrozen_indices) or "none"
    if unfreeze_final_norm and hasattr(encoder, "ln"):
        pretty += " + encoder.ln"
    print(f"[model] vit: unfroze last {n} encoder block(s) -> {pretty} "
          f"(of {n_blocks}); {trainable:,}/{total:,} parameters now trainable")
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
    if name.startswith("vit_"):
        raise ValueError(
            f"Grad-CAM is not defined for '{name}'. A Vision Transformer has no final "
            "convolutional feature map: its encoder emits a (B, 1+N_patches, hidden) token "
            "sequence, so the CNN Grad-CAM hook in src.classification.gradcam does not apply. "
            "Use an attention-rollout or transformer-specific CAM instead, and do not "
            "reuse another model's Grad-CAM panel for it."
        )
    raise ValueError(f"No Grad-CAM target layer defined for model '{name}'")
