"""U-Net for binary brain-tumour segmentation (Ronneberger et al., 2015).

Architecture
------------
A symmetric encoder-decoder. The encoder halves the spatial size and doubles the
channel count four times; the decoder mirrors it with transposed convolutions.
**Skip connections** concatenate each encoder feature map into the matching
decoder stage - this is what lets the network combine "what" (deep semantic
features) with "where" (high-resolution boundary detail), and it is why U-Net
still outperforms plain encoder-decoders on small medical datasets.

The model outputs raw logits of shape ``(B,1,H,W)``. Apply sigmoid in the
metric/inference code, not in ``forward`` - the losses need logits.

``ResNetUNet`` offers an ImageNet-pretrained encoder as a stronger comparison
point, useful when the course dataset is small.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv 3x3 -> BatchNorm -> ReLU) x2, the basic U-Net building block."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.insert(3, nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Classic U-Net.

    Parameters
    ----------
    in_channels:
        3 when slices are loaded as RGB (the usual case for the LGG dataset,
        whose TIFFs carry three MRI sequences in the colour channels), 1 for
        true grayscale.
    out_channels:
        1 for binary tumour/background segmentation.
    features:
        Channel widths per encoder level. Halve them if you run out of GPU memory.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1,
                 features: tuple[int, ...] = (64, 128, 256, 512), dropout: float = 0.0):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        prev = in_channels
        for width in features:
            self.encoders.append(DoubleConv(prev, width, dropout=dropout))
            prev = width

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, dropout=dropout)

        for width in reversed(features):
            self.upsamples.append(nn.ConvTranspose2d(width * 2, width, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(width * 2, width, dropout=dropout))

        self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for upsample, decoder, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = upsample(x)
            if x.shape[-2:] != skip.shape[-2:]:
                # Guard against odd input sizes losing a pixel through pool/upsample.
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat([skip, x], dim=1))

        return self.head(x)


class ResNetUNet(nn.Module):
    """U-Net with an ImageNet-pretrained ResNet-34 encoder.

    Transfer learning usually converges faster and scores higher than the
    from-scratch U-Net when the dataset is small - a worthwhile ablation for the
    report. Requires ``in_channels=3``.
    """

    def __init__(self, out_channels: int = 1, pretrained: bool = True, dropout: float = 0.0):
        super().__init__()
        from torchvision import models as tvm

        backbone = tvm.resnet34(weights="DEFAULT" if pretrained else None)
        self.input_block = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # /2, 64ch
        self.pool = backbone.maxpool                                                    # /4
        self.enc1, self.enc2 = backbone.layer1, backbone.layer2                         # 64ch /4, 128ch /8
        self.enc3, self.enc4 = backbone.layer3, backbone.layer4                         # 256ch /16, 512ch /32

        self.up4 = self._up(512, 256)
        self.dec4 = DoubleConv(256 + 256, 256, dropout)
        self.up3 = self._up(256, 128)
        self.dec3 = DoubleConv(128 + 128, 128, dropout)
        self.up2 = self._up(128, 64)
        self.dec2 = DoubleConv(64 + 64, 64, dropout)
        self.up1 = self._up(64, 64)
        self.dec1 = DoubleConv(64 + 64, 64, dropout)
        self.up0 = self._up(64, 32)
        self.dec0 = DoubleConv(32, 32, dropout)
        self.head = nn.Conv2d(32, out_channels, kernel_size=1)

    @staticmethod
    def _up(in_ch: int, out_ch: int) -> nn.ConvTranspose2d:
        return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    @staticmethod
    def _cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([skip, x], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.input_block(x)          # /2
        e1 = self.enc1(self.pool(s0))     # /4
        e2 = self.enc2(e1)                # /8
        e3 = self.enc3(e2)                # /16
        e4 = self.enc4(e3)                # /32

        d4 = self.dec4(self._cat(self.up4(e4), e3))
        d3 = self.dec3(self._cat(self.up3(d4), e2))
        d2 = self.dec2(self._cat(self.up2(d3), e1))
        d1 = self.dec1(self._cat(self.up1(d2), s0))
        d0 = self.dec0(self.up0(d1))
        return self.head(d0)


def build_segmentation_model(cfg) -> nn.Module:
    """Build the model described by ``cfg.model``."""
    name = str(cfg.get("model.name", "unet")).lower()
    in_channels = int(cfg.get("model.in_channels", 3))
    out_channels = int(cfg.get("model.out_channels", 1))
    dropout = float(cfg.get("model.dropout", 0.0))

    if name in {"unet", "u-net"}:
        features = tuple(cfg.get("model.features", (64, 128, 256, 512)))
        model = UNet(in_channels, out_channels, features=features, dropout=dropout)
    elif name in {"resnet_unet", "resnet34_unet", "unet_resnet34"}:
        if in_channels != 3:
            raise ValueError("resnet_unet requires model.in_channels: 3 (ImageNet encoder).")
        model = ResNetUNet(out_channels, pretrained=bool(cfg.get("model.pretrained", True)),
                           dropout=dropout)
    else:
        raise ValueError(f"Unknown model.name={name!r}. Use 'unet' or 'resnet_unet'.")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {name}: {total:,} parameters ({trainable:,} trainable), "
          f"in={in_channels} out={out_channels}")
    return model
