"""2D slice-wise U-Net for MRI whole-tumour segmentation.

Four input channels (FLAIR, T1w, T1gd, T2w), one binary output channel
(whole tumour vs background). Every convolution is 2D and every sample is a
single axial slice — this is **not** a 3D U-Net and must never be described as one.

Architecture: classic Ronneberger U-Net with BatchNorm, four downsampling
stages, transposed-convolution upsampling, and skip connections. The head emits
raw logits; sigmoid is applied only in metrics/inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(3x3 conv -> BN -> ReLU) x 2."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.insert(3, nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MRIUNet2D(nn.Module):
    """2D slice-wise U-Net using four MRI sequences."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        features: tuple[int, ...] = (32, 64, 128, 256),
        bottleneck: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = tuple(features)

        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f

        bottleneck = bottleneck or features[-1] * 2
        self.bottleneck = DoubleConv(features[-1], bottleneck, dropout=dropout)

        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        ch = bottleneck
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(ch, f, kernel_size=2, stride=2))
            self.up_convs.append(DoubleConv(f * 2, f))
            ch = f

        self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up, conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                # Only triggers for input sizes not divisible by 16.
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = conv(torch.cat([skip, x], dim=1))

        return self.head(x)  # raw logits


def build_mri_unet(
    *,
    in_channels: int = 4,
    out_channels: int = 1,
    features: tuple[int, ...] | list[int] = (32, 64, 128, 256),
    dropout: float = 0.0,
) -> MRIUNet2D:
    return MRIUNet2D(
        in_channels=in_channels,
        out_channels=out_channels,
        features=tuple(features),
        dropout=dropout,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
