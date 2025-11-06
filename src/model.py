# src/model.py
"""
UNet3D (volumetric regression)

- Fully parameterized input/output channels: UNet3D(in_ch=..., out_ch=...)
- Final skip connects decoder to the original input with 'in_ch' channels.
- Kaiming init for Conv3d; BatchNorm3d(γ=1, β=0); Identity head for regression.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlockEnc(nn.Module):
    """Encoder: ReplicationPad3d(2) → Conv3d(k=5,s=2) → BN → ReLU."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, stride: int = 2):
        super().__init__()
        assert kernel_size == 5, "This block assumes k=5 (pad=2)."
        self.pad = nn.ReplicationPad3d(2)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=0, bias=True)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class ConvBlockDec(nn.Module):
    """Decoder: Upsample(x2) → concat(skip) → RepPad3d(1) → Conv3d(k=3) → BN → ReLU."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        assert kernel_size == 3, "This block assumes k=3 (pad=1)."
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.pad = nn.ReplicationPad3d(1)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=1, padding=0, bias=True)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        # Auto spatial align (handles off-by-one after upsampling)
        if x.shape[2:] != skip.shape[2:]:
            dz, dy, dx = (s - t for s, t in zip(skip.shape[2:], x.shape[2:]))
            x = F.pad(x, [dx // 2, dx - dx // 2,
                          dy // 2, dy - dy // 2,
                          dz // 2, dz - dz // 2])

        x = torch.cat([x, skip], dim=1)
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class UNet3D(nn.Module):
    """
    3D U-Net for voxel-wise regression.

    Args
    ----
    in_ch  : int
        Number of input channels (e.g., 1 for single input, 2 for [ngal, vpec]).
    out_ch : int
        Number of output channels (typically 1 for scalar fields like ρ or φ).

    Input shape
    -----------
    (B, in_ch, D, H, W)

    Output shape
    ------------
    (B, out_ch, D, H, W)
    """
    def __init__(self, in_ch: int = 2, out_ch: int = 1):
        super().__init__()
        if in_ch < 1:
            raise ValueError(f"in_ch must be >= 1, got {in_ch}")
        if out_ch < 1:
            raise ValueError(f"out_ch must be >= 1, got {out_ch}")

        # Encoder
        self.enc1 = ConvBlockEnc(in_ch,   32)   # (D,H,W) -> /2
        self.enc2 = ConvBlockEnc(32,      64)   # -> /4
        self.enc3 = ConvBlockEnc(64,      128)  # -> /8
        self.enc4 = ConvBlockEnc(128,     256)  # -> /16
        self.enc5 = ConvBlockEnc(256,     512)  # -> /32 (bottleneck in)

        # Decoder (concat with encoder skips)
        self.dec4 = ConvBlockDec(512 + 256, 256)
        self.dec3 = ConvBlockDec(256 + 128, 128)
        self.dec2 = ConvBlockDec(128 + 64,   64)
        self.dec1 = ConvBlockDec(64  + 32,   32)

        # Final: concat with original input (skip from very first stage)
        self.out  = ConvBlockDec(32 + in_ch, out_ch)

        # Identity head (linear) for regression
        self.final_activation = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect x: (B, in_ch, D, H, W)
        if x.ndim != 5:
            raise ValueError(f"Expected 5D input (B,C,D,H,W), got shape {tuple(x.shape)}")

        x1 = x
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)
        x6 = self.enc5(x5)

        d5 = self.dec4(x6, x5)
        d4 = self.dec3(d5, x4)
        d3 = self.dec2(d4, x3)
        d2 = self.dec1(d3, x2)
        d1 = self.out(d2, x1)

        return self.final_activation(d1)
