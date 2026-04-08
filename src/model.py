# src/model.py
"""
UNet3D (volumetric regression)

- Fully parameterized input/output channels: UNet3D(in_ch=..., out_ch=...)
- Final skip connects decoder to the original input with 'in_ch' channels.
- Hidden decoder blocks use Conv + BN + ReLU.
- Final output head is linear: Conv only + Identity (no ReLU, no BN).
- Kaiming init for Conv3d; BatchNorm3d(γ=1, β=0).

Reference:
    Inspired by the V-Net and U-Net architectures for 3D medical image segmentation
    and adapted for scientific data modeling.
    Adapted from the TensorFlow architecture script:
    https://github.com/redeostm/ML_LocalEnv/blob/main/generatorSingle.py
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlockEnc(nn.Module):
    """Encoder: ReplicationPad3d(2) -> Conv3d(k=5,s=2) -> BN -> ReLU."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, stride: int = 2):
        super().__init__()
        assert kernel_size == 5, "This block assumes k=5 (pad=2)."
        self.pad = nn.ReplicationPad3d(2)
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=True,
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class ConvBlockDec(nn.Module):
    """Decoder hidden block: Upsample(x2) -> concat(skip) -> RepPad3d(1) -> Conv3d(k=3) -> BN -> ReLU."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        assert kernel_size == 3, "This block assumes k=3 (pad=1)."
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.pad = nn.ReplicationPad3d(1)
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            bias=True,
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        # Auto spatial align (handles off-by-one after upsampling)
        if x.shape[2:] != skip.shape[2:]:
            dz, dy, dx = (s - t for s, t in zip(skip.shape[2:], x.shape[2:]))
            x = F.pad(
                x,
                [
                    dx // 2, dx - dx // 2,
                    dy // 2, dy - dy // 2,
                    dz // 2, dz - dz // 2,
                ],
            )

        x = torch.cat([x, skip], dim=1)
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class FinalHead(nn.Module):
    """
    Final linear output head:
    Upsample(x2) -> concat(skip=input) -> RepPad3d(1) -> Conv3d(k=3) -> Identity

    No BN, no ReLU.
    This allows negative outputs for normalized regression targets.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        assert kernel_size == 3, "This block assumes k=3 (pad=1)."
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.pad = nn.ReplicationPad3d(1)
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            bias=True,
        )
        self.final_activation = nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        if x.shape[2:] != skip.shape[2:]:
            dz, dy, dx = (s - t for s, t in zip(skip.shape[2:], x.shape[2:]))
            x = F.pad(
                x,
                [
                    dx // 2, dx - dx // 2,
                    dy // 2, dy - dy // 2,
                    dz // 2, dz - dz // 2,
                ],
            )

        x = torch.cat([x, skip], dim=1)
        x = self.pad(x)
        x = self.conv(x)
        return self.final_activation(x)


class UNet3D(nn.Module):
    """
    3D U-Net for voxel-wise regression.

    Args
    ----
    in_ch : int
        Number of input channels (e.g., 1 for single input, 2 for [ngal, vpec]).
    out_ch : int
        Number of output channels (typically 1 for scalar fields like rho or phi).

    Input shape
    -----------
    (B, in_ch, D, H, W)

    Output shape
    ------------
    (B, out_ch, D, H, W)
    """
    def __init__(self, in_ch: int = 2, out_ch: int = 1, BASE: int = 32):
        super().__init__()
        if in_ch < 1:
            raise ValueError(f"in_ch must be >= 1, got {in_ch}")
        if out_ch < 1:
            raise ValueError(f"out_ch must be >= 1, got {out_ch}")

        # Encoder
        self.enc1 = ConvBlockEnc(in_ch,   BASE)
        self.enc2 = ConvBlockEnc(BASE,    BASE * 2)
        self.enc3 = ConvBlockEnc(BASE * 2, BASE * 4)
        self.enc4 = ConvBlockEnc(BASE * 4, BASE * 8)
        self.enc5 = ConvBlockEnc(BASE * 8, BASE * 16)

        # Decoder hidden blocks (with ReLU)
        self.dec4 = ConvBlockDec(BASE * 16 + BASE * 8, BASE * 8)
        self.dec3 = ConvBlockDec(BASE * 8 + BASE * 4,  BASE * 4)
        self.dec2 = ConvBlockDec(BASE * 4 + BASE * 2,  BASE * 2)
        self.dec1 = ConvBlockDec(BASE * 2 + BASE,      BASE)

        # Final linear head (no BN / no ReLU)
        self.out_head = FinalHead(BASE + in_ch, out_ch)

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

        out = self.out_head(d2, x1)
        return out