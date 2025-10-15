from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CrossModalAttention
from .gdnet_unet import GDNetUNet


class CrossModalGDNet(nn.Module):
    def __init__(self, thermal_in_channels: int = 2, optical_in_channels: int = 3, base_channels: int = 128, scale_factor: int = 2) -> None:
        super().__init__()
        self.thermal_in_channels = thermal_in_channels
        self.optical_in_channels = optical_in_channels
        self.cross_att_thermal = CrossModalAttention(
            query_channels=thermal_in_channels, key_value_channels=optical_in_channels)
        self.cross_att_optical = CrossModalAttention(
            query_channels=optical_in_channels, key_value_channels=thermal_in_channels)
        self.reduce_thermal = nn.Conv2d(
            thermal_in_channels * 2, thermal_in_channels, 1)
        self.reduce_optical = nn.Conv2d(
            optical_in_channels * 2, optical_in_channels, 1)
        self.gdnet = GDNetUNet(
            thermal_in_channels=thermal_in_channels,
            optical_in_channels=optical_in_channels,
            base_channels=base_channels,
            num_levels=3,
            num_rmag=4,
            scale_factor=scale_factor,
        )

    def forward(self, thermal_lr: torch.Tensor, optical: torch.Tensor) -> torch.Tensor:
        if optical.shape[2:] != thermal_lr.shape[2:]:
            optical = F.interpolate(
                optical, size=thermal_lr.shape[2:], mode="bilinear", align_corners=False)
        thermal_att = self.cross_att_thermal(thermal_lr, optical)
        optical_att = self.cross_att_optical(optical, thermal_lr)
        thermal_enhanced = thermal_lr + self.reduce_thermal(thermal_att)
        optical_enhanced = optical + self.reduce_optical(optical_att)
        return self.gdnet(thermal_enhanced, optical_enhanced)


class SimplifiedCrossModalGDNet(nn.Module):
    def __init__(self, thermal_in_channels: int = 2, optical_in_channels: int = 3, base_channels: int = 128, scale_factor: int = 2) -> None:
        super().__init__()
        self.input_channels = thermal_in_channels + optical_in_channels
        self.base_channels = base_channels
        self.scale_factor = scale_factor
        self.cross_modal_att = CrossModalAttention(self.input_channels)
        self.decoder = nn.Sequential(
            nn.Conv2d(self.input_channels * 2, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, thermal_in_channels *
                      (scale_factor**2), 3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.Tanh(),
        )

    def forward(self, thermal_lr: torch.Tensor, optical: torch.Tensor) -> torch.Tensor:
        x = torch.cat([thermal_lr, optical], dim=1)
        fused = self.cross_modal_att(x, x)
        return self.decoder(fused)
