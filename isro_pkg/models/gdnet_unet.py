from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import AGM, AFM, RMAG


class GDNetUNet(nn.Module):
    def __init__(
        self,
        thermal_in_channels: int = 2,
        optical_in_channels: int = 3,
        base_channels: int = 64,
        num_levels: int = 3,
        num_rmag: int = 4,
        scale_factor: int = 2,
    ) -> None:
        super().__init__()
        self.base_channels = base_channels
        self.num_levels = num_levels

        self.agm = AGM(optical_in_channels, base_channels)
        self.afm = AFM(base_channels)
        self.input_conv = nn.Conv2d(
            thermal_in_channels + base_channels, base_channels, 3, padding=1)

        self.encoders = nn.ModuleList()
        ch = base_channels
        for _ in range(num_levels):
            self.encoders.append(
                nn.Sequential(
                    nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1),
                    nn.BatchNorm2d(ch * 2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ch * 2, ch * 2, 3, padding=1),
                    nn.BatchNorm2d(ch * 2),
                    nn.ReLU(inplace=True),
                )
            )
            ch = ch * 2

        self.bottleneck = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True))
        self.rmags = nn.ModuleList(
            [RMAG(channels=ch, guidance_channels=base_channels) for _ in range(num_rmag)])

        self.decoders = nn.ModuleList()
        for _ in range(num_levels):
            self.decoders.append(
                nn.Sequential(
                    nn.ConvTranspose2d(ch, ch // 2, 2, stride=2),
                    nn.BatchNorm2d(ch // 2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ch // 2, ch // 2, 3, padding=1),
                    nn.BatchNorm2d(ch // 2),
                    nn.ReLU(inplace=True),
                )
            )
            ch = ch // 2

        self.final = nn.Sequential(
            nn.Conv2d(ch, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels *
                      (scale_factor**2), 3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, thermal_in_channels, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, thermal_input: torch.Tensor, optical_input: torch.Tensor) -> torch.Tensor:
        fn, fl, ff = self.agm(optical_input)
        fused_guidance = self.afm(fn, fl, ff)
        if fused_guidance.shape[2:] != thermal_input.shape[2:]:
            fused_guidance = F.interpolate(
                fused_guidance, size=thermal_input.shape[2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([thermal_input, fused_guidance], dim=1)
        x = self.input_conv(x)
        for enc in self.encoders:
            x = enc(x)
        x = self.bottleneck(x)
        for rmag in self.rmags:
            x = rmag(x, fused_guidance)
        for dec in self.decoders:
            x = dec(x)
        return self.final(x)
