from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 128, num_layers: int = 6) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = in_channels
        for i in range(num_layers):
            current_out = out_channels if i == 0 else out_channels
            layers.append(nn.Conv2d(channels, current_out,
                          kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(current_out))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(current_out, current_out,
                          kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(current_out))
            layers.append(nn.ReLU(inplace=True))
            channels = current_out
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class AGM(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 128) -> None:
        super().__init__()
        self.nc = FeatureExtractor(in_channels, base_channels)
        self.li = FeatureExtractor(in_channels, base_channels)
        self.fo = FeatureExtractor(in_channels, base_channels)

    def forward(self, x: torch.Tensor):
        fn = self.nc(x)
        fl = self.li(x)
        ff = self.fo(x)
        return fn, fl, ff


class AFM(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 3, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 3, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, fn: torch.Tensor, fl: torch.Tensor, ff: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([fn, fl, ff], dim=1)
        weights = self.channel_att(concat)
        w_fn = weights[:, 0:1, :, :].expand_as(fn)
        w_fl = weights[:, 1:2, :, :].expand_as(fl)
        w_ff = weights[:, 2:3, :, :].expand_as(ff)
        return w_fn * fn + w_fl * fl + w_ff * ff


class RMAG(nn.Module):
    def __init__(self, channels: int = 128, guidance_channels: int = 128) -> None:
        super().__init__()
        self.channels = channels
        self.channel_proj = nn.Conv2d(
            guidance_channels, channels, kernel_size=1, bias=False)
        self.block = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.relu = nn.ReLU(inplace=True)
        self.W = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        if guidance.size(2) != x.size(2) or guidance.size(3) != x.size(3):
            guidance = F.interpolate(guidance, size=(
                x.size(2), x.size(3)), mode="bilinear", align_corners=False)
        if guidance.size(1) != self.channels:
            guidance = self.channel_proj(guidance)
        combined = torch.cat([x, guidance], dim=1)
        out = self.block(combined)
        out = out * self.W + x
        return self.relu(out)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels,
                              1) if in_channels != out_channels else None
        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x) if self.proj is not None else x
        out = self.relu(self.conv1(residual))
        out = self.conv2(out)
        return residual + out
