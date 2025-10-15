from __future__ import annotations

import torch
import torch.nn as nn


class CrossModalAttention(nn.Module):
    def __init__(self, query_channels: int, key_value_channels: int | None = None) -> None:
        super().__init__()
        if key_value_channels is None:
            key_value_channels = query_channels
        embed_dim = max((query_channels // 2) * 2,
                        (key_value_channels // 2) * 2)
        self.embed_dim = embed_dim
        self.theta = nn.Conv2d(query_channels, embed_dim,
                               kernel_size=1, bias=False)
        self.phi = nn.Conv2d(key_value_channels, embed_dim,
                             kernel_size=1, bias=False)
        self.g = nn.Conv2d(key_value_channels, embed_dim,
                           kernel_size=1, bias=False)
        self.out_conv = nn.Conv2d(
            embed_dim, query_channels, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, C: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        bsz, _, h, w = C.size()
        theta_C = self.theta(C).reshape(
            bsz, self.embed_dim, -1).permute(0, 2, 1)
        phi_P = self.phi(P).reshape(bsz, self.embed_dim, -1)
        g_P = self.g(P).reshape(bsz, self.embed_dim, -1)
        attn = torch.bmm(theta_C, phi_P) / (self.embed_dim ** 0.5)
        attn = self.softmax(attn)
        out = torch.bmm(g_P, attn.permute(0, 2, 1)).view(
            bsz, self.embed_dim, h, w)
        out = self.out_conv(out)
        return torch.cat([C, out], dim=1)
