from math import ceil

import torch
from torch import nn


class FFTTransformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.fft.rfft(x, dim=-1)


class ComplexDotProduct(nn.Module):
    def __init__(self, n_channels: int, n_samp: int, n_out: int):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(1, n_out, n_channels, n_samp, dtype=torch.cfloat)
        )
        self.bias = nn.Parameter(torch.randn(1, n_out, 1, dtype=torch.cfloat))

    def forward(self, x):

        out_channel_acts = [
            torch.sum(w_ * x, dim=1) for w_ in torch.unbind(self.weight, 1)
        ]
        return torch.stack(out_channel_acts, dim=1) + self.bias


class CustomNorm(nn.Module):
    def __init__(self, n_channels: int, affine: bool = True):
        super().__init__()
        self.affine = affine
        self.scale = nn.Parameter(torch.ones(1, n_channels, 1, dtype=torch.cfloat))
        self.offset = nn.Parameter(torch.zeros(1, n_channels, 1, dtype=torch.cfloat))

    def forward(self, x):

        z_scored = (x - torch.mean(x, dim=[0, -1], keepdim=True)) / torch.std(
            x, dim=[0, -1], keepdim=True
        )

        if self.affine:
            return z_scored * self.scale + self.offset
        return z_scored


class ComplexFFTNet(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_hidden: int,
        n_out: int = 1,
        depth: int = 6,
    ):
        super().__init__()

        self.fft_transform = FFTTransformer()

        layers = []
        layers.extend(  # First group:
            [
                CustomNorm(n_channels),
                ComplexDotProduct(n_channels, ceil(n_samples / 2 + 1), n_hidden),
            ]
        )

        for _ in range(depth):
            layers.extend(
                [
                    CustomNorm(n_hidden),
                    ComplexDotProduct(n_hidden, ceil(n_samples / 2 + 1), n_hidden),
                ]
            )

        layers.append(nn.AdaptiveAvgPool1d(1))

        self.body = nn.Sequential(*layers)

        self.cls = nn.Sequential(nn.Linear(n_hidden * 2, n_out))

    def forward(self, x):

        x_ = self.fft_transform(x)
        x_ = self.body(x_).squeeze()

        mag, angle = torch.abs(x_), torch.angle(x_)

        return self.cls(torch.concat((mag, angle), 1))
