from __future__ import annotations



from dataclasses import dataclass
from typing import List, Optional, Tuple

from torch.utils.data import Dataset, DataLoader

import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary

# ---------------------------------------
# Utils
# ---------------------------------------

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    backends.cudnn.deterministic = True
    backends.cudnn.benchmark = False


def _choose_groupnorm_groups(num_channels: int, max_groups: int) -> int:
    # num_ch % g == 0
    # fallback to 1

    g = min(max_groups, num_channels)
    while g > 1 and (num_channels % g) != 0:
        g -= 1
    return g


# ---------------------------------------
# SyntheticBlobs 2D : Datasets
# ---------------------------------------
class SyntheticBlobs2D(Dataset):

    def __init__(
            self,
            n_samples: int = 1024,
            size: Tuple[int, int] = (64, 64),
            n_blobs_range: Tuple[int, int] = (1, 4),
            radius_range: Tuple[float, float] = (6.0, 14.0),
            noise_std: float = 0.25,
            seed: int = 12,
    ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.size = size
        self.n_blobs_range = n_blobs_range
        self.radius_range = radius_range
        self.noise_std = noise_std
        self.rng = random.Random(seed)

        self.torch_gen = torch.Generator().manual_seed(seed)

        # normalized coordinate grid (D, L, W, 3)
        L, W = size
        ys = torch.linspace(-1.0, 1.0, L)
        xs = torch.linspace(-1.0, 1.0, W)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        self.grid = torch.stack([yy, xx], dim=-1)  # (L, W, 3)

    def __len__(self) -> int:
        return self.n_samples

    def _rand_uniform(self, a: float, b: float) -> float:
        return a + (b - a) * self.rng.random()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        L, W = self.size

        # empty mask
        mask = torch.zeros((L, W), dtype=torch.float32)

        # add a new spherical blobs
        n_blobs = self.rng.randint(self.n_blobs_range[0], self.n_blobs_range[1])

        for _ in range(n_blobs):
            cy = self._rand_uniform(-.6, .6)
            cx = self._rand_uniform(-.6, .6)
            radius = self._rand_uniform(self.radius_range[0], self.radius_range[1])

            # normalized scale ~ 2 / size_axis
            avg_axis = (L + W) / 3.0
            r_norm = radius * (2.0 / avg_axis)

            center = torch.tensor([cy, cx], dtype=torch.float32)
            dist = torch.norm(self.grid - center, dim=-1)  # (L, W)
            blob = (dist <= r_norm).float()
            mask = torch.maximum(mask, blob)

        # build intensity image correlated with the mask
        # brighter foreground and noise added
        img = .15 * torch.randn((L, W), generator=self.torch_gen)
        img = img + 1.0 * mask
        img = img + self.noise_std * torch.randn((L, W), generator=self.torch_gen)

        # add channel dim for input/output : (C, L, W)
        img = img.unsqueeze(0)  # (1, L, W)
        mask = mask.unsqueeze(0)  # (1, L, W)

        return img, mask


class ConvNormAct2d(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            padding: int = 1,
            groups: int = 8,
            dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        g = _choose_groupnorm_groups(out_channels, groups)
        self.norm = nn.GroupNorm(num_groups=g, num_channels=out_channels)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)

        return x


class ResNetBlock2D(nn.Module):
    """
    two ConvNormAct2d blocks
    residual skip with 1x1 conv if channels changes
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0, groups: int = 8) -> None:
        super().__init__()
        self.block1 = ConvNormAct2d(in_channels, out_channels, dropout=dropout, groups=groups)
        self.block2 = ConvNormAct2d(out_channels, out_channels, dropout=dropout, groups=groups)
        self.skip = (nn.Conv2d(in_channels, out_channels, kernel_size=1,
                              bias=False) if in_channels != out_channels else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l = self.block1(x)
        l = self.block2(l)

        return l + self.skip(x)


class Downsample2D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.down = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class Upsample2D(nn.Module):
    """
    Bilinear upsample > ConvNormAct2d
    """

    def __init__(self, in_channels: int, out_channels: int, scale: int = 2, dropout: float = 0.0, groups: int = 8):
        super().__init__()
        self.scale = scale
        self.post = ConvNormAct2d(in_channels, out_channels, dropout=dropout, groups=groups)

    def forward(self, x: torch.Tensor, size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        if size is None:
            x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        else:
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        x = self.post(x)

        return x
    
# ---------------------------------------
# UNet2D
# ---------------------------------------

class UNet2D(nn.Module):
    """
    Lightweight 2D Unet with ResNet blocks and GroupNorm
    Input: [B, C, L, W]
    output: [B, out_channels, L, W]
    """

    def __init__(
            self,
            in_channels: int = 1,
            base_channels: int = 32,
            num_levels: int = 4,
            dropout: float = 0.0,
            groups: int = 8,
            out_channels: int = 1,
    ) -> None:
        super().__init__()
        assert num_levels >= 2, f'num_levels should be >= 2, got {num_levels}'

        channels = [base_channels * (2 ** i) for i in range(num_levels)]

        #
        self.in_conv = ResNetBlock2D(in_channels, channels[0], dropout=dropout, groups=groups)
        self.downs = nn.ModuleList()
        self.encs = nn.ModuleList()

        for i in range(1, num_levels):
            self.downs.append(Downsample2D(channels[i - 1]))
            self.encs.append(ResNetBlock2D(channels[i - 1], channels[i], dropout=dropout, groups=groups))

        self.bottleneck = ResNetBlock2D(channels[-1], channels[-1], dropout=dropout, groups=groups)

        # decoder
        self.ups = nn.ModuleList()
        self.decs = nn.ModuleList()
        for i in reversed(range(1, num_levels)):
            self.ups.append(Upsample2D(channels[i], channels[i - 1], dropout=dropout, groups=groups))
            self.decs.append(ResNetBlock2D(channels[i - 1] * 2, channels[i - 1], dropout=dropout, groups=groups))

        self.out = nn.Conv2d(channels[0], out_channels, kernel_size=1) # Changed from nn.Conv3d to nn.Conv2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        l = self.in_conv(x)
        skips.append(l)

        for down, enc in zip(self.downs, self.encs):
            l = down(l)
            l = enc(l)
            skips.append(l)

        l = self.bottleneck(l)

        # decode : pop skips from end (the last = the deepest skip)
        skips.pop()
        for up, dec in zip(self.ups, self.decs):
            skip = skips.pop()
            l = up(l, size=skip.shape[-2:])
            l = torch.cat([l, skip], dim=1)
            l = dec(l)

        return self.out(l)
    
def model_sanity_check(model, x):
    m = model(in_channels=1, out_channels=1, base_channels=16, num_levels=4).to('cuda').eval()
    x = x.to('cuda')

    print("x imported:", x.shape, x.ndim)


    summary(m, input_size=x.shape, device='cuda')