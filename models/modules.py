"""
Author : Conor.k
"""
import os
import sys

__file__path = os.path.abspath("")
print(__file__path)
sys.path.append(__file__path)

import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# GroupNorm
# $$y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta$$
# class Normalize(nn.Module):
#     def __init__(self, in_channels, num_groups=32):
#         super().__init__()
#         self.norm = torch.nn.GroupNorm(
#             num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
#         ).to(DEVICE)

#     def forward(self, x):
#         return self.norm(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    ).to(DEVICE)


class Combine(nn.Module):
    def __init__(self, method="cat"):
        super().__init__()
        self.method = method

    def forward(self, x, y):
        frac = 1
        if x.size()[1] != y.size()[1]:
            self.conv = torch.nn.Conv2d(
                y.size()[1], x.size()[1], kernel_size=1, stride=1, padding=0
            ).to(DEVICE)
        else:
            self.conv = None

        if x.size()[-1] > y.size()[-1]:
            frac = x.size()[-1] // y.size()[-1]
            change_grid = Upsample(y.size()[1])

        elif x.size()[-1] < y.size()[-1]:
            frac = y.size()[-1] // x.size()[-1]
            change_grid = Downsample(y.size()[1])

        if y.size()[1:] == x.size()[1:]:
            return torch.cat([x, y], dim=1)

        else:
            if frac == 1:
                _y = y
            elif frac == 2:
                _y = change_grid(y)
            else:
                _y = change_grid(change_grid(y))

            if self.conv is not None:
                _y = self.conv(_y)

            return torch.cat([x, _y], dim=1)


"""NEED REFACTORING"""


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv=False):
        super().__init__()

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv=False):
        super().__init__()

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x
