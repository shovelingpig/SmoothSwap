"""
Author : Conor.k
"""

import os
import sys

__file__path = os.path.abspath("")
print(__file__path)
sys.path.append(__file__path)

sys.path.append("./")
# Configuration
from omegaconf import DictConfig, OmegaConf
import hydra
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    For discriminator
    """

    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        self.conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        h = F.leaky_relu(self.conv1(x))
        h = F.leaky_relu(self.conv2(h))

        sk = self.skip(x)

        return (h + sk) / math.sqrt(2)


class DiscriminatorCustom(nn.Module):
    """
    Implement it refered StyleGanV2(It might not exactly equal \
    with the original one, such as fused_lrelu -> relu and etc.)
    """

    def __init__(self, size, channel_multiplier=2):
        super().__init__()
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }
        conv_stem = nn.Conv2d(3, channels[size], kernel_size=1, stride=1, padding=0)

        log_size = int(math.log(size, 2))
        in_channel = channels[size]

        net = [conv_stem]
        for i in range(log_size, 2, -1):
            out_channel = channels[int(2 ** (i - 1))]
            net.append(ResBlock(in_channel, out_channel))
            in_channel = out_channel

        self.extractor = nn.Sequential(*net)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = nn.Conv2d(
            in_channel + 1, channels[4], kernel_size=3, stride=1, padding=1
        )
        self.final_linear = nn.Sequential(
            nn.Linear(channels[4] * 4 * 4, channels[4]),
            nn.LeakyReLU(),
            nn.Linear(channels[4], 1),
        )

    def forward(self, x):
        h = self.extractor(x)

        batch, channel, height, width = h.shape
        group = min(batch, self.stddev_group)
        stddev = h.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )

        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)

        h = torch.cat([h, stddev], dim=1)
        h = self.final_conv(h)
        h = h.view(batch, -1)
        return self.final_linear(h)


class Discriminator(nn.Module):
    """
    Implement it refered StyleGanV2(It might not exactly equal \
    with the original one, such as fused_lrelu -> relu and etc.)
    """

    def __init__(self, size, channel_multiplier=2):
        super().__init__()
        resnet = torch.hub.load(
            "pytorch/vision:v0.10.0",
            "resnet50",
            pretrained=True,
        )
        self.extractor = nn.Sequential(*list(resnet.children())[:-2])

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = nn.Conv2d(
            512 * 4 + 1, 512, kernel_size=3, stride=1, padding=1
        )
        self.final_linear = nn.Sequential(
            nn.Linear(512 * 4 * 4 * 4, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        h = self.extractor(x)

        batch, channel, height, width = h.shape
        group = min(batch, self.stddev_group)
        stddev = h.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )

        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)

        h = torch.cat([h, stddev], dim=1)
        h = self.final_conv(h)
        h = h.view(batch, -1)
        return self.final_linear(h)


config_path = "../configs"
config_name = "config.yaml"


@hydra.main(version_base=None, config_path=config_path, config_name=config_name)
def test_Discriminator(cfg):
    # D = DiscriminatorCustom(cfg.discriminator.image_size)
    D = Discriminator(cfg.discriminator.image_size)
    x = torch.rand([2, 3, 256, 256])
    out = D(x)
    print("prediction: ", out, "\n", out.shape)


if __name__ == "__main__":
    print("#" * 7, " distcriminator Test ", "*" * 7)
    test_Discriminator()
