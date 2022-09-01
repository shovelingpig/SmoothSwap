"""
Author : Conor.k
"""

import os
import sys

__file__path = os.path.abspath("")
print(__file__path)
sys.path.append(__file__path)

# Configuration
from omegaconf import DictConfig, OmegaConf
import hydra

import math
import torch
import torch.nn as nn

from models.modules import Combine, Upsample, Downsample

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Normalize(nn.Module):
    def __init__(self, in_channels, num_groups=32):
        super().__init__()
        self.norm = torch.nn.GroupNorm(
            num_groups=num_groups,
            num_channels=in_channels,
            eps=torch.tensor(1e-6).to(DEVICE),
            affine=True,
        )

    def forward(self, x):
        return self.norm(x)


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        skip_connection=None,  # one of ["give", "take"]
        up_down=None,  # one of ["up", "down", None]
        dropout=0.1,
        temb_channels=512,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.ch_scale = 1

        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.use_conv_shortcut = conv_shortcut
        self.skip_connection = skip_connection
        self.up_down = up_down

        if self.skip_connection is not None:
            if self.skip_connection == "take":
                self.ch_scale = 2
                self.take_skip_connection = Combine(method="cat").to(DEVICE)

        self.norm1 = Normalize(in_channels * self.ch_scale).to(DEVICE)
        if self.up_down is not None:
            if self.up_down.lower() == "up":
                self.change_grid = Upsample(in_channels * self.ch_scale).to(DEVICE)
            else:
                self.change_grid = Downsample(in_channels * self.ch_scale).to(DEVICE)

        self.conv1 = torch.nn.Conv2d(
            in_channels * self.ch_scale,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        ).to(DEVICE)

        # for embedding -> after this, broadcast has to be done to add it to Conv output in ResBlock
        self.temb_proj = torch.nn.Linear(temb_channels, out_channels).to(DEVICE)

        self.norm2 = Normalize(out_channels).to(DEVICE)
        self.dropout = torch.nn.Dropout(dropout).to(DEVICE)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        ).to(DEVICE)

        self.use_shortcut = in_channels != out_channels
        if self.use_shortcut:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                ).to(DEVICE)
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                ).to(DEVICE)

    def forward(self, x, temb, skip=None):

        h = x
        _x = x
        if self.skip_connection is not None:
            if self.skip_connection == "take":
                h = self.take_skip_connection(h, skip)

        h = self.norm1(h)
        h = nonlinearity(h)

        if self.up_down is not None:
            h = self.change_grid(h)
            _x = self.change_grid(_x)

        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                _x = self.conv_shortcut(_x)
            else:
                _x = self.nin_shortcut(_x)

        if self.skip_connection is not None:
            if self.skip_connection == "give":
                return (_x + h) / math.sqrt(2), x
        return (_x + h) / math.sqrt(2)


def test_ResBlock():
    try:
        print(ResBlock(64, skip_connection="give", up_down="down"))
    except Exception as e:
        print("ResBlock has a problem\n\n {}".format(e))


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels).to(DEVICE)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        ).to(DEVICE)
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        ).to(DEVICE)
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        ).to(DEVICE)
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        ).to(DEVICE)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,"c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


def test_AttnBlock():
    try:
        at_block = AttnBlock(128)
        sample = torch.rand((2, 128, 32, 32))
        sample_out = at_block.forward(sample)
        print(sample_out.size())

    except Exception as e:
        print("AttnBlock has a problem \n\n {}".format(e))


class Generator(nn.Module):
    """
    The Generator model
    """

    def __init__(self, args):
        super(Generator, self).__init__()

        self.device = (
            torch.device("cpu")
            if args.train.device is None
            else torch.device(args.train.device)
        )
        self.image_size = args.generator.image_size

        out_channels = args.generator.num_feature_init

        self.stem_conv = torch.nn.Conv2d(
            3, out_channels, kernel_size=3, stride=1, padding=1  # 64
        ).to(DEVICE)

        current_resolution = self.image_size

        down_procedure = {}
        channels = [1, 2, 1]
        for ch_mul in channels:
            down_procedure[current_resolution] = []
            in_channels = out_channels
            out_channels = in_channels * ch_mul

            down_procedure[current_resolution].append(
                ResBlock(in_channels, skip_connection="give", up_down="down").to(
                    self.device
                )
            )
            down_procedure[current_resolution].append(
                ResBlock(in_channels, skip_connection="give").to(self.device)
            )
            down_procedure[current_resolution].append(
                ResBlock(in_channels, out_channels, skip_connection="give").to(
                    self.device
                )
            )
            current_resolution = current_resolution // 2

        down_procedure[current_resolution] = []
        in_channels = out_channels
        out_channels = in_channels
        down_procedure[current_resolution].append(
            ResBlock(in_channels, skip_connection="give").to(self.device)
        )
        down_procedure[current_resolution].append(
            ResBlock(in_channels, skip_connection="give").to(self.device)
        )
        down_procedure[current_resolution].append(
            ResBlock(in_channels, out_channels, skip_connection="give").to(self.device)
        )

        # attention
        self.attention_block = AttnBlock(out_channels).to(self.device)

        # Up
        in_channels = out_channels
        out_channels = out_channels // 2

        up_procedure = {}
        up_procedure[current_resolution] = []

        # ResBlock x 5
        up_procedure[current_resolution].append(ResBlock(in_channels).to(self.device))
        up_procedure[current_resolution].append(
            ResBlock(
                in_channels,
                skip_connection="take",
                up_down="up",
            ).to(self.device)
        )
        up_procedure[current_resolution].append(
            ResBlock(
                in_channels,
                skip_connection="take",
            ).to(self.device)
        )
        up_procedure[current_resolution].append(
            ResBlock(
                in_channels,
                skip_connection="take",
            ).to(self.device)
        )
        up_procedure[current_resolution].append(ResBlock(in_channels).to(self.device))

        # ResBlock x 4
        current_resolution = current_resolution * 2
        up_procedure[current_resolution] = []

        up_procedure[current_resolution].append(
            ResBlock(
                in_channels,
                skip_connection="take",
                up_down="up",
            ).to(self.device)
        )
        up_procedure[current_resolution].append(
            ResBlock(
                in_channels,
                skip_connection="take",
            ).to(self.device)
        )
        up_procedure[current_resolution].append(
            ResBlock(
                in_channels,
                skip_connection="take",
            ).to(self.device)
        )
        up_procedure[current_resolution].append(ResBlock(in_channels).to(self.device))

        # ResBlock x 4
        current_resolution = current_resolution * 2
        up_procedure[current_resolution] = []

        up_procedure[current_resolution].append(
            ResBlock(
                in_channels,
                skip_connection="take",
                up_down="up",
            ).to(self.device)
        )
        up_procedure[current_resolution].append(
            ResBlock(in_channels, skip_connection="take").to(self.device)
        )
        up_procedure[current_resolution].append(
            ResBlock(in_channels, skip_connection="take").to(self.device)
        )
        up_procedure[current_resolution].append(ResBlock(in_channels).to(self.device))

        # ResBlock x 3
        current_resolution = current_resolution * 2
        up_procedure[current_resolution] = []

        up_procedure[current_resolution].append(
            ResBlock(in_channels, skip_connection="take").to(self.device)
        )
        up_procedure[current_resolution].append(
            ResBlock(in_channels, skip_connection="take").to(self.device)
        )
        up_procedure[current_resolution].append(
            ResBlock(in_channels, out_channels, skip_connection="take").to(self.device)
        )
        self.down_procedure = down_procedure
        self.up_procedure = up_procedure
        # make_an_image_difference
        self.last_conv = torch.nn.Conv2d(
            out_channels, 3, kernel_size=3, stride=1, padding=1
        ).to(self.device)

    def forward(self, x_tar, temb):
        h = self.stem_conv(x_tar)

        current_resolution = self.image_size

        skips = []
        ####### downsampling #######
        for i in range(len(self.down_procedure)):
            procedure = nn.ModuleList(self.down_procedure[current_resolution])

            for i in range(len(procedure)):
                h, skip = procedure[i].to(DEVICE)(h, temb)
                skips.append(skip)

            current_resolution = current_resolution // 2

        current_resolution = current_resolution * 2
        ####### attention block #######
        h = self.attention_block(h)

        ####### upsampling #######
        skip_idx = -1

        ### ResBlock x5
        procedure = nn.ModuleList(self.up_procedure[current_resolution]).to(
            DEVICE
        )  # 32

        h = procedure[0](h, temb)

        h = procedure[1](h, temb, skips[skip_idx])
        skip_idx -= 1
        h = procedure[2](h, temb, skips[skip_idx])
        skip_idx -= 1
        h = procedure[3](h, temb, skips[skip_idx])
        skip_idx -= 1
        h = procedure[4](h, temb)

        current_resolution = current_resolution * 2  # 64

        ### ResBlock x4
        procedure = nn.ModuleList(self.up_procedure[current_resolution]).to(DEVICE)
        h = procedure[0](h, temb, skips[skip_idx])
        skip_idx -= 1
        h = procedure[1](h, temb, skips[skip_idx])
        skip_idx -= 1
        h = procedure[2](h, temb, skips[skip_idx])
        skip_idx -= 1
        h = procedure[3](h, temb)
        current_resolution = current_resolution * 2  # 128

        ### ResBlock x4
        procedure = nn.ModuleList(self.up_procedure[current_resolution]).to(DEVICE)
        h = procedure[0](h, temb, skips[skip_idx])
        skip_idx -= 1
        h = procedure[1](h, temb, skips[skip_idx])
        skip_idx -= 1
        h = procedure[2](h, temb, skips[skip_idx])
        skip_idx -= 1
        h = procedure[3](h, temb)
        current_resolution = current_resolution * 2  # 256

        ### ResBlock x3
        procedure = nn.ModuleList(self.up_procedure[current_resolution]).to(DEVICE)
        h = procedure[0](h, temb, skips[skip_idx])
        skip_idx -= 1
        h = procedure[1](h, temb, skips[skip_idx])
        skip_idx -= 1
        h = procedure[2](h, temb, skips[skip_idx])

        # last conv
        h = self.last_conv(h)

        return h + x_tar


config_path = "../configs"
config_name = "config.yaml"


@hydra.main(version_base=None, config_path=config_path, config_name=config_name)
def test_Generator(cfg):
    print(cfg)

    G = Generator(cfg)
    temb, x_tar = torch.rand([2, 512]), torch.rand([2, 3, 256, 256])
    out = G(x_tar, temb)
    print(out.shape)


if __name__ == "__main__":
    print("\n#" * 7, " ResBlock Test ", "*" * 7)
    test_ResBlock()
    print("\n#" * 7, " AttnBlock Test ", "*" * 7)
    test_AttnBlock()
    print("\n#" * 7, " Geneartor Test ", "*" * 7)
    test_Generator()
