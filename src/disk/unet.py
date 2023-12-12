import torch
from torch import nn, Tensor

import torch, functools
import torch.nn as nn
import torch.nn.functional as F


def cut_to_match(reference, t, n_pref=2):
    """
    Slice tensor `t` along spatial dimensions to match `reference`, by
    picking the central region. Ignores first `n_pref` axes
    """

    if reference.shape[n_pref:] == t.shape[n_pref:]:
        # sizes match, no slicing necessary
        return t

    # compute the difference along all spatial axes
    diffs = [s - r for s, r in zip(t.shape[n_pref:], reference.shape[n_pref:])]

    # check if diffs are even, which is necessary if we want a truly centered crop
    if not all(d % 2 == 0 for d in diffs) and all(d >= 0 for d in diffs):
        fmt = "Tried to slice `t` of size {} to match `reference` of size {}"
        msg = fmt.format(t.shape, reference.shape)
        raise RuntimeError(msg)

    # pick the full extent of `batch` and `feature` axes
    slices = [slice(None, None)] * n_pref

    # for all remaining pick between diff//2 and size-diff//2
    for d in diffs:
        if d > 0:
            slices.append(slice(d // 2, -(d // 2)))
        elif d == 0:
            slices.append(slice(None, None))

    if slices == []:
        return t
    else:
        return t[slices]


def size_is_pow2(t):
    """Check if the trailing spatial dimensions are powers of 2"""
    return all(s % 2 == 0 for s in t.size()[-2:])


class AttentionGate(nn.Module):
    def __init__(self, n_features):
        super(AttentionGate, self).__init__()
        self.n_features = n_features

        self.seq = nn.Sequential(
            nn.Conv2d(self.n_features, self.n_features, 1), nn.Sigmoid()
        )

    def forward(self, inp):
        g = self.seq(inp)

        return g * inp


class TrivialUpsample(nn.Module):
    def __init__(self, *args, **kwargs):
        super(TrivialUpsample, self).__init__()

    def forward(self, x):
        r = F.interpolate(x, scale_factor=2, mode="nearest")
        return r


class TrivialDownsample(nn.Module):
    def __init__(self, *args, **kwargs):
        super(TrivialDownsample, self).__init__()

    def forward(self, x):
        if not size_is_pow2(x):
            msg = f"Trying to downsample feature map of size {x.size()}"
            raise RuntimeError(msg)

        return F.avg_pool2d(x, 2)


class NoOp(nn.Module):
    def __init__(self, *args, **kwargs):
        super(NoOp, self).__init__()

    def forward(self, x):
        return x


class UGroupNorm(nn.GroupNorm):
    def __init__(self, in_channels, group_size):
        group_size = max(1, min(group_size, in_channels))

        if in_channels % group_size != 0:
            for upper in range(group_size + 1, in_channels + 1):
                if in_channels % upper == 0:
                    break

            for lower in range(group_size - 1, 0, -1):
                if in_channels % lower == 0:
                    break

            if upper - group_size < group_size - lower:
                group_size = upper
            else:
                group_size = lower

        assert in_channels % group_size == 0
        num_groups = in_channels // group_size

        super(UGroupNorm, self).__init__(num_groups, in_channels)


def u_group_norm(group_size):
    return functools.partial(UGroupNorm, group_size=group_size)


class Conv(nn.Sequential):
    def __init__(self, in_, out_, size):
        norm = nn.InstanceNorm2d(in_)
        nonl = nn.PReLU(in_)
        conv = nn.Conv2d(in_, out_, size, padding="same", bias=True)

        super(Conv, self).__init__(norm, nonl, conv)


class ThinUnetDownBlock(nn.Sequential):
    def __init__(self, in_, out_, size=5, is_first=False):
        self.in_ = in_
        self.out_ = out_

        if is_first:
            downsample = NoOp()
            conv = Conv(in_, out_, size)
        else:
            downsample = TrivialDownsample(in_, size)
            conv = Conv(in_, out_, size)

        super(ThinUnetDownBlock, self).__init__(downsample, conv)


class ThinUnetUpBlock(nn.Module):
    def __init__(self, bottom_, horizontal_, out_, size=5):
        super(ThinUnetUpBlock, self).__init__()

        self.bottom_ = bottom_
        self.horizontal_ = horizontal_
        self.cat_ = bottom_ + horizontal_
        self.out_ = out_

        self.upsample = ThinUnetUpBlock(bottom_, size)
        self.conv = Conv(self.cat_, self.out_, size)

    def forward(self, bot: Tensor, hor: Tensor) -> Tensor:
        bot_big = self.upsample(bot)
        hor = cut_to_match(bot_big, hor, n_pref=2)
        combined = torch.cat([bot_big, hor], dim=1)

        return self.conv(combined)


class Unet(nn.Module):
    def __init__(self, in_features=1, up=None, down=None, size=5):
        super(Unet, self).__init__()

        if not len(down) == len(up) + 1:
            raise ValueError("`down` must be 1 item longer than `up`")

        self.up = up
        self.down = down
        self.in_features = in_features

        down_dims = [in_features] + down
        self.path_down = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(down_dims[:-1], down_dims[1:])):
            block = ThinUnetDownBlock(
                d_in,
                d_out,
                size=size,
                is_first=i == 0,
            )
            self.path_down.append(block)

        bot_dims = [down[-1]] + up
        hor_dims = down_dims[-2::-1]
        self.path_up = nn.ModuleList()
        for i, (d_bot, d_hor, d_out) in enumerate(zip(bot_dims, hor_dims, up)):
            block = ThinUnetUpBlock(d_bot, d_hor, d_out, size=size)
            self.path_up.append(block)

        self.n_params = 0
        for param in self.parameters():
            self.n_params += param.numel()

    def forward(self, inp: Tensor) -> Tensor:
        if inp.size(1) != self.in_features:
            fmt = "Expected {} feature channels in input, got {}"
            msg = fmt.format(self.in_features, inp.size(1))
            raise ValueError(msg)

        features = [inp]
        for i, layer in enumerate(self.path_down):
            features.append(layer(features[-1]))

        f_bot = features[-1]
        features_horizontal = features[-2::-1]

        for layer, f_hor in zip(self.path_up, features_horizontal):
            f_bot = layer(f_bot, f_hor)

        return f_bot
