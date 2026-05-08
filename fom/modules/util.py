"""CNN blocks, hourglass, keypoint Gaussian maps, anti-alias resize."""
from torch import nn
import torch
import torch.nn.functional as F

from ..sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d


def kp2gaussian(kp, spatial_size, kp_variance):
    mean = kp["value"]
    nd = len(mean.shape) - 1
    grid = make_coordinate_grid(spatial_size, mean.type())
    grid = grid.view(*((1,) * nd + spatial_size + (2,))).repeat(*(mean.shape[:nd] + (1, 1, 1)))
    mean = mean.view(*(mean.shape[:nd] + (1, 1, 2)))
    return torch.exp(-0.5 * ((grid - mean) ** 2).sum(-1) / kp_variance)


def make_coordinate_grid(spatial_size, type):  # noqa: A002
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)
    x = 2 * (x / (w - 1)) - 1
    y = 2 * (y / (h - 1)) - 1
    xx = x.view(1, -1).repeat(h, 1)
    yy = y.view(-1, 1).repeat(1, w)
    return torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)


class ResBlock2d(nn.Module):
    def __init__(self, in_features, kernel_size, padding):
        super().__init__()
        c1 = nn.Conv2d(in_features, in_features, kernel_size, padding=padding)
        c2 = nn.Conv2d(in_features, in_features, kernel_size, padding=padding)
        n1 = BatchNorm2d(in_features, affine=True)
        n2 = BatchNorm2d(in_features, affine=True)
        self.conv1, self.conv2, self.norm1, self.norm2 = c1, c2, n1, n2

    def forward(self, x):
        y = self.conv2(F.relu(self.norm2(self.conv1(F.relu(self.norm1(x))))))
        return y + x


class UpBlock2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size, padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        return F.relu(self.norm(self.conv(F.interpolate(x, scale_factor=2))))


class DownBlock2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size, padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d((2, 2))

    def forward(self, x):
        return self.pool(F.relu(self.norm(self.conv(x))))


class SameBlock2d(nn.Module):
    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size, padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        return F.relu(self.norm(self.conv(x)))


class Encoder(nn.Module):
    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super().__init__()
        blocks = []
        for i in range(num_blocks):
            inf = in_features if i == 0 else min(max_features, block_expansion * (2 ** i))
            ouf = min(max_features, block_expansion * (2 ** (i + 1)))
            blocks.append(DownBlock2d(inf, ouf, 3, 1))
        self.down_blocks = nn.ModuleList(blocks)

    def forward(self, x):
        out = [x]
        for d in self.down_blocks:
            out.append(d(out[-1]))
        return out


class Decoder(nn.Module):
    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super().__init__()
        blocks = []
        for i in range(num_blocks - 1, -1, -1):
            inf = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            ouf = min(max_features, block_expansion * (2 ** i))
            blocks.append(UpBlock2d(inf, ouf, 3, 1))
        self.up_blocks = nn.ModuleList(blocks)
        self.out_filters = block_expansion + in_features

    def forward(self, x):
        out = x.pop()
        for u in self.up_blocks:
            out = torch.cat([u(out), x.pop()], dim=1)
        return out


class Hourglass(nn.Module):
    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super().__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AntiAliasInterpolation2d(nn.Module):
    def __init__(self, channels, scale):
        super().__init__()
        sigma = (1 / scale - 1) / 2
        ksz = 2 * round(sigma * 4) + 1
        self.ka = ksz // 2
        self.kb = self.ka - 1 if ksz % 2 == 0 else self.ka
        kernel_size, sig = [ksz, ksz], [sigma, sigma]
        grids = torch.meshgrid(*[torch.arange(sz, dtype=torch.float32) for sz in kernel_size])
        k = grids[0].new_ones(())
        for sz, std, mg in zip(kernel_size, sig, grids):
            mean = (sz - 1) / 2
            k = k * torch.exp(-(mg - mean) ** 2 / (2 * std ** 2))
        k = k / k.sum()
        k = k.view(1, 1, *k.shape)
        k = k.repeat(channels, *([1] * (k.dim() - 1)))
        self.register_buffer("weight", k)
        self.groups = channels
        self.scale = scale
        self.int_inv_scale = int(1 / scale)

    def forward(self, inp):
        if self.scale == 1.0:
            return inp
        x = F.pad(inp, (self.ka, self.kb, self.ka, self.kb))
        x = F.conv2d(x, weight=self.weight, groups=self.groups)
        return x[:, :, :: self.int_inv_scale, :: self.int_inv_scale]
