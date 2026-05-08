from torch import nn
import torch
import torch.nn.functional as F
from .util import kp2gaussian


class DownBlock2d(nn.Module):
    def __init__(self, in_features, out_features, norm=False, kernel_size=4, pool=False, sn=False):
        super().__init__()
        conv = nn.Conv2d(in_features, out_features, kernel_size)
        self.conv = nn.utils.spectral_norm(conv) if sn else conv
        self.norm = nn.InstanceNorm2d(out_features, affine=True) if norm else None
        self.pool = pool

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        x = F.leaky_relu(x, 0.2)
        return F.avg_pool2d(x, (2, 2)) if self.pool else x


class Discriminator(nn.Module):
    def __init__(self, num_channels=3, block_expansion=64, num_blocks=4, max_features=512,
                 sn=False, use_kp=False, num_kp=10, kp_variance=0.01, **kwargs):
        super().__init__()
        blocks = []
        for i in range(num_blocks):
            inc = num_channels + num_kp * use_kp if i == 0 else min(max_features, block_expansion * (2 ** i))
            ouc = min(max_features, block_expansion * (2 ** (i + 1)))
            blocks.append(DownBlock2d(inc, ouc, norm=(i != 0), kernel_size=4, pool=(i != num_blocks - 1), sn=sn))
        self.down_blocks = nn.ModuleList(blocks)
        last_c = self.down_blocks[-1].conv.out_channels
        out = nn.Conv2d(last_c, 1, 1)
        self.conv = nn.utils.spectral_norm(out) if sn else out
        self.use_kp, self.kp_variance = use_kp, kp_variance

    def forward(self, x, kp=None):
        fmaps, out = [], x
        if self.use_kp:
            out = torch.cat([out, kp2gaussian(kp, x.shape[2:], self.kp_variance)], dim=1)
        for db in self.down_blocks:
            fmaps.append(db(out))
            out = fmaps[-1]
        return fmaps, self.conv(out)


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, scales=(), **kwargs):
        super().__init__()
        self.scales = scales
        self.discs = nn.ModuleDict({str(s).replace(".", "-"): Discriminator(**kwargs) for s in scales})

    def forward(self, x, kp=None):
        out = {}
        for k, disc in self.discs.items():
            s = k.replace("-", ".")
            fmaps, pred = disc(x["prediction_" + s], kp)
            out["feature_maps_" + s] = fmaps
            out["prediction_map_" + s] = pred
        return out
