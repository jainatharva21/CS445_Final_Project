from torch import nn
import torch
import torch.nn.functional as F
from .util import Hourglass, make_coordinate_grid, AntiAliasInterpolation2d


class KPDetector(nn.Module):
    def __init__(self, block_expansion, num_kp, num_channels, max_features, num_blocks, temperature,
                 estimate_jacobian=False, scale_factor=1, single_jacobian_map=False, pad=0):
        super().__init__()
        self.predictor = Hourglass(block_expansion, num_channels, num_blocks, max_features)
        self.kp = nn.Conv2d(self.predictor.out_filters, num_kp, 7, padding=pad)
        self.jacobian = None
        if estimate_jacobian:
            self.num_jacobian_maps = 1 if single_jacobian_map else num_kp
            jc = nn.Conv2d(self.predictor.out_filters, 4 * self.num_jacobian_maps, 7, padding=pad)
            jc.weight.data.zero_()
            jc.bias.data.copy_(torch.tensor([1, 0, 0, 1] * self.num_jacobian_maps, dtype=torch.float))
            self.jacobian = jc
        self.temperature = temperature
        self.down = AntiAliasInterpolation2d(num_channels, scale_factor) if scale_factor != 1 else None

    def gaussian2kp(self, heatmap):
        g = make_coordinate_grid(heatmap.shape[2:], heatmap.dtype).unsqueeze(0).unsqueeze(0).to(heatmap.device)
        h = heatmap.unsqueeze(-1)
        return {"value": (h * g).sum(dim=(2, 3))}

    def forward(self, x):
        if self.down:
            x = self.down(x)
        fm = self.predictor(x)
        pred = self.kp(fm)
        fs = pred.shape
        hmap = pred.view(fs[0], fs[1], -1).div(self.temperature).softmax(dim=2).view(fs)
        out = self.gaussian2kp(hmap)
        if self.jacobian is not None:
            jm = self.jacobian(fm).reshape(fs[0], self.num_jacobian_maps, 4, fs[2], fs[3])
            j = (hmap.unsqueeze(2) * jm).reshape(fs[0], fs[1], 4, -1).sum(dim=-1).reshape(fs[0], fs[1], 2, 2)
            out["jacobian"] = j
        return out
