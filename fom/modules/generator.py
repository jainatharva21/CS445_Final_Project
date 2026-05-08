import torch
from torch import nn
import torch.nn.functional as F
from .util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d
from .dense_motion import DenseMotionNetwork


class OcclusionAwareGenerator(nn.Module):
    def __init__(self, num_channels, num_kp, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks, estimate_occlusion_map=False, dense_motion_params=None, estimate_jacobian=False):
        super().__init__()
        self.dense_motion_network = (
            DenseMotionNetwork(num_kp=num_kp, num_channels=num_channels, estimate_occlusion_map=estimate_occlusion_map,
                               **dense_motion_params) if dense_motion_params else None)
        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=7, padding=3)
        downs, ups = [], []
        for i in range(num_down_blocks):
            downs.append(DownBlock2d(
                min(max_features, block_expansion * (2 ** i)),
                min(max_features, block_expansion * (2 ** (i + 1))), 3, 1))
        for i in range(num_down_blocks):
            ups.append(UpBlock2d(
                min(max_features, block_expansion * (2 ** (num_down_blocks - i))),
                min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1))), 3, 1))
        self.down_blocks, self.up_blocks = nn.ModuleList(downs), nn.ModuleList(ups)
        bf = min(max_features, block_expansion * (2 ** num_down_blocks))
        self.bottleneck = nn.Sequential(*[ResBlock2d(bf, 3, 1) for _ in range(num_bottleneck_blocks)])
        self.final = nn.Conv2d(block_expansion, num_channels, 7, padding=3)
        self.num_channels = num_channels

    def deform_input(self, inp, deformation):
        _, _, h, w = inp.shape
        d = deformation
        if d.shape[1:3] != (h, w):
            d = F.interpolate(d.permute(0, 3, 1, 2), size=(h, w), mode="bilinear").permute(0, 2, 3, 1)
        return F.grid_sample(inp, d)

    def forward(self, source_image, kp_driving, kp_source):
        out = self.first(source_image)
        for d in self.down_blocks:
            out = d(out)
        od = {}
        if self.dense_motion_network is not None:
            dm = self.dense_motion_network(source_image=source_image, kp_driving=kp_driving, kp_source=kp_source)
            od["mask"] = dm["mask"]
            od["sparse_deformed"] = dm["sparse_deformed"]
            occ = dm.get("occlusion_map")
            if occ is not None:
                od["occlusion_map"] = occ
            defm = dm["deformation"]
            out = self.deform_input(out, defm)
            if occ is not None:
                if out.shape[2:] != occ.shape[2:]:
                    occ = F.interpolate(occ, size=out.shape[2:], mode="bilinear")
                out = out * occ
            od["deformed"] = self.deform_input(source_image, defm)
        out = self.bottleneck(out)
        for u in self.up_blocks:
            out = u(out)
        od["prediction"] = torch.sigmoid(self.final(out))
        return od
