from torch import nn
import torch
import torch.nn.functional as F
from .util import Hourglass, AntiAliasInterpolation2d, make_coordinate_grid, kp2gaussian


class DenseMotionNetwork(nn.Module):
    def __init__(self, block_expansion, num_blocks, max_features, num_kp, num_channels, estimate_occlusion_map=False,
                 scale_factor=1, kp_variance=0.01):
        super().__init__()
        inch = (num_kp + 1) * (num_channels + 1)
        self.hourglass = Hourglass(block_expansion, inch, num_blocks, max_features)
        self.mask = nn.Conv2d(self.hourglass.out_filters, num_kp + 1, 7, padding=3)
        self.occlusion = nn.Conv2d(self.hourglass.out_filters, 1, 7, padding=3) if estimate_occlusion_map else None
        self.num_kp = num_kp
        self.scale_factor = scale_factor
        self.kp_variance = kp_variance
        self.down = AntiAliasInterpolation2d(num_channels, scale_factor) if scale_factor != 1 else None

    def create_heatmap_representations(self, source_image, kp_driving, kp_source):
        spatial_size = source_image.shape[2:]
        h = kp2gaussian(kp_driving, spatial_size, self.kp_variance) - kp2gaussian(kp_source, spatial_size, self.kp_variance)
        z = h.new_zeros(h.shape[0], 1, *spatial_size)
        return torch.cat([z, h], dim=1).unsqueeze(2)

    def create_sparse_motions(self, source_image, kp_driving, kp_source):
        bs, _, h, w = source_image.shape
        t = kp_source["value"].type()
        ig = make_coordinate_grid((h, w), t).view(1, 1, h, w, 2)
        cg = ig - kp_driving["value"].view(bs, self.num_kp, 1, 1, 2)
        if "jacobian" in kp_driving:
            J = torch.matmul(kp_source["jacobian"], torch.inverse(kp_driving["jacobian"]))
            J = J.unsqueeze(-3).unsqueeze(-3).expand(-1, -1, h, w, -1, -1)
            cg = torch.matmul(J, cg.unsqueeze(-1)).squeeze(-1)
        sparse = torch.cat([ig.expand(bs, -1, -1, -1, -1), cg + kp_source["value"].view(bs, self.num_kp, 1, 1, 2)], dim=1)
        return sparse

    def create_deformed_source_image(self, source_image, sparse_motions):
        bs, _, h, w = source_image.shape
        k1 = self.num_kp + 1
        src = source_image.unsqueeze(1).unsqueeze(1).expand(-1, k1, -1, -1, -1, -1).reshape(bs * k1, -1, h, w)
        sm = sparse_motions.reshape(bs * k1, h, w, -1)
        return F.grid_sample(src, sm).view(bs, k1, -1, h, w)

    def forward(self, source_image, kp_driving, kp_source):
        if self.down is not None:
            source_image = self.down(source_image)
        bs, _, h, w = source_image.shape
        heat = self.create_heatmap_representations(source_image, kp_driving, kp_source)
        sparse = self.create_sparse_motions(source_image, kp_driving, kp_source)
        deformed = self.create_deformed_source_image(source_image, sparse)
        x = torch.cat([heat, deformed], dim=2).view(bs, -1, h, w)
        pred = self.hourglass(x)
        m = F.softmax(self.mask(pred), dim=1)
        out = {"sparse_deformed": deformed, "mask": m,
               "deformation": (sparse.permute(0, 1, 4, 2, 3) * m.unsqueeze(2)).sum(1).permute(0, 2, 3, 1)}
        if self.occlusion is not None:
            out["occlusion_map"] = torch.sigmoid(self.occlusion(pred))
        return out
