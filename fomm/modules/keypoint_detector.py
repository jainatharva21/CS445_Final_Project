import torch
from torch import nn
import torch.nn.functional as F

from modules.util import Hourglass, make_coordinate_grid

class KPDetector(nn.Module):
    def __init__(self, 
        block_expansion,
        num_kp,
        num_channels,
        max_features,
        num_blocks,
        temperature,
        estimate_jacobian=False,
        scale_factor=1,
        **kwargs
    ):
        super().__init__()

        self.num_kp = num_kp
        self.temperature = temperature
        self.scale_factor = scale_factor
        self.estimate_jacobian = estimate_jacobian

        self.predictor = Hourglass(
            block_expansion=block_expansion,
            in_features=num_channels,
            max_features=max_features,
            num_blocks=num_blocks
        )

        feature_channels = self.predictor.out_filters

        self.kp = nn.Conv2d(
            in_channels=feature_channels,
            out_channels=num_kp,
            kernel_size=7,
            padding=3
        )

        if estimate_jacobian:
            self.jacobian_head = nn.Conv2d(
                in_channels=feature_channels,
                out_channels=4 * num_kp,
                kernel_size=7,
                padding=3
            )

            self._init_jacobian_to_identity()
        else:
            self.jacobian_head = None

    def _init_jacobian_to_identity(self):
        """
        Initializes each predicted 2x2 Jacobian near identity.
        """
        with torch.no_grad():
            self.jacobian_head.weight.zero_()

            identity_bias = torch.tensor(
                [1.0, 0.0, 0.0, 1.0] * self.num_kp,
                dtype=self.jacobian_head.bias.dtype,
                device=self.jacobian_head.bias.device
            )

            self.jacobian_head.bias.copy_(identity_bias)

    def _resize_input(self, x):
        """
        Runs the detector at a lower spatial resolution.

        scale_factor=0.25, would turn a 256x256 input into 64x64.
        Keypoints are still returned in normalized [-1, 1] coordinates.
        """
        if self.scale_factor == 1:
            return x

        return F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=True
        )
    
    def _spatial_softmax(self, logits):
        """
        Converts raw heatmap logits into spatial probability maps.

        logits:  [B, K, H, W]
        returns: [B, K, H, W]
        """
        b, k, h, w = logits.shape

        flat_logits = logits.reshape(b, k, h * w)

        try:
            flat_probs = F.softmax(flat_logits / self.temperature, dim=-1)
        except:
            print("ValueError in _spatial_softmax, self.temperature is probably 0")

        
        return flat_probs.reshape(b, k, h, w)

    def _heatmap_to_points(self, heatmap):
        """
        Computes soft-argmax keypoint coordinates from heatmaps.

        heatmap: [B, K, H, W]
        returns: [B, K, 2]
        """
        _, _, h, w = heatmap.shape

        coordinate_grid = make_coordinate_grid((h, w), heatmap.type())
        coordinate_grid = coordinate_grid.reshape(1, 1, h, w, 2)

        weighted_grid = heatmap.unsqueeze(-1) * coordinate_grid

        return weighted_grid.sum(dim=(2, 3))
    
    def _estimate_jacobians(self, feature_map, heatmap):
        """
        Predicts one local 2x2 affine transform per keypoint.

        feature_map: [B, F, H, W]
        heatmap:     [B, K, H, W]
        returns:     [B, K, 2, 2]
        """
        b, _, h, w = feature_map.shape

        jacobian_pixels = self.jacobian_head(feature_map)
        jacobian_pixels = jacobian_pixels.reshape(
            b,
            self.num_kp,
            4,
            h,
            w
        )

        weighted_jacobian = heatmap.unsqueeze(2) * jacobian_pixels
        jacobian = weighted_jacobian.sum(dim=(3, 4))

        return jacobian.reshape(b, self.num_kp, 2, 2)
    
    def forward(self, x):
        x = self._resize_input(x)

        feature_map = self.predictor(x)

        heatmap_logits = self.kp(feature_map)
        heatmap = self._spatial_softmax(heatmap_logits)

        output = {
            "value": self._heatmap_to_points(heatmap)
        }

        if self.jacobian_head is not None:
            output["jacobian"] = self._estimate_jacobians(feature_map, heatmap)

        return output