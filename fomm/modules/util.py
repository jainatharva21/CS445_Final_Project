import torch
from torch import nn
import torch.nn.functional as F

def make_coordinate_grid(spatial_size, type):
    """
    Creates a normalized coordinate grid in the range [-1, 1].

    spatial_size: tuple/list (height, width)
    type: tensor type, usually heatmap.type()

    returns: [H, W, 2], where the last dimension is (x, y)
    """
    h, w = spatial_size

    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(h, 1).repeat(1, w)
    xx = x.view(1, w).repeat(h, 1)

    coordinate_grid = torch.stack([xx, yy], dim=-1)

    return coordinate_grid

def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform keypoints into Gaussian-like heatmap representations.
    """
    mean = kp["value"]

    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())

    number_of_leading_dimensions = len(mean.shape) - 1

    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)

    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    mean = mean.view(*shape)

    mean_sub = coordinate_grid - mean

    out = torch.exp(
        -0.5 * (mean_sub ** 2).sum(-1) / kp_variance
    )

    return out

class AntiAliasInterpolation2d(nn.Module):
    """
    Downsamples an image with anti-aliasing.

    This is used when scale_factor != 1. Before resizing, it applies
    a Gaussian blur so that high-frequency details do not alias badly
    during downsampling.

    Input:
        x: [B, C, H, W]

    Output:
        resized tensor [B, C, new_H, new_W]
    """
    def __init__(self, channels, scale):
        super().__init__()

        self.scale = scale
        self.channels = channels

        if scale == 1.0:
            self.ka = 0
            self.kb = 0
            self.register_buffer("weight", torch.zeros(1))
            return

        sigma = (1 / scale - 1) / 2

        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]

        kernel = 1

        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ],
            indexing="ij"
        )

        for size, std, grid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(
                -((grid - mean) ** 2) / (2 * std ** 2)
            )

        kernel = kernel / torch.sum(kernel)

        kernel = kernel.view(1, 1, kernel_size[0], kernel_size[1])
        kernel = kernel.repeat(channels, 1, 1, 1)

        self.register_buffer("weight", kernel)

    def forward(self, x):
        if self.scale == 1.0:
            return x

        out = F.pad(x, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.channels)

        out = F.interpolate(
            out,
            scale_factor=(self.scale, self.scale),
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=True
        )

        return out

class SameBlock2d(nn.Module):
    """
    Conv block that keeps the same spatial resolution.
    """
    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            padding=padding
        )

        self.norm = nn.BatchNorm2d(out_features)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)

        return out


class DownBlock2d(nn.Module):
    """
    Downsampling block used in the encoder.
    Reduces H and W by a factor of 2.
    """
    def __init__(self, in_features, out_features):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=3,
            padding=1
        )

        self.norm = nn.BatchNorm2d(out_features)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = F.avg_pool2d(out, kernel_size=2)

        return out


class UpBlock2d(nn.Module):
    """
    Upsampling block used in the decoder.
    Increases H and W by a factor of 2.
    """
    def __init__(self, in_features, out_features):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=3,
            padding=1
        )

        self.norm = nn.BatchNorm2d(out_features)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2, mode="nearest")
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)

        return out


class Encoder(nn.Module):
    """
    Encoder part of the hourglass.

    Saves intermediate outputs for skip connections.
    """
    def __init__(self, block_expansion, in_features, num_blocks, max_features):
        super().__init__()

        down_blocks = []

        for i in range(num_blocks):
            input_channels = in_features if i == 0 else min(max_features, block_expansion * (2 ** i))
            output_channels = min(max_features, block_expansion * (2 ** (i + 1)))

            down_blocks.append(
                DownBlock2d(input_channels, output_channels)
            )

        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outputs = [x]

        out = x
        for down_block in self.down_blocks:
            out = down_block(out)
            outputs.append(out)

        return outputs


class Decoder(nn.Module):
    """
    Decoder part of the hourglass.

    Uses skip connections from the encoder.
    """
    def __init__(self, block_expansion, in_features, num_blocks, max_features):
        super().__init__()

        up_blocks = []

        for i in range(num_blocks)[::-1]:
            input_channels = min(max_features, block_expansion * (2 ** (i + 1)))
            output_channels = min(max_features, block_expansion * (2 ** i))

            up_blocks.append(
                UpBlock2d(input_channels, output_channels)
            )

        self.up_blocks = nn.ModuleList(up_blocks)

        # Final decoder output has the last upsampled features concatenated
        # with the original input through a skip connection.
        self.out_filters = block_expansion + in_features

    def forward(self, encoder_outputs):
        out = encoder_outputs.pop()

        for up_block in self.up_blocks:
            out = up_block(out)

            skip = encoder_outputs.pop()
            out = torch.cat([out, skip], dim=1)

        return out


class Hourglass(nn.Module):
    """
    Hourglass network used by the keypoint detector.

    The encoder downsamples the image and the decoder upsamples it back,
    using skip connections to preserve spatial details.
    """
    def __init__(self, block_expansion, in_features, num_blocks, max_features):
        super().__init__()

        self.encoder = Encoder(
            block_expansion=block_expansion,
            in_features=in_features,
            num_blocks=num_blocks,
            max_features=max_features
        )

        self.decoder = Decoder(
            block_expansion=block_expansion,
            in_features=in_features,
            num_blocks=num_blocks,
            max_features=max_features
        )

        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        encoder_outputs = self.encoder(x)
        out = self.decoder(encoder_outputs)

        return out