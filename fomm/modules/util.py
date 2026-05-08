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