"""Backward-compatible imports — implementations live in ``fom.data``."""

from .data.datasets import DatasetRepeater, FramesDataset, PairedDataset
from .data.video_io import read_video

__all__ = ["DatasetRepeater", "FramesDataset", "PairedDataset", "read_video"]
