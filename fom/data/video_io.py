"""
Decode on-disk clips into float32 numpy arrays (``T × H × W × C`` in ``[0, 1]``).

Separated from dataset indexing so loaders can be unit-tested or reused by other
pipelines without importing PyTorch ``Dataset`` types.
"""
from __future__ import annotations

import os

import numpy as np
from imageio import mimread
from skimage import io, img_as_float32
from skimage.color import gray2rgb


def _load_directory_as_stack(folder: str) -> np.ndarray:
    frame_files = sorted(os.listdir(folder))
    return np.stack([img_as_float32(io.imread(os.path.join(folder, frame_files[i]))) for i in range(len(frame_files))])


def _load_png_filmstrip(path: str, frame_shape: tuple[int, int, int]) -> np.ndarray:
    image = io.imread(path)
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = gray2rgb(image)
    if image.shape[2] == 4:
        image = image[..., :3]
    image = img_as_float32(image)
    height, width, channels = frame_shape
    video = np.moveaxis(image, 1, 0).reshape((-1, height, width, channels))
    return np.moveaxis(video, 1, 2)


def _load_container_clip(path: str) -> np.ndarray:
    clip = mimread(path)
    arr = np.array(clip)
    if len(arr.shape) == 3:
        arr = np.array([gray2rgb(f) for f in clip])
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return img_as_float32(arr)


def read_video(name: str, frame_shape: tuple[int, int, int]) -> np.ndarray:
    """Load a clip as float32 ``T x H x W x C`` in ``[0, 1]``."""
    if os.path.isdir(name):
        return _load_directory_as_stack(name)

    lower = name.lower()
    if lower.endswith((".png", ".jpg")):
        return _load_png_filmstrip(name, frame_shape)

    if lower.endswith((".gif", ".mp4", ".mov")):
        return _load_container_clip(name)

    raise RuntimeError(f"Unsupported format: {name}")
