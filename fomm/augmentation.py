import numpy as np
from skimage.color import rgb2hsv, hsv2rgb


class FlipHorizontal:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, video):
        if np.random.rand() < self.p:
            return np.ascontiguousarray(video[:, :, ::-1, :])
        return video


class TimeFlip:
    """Reverse frame order. Swaps source and driving in (source, driving) pairs."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, video):
        if np.random.rand() < self.p:
            return np.ascontiguousarray(video[::-1])
        return video


class ColorJitter:
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, video):
        b = 1.0 + np.random.uniform(-self.brightness, self.brightness) if self.brightness else 1.0
        c = 1.0 + np.random.uniform(-self.contrast, self.contrast) if self.contrast else 1.0
        s = 1.0 + np.random.uniform(-self.saturation, self.saturation) if self.saturation else 1.0
        h = np.random.uniform(-self.hue, self.hue) if self.hue else 0.0

        out = video
        if b != 1.0:
            out = np.clip(out * b, 0, 1)
        if c != 1.0:
            mean = out.mean(axis=(1, 2), keepdims=True)
            out = np.clip((out - mean) * c + mean, 0, 1)
        if s != 1.0 or h != 0.0:
            hsv = np.stack([rgb2hsv(f) for f in out])
            if h != 0.0:
                hsv[..., 0] = (hsv[..., 0] + h) % 1.0
            if s != 1.0:
                hsv[..., 1] = np.clip(hsv[..., 1] * s, 0, 1)
            out = np.stack([hsv2rgb(f) for f in hsv]).astype(np.float32)
        return out


class AllAugmentationTransform:
    def __init__(self, flip_horizontal=True, time_flip=True, color_jitter=None):
        self.transforms = []
        if flip_horizontal:
            self.transforms.append(FlipHorizontal(p=0.5))
        if time_flip:
            self.transforms.append(TimeFlip(p=0.5))
        if color_jitter is not None:
            self.transforms.append(ColorJitter(**color_jitter))

    def __call__(self, video):
        for t in self.transforms:
            video = t(video)
        return video
