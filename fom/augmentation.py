"""
Backward-compatible augmentation module.

New code should import from ``fom.data.spatial_augmentation`` (or ``fom.data``).
"""
from .data.spatial_augmentation import *  # noqa: F401,F403
