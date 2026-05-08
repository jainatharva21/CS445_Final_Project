"""Construct bare generator / discriminator / KP towers from ``model_params``."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

import torch

from ..modules.discriminator import MultiScaleDiscriminator
from ..modules.generator import OcclusionAwareGenerator
from ..modules.keypoint_detector import KPDetector


def build_core_networks(model_params: Mapping[str, Any]) -> Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    """Return ``(generator, discriminator, kp_detector)`` without moving devices."""
    common = model_params["common_params"]
    generator = OcclusionAwareGenerator(**model_params["generator_params"], **common)
    discriminator = MultiScaleDiscriminator(**model_params["discriminator_params"], **common)
    kp_detector = KPDetector(**model_params["kp_detector_params"], **common)
    return generator, discriminator, kp_detector


def place_on_devices(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    kp_detector: torch.nn.Module,
    *,
    use_cuda: bool,
    primary_device: int,
) -> None:
    if use_cuda:
        generator.to(primary_device)
        discriminator.to(primary_device)
        kp_detector.to(primary_device)
