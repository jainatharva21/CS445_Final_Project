"""Backward-compatible import — implementation lives in ``fom.services.inference``."""

from .services.inference import animate, normalize_kp

__all__ = ["animate", "normalize_kp"]
