"""
Application entry: load YAML, build networks, dispatch train / reconstruction / animate.

Routing and filesystem helpers live here; training and inference bodies are under
``fom.services``, datasets under ``fom.data``. ``fom.modules`` stays aligned with the
published FOMM implementation for weight compatibility.
"""
from __future__ import annotations

import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
from shutil import copy
from time import gmtime, strftime
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import torch
import yaml

from .data.datasets import FramesDataset
from .services import animate, reconstruction, train
from .services.model_factory import build_core_networks, place_on_devices


def _timestamped_run_dir(log_root: Path, config_path: Path) -> Path:
    stem = config_path.stem
    stamp = strftime("%d_%m_%y_%H.%M.%S", gmtime())
    return log_root / f"{stem} {stamp}"


def _resolve_log_directory(
    *,
    log_root: str,
    config_path: Path,
    checkpoint_path: str | None,
) -> Path:
    if checkpoint_path:
        ckpt = Path(checkpoint_path).resolve()
        return ckpt.parent
    return _timestamped_run_dir(Path(log_root), config_path)


def _snapshot_config(config_src: Path, log_dir: Path) -> None:
    dest = log_dir / config_src.name
    if not dest.exists():
        copy(str(config_src), dest)


def main(argv: Sequence[str] | None = None) -> None:
    if sys.version_info < (3, 8):
        sys.exit("Python 3.8 or newer is required.")

    parser = ArgumentParser(
        description="First-order motion: train, reconstruction self-test, or cross-clip animation.",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
Examples (run from this project root so dataset paths in YAML resolve):
  python cli.py train --config config/mgif-256-local.yaml --log_dir logs
  python cli.py reconstruction --config config/mgif-e2e-smoke.yaml --checkpoint path/to/epoch.pth.tar
""",
    )
    parser.add_argument("--config", required=True, help="Path to experiment YAML.")
    parser.add_argument(
        "--mode",
        default="train",
        choices=("train", "reconstruction", "animate"),
        help="train = fit models; reconstruction = self-reenactment on test; animate = paired transfer.",
    )
    parser.add_argument("--log_dir", default="log", help="Directory for new training runs (ignored if --checkpoint is set).")
    parser.add_argument("--checkpoint", default=None, help="Resume path, or required for reconstruction / animate.")
    parser.add_argument(
        "--device_ids",
        default="0",
        type=lambda s: [int(x) for x in s.split(",") if x.strip() != ""],
        help="Comma-separated CUDA device indices (first device hosts module copies).",
    )
    parser.add_argument("--verbose", action="store_true", help="Print module summaries.")
    opt = parser.parse_args(list(argv) if argv is not None else None)

    config_path = Path(opt.config).resolve()
    with config_path.open(encoding="utf-8") as fh:
        cfg: Mapping[str, Any] = yaml.safe_load(fh)

    log_dir = _resolve_log_directory(
        log_root=opt.log_dir,
        config_path=config_path,
        checkpoint_path=opt.checkpoint,
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    _snapshot_config(config_path, log_dir)

    mp = cfg["model_params"]
    use_cuda = torch.cuda.is_available()

    generator, discriminator, kp_detector = build_core_networks(mp)
    place_on_devices(generator, discriminator, kp_detector, use_cuda=use_cuda, primary_device=opt.device_ids[0])

    if opt.verbose:
        print(generator)
        print(discriminator)
        print(kp_detector)

    dataset = FramesDataset(is_train=(opt.mode == "train"), **cfg["dataset_params"])

    if opt.mode == "train":
        print("Training...")
        train(cfg, generator, discriminator, kp_detector, opt.checkpoint, str(log_dir), dataset, opt.device_ids)
    elif opt.mode == "reconstruction":
        print("Reconstruction...")
        reconstruction(cfg, generator, kp_detector, opt.checkpoint, str(log_dir), dataset)
    else:
        print("Animate...")
        animate(cfg, generator, kp_detector, opt.checkpoint, str(log_dir), dataset)
