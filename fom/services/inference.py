"""
Frozen-network evaluation: reconstruction (self-reenactment) and cross-clip animation.

Shared concerns (checkpoint load, DataParallel eval mode, montage + movie export) live
here as helpers so the two modes stay in sync when export settings change.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import imageio
import numpy as np
import torch
from scipy.spatial import ConvexHull
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.datasets import PairedDataset
from ..logger import Logger, Visualizer
from ..sync_batchnorm import DataParallelWithCallback


def _maybe_parallel_eval(gen: torch.nn.Module, kp: torch.nn.Module) -> tuple[torch.nn.Module, torch.nn.Module]:
    if torch.cuda.is_available():
        return DataParallelWithCallback(gen), DataParallelWithCallback(kp)
    return gen, kp


def normalize_kp(
    kp_source: Mapping[str, torch.Tensor],
    kp_driving: Mapping[str, torch.Tensor],
    kp_driving_initial: Mapping[str, torch.Tensor],
    adapt_movement_scale: bool = False,
    use_relative_movement: bool = False,
    use_relative_jacobian: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Map driving keypoints into the coordinate frame of the source clip.

    Kept under this name because ``fom.demo`` imports it.
    """
    if adapt_movement_scale:
        source_np = kp_source["value"][0].detach().cpu().numpy()
        driving_init_np = kp_driving_initial["value"][0].detach().cpu().numpy()
        hull_src = ConvexHull(source_np).volume
        hull_drv0 = ConvexHull(driving_init_np).volume
        motion_scale = float(np.sqrt(hull_src / hull_drv0))
    else:
        motion_scale = 1.0

    adjusted: dict[str, torch.Tensor] = dict(kp_driving)

    if use_relative_movement:
        delta_xy = (kp_driving["value"] - kp_driving_initial["value"]) * motion_scale
        adjusted["value"] = delta_xy + kp_source["value"]
        if use_relative_jacobian:
            j_rel = torch.matmul(kp_driving["jacobian"], torch.inverse(kp_driving_initial["jacobian"]))
            adjusted["jacobian"] = torch.matmul(j_rel, kp_source["jacobian"])

    return adjusted


def reconstruction(
    config: Mapping[str, Any],
    generator: torch.nn.Module,
    kp_detector: torch.nn.Module,
    checkpoint: str | None,
    log_dir: str,
    dataset: torch.utils.data.Dataset,
) -> None:
    if checkpoint is None:
        raise ValueError("A checkpoint path is required for reconstruction.")

    log_path = Path(log_dir)
    recon_root = log_path / "reconstruction"
    png_dir = recon_root / "png"

    Logger.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    recon_root.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    visualizer = Visualizer(**config["visualizer_params"])
    recon_cfg = config["reconstruction_params"]
    max_clips = recon_cfg.get("num_videos")

    use_cuda = torch.cuda.is_available()
    generator, kp_detector = _maybe_parallel_eval(generator, kp_detector)
    generator.eval()
    kp_detector.eval()

    per_frame_errors: list[float] = []

    for clip_index, batch in tqdm(enumerate(loader)):
        if max_clips is not None and clip_index >= max_clips:
            break

        video = batch["video"].cuda() if use_cuda else batch["video"]
        clip_name = batch["name"][0]

        with torch.no_grad():
            predicted_frames: list[np.ndarray] = []
            visualization_rows: list[np.ndarray] = []

            kp_reference = kp_detector(video[:, :, 0])
            num_frames = video.shape[2]

            for frame_index in range(num_frames):
                appearance = video[:, :, 0]
                target = video[:, :, frame_index]
                kp_target = kp_detector(target)
                outputs = generator(appearance, kp_source=kp_reference, kp_driving=kp_target)
                outputs["kp_source"], outputs["kp_driving"] = kp_reference, kp_target
                outputs.pop("sparse_deformed", None)

                predicted_frames.append(outputs["prediction"].detach().cpu().numpy().transpose(0, 2, 3, 1)[0])
                visualization_rows.append(visualizer.visualize(target, appearance, outputs))
                per_frame_errors.append(float((outputs["prediction"] - target).abs().mean().cpu().numpy()))

        strip = np.concatenate(predicted_frames, axis=1)
        stem = Path(clip_name).stem
        imageio.imwrite(png_dir / f"{stem}.png", (255 * strip).astype(np.uint8))
        movie_path = recon_root / f"{clip_name}{recon_cfg['format']}"
        imageio.mimsave(str(movie_path), visualization_rows)

    print("Reconstruction loss: %s" % np.mean(per_frame_errors))


def animate(
    config: Mapping[str, Any],
    generator: torch.nn.Module,
    kp_detector: torch.nn.Module,
    checkpoint: str | None,
    log_dir: str,
    dataset: torch.utils.data.Dataset,
) -> None:
    if checkpoint is None:
        raise ValueError("A checkpoint path is required for animation.")

    log_path = Path(log_dir)
    animation_root = log_path / "animation"
    png_dir = animation_root / "png"
    animation_params = config["animate_params"]

    Logger.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector)
    paired = PairedDataset(dataset, animation_params["num_pairs"])
    loader = DataLoader(paired, batch_size=1, shuffle=False, num_workers=1)
    animation_root.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    visualizer = Visualizer(**config["visualizer_params"])
    use_cuda = torch.cuda.is_available()
    generator, kp_detector = _maybe_parallel_eval(generator, kp_detector)
    generator.eval()
    kp_detector.eval()

    for _, sample in tqdm(enumerate(loader)):
        driving = sample["driving_video"].cuda() if use_cuda else sample["driving_video"]
        source = sample["source_video"].cuda() if use_cuda else sample["source_video"]

        with torch.no_grad():
            frame_predictions: list[np.ndarray] = []
            visualization_rows: list[np.ndarray] = []

            kp_source = kp_detector(source[:, :, 0, :, :])
            kp_driving_t0 = kp_detector(driving[:, :, 0])

            num_frames = driving.shape[2]
            cap = animation_params.get("max_driving_frames")
            if cap is not None:
                num_frames = min(num_frames, int(cap))

            for frame_idx in range(num_frames):
                driving_frame = driving[:, :, frame_idx]
                kp_driving = kp_detector(driving_frame)
                kp_for_gen = normalize_kp(
                    kp_source,
                    kp_driving,
                    kp_driving_t0,
                    **animation_params["normalization_params"],
                )
                outputs = generator(source[:, :, 0, :, :], kp_source=kp_source, kp_driving=kp_for_gen)
                outputs.update({"kp_driving": kp_driving, "kp_source": kp_source, "kp_norm": kp_for_gen})
                outputs.pop("sparse_deformed", None)

                pred_hwc = outputs["prediction"].detach().cpu().numpy().transpose(0, 2, 3, 1)[0]
                frame_predictions.append(pred_hwc)
                visualization_rows.append(visualizer.visualize(driving_frame, source[:, :, 0, :, :], outputs))

        pair_tag = "-".join((sample["driving_name"][0], sample["source_name"][0]))
        montage = np.concatenate(frame_predictions, axis=1)
        imageio.imwrite(png_dir / f"{pair_tag}.png", (255 * montage).astype(np.uint8))
        out_movie = animation_root / f"{pair_tag}{animation_params['format']}"
        imageio.mimsave(str(out_movie), visualization_rows)
