#!/usr/bin/env python3
"""
Reconstruction evaluation harness (course extension).

Computes per-test-clip and pooled metrics without writing GIFs:
  - mae_per_frame: mean |pred - gt| in image space (same as training-time range)
  - temporal_pred: mean over time of mean spatial |pred_{t+1} - pred_t|
  - temporal_gt: same for ground-truth frames
  - temporal_ratio: temporal_pred / (temporal_gt + 1e-8)  (>1 often means extra flicker vs GT motion)

Run from project root:
  python scripts/eval_reconstruction_metrics.py \\
    --config config/mgif-e2e-smoke.yaml \\
    --checkpoint logs/e2e/<run>/00000000-checkpoint.pth.tar \\
    --max_videos 20 --out_csv logs/e2e/metrics_baseline.csv

Use two checkpoints (baseline vs ablation) and compare CSVs in the report.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

NC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NC))

from fom.frames_dataset import FramesDataset
from fom.logger import Logger
from fom.modules.generator import OcclusionAwareGenerator
from fom.modules.keypoint_detector import KPDetector
from fom.sync_batchnorm import DataParallelWithCallback


@torch.no_grad()
def _clip_metrics(generator, kp_detector, vid: torch.Tensor) -> dict:
    """vid: 1 x 3 x T x H x W"""
    kp0 = kp_detector(vid[:, :, 0])
    preds, drvs = [], []
    t = vid.shape[2]
    for fi in range(t):
        src, drv = vid[:, :, 0], vid[:, :, fi]
        kd = kp_detector(drv)
        o = generator(src, kp_source=kp0, kp_driving=kd)
        o.pop("sparse_deformed", None)
        preds.append(o["prediction"])
        drvs.append(drv)
    P = torch.stack(preds, dim=2)  # 1,3,T,H,W
    D = torch.stack(drvs, dim=2)

    mae = (P - D).abs().mean().item()

    if t < 2:
        tp = tg = 0.0
        ratio = float("nan")
    else:
        dtp = (P[:, :, 1:] - P[:, :, :-1]).abs().mean(dim=(1, 2, 3, 4)).item()
        dtg = (D[:, :, 1:] - D[:, :, :-1]).abs().mean(dim=(1, 2, 3, 4)).item()
        tp, tg = dtp, dtg
        ratio = tp / (tg + 1e-8)

    return {"mae_mean": mae, "temporal_pred_mae": tp, "temporal_gt_mae": tg, "temporal_ratio": ratio, "num_frames": t}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="YAML config (must match checkpoint architecture)")
    ap.add_argument("--checkpoint", required=True, help="Path to .pth.tar")
    ap.add_argument("--max_videos", type=int, default=None, help="Cap number of test clips (default: all)")
    ap.add_argument("--out_csv", default=None, help="Write per-video rows to this CSV")
    ap.add_argument("--device_ids", default="0", type=lambda s: list(map(int, s.split(","))))
    opt = ap.parse_args()

    cfg_path = os.path.abspath(opt.config)
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)

    mp = cfg["model_params"]
    cuda = torch.cuda.is_available()
    generator = OcclusionAwareGenerator(**mp["generator_params"], **mp["common_params"])
    kp_detector = KPDetector(**mp["kp_detector_params"], **mp["common_params"])
    if cuda:
        generator.to(opt.device_ids[0])
        kp_detector.to(opt.device_ids[0])

    Logger.load_cpk(opt.checkpoint, generator=generator, kp_detector=kp_detector)
    if cuda:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)
    generator.eval()
    kp_detector.eval()

    ds = FramesDataset(is_train=False, **cfg["dataset_params"])
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    rows = []
    for it, x in tqdm(enumerate(dl), total=len(dl) if opt.max_videos is None else min(opt.max_videos, len(dl))):
        if opt.max_videos is not None and it >= opt.max_videos:
            break
        vid = x["video"].cuda() if cuda else x["video"]
        name = x["name"][0] if isinstance(x["name"], (list, tuple)) else x["name"]
        m = _clip_metrics(generator, kp_detector, vid)
        m["name"] = name
        rows.append(m)

    if not rows:
        print("No clips evaluated (empty dataset or max_videos=0).")
        return

    maes = [r["mae_mean"] for r in rows]
    ratios = [r["temporal_ratio"] for r in rows if not np.isnan(r["temporal_ratio"])]

    print("\n=== Reconstruction metrics (%d clips) ===" % len(rows))
    print("mae_mean pooled: %.6f (median %.6f, std %.6f)" % (float(np.mean(maes)), float(np.median(maes)), float(np.std(maes))))
    if ratios:
        print("temporal_ratio pooled: %.4f (median %.4f)  [>1 suggests more frame-to-frame change than GT]" % (float(np.nanmean(ratios)), float(np.median(ratios))))

    if opt.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(opt.out_csv)) or ".", exist_ok=True)
        with open(opt.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["name", "mae_mean", "temporal_pred_mae", "temporal_gt_mae", "temporal_ratio", "num_frames"])
            w.writeheader()
            w.writerows(rows)
        print("Wrote", opt.out_csv)


if __name__ == "__main__":
    main()
