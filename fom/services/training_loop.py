"""
Training session: one entrypoint that wires dataloaders, composite loss models, and logging.

The optimization pattern (generator + KP step, then optional discriminator step) follows
FOMM; splitting it here isolates orchestration from raw ``nn.Module`` definitions in
``fom.modules``.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import trange

from ..data.datasets import DatasetRepeater
from ..logger import Logger
from ..modules.model import DiscriminatorFullModel, GeneratorFullModel
from ..sync_batchnorm import DataParallelWithCallback

_NUM_DATALOADER_WORKERS = 6


def train(
    config: Mapping[str, Any],
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    kp_detector: torch.nn.Module,
    checkpoint: str | None,
    log_dir: str,
    dataset: torch.utils.data.Dataset,
    device_ids: list[int],
) -> None:
    train_cfg = config["train_params"]

    opt_generator = torch.optim.Adam(generator.parameters(), lr=train_cfg["lr_generator"], betas=(0.5, 0.999))
    opt_discriminator = torch.optim.Adam(
        discriminator.parameters(), lr=train_cfg["lr_discriminator"], betas=(0.5, 0.999)
    )
    opt_keypoints = torch.optim.Adam(kp_detector.parameters(), lr=train_cfg["lr_kp_detector"], betas=(0.5, 0.999))

    if checkpoint:
        start_epoch = Logger.load_cpk(
            checkpoint,
            generator,
            discriminator,
            kp_detector,
            opt_generator,
            opt_discriminator,
            None if train_cfg["lr_kp_detector"] == 0 else opt_keypoints,
        )
    else:
        start_epoch = 0

    milestones = train_cfg["epoch_milestones"]
    sched_generator = MultiStepLR(opt_generator, milestones, gamma=0.1, last_epoch=start_epoch - 1)
    sched_discriminator = MultiStepLR(opt_discriminator, milestones, gamma=0.1, last_epoch=start_epoch - 1)
    kp_warm = -1 + start_epoch * (train_cfg["lr_kp_detector"] != 0)
    sched_keypoints = MultiStepLR(opt_keypoints, milestones, gamma=0.1, last_epoch=kp_warm)

    if train_cfg.get("num_repeats", 1) != 1:
        dataset = DatasetRepeater(dataset, train_cfg["num_repeats"])

    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=_NUM_DATALOADER_WORKERS,
        drop_last=True,
    )

    full_generator = GeneratorFullModel(kp_detector, generator, discriminator, train_cfg)
    full_discriminator = DiscriminatorFullModel(kp_detector, generator, discriminator, train_cfg)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        full_generator = DataParallelWithCallback(full_generator, device_ids=device_ids)
        full_discriminator = DataParallelWithCallback(full_discriminator, device_ids=device_ids)

    loss_weights = train_cfg["loss_weights"]
    visual_cfg = config["visualizer_params"]

    with Logger(log_dir, checkpoint_freq=train_cfg["checkpoint_freq"], visualizer_params=visual_cfg) as logger:
        for epoch in trange(start_epoch, train_cfg["num_epochs"]):
            last_batch = None
            last_gen_out = None

            for batch in loader:
                last_batch = batch

                gen_losses, last_gen_out = full_generator(batch)
                total_gen = sum(v.mean() for v in gen_losses.values())
                total_gen.backward()
                opt_generator.step()
                opt_generator.zero_grad()
                opt_keypoints.step()
                opt_keypoints.zero_grad()

                if loss_weights["generator_gan"] != 0:
                    opt_discriminator.zero_grad()
                    disc_losses = full_discriminator(batch, last_gen_out)
                    total_disc = sum(v.mean() for v in disc_losses.values())
                    total_disc.backward()
                    opt_discriminator.step()
                    opt_discriminator.zero_grad()
                else:
                    disc_losses = {}
                gen_losses.update(disc_losses)

                logger.log_iter({k: v.mean().detach().cpu().numpy() for k, v in gen_losses.items()})

            sched_generator.step()
            sched_discriminator.step()
            sched_keypoints.step()

            state = {
                "generator": generator,
                "discriminator": discriminator,
                "kp_detector": kp_detector,
                "optimizer_generator": opt_generator,
                "optimizer_discriminator": opt_discriminator,
                "optimizer_kp_detector": opt_keypoints,
            }
            logger.log_epoch(epoch, state, inp=last_batch, out=last_gen_out)
