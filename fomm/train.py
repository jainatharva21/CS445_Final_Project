import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from frames_dataset import FramesDataset
from logger import Logger


def _move_to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def train(config, generator, discriminator, kp_detector,
          generator_full, discriminator_full, device='cuda'):
    """Train FOMM. Models are passed in pre-built — built by run.py.

    `generator_full(batch)` must return `(losses_dict, generated_dict)`.
    `discriminator_full(batch, generated_dict)` must return `losses_dict`.
    Pass `discriminator=None` and `discriminator_full=None` to disable the GAN step.
    """
    dp = config['dataset_params']
    tp = config['train_params']

    dataset = FramesDataset(
        root_dir=dp['root_dir'],
        frame_shape=tuple(dp['frame_shape']),
        is_train=True,
        id_sampling=dp.get('id_sampling', False),
        augmentation_params=dp.get('augmentation_params'),
        sampling_mode='pair',
    )
    dataloader = DataLoader(
        dataset,
        batch_size=tp['batch_size'],
        shuffle=True,
        num_workers=tp.get('num_workers', 4),
        drop_last=True,
        pin_memory=(device != 'cpu'),
    )

    opt_generator = torch.optim.Adam(
        generator.parameters(), lr=tp['lr_generator'], betas=(0.5, 0.999)
    )
    opt_kp = torch.optim.Adam(
        kp_detector.parameters(), lr=tp['lr_kp_detector'], betas=(0.5, 0.999)
    )
    opt_discriminator = None
    if discriminator is not None:
        opt_discriminator = torch.optim.Adam(
            discriminator.parameters(), lr=tp['lr_discriminator'], betas=(0.5, 0.999)
        )

    total_iters = tp['num_iters']
    milestones = [int(total_iters * 0.5), int(total_iters * 0.75)]
    sched_g = MultiStepLR(opt_generator, milestones=milestones, gamma=0.1)
    sched_kp = MultiStepLR(opt_kp, milestones=milestones, gamma=0.1)
    sched_d = MultiStepLR(opt_discriminator, milestones=milestones, gamma=0.1) if opt_discriminator else None

    logger = Logger(tp['log_dir'], log_freq=tp.get('log_freq', 100))

    start_step = 0
    ckpt = logger.load_checkpoint(map_location=device)
    if ckpt is not None:
        start_step = ckpt['step'] + 1
        generator.load_state_dict(ckpt['generator'])
        kp_detector.load_state_dict(ckpt['kp_detector'])
        opt_generator.load_state_dict(ckpt['opt_generator'])
        opt_kp.load_state_dict(ckpt['opt_kp'])
        if discriminator is not None and 'discriminator' in ckpt:
            discriminator.load_state_dict(ckpt['discriminator'])
            opt_discriminator.load_state_dict(ckpt['opt_discriminator'])
        print(f'Resumed from step {start_step}')

    image_freq = tp.get('image_freq', 1000)
    ckpt_freq = tp.get('ckpt_freq', 5000)

    generator.train()
    kp_detector.train()
    if discriminator is not None:
        discriminator.train()

    step = start_step
    while step < total_iters:
        for batch in dataloader:
            if step >= total_iters:
                break
            batch = _move_to_device(batch, device)

            losses_g, generated = generator_full(batch)
            loss_g = sum(losses_g.values())
            loss_g.backward()
            opt_generator.step()
            opt_kp.step()
            opt_generator.zero_grad()
            opt_kp.zero_grad()

            losses_d = {}
            if discriminator_full is not None and opt_discriminator is not None:
                opt_discriminator.zero_grad()
                losses_d = discriminator_full(batch, generated)
                loss_d = sum(losses_d.values())
                loss_d.backward()
                opt_discriminator.step()

            scalars = {f'g/{k}': v for k, v in losses_g.items()}
            scalars.update({f'd/{k}': v for k, v in losses_d.items()})
            logger.log_scalars(scalars, step)

            if step > 0 and step % image_freq == 0:
                with torch.no_grad():
                    src = batch['source'][:4]
                    drv = batch['driving'][:4]
                    logger.log_images(torch.cat([src, drv], dim=-1), step, name='input')
                    if isinstance(generated, dict) and 'prediction' in generated:
                        pred = generated['prediction'][:4].clamp(0, 1)
                        logger.log_images(pred, step, name='prediction')

            if step > 0 and step % ckpt_freq == 0:
                state = {
                    'generator': generator.state_dict(),
                    'kp_detector': kp_detector.state_dict(),
                    'opt_generator': opt_generator.state_dict(),
                    'opt_kp': opt_kp.state_dict(),
                }
                if discriminator is not None:
                    state['discriminator'] = discriminator.state_dict()
                    state['opt_discriminator'] = opt_discriminator.state_dict()
                logger.save_checkpoint(state, step)

            sched_g.step()
            sched_kp.step()
            if sched_d is not None:
                sched_d.step()
            step += 1

    logger.close()
