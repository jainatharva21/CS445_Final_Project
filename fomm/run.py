import argparse

import torch
import yaml

from frames_dataset import FramesDataset, PairedDataset


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f'Config root must be a mapping: {path}')
    return cfg


def build_models(config, device):
    """Build the real models. Imports come from teammates' modules."""
    try:
        from modules.generator import OcclusionAwareGenerator       # Person 2
        from modules.discriminator import MultiScaleDiscriminator   # Person 3
        from modules.keypoint_detector import KPDetector            # Person 2
        from modules.model import GeneratorFullModel, DiscriminatorFullModel  # Person 3
    except ImportError as e:
        raise ImportError(
            "Model modules not yet available. For pipeline testing run with --stub-models. "
            f"Underlying error: {e}"
        )

    mp = config['model_params']
    common = mp['common_params']
    generator = OcclusionAwareGenerator(**mp['generator_params'], **common).to(device)
    discriminator = MultiScaleDiscriminator(**mp['discriminator_params'], **common).to(device)
    kp_detector = KPDetector(**mp['kp_detector_params'], **common).to(device)
    generator_full = GeneratorFullModel(kp_detector, generator, discriminator, config['train_params'])
    discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, config['train_params'])
    return generator, discriminator, kp_detector, generator_full, discriminator_full


def build_stub_models(config, device):
    from _stubs import (
        StubGenerator, StubDiscriminator, StubKPDetector,
        StubGeneratorFull, StubDiscriminatorFull,
    )
    generator = StubGenerator().to(device)
    discriminator = StubDiscriminator().to(device)
    kp_detector = StubKPDetector().to(device)
    generator_full = StubGeneratorFull(generator, discriminator, kp_detector)
    discriminator_full = StubDiscriminatorFull(generator, discriminator, kp_detector)
    return generator, discriminator, kp_detector, generator_full, discriminator_full


def _build(config, device, use_stubs):
    return build_stub_models(config, device) if use_stubs else build_models(config, device)


def cmd_train(config, args):
    from train import train
    g, d, kp, gf, df = _build(config, args.device, args.stub_models)
    train(config, g, d, kp, gf, df, device=args.device)


def cmd_reconstruct(config, args):
    """Video reconstruction proxy task. Dispatches to Person 3's reconstruction.py."""
    try:
        from reconstruction import reconstruction
    except ImportError as e:
        print(f"reconstruction.py not yet implemented (Person 3's deliverable). Error: {e}")
        return

    g, _, kp, _, _ = _build(config, args.device, args.stub_models)
    dp = config['dataset_params']
    dataset = FramesDataset(
        root_dir=dp['root_dir'],
        frame_shape=tuple(dp['frame_shape']),
        is_train=False,
        sampling_mode='full',
    )
    log_dir = config['train_params']['log_dir']
    reconstruction(config, g, kp, args.checkpoint, log_dir, dataset)


def cmd_animate(config, args):
    """Image animation. Dispatches to Person 3's animate.py."""
    try:
        from animate import animate
    except ImportError as e:
        print(f"animate.py not yet implemented (Person 3's deliverable). Error: {e}")
        return

    g, _, kp, _, _ = _build(config, args.device, args.stub_models)
    dp = config['dataset_params']
    base = FramesDataset(
        root_dir=dp['root_dir'],
        frame_shape=tuple(dp['frame_shape']),
        is_train=False,
        sampling_mode='full',
    )
    num_pairs = config.get('animate_params', {}).get('num_pairs', 50)
    dataset = PairedDataset(base, num_pairs=num_pairs)
    log_dir = config['train_params']['log_dir']
    animate(config, g, kp, args.checkpoint, log_dir, dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=['train', 'reconstruct', 'animate'], default='train')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint', default=None,
                        help='Checkpoint path (used by reconstruct / animate).')
    parser.add_argument('--stub-models', action='store_true',
                        help='Use dummy models (only for verifying the pipeline).')
    args = parser.parse_args()

    config = load_config(args.config)
    {'train': cmd_train, 'reconstruct': cmd_reconstruct, 'animate': cmd_animate}[args.mode](config, args)


if __name__ == '__main__':
    main()
