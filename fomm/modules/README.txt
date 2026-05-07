Module ownership map (from team responsibilities document)
===========================================================

Person 1 (data + orchestration) — top level files:
    frames_dataset.py, augmentation.py, train.py, logger.py, run.py, config/*.yaml

Person 2 (core networks, forward pass only):
    modules/keypoint_detector.py    -> class KPDetector(...)
    modules/dense_motion.py         -> class DenseMotionNetwork(...)   (used inside generator)
    modules/generator.py            -> class OcclusionAwareGenerator(...)
    modules/util.py                 -> ResBlock, Hourglass, grid helpers, etc.

Person 3 (losses, GAN, eval):
    modules/discriminator.py        -> class MultiScaleDiscriminator(...)
    modules/model.py                -> class GeneratorFullModel(kp_detector, generator, discriminator, train_params)
                                       class DiscriminatorFullModel(kp_detector, generator, discriminator, train_params)
    reconstruction.py               -> def reconstruction(config, generator, kp_detector, checkpoint, log_dir, dataset)
    animate.py                      -> def animate(config, generator, kp_detector, checkpoint, log_dir, dataset)
                                       def normalize_kp(...)   (relative vs absolute kp transfer)
    demo.py                         -> CLI for one-off animation


Contracts that train.py and run.py rely on
==========================================

run.py imports and constructs:
    KPDetector(**kp_detector_params, **common_params)
    OcclusionAwareGenerator(**generator_params, **common_params)
    MultiScaleDiscriminator(**discriminator_params, **common_params)
    GeneratorFullModel(kp_detector, generator, discriminator, train_params)
    DiscriminatorFullModel(kp_detector, generator, discriminator, train_params)

The Full* wrappers must satisfy the train.py contract:

    losses_g, generated = generator_full(batch)
        batch:     {'source': [B,3,H,W], 'driving': [B,3,H,W], 'name': [...]}
        losses_g:  dict[str, Tensor scalar]   (each value is a loss term to sum)
        generated: dict[str, Tensor]          (must include 'prediction' = [B,3,H,W])

    losses_d = discriminator_full(batch, generated)
        losses_d:  dict[str, Tensor scalar]
        (must detach generated['prediction'] internally before passing to the discriminator)

reconstruction.py / animate.py contract (called by run.py):

    reconstruction(config, generator, kp_detector, checkpoint, log_dir, dataset)
        dataset is a FramesDataset(sampling_mode='full') over the test split.

    animate(config, generator, kp_detector, checkpoint, log_dir, dataset)
        dataset is a PairedDataset wrapping FramesDataset(sampling_mode='full').

Until modules/ is in place, run.py works with --stub-models for pipeline testing.
