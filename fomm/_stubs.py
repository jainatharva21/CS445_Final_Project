"""Dummy models that satisfy the contract train.py expects.

ONLY for verifying the data + training pipeline runs end-to-end. Delete once
your teammates' real model code (modules/) is in place.
"""
import torch
import torch.nn as nn


class StubGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class StubDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class StubKPDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Conv2d(3, 10, 1)

    def forward(self, x):
        return self.net(x)


class StubGeneratorFull(nn.Module):
    """Returns (losses_dict, generated_dict). Mimics the real GeneratorFullModel."""

    def __init__(self, generator, discriminator, kp_detector):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.kp_detector = kp_detector

    def forward(self, batch):
        src, drv = batch['source'], batch['driving']
        # Use kp_detector so its parameters get gradients.
        kp_pen = self.kp_detector(drv).abs().mean() * 1e-3
        prediction = self.generator(src)
        recon = ((prediction - drv) ** 2).mean()
        losses = {'recon': recon, 'kp_reg': kp_pen}
        return losses, {'prediction': prediction}


class StubDiscriminatorFull(nn.Module):
    """Returns losses_dict only. Mimics the real DiscriminatorFullModel."""

    def __init__(self, generator, discriminator, kp_detector):
        super().__init__()
        self.discriminator = discriminator

    def forward(self, batch, generated):
        drv = batch['driving']
        d_real = self.discriminator(drv).mean()
        d_fake = self.discriminator(generated['prediction'].detach()).mean()
        return {'gan': (1.0 - d_real).clamp(min=0).mean() + d_fake.clamp(min=0).mean()}
