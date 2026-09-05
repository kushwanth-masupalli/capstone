"""
DCGAN for chest X-ray synthetic augmentation (one GAN per class)
================================================================

Generator:  latent z (default 100) -> 224x224 single-channel image, tanh output
            (values in [-1, 1], matching the normalized training images).
Discriminator: 224x224 single-channel image -> scalar logit (BCEWithLogitsLoss,
            no sigmoid inside the model).

Both use the standard DCGAN recipe (ConvTranspose/Conv with stride 2,
BatchNorm, ReLU / LeakyReLU, Adam betas (0.5, 0.999)).

Usage:
    from src.gan.dcgan import Generator, Discriminator
    G = Generator(latent_dim=100)
    D = Discriminator()
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Maps z (latent_dim,) -> (1, 224, 224) in [-1, 1].

    224 = 7 * 2^5, so we start from a 7x7 feature map and upsample 5 times.
    """

    def __init__(self, latent_dim=100, ngf=64, out_channels=1):
        super().__init__()
        self.latent_dim = latent_dim
        # 7x7 base feature map
        self.init = nn.Sequential(
            nn.Linear(latent_dim, ngf * 8 * 7 * 7),
            nn.BatchNorm1d(ngf * 8 * 7 * 7),
            nn.ReLU(True),
        )
        self.main = nn.Sequential(
            # 7 -> 14
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 14 -> 28
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 28 -> 56
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 56 -> 112
            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),
            # 112 -> 224
            nn.ConvTranspose2d(ngf // 2, out_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.init(z)
        out = out.view(z.size(0), -1, 7, 7)
        return self.main(out)


class Discriminator(nn.Module):
    """
    Maps (1, 224, 224) -> scalar logit.

    224 -> 112 -> 56 -> 28 -> 14 -> 7, then flatten + Linear.
    """

    def __init__(self, ndf=64, in_channels=1):
        super().__init__()
        self.main = nn.Sequential(
            # 224 -> 112
            nn.Conv2d(in_channels, ndf // 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 112 -> 56
            nn.Conv2d(ndf // 2, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # 56 -> 28
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 28 -> 14
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 14 -> 7
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(ndf * 8 * 7 * 7, 1)

    def forward(self, x):
        out = self.main(x)
        return self.fc(out.flatten(1))