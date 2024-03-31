import torch
import torch.nn as nn

from .common import *
from diffusers.models import AutoencoderKL


# Custom Vae model

class VaeADModel(nn.Module):
    """ A little wrapper around VQ-VAE from diffusers """
    def __init__(self, in_channels=3, out_channels=3, **kwargs):
        super().__init__()
        self.vae = AutoencoderKL(in_channels=in_channels, out_channels=out_channels, **kwargs)

    def encode(self, x: torch.FloatTensor):
        enc = self.vae.encode(x)
        mu, logvar = enc.latent_dist.mean, enc.latent_dist.logvar
        return mu, logvar

    def decode(self, z: torch.FloatTensor):
        dec = self.vae.decode(z)
        return dec.sample

    def forward(self, x):
        mu, logvar = self.encode(x)

        if self.training:
            z = mu + (0.5 * logvar).exp() * torch.randn_like(logvar)
        else:
            z = mu

        x_recon = self.decode(z)
        alpha = x - x_recon
        score = alpha.norm(p=2, dim=(1,2,3)) ** 2
        return ADModelOutput(
            score = score,
            alpha = alpha,
            other = {
                "x_recon": x_recon,
                "mu": mu,
                "logvar": logvar,
            }
        )


