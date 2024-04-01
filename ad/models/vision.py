import torch
import torch.nn as nn

from .common import *
from diffusers import AutoencoderKL


class VaeADModel(nn.Module):
    """ A little wrapper around AutoencoderKL Vae from diffusers """
    def __init__(self, image_channels: int = 3, **kwargs):
        super().__init__()
        self.vae = AutoencoderKL(
            in_channels = image_channels,
            out_channels = image_channels,
            **kwargs
        )

    def forward(self, x: torch.FloatTensor):
        enc = self.vae.encode(x)
        mu, logvar = enc.latent_dist.mean, enc.latent_dist.logvar

        if self.training:
            z = mu + (0.5 * logvar).exp() * torch.randn_like(logvar)
        else:
            z = mu

        dec = self.vae.decode(z)
        x_recon = dec.sample
        alpha = (x - x_recon).abs()
        score = alpha.norm(p=2, dim=(1,2,3)) ** 2
        return ADModelOutput(
            score = score,
            alpha = alpha,
            others = {
                "x_recon": x_recon,
                "z": z,
                "mu": mu,
                "logvar": logvar,
                "enc": enc,
                "dec": dec,
            }
        )


