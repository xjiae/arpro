import torch
import torch.nn as nn

from ..common import *
from diffusers.models import AutoencoderKL



class VaeFixerModel(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        **kwargs
    ):
        super().__init__()
        self.image_channels = image_channels
        self.in_channels = 3 * image_channels   # image, alpha, omega
        self.vae = AutoencoderKL(
            in_channels = in_channels,
            out_channels = image_channels,
            **kwargs
        )

    def forward(
        self,
        x_bad: torch.FloatTensor,
        alpha: torch.FloatTensor,
        omega: torch.LongTensor
    ):
        xx = torch.cat([x_bad, alpha, omega.float()], dim=1)
        enc = self.vae.encode(xx)
        mu, logvar = enc.latent_dist.mean, enc.latent_dist.logvar

        if self.training:
            z = mu + (0.5 * logvar).exp() * torch.randn_like(logvar)
        else:
            z = mu

        dec = self.vae.decode(z)
        x_fix = dec.sample
        return FixerModelOutput(
            x_fix = x_fix,
            others = {
                "z": z,
                "mu": mu,
                "logvar": logvar,
                "enc": enc,
                "dec": dec
            }
        )



