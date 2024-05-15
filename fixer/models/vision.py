from typing import Optional, Tuple
import torch
import torch.nn as nn
from tqdm import tqdm

from .common import *
from diffusers.models import AutoencoderKL, UNet2DModel
from diffusers import DDPMPipeline, DDPMScheduler


class VaeFixerModel(nn.Module):
    def __init__(self, image_channels: int = 3, **kwargs):
        super().__init__()
        self.image_channels = image_channels
        self.vae_in_channels = image_channels + 1   # masked_image, omega
        self.vae = AutoencoderKL(
            in_channels = self.vae_in_channels,
            out_channels = image_channels,
            **kwargs
        )

    def forward(self, x_goods: torch.FloatTensor, anom_parts: torch.LongTensor):
        """ (x_goods == 0) ~ anom_parts """
        xx = torch.cat([x_goods, anom_parts.float()], dim=1)
        enc = self.vae.encode(xx)
        mu, logvar = enc.latent_dist.mean, enc.latent_dist.logvar

        if self.training: z = mu + (0.5 * logvar).exp() * torch.randn_like(logvar)
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


class MyDiffusionModel(nn.Module):
    """
    This is a wrapper around a DDPM diffusion pipeline for convenience.
    We should be able to do the following tasks:
        * Unconditional image generation
        * "Conditional" image generation (i.e. starting from initial x)
        * Add noise: generate x_t given x_0
        * Estimate noise: estimate the noise used
        * Backward diffusion step: generate x_{t-1} from x_t
    """
    def __init__(
        self,
        image_size: int = 256,
        image_channels: int = 3,
        **kwargs
    ):
        super().__init__()
        self.image_size = image_size
        self.image_channels = image_channels

        # Taken from: https://huggingface.co/docs/diffusers/en/tutorials/basic_training
        self.unet = UNet2DModel(
            sample_size = image_size,
            in_channels = image_channels,
            out_channels = image_channels,
            layers_per_block = 2,
            block_out_channels = (128, 128, 256, 256, 512, 512),
            down_block_types = (
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ),
            up_block_types = (
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

        self.scheduler = DDPMScheduler(**kwargs)

    @property
    def num_timesteps(self):
        return len(self.scheduler.timesteps)

    def add_noise(self, x: torch.FloatTensor, noise: torch.FloatTensor, t: torch.LongTensor):
        return self.scheduler.add_noise(x, noise, t)

    def estimate_noise(self, xt, t):
        return self.unet(xt, t).sample

    def forward(
        self,
        x: Optional[torch.FloatTensor] = None,
        t: Optional[torch.FloatTensor] = None,
        batch_size: Optional[int] = None,
        num_inference_steps: int = 1000,
        progress_bar: bool = False,
        enable_grad: bool = False
    ):
        """
        Copied code from the __call__ function of DDPM pipeline
        """

        # If the seed is not provided, start from some random stuff
        if x is None:
            assert batch_size is not None
            image = torch.randn(
                batch_size, self.image_channels, self.image_size, self.image_size,
                device = next(self.unet.parameters()).device
            )

        # Otherwise we start from the seed x, and noise is to t steps
        else:
            noise = torch.randn_like(x)
            if t is None:
                t = torch.LongTensor([self.num_timesteps-1]).to(x.device)
            elif isinstance(t, int):
                t = torch.LongTensor([t]).to(x.device)

            image = self.add_noise(x, noise, t)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        pbar = tqdm(self.scheduler.timesteps) if progress_bar else self.scheduler.timesteps

        with torch.set_grad_enabled(enable_grad):
            for t in pbar:
                # 1. predict nosie model_output
                noise_pred = self.estimate_noise(image, t)

                # 2. compute previous image: x_{t} -> x_{t-1}
                image = self.scheduler.step(noise_pred, t, image).prev_sample

        return image


