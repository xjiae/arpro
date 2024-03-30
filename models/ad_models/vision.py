import torch
import torch.nn as nn

from .common import *

# Custom Vae model

class VaeResDown(nn.Module):
    """ Residual down sampling block for the encoder """
    def __init__(self, channel_in, channel_out, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_in, channel_out // 2, kernel_size, 2, kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(channel_out // 2, eps=1e-4)
        self.conv2 = nn.Conv2d(channel_out // 2, channel_out, kernel_size, 1, kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(channel_out, eps=1e-4)
        self.conv3 = nn.Conv2d(channel_in, channel_out, kernel_size, 2, kernel_size // 2)
        self.act_fnc = nn.ELU()

    def forward(self, x):
        skip = self.conv3(x)
        x = self.act_fnc(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return self.act_fnc(self.bn2(x + skip))


class VaeResUp(nn.Module):
    """ Residual up sampling block for the decoder """

    def __init__(self, channel_in, channel_out, kernel_size=3, scale_factor=2):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_in, channel_in // 2, kernel_size, 1, kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(channel_in // 2, eps=1e-4)
        self.conv2 = nn.Conv2d(channel_in // 2, channel_out, kernel_size, 1, kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(channel_out, eps=1e-4)
        self.conv3 = nn.Conv2d(channel_in, channel_out, kernel_size, 1, kernel_size // 2)
        self.up_nn = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = self.up_nn(x)
        skip = self.conv3(x)
        x = self.act_fnc(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return self.act_fnc(self.bn2(x + skip))


class VaeADModel(nn.Module):
    """ Vae network, uses the above encoder and decoder blocks """
    def __init__(self, in_channels=3, ch=64, latent_channels=256):
        super().__init__()
        self.ch = ch
        self.latent_channels = latent_channels

        self.enc_conv_in = nn.Conv2d(in_channels, ch, 7, 1, 3)
        self.enc_res_down_block1 = VaeResDown(ch, 2 * ch)
        self.enc_res_down_block2 = VaeResDown(2 * ch, 4 * ch)
        self.enc_res_down_block3 = VaeResDown(4 * ch, 8 * ch)
        self.enc_res_down_block4 = VaeResDown(8 * ch, 16 * ch)
        self.enc_conv_mu = nn.Conv2d(16 * ch, latent_channels, 16, 1)
        self.enc_conv_logvar = nn.Conv2d(16 * ch, latent_channels, 16, 1)
        self.enc_act_fnc = nn.ELU()

        self.dec_conv_t_up = nn.ConvTranspose2d(latent_channels, ch * 16, 16, 1)
        self.dec_res_up_block1 = VaeResUp(ch * 16, ch * 8)
        self.dec_res_up_block2 = VaeResUp(ch * 8, ch * 4)
        self.dec_res_up_block3 = VaeResUp(ch * 4, ch * 2)
        self.dec_res_up_block4 = VaeResUp(ch * 2, ch)
        self.dec_conv_out = nn.Conv2d(ch, in_channels, 3, 1, 1)
        self.dec_act_fnc = nn.ELU()

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def encode(self, x):
        x = self.enc_act_fnc(self.enc_conv_in(x)) # 256
        x = self.enc_res_down_block1(x)  # 128
        x = self.enc_res_down_block2(x)  # 64
        x = self.enc_res_down_block3(x)  # 32
        x = self.enc_res_down_block4(x)  # 16

        mu = self.enc_conv_mu(x)  # 1
        logvar = self.enc_conv_logvar(x)  # 1
        z = self.sample(mu, logvar) if self.training else mu
        return z, mu, logvar
    
    def decode(self, z):
        z = self.dec_act_fnc(self.dec_conv_t_up(z))  # 16
        z = self.dec_res_up_block1(z)  # 32
        z = self.dec_res_up_block2(z)  # 64
        z = self.dec_res_up_block3(z)  # 128
        z = self.dec_res_up_block4(z)  # 256
        xhat = self.dec_conv_out(z)
        # xhat = torch.tanh(self.dec_conv_out(z))
        return xhat

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        xhat = self.decode(z)
        alpha = (x - xhat).abs()
        score = alpha.flatten(1).norm(p=2, dim=-1)

        return ADModelOutput(
            score = score,
            alpha = alpha,
            other = {
                "xhat": xhat,
                "mu": mu,
                "logvar": logvar
            }
        )


