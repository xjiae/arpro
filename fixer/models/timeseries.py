import torch
import torch.nn as nn
from tqdm import tqdm
from functools import partial
from typing import Optional, Tuple
from diffusers import DDPMPipeline, DDPMScheduler
from transformers.models.gpt2.modeling_gpt2 import GPT2Model

from .common import *
from .timeseries_external.gaussian_diffusion import Diffusion_TS


class MyTimeDiffusionModel(nn.Module):
    def __init__(
        self,
        window_size: int,
        feature_dim: int,
        **kwargs
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.window_size = window_size
        self.dts_model = Diffusion_TS(seq_length=window_size, feature_size=feature_dim, **kwargs)
        self.num_timesteps = self.dts_model.num_timesteps

    def compute_loss(self, x):
        """ Does some computation with the Fourier loss """
        return self.dts_model(x)

    def add_noise(self, x: torch.FloatTensor, noise: torch.FloatTensor, t: torch.LongTensor, **kwargs):
        return self.dts_model.q_sample(x, t, noise, **kwargs)

    def estimate_noise_and_x0(self, xt, t, **kwargs):
        """ returns (noise_pred, x0_pred) """
        return self.dts_model.model_predictions(xt, t, **kwargs)

    @torch.no_grad()
    def forward(
        self,
        x: Optional[torch.FloatTensor] = None,
        t: Optional[torch.FloatTensor] = None,
        num_inference_steps: Optional[int] = None,
        batch_size: Optional[int] = None,
        clip_denoised: bool = True,
        progress_bar: bool = False,
    ):
        """ Adapted from Diffusion_TS.fast_sample """
        device = next(self.dts_model.parameters()).device
        # Unconditional generation
        if x is None:
            batch_size = 1 if batch_size is None else batch_size
            signal = torch.randn(batch_size, self.window_size, self.feature_dim, device=device)

        # Conditional generation
        else:
            noise = torch.randn_like(x)
            if t is None:
                t = torch.LongTensor([self.num_timesteps-1]).to(x.device)
            elif isinstance(t, int):
                t = torch.LongTensor([t]).to(x.device)
            signal = self.add_noise(x, noise, t)

        batch_size = signal.size(0)

        # Set step values
        eta = self.dts_model.eta
        sampling_timesteps = \
            self.dts_model.sampling_timesteps if num_inference_steps is None else num_inference_steps

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == num_timesteps
        times = torch.linspace(-1, self.num_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        pbar = tqdm(time_pairs, desc="sampling loop time steps") if progress_bar else time_pairs
        for time, time_next in pbar:
            time_cond = torch.full((batch_size,), time, device=device, dtype=torch.long)
            pred_noise, x0_pred = self.estimate_noise_and_x0(signal, time_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                signal = x0_pred
                continue

            alpha = self.dts_model.alphas_cumprod[time]
            alpha_next = self.dts_model.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(signal)
            signal = x0_pred * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
        
        return signal
    
    def get_prev_sample(
        self, 
        signal: torch.FloatTensor, 
        time: int,
        time_next: int,
        time_cond: torch.LongTensor,
        target: torch.FloatTensor,
        partial_mask: torch.LongTensor,
        clip_denoised: Optional[bool] = True,
        model_kwargs: Optional[dict] = None
    ):
        pred_noise, x_start, *_ = self.dts_model.model_predictions(signal, time_cond, clip_x_start=clip_denoised)
        if time_next < 0:
            return x_start
        alpha = self.dts_model.alphas_cumprod[time]
        alpha_next = self.dts_model.alphas_cumprod[time_next]
        sigma = self.dts_model.eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()
        pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
        noise = torch.randn_like(signal)

        signal = pred_mean + sigma * noise
        signal = self.dts_model.langevin_fn(sample=signal, mean=pred_mean, sigma=sigma, t=time_cond,
                                tgt_embs=target, partial_mask=partial_mask, **model_kwargs)
        target_t = self.dts_model.q_sample(target, t=time_cond)
        signal[partial_mask] = target_t[partial_mask]

        return signal

    @torch.no_grad()
    def repair(
        self,
        x: torch.FloatTensor = None,
        target: torch.FloatTensor = None,
        partial_mask: torch.LongTensor = None,
        noise_level: int = 500,
        num_inference_steps: int = 1000,
        clip_denoised: Optional[bool] = True,
        model_kwargs: Optional[dict] = None
    ):
        """ Adapted from Diffusion_TS.sample """
        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        noise = torch.randn_like(x)
        signal = self.add_noise(x, noise, torch.tensor([noise_level]).to(x.device))
        times = reversed(range(0, num_inference_steps))
        for time in tqdm(times, desc='baseline'):
            signal = self.dts_model.p_sample_infill(x, target=target, t=time, partial_mask=partial_mask, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
            # signal, _ = self.dts_model.p_sample(signal, time)
        # signal[partial_mask] = target[partial_mask]
        return signal


    def prev_sample_with_grad_update(
        self,
        x,
        target,
        t: int,
        grad_update,
        partial_mask=None,
        clip_denoised=True,
        model_kwargs=None
    ):
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, _ = \
            self.dts_model.p_mean_variance(x=x, t=batched_times, clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        sigma = (0.5 * model_log_variance).exp()
        pred_img = model_mean + sigma * noise
        # apply the update
        pred_img = pred_img - grad_update
        pred_img = self.dts_model.langevin_fn(sample=pred_img, mean=model_mean, sigma=sigma, t=batched_times,
                                    tgt_embs=target, partial_mask=partial_mask, **model_kwargs)
        
        target_t = self.dts_model.q_sample(target, t=batched_times)
        pred_img[partial_mask] = target_t[partial_mask]
        # return pred_img
        # batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        # model_mean, _, model_log_variance, x_start = \
        #     self.dts_model.p_mean_variance(x=x, t=batched_times, clip_denoised=clip_denoised)
        # noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        # pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        # pred_img += grad_update
        return pred_img

    
    


### Some custom stuff

class MyTSDiffusionModel(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        **kwargs
    ):
        super().__init__()
        self.feature_dim = feature_dim

        # Load GPT2
        self.gpt2 = GPT2Model.from_pretrained("gpt2", output_attentions=True, output_hidden_states=True)
        self.embed_dim = self.gpt2.wte.embedding_dim    # 768
        self.max_window_size = self.gpt2.wpe.num_embeddings # 1024
        

        # Set up some stuff with the scheduler
        self.scheduler = DDPMScheduler(**kwargs)
        self.num_timesteps = len(self.scheduler.timesteps)

        # Embedding functions
        self.embed_input = nn.Linear(feature_dim, self.embed_dim)
        self.embed_timestep = nn.Embedding(self.num_timesteps, self.embed_dim)
        self.unembed = nn.Linear(self.embed_dim, feature_dim)
        self.ap1d = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)

    def add_noise(self, x: torch.FloatTensor, noise: torch.FloatTensor, t: torch.LongTensor):
        return self.scheduler.add_noise(x, noise, t)

    def estimate_noise(self, xt, t):
        t = t.to(xt.device)
        xt_emb = self.embed_input(xt)   # (N,L,768)
        t_emb = self.embed_timestep(t)  # (N,768) or (768)
        zt = xt_emb + t_emb.view(-1,1,self.embed_dim) # (N,L,768)
        gpt2_out = self.gpt2(inputs_embeds=zt)
        noise_pred = self.unembed(gpt2_out.last_hidden_state)   # (N,L,768) -> (N,L,d)
        # noise_pred = self.ap1d(noise_pred)
        return noise_pred

    def forward(
        self,
        x: Optional[torch.FloatTensor] = None,
        t: Optional[torch.FloatTensor] = None,
        batch_size: Optional[int] = None,
        window_size: Optional[int] = None,
        num_inference_steps: int = 1000,
        progress_bar: bool = False,
        enable_grad: bool = False
    ):
        # Unconditional generation
        if x is None:
            assert batch_size is not None
            window_size = self.max_window_size if window_size is None else window_size

            signal = torch.randn(
                batch_size, window_size, self.feature_dim,
                device = next(self.gpt2.parameters()).device
            )

        # Conditional generation
        else:
            noise = torch.randn_like(x)
            if t is None:
                t = torch.LongTensor([self.num_timesteps-1]).to(x.device)
            elif isinstance(t, int):
                t = torch.LongTensor([t]).to(x.device)
            signal = self.add_noise(x, noise, t)

        # Set step values
        self.scheduler.set_timesteps(num_inference_steps)
        pbar = tqdm(self.scheduler.timesteps) if progress_bar else self.scheduler.timesteps

        with torch.set_grad_enabled(enable_grad):
            for t in pbar:
                # 1. Predict the noise
                noise_pred = self.estimate_noise(signal, t)
                # print(noise_pred.shape)

                # 2. Compute the previous: x_{t} -> x_{t-1}
                signal = self.scheduler.step(noise_pred, t, signal).prev_sample

        return signal



