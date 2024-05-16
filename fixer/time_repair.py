import sys
import math
import json
import torch
import numpy as np
from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass

from ad import *
from datasets import *
from fixer.models.timeseries import *

sys.path.insert(0, "..")

@dataclass
class TimeRepairConfig:
    lr: float 
    batch_size: int
    device: Optional[str] = "cuda"
    output_dir: Optional[str] = None
    eval_every: int = 5
    ad_model_path: Optional[str] = None
    prop1_scale: float = 1e-3
    prop2_scale: float = 10.
    prop3_scale: float = 1.0
    prop4_scale: float = 1.0
    wandb_project: str = "arpro"
    sampling_batch_size: int = 8
    sample_step: int = 3
    noise_scale: int = 600
    diffusion_beta_start: float = 0.0001
    diffusion_beta_end: float = 0.02
    num_diffusion_timesteps: int = 1000
    guide_scale_start: float = 0.0
    guide_scale_end: float = 1e-3
    mask_scale_start: float = 1e-4
    mask_scale_end: float = 5e-3

def L(x_fix, x_fix_ad_out, x, ad_out, good_parts, anom_parts):
    prop1_loss = x_fix_ad_out.score.mean()
    prop2_loss = F.mse_loss(good_parts * x_fix, good_parts * x, reduction="sum") / good_parts.sum()
    prop3_loss = F.relu((x_fix_ad_out.alpha - ad_out.alpha) * anom_parts).sum() / anom_parts.sum()
    prop4_loss = F.relu((x_fix_ad_out.alpha - ad_out.alpha) * good_parts).sum() / good_parts.sum()
    return prop1_loss, prop2_loss, prop3_loss, prop4_loss

@torch.no_grad()
def time_repair(
    x_bad: torch.FloatTensor,
    anom_parts: torch.LongTensor,
    ad_model: GPT2ADModel,
    mydiff_model: MyTimeDiffusionModel,
    config: TimeRepairConfig,
    noise_level: int,
):
    
    ad_out = ad_model(x_bad)
    good_parts = (1 - anom_parts).long()

    noise = torch.randn_like(x_bad)
    noise_amt = torch.LongTensor([noise_level]).to(x_bad.device)
    x_fix = mydiff_model.add_noise(x_bad, noise, noise_amt)
    model_kwargs = {
        "coef": 1e-2,
        "learning_rate": 5e-2
    }
    num_iters = 0
    cum_prop_loss = 0.0
    cum_L1, cum_L2, cum_L3, cum_L4 = 0.0, 0.0, 0.0, 0.0
    pbar = tqdm(mydiff_model.num_timesteps)
    guide_scales = torch.linspace(config.guide_scale_end, config.guide_scale_start, 1000) 
    masked_scales = torch.linspace(config.mask_scale_end, config.mask_scale_start, 1000)
    
    times = torch.linspace(-1, noise_level - 1, steps=noise_level + 1)
    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:]))
    for time, time_next in tqdm(time_pairs): # This is already reversed from 999, 998, ..., 1, 0
       
        time_cond = torch.full((config.batch_size,), time, device=config.device, dtype=torch.long)
        xbm_noised = mydiff_model.add_noise(x_bad, torch.randn_like(x_bad), torch.tensor([time]).to(x_bad.device))
        x_fix = mydiff_model.get_prev_sample(x_fix, time, time_next, time_cond, x_bad * good_parts, good_parts, model_kwargs=model_kwargs)
        x_fix = (masked_scales[time] * xbm_noised + (1-masked_scales[time]) * x_fix) * good_parts + x_fix * anom_parts
        # x_fix = xbm_noised * good_parts + x_fix * anom_parts

        with torch.enable_grad():
            x_fix.requires_grad_(True)
            x_fix_ad_out = ad_model(x_fix)
            L1, L2, L3, L4 = L(x_fix, x_fix_ad_out, x_bad, ad_out, good_parts, anom_parts)
            prop_loss = config.prop1_scale * L1 \
                        + config.prop2_scale * L2 \
                        + config.prop3_scale * L3 \
                        + config.prop4_scale * L4
            if config.guide_scale_end > 0:
                prop_loss.backward()
                x_fix = x_fix - guide_scales[time] * x_fix.grad.data
            x_fix.detach()
        num_iters += 1
        cum_prop_loss += prop_loss.detach()
        avg_prop_loss = cum_prop_loss / num_iters

        cum_L1 += L1.detach()
        cum_L2 += L2.detach()
        cum_L3 += L3.detach()
        cum_L4 += L4.detach()

        avg_L1 = L1 / num_iters
        avg_L2 = L2 / num_iters
        avg_L3 = L3 / num_iters
        avg_L4 = L4 / num_iters

        end_num = math.log10(config.guide_scale_end) if config.guide_scale_end > 0 else 0
        desc_str = f"gs {end_num:.2f}, "
        desc_str += f"avg_L {avg_prop_loss:.2f}, "
        desc_str += f"curr_L {prop_loss:.2f}, "
        desc_str += f"L1 {(config.prop1_scale * L1):.2f} {L1:.2f}, "
        desc_str += f"L2 {(config.prop2_scale * L2):.2f} {L2:.2f}, "
        desc_str += f"L3 {(config.prop3_scale * L3):.2f} {L3:.2f}, "
        desc_str += f"L4 {(config.prop4_scale * L4):.2f} {L4:.2f}, "
        pbar.set_description(desc_str)
    return {"x_fix": x_fix, 
            "prop_loss": prop_loss, 
            "l1": L1,
            "l2": L2,
            "l3": L3,
            "l4": L4,
            }

