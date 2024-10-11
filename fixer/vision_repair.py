import sys
import math
import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Optional
import torchvision.utils as tvu
import torch.nn.functional as F
from dataclasses import dataclass

from ad import *
from mydatasets import *
from fixer.models.vision import *

sys.path.insert(0, "..")

@dataclass
class VisionRepairConfig:
    category: str
    lr: float 
    batch_size: int
    device: Optional[str] = "cuda"
    output_dir: Optional[str] = None
    image_folder: str = str(Path(Path(__file__).parent.resolve(), "../_dump/edit/"))
    image_channels: int = 3
    eval_every: int = 5
    ad_model_path: Optional[str] = None
    prop1_scale: float = 0.1
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
    guide_scale: float = 5.0


def vision_repair(
    x_bad: torch.FloatTensor,
    anom_parts: torch.FloatTensor,
    ad_model: FastflowAdModel,
    mydiff_model: MyDiffusionModel,
    config: VisionRepairConfig,
    noise_level: int,
):
    ad_out = ad_model(x_bad)
    good_parts = (1 - anom_parts).long()
    average_colors = (x_bad * anom_parts).sum(dim=(-1,-2)) / (anom_parts.sum(dim=(-1,-2)))
    x_bad_masked = good_parts * x_bad + anom_parts * (average_colors.view(-1,3,1,1))

    noise = torch.randn_like(x_bad_masked)
    noise_amt = torch.LongTensor([noise_level]).to(x_bad.device)
    x_fix = mydiff_model.add_noise(x_bad_masked, noise, noise_amt)

    num_iters = 0
    cum_prop_loss = 0.0
    cum_L1, cum_L2, cum_L3, cum_L4 = 0.0, 0.0, 0.0, 0.0
    pbar = tqdm(mydiff_model.scheduler.timesteps)
    for t in pbar: # This is already reversed from 999, 998, ..., 1, 0
        # if guide_scale > 0, then we use our properties
        guide_vec = torch.zeros_like(x_fix)
        prop_loss, L1, L2, L3, L4 = 0., 0., 0., 0., 0.
        if config.guide_scale > 0:
            with torch.enable_grad():
                x_fix.requires_grad_(True)
                x_fix_ad_out = ad_model(x_fix)
                L1, L2, L3, L4 = L(x_fix, x_fix_ad_out, x_bad, ad_out, good_parts, anom_parts)
                prop_loss = config.prop1_scale * L1 \
                            + config.prop2_scale * L2 \
                            + config.prop3_scale * L3 \
                            + config.prop4_scale * L4
                prop_loss.backward(retain_graph=True)
                prop_loss = prop_loss.detach()
                L1, L2, L3, L4 = L1.detach(), L2.detach(), L3.detach(), L4.detach()
                guide_vec = x_fix.grad.data.detach()

        with torch.no_grad():
            xb_noised = mydiff_model.add_noise(x_bad, torch.randn_like(x_bad), t)
            noise_pred = mydiff_model.estimate_noise(x_fix, t)

            noise_pred = noise_pred - \
                (1 - mydiff_model.scheduler.alphas_cumprod[t]).sqrt() * config.guide_scale * guide_vec

            # noise_pred = noise_pred - \
            #     mydiff_model.scheduler.betas[t] * config.guide_scale * guide_vec


            x_fix = mydiff_model.scheduler.step(noise_pred, t, x_fix).prev_sample
            x_fix = good_parts * xb_noised + anom_parts * x_fix
            x_fix.detach()

        num_iters += 1
        cum_prop_loss += prop_loss
        avg_prop_loss = cum_prop_loss / num_iters

        cum_L1 += L1
        cum_L2 += L2
        cum_L3 += L3
        cum_L4 += L4

        avg_L1 = cum_L1 / num_iters
        avg_L2 = cum_L2 / num_iters
        avg_L3 = cum_L3 / num_iters
        avg_L4 = cum_L4 / num_iters

        end_num = math.log10(config.guide_scale) if config.guide_scale > 0 else 0
        desc_str = f"gs {end_num:.2f}, "
        desc_str += f"avg_L {avg_prop_loss:.2f}, "
        desc_str += f"curr_L {prop_loss:.2f}, "
        desc_str += f"L1 {(config.prop1_scale * avg_L1):.2f} {avg_L1:.2f}, "
        desc_str += f"L2 {(config.prop2_scale * avg_L2):.2f} {avg_L2:.2f}, "
        desc_str += f"L3 {(config.prop3_scale * avg_L3):.2f} {avg_L3:.2f}, "
        desc_str += f"L4 {(config.prop4_scale * avg_L4):.2f} {avg_L4:.2f}, "
        pbar.set_description(desc_str)
    return {"x_fix": x_fix}


def L(x_fix, x_fix_ad_out, x, ad_out, good_parts, anom_parts):
    prop1_loss = x_fix_ad_out.score.mean()
    # prop1_loss = torch.norm(x_fix_ad_out.score, p=2)
    prop2_loss = F.mse_loss(good_parts * x_fix, good_parts * x, reduction="sum") / good_parts.sum()
    prop3_loss = F.relu((x_fix_ad_out.alpha - ad_out.alpha) * anom_parts).sum() / anom_parts.sum()
    prop4_loss = F.relu((x_fix_ad_out.alpha - ad_out.alpha) * good_parts).sum() / good_parts.sum()
    return prop1_loss, prop2_loss, prop3_loss, prop4_loss

