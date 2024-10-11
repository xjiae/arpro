import sys
import math
import json
import torch
import numpy as np
from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass

from ad import *
from mydatasets import *
from fixer.models.language import *

sys.path.insert(0, "..")

@dataclass
class TextRepairConfig:
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
    good_parts = good_parts.squeeze(-1)
    anom_parts = anom_parts.squeeze(-1)
    prop3_loss = F.relu((x_fix_ad_out.alpha - ad_out.alpha) * anom_parts).sum() / anom_parts.sum()
    prop4_loss = F.relu((x_fix_ad_out.alpha - ad_out.alpha) * good_parts).sum() / good_parts.sum()

    return prop1_loss, prop2_loss, prop3_loss, prop4_loss


def text_repair(
    x_bad: torch.FloatTensor,
    anom_parts: torch.LongTensor,
    ad_model: RobertaADModel,
    mydiff_model: MyTextDiffusionModel,
    config: TextRepairConfig,
    noise_level: int,
    num_inference_steps: int=1000,
    progress_bar: bool=True
):
    
    ad_out = ad_model(x_bad)
    anom_parts = anom_parts[:,:, None]
    good_parts = (1 - anom_parts)

    x_bad = mydiff_model.dlm_model.get_embeddings(x_bad)
    noise = torch.randn_like(x_bad)
    noise_amt = torch.LongTensor([noise_level]).to(x_bad.device)
    x_fix = mydiff_model.add_noise(x_bad, noise, noise_amt)

    num_iters = 0
    cum_prop_loss = 0.0
    cum_L1, cum_L2, cum_L3, cum_L4 = 0.0, 0.0, 0.0, 0.0
    pbar = tqdm(num_inference_steps)
    guide_scales = torch.linspace(config.guide_scale_start, config.guide_scale_end, 1000).to(config.device)
    masked_scales = torch.linspace(config.mask_scale_start, config.mask_scale_end, 1000).to(config.device)

    x_t = x_fix
    x_estimation = torch.zeros_like(x_t)

    t_now = torch.ones(x_bad.shape[0], dtype=x_t.dtype, device=x_t.device, requires_grad=False)
    diff_model = mydiff_model.dlm_model.diffusion
    pbar = tqdm(range(num_inference_steps)) if progress_bar else range(num_inference_steps)
    for step in pbar:
        ### get previous step
        if not diff_model.self_conditioning:
            x_estimation = torch.zeros_like(x_t)
        if diff_model.normalize:
            x_t = x_t / x_t.std(dim=-1, keepdim=True)
        x_estimation, latent = diff_model.estimator(torch.cat([x_t, torch.zeros_like(x_t), x_estimation], dim=-1), t_now)
        if diff_model.interpolate is not None:
            x_estimation = diff_model.interpolate(latent)
        t_next = torch.clamp(t_now - 1 / num_inference_steps, 0.0, 1.0)
        xbm_noised = mydiff_model.add_noise(x_bad, torch.randn_like(x_bad), t_now)
        x_t = diff_model.diff_lm_step(x_t, x_estimation, t_now, t_next)
        x_t = (masked_scales[step] * xbm_noised + (1-masked_scales[step]) * x_t) * good_parts + x_t * anom_parts
        x_estimation = x_estimation.detach()
        xbm_noised = xbm_noised.detach()
        with torch.enable_grad():
            x_t = x_t.detach()
            latent = latent.detach()
            x_bad = x_bad.detach()
            good_parts = good_parts.detach()
            anom_parts = anom_parts.detach()

            x_t.requires_grad_(True)
            x_t_logits = mydiff_model.dlm_model.get_logits(latent)
            x_t_ids = x_t_logits.argmax(dim=-1)
            
            x_t_ad_out = ad_model(x_t_ids)
            
            L1, L2, L3, L4 = L(x_t, x_t_ad_out, x_bad, ad_out, good_parts, anom_parts)
            prop_loss = config.prop1_scale * L1 \
                        + config.prop2_scale * L2 \
                        + config.prop3_scale * L3 \
                        + config.prop4_scale * L4
            if config.guide_scale_end > 0:
                prop_loss.backward(retain_graph=True)
                # prop_loss.backward()
                x_t = x_t - guide_scales[step] * x_t.grad.data

            x_t = x_t.detach()
        
        t_now = t_next.detach()
        num_iters += 1
        cum_prop_loss += prop_loss.detach().item()
        avg_prop_loss = cum_prop_loss / num_iters

        cum_L1 += L1.detach().item()
        cum_L2 += L2.detach().item()
        cum_L3 += L3.detach().item()
        cum_L4 += L4.detach().item()

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
    t_final = torch.zeros(x_bad.shape[0], device=x_bad.device)
    _, latent = diff_model.estimator(torch.cat([x_t, torch.zeros_like(x_t), x_estimation], dim=-1), t_final)
    x_fix = mydiff_model.dlm_model.get_logits(latent).argmax(dim=-1)
    return {"x_fix": x_fix, 
            "prop_loss": prop_loss, 
            "l1": L1,
            "l2": L2,
            "l3": L3,
            "l4": L4,
            }
        
    



    
    