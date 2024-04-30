import sys
sys.path.insert(0, "..")
import torch
import math
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Optional
import torchvision.utils as tvu
import torch.nn.functional as F
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
from ad import *
from fixer.models.vision import *
from datasets import *
import json
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

def calculate_averages(dictionary):
    return {k: sum(v) / len(v) for k, v in dictionary.items()}

def prop_loss_plot(noise_level=900, mvtec_category="transistor", num_sample=5):
    
    model_path = "../_dump/fixer_diffusion_mvtec_transistor_best.pt"
    model_dict = torch.load(model_path)['model_state_dict']
    mydiff = MyDiffusionModel()
    mydiff.load_state_dict(model_dict)
    mydiff.eval().cuda()

    ad = FastflowAdModel()
    state_dict = torch.load(f"../_dump/ad_fast_mvtec_transistor_best.pt")["model_state_dict"]
    ad.load_state_dict(state_dict)
    ad.eval().cuda()

    torch.manual_seed(1234)
    dataloader = get_fixer_dataloader("mvtec", batch_size=8, category="transistor", split="test")
    for batch in dataloader:
        break
    x_bad = batch["image"][[3,7]].cuda()
   
    # end_scales = [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
    end_scales = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 3e-1, 5e-1, 1.0]
    prop_losses = defaultdict(list)
    l1_losses = defaultdict(list)
    l2_losses = defaultdict(list)
    l3_losses = defaultdict(list)
    l4_losses = defaultdict(list)
    for s in range(num_sample):
        for end in end_scales:
            config = VisionRepairConfig(mvtec_category=mvtec_category, lr=1e-5, batch_size=2, guide_scale_end=end)
            out = vision_repair(x_bad, ad, mydiff, config, noise_level)
            prop_losses[end].append(out['prop_loss'].item())
            l1_losses[end].append(out['l1'].item())
            l2_losses[end].append(out['l2'].item())
            l3_losses[end].append(out['l3'].item())
            l4_losses[end].append(out['l4'].item())

            x_fix = out['x_fix']
            x_fix = (x_fix+1) * 0.5
            x_fix = x_fix.clamp(0,1).detach().cpu()
            plt.clf()
            fig, ax = plt.subplots(2,2)
            ax[0,0].imshow(x_bad[0].cpu().numpy().transpose(1,2,0))
            ax[0,1].imshow(x_bad[1].cpu().numpy().transpose(1,2,0))
            ax[1,0].imshow(x_fix[0].numpy().transpose(1,2,0))
            ax[1,1].imshow(x_fix[1].numpy().transpose(1,2,0))
            end_num = math.log10(end) if end > 0 else 0
            plt.savefig(config.image_folder+f"/iter{s}_end{end_num:.2f}.png")
            plt.close()

            # Save dictionaries as JSON after each update
            json_path = config.image_folder + f"/loss_data_iter{s}.json"
            with open(json_path, 'w') as f:
                json.dump({
                    'prop_losses': prop_losses,
                    'l1_losses': l1_losses,
                    'l2_losses': l2_losses,
                    'l3_losses': l3_losses,
                    'l4_losses': l4_losses
                }, f)
    plot(json_path)


def plot(file_path):
    with open(file_path, 'r') as file:
        all_losses = json.load(file)
        prop_losses = all_losses['prop_losses']
        l1_losses = all_losses['l1_losses']
        l2_losses = all_losses['l2_losses']
        l3_losses = all_losses['l3_losses']
        l4_losses = all_losses['l4_losses']
    avg_of_prop_losses = calculate_averages(prop_losses)
    avg_of_l1_losses = calculate_averages(l1_losses)
    avg_of_l2_losses = calculate_averages(l2_losses)
    avg_of_l3_losses = calculate_averages(l3_losses)
    avg_of_l4_losses = calculate_averages(l4_losses)
    
    
    fig, ax = plt.subplots()
    ax.plot(avg_of_prop_losses.keys(), avg_of_prop_losses.values(), label='Prop Losses', marker='o')
    ax.set_xlabel('End Scale')
    ax.set_ylabel('Average Loss')
    ax.legend()
    plt.title('Comparison of Average Losses')
    plt.savefig(f'/home/antonxue/foo/arpro/_dump/edit/average_prop_losses_plot.png') 
    plt.close()

    fig, ax = plt.subplots(2, 2,  figsize=(10, 6))

    ax[0, 0].plot(avg_of_l1_losses.keys(), avg_of_l1_losses.values(), label='l1 Losses', marker='o')
    ax[0, 0].set_ylabel('Average Loss')
    ax[0, 0].legend()

    ax[0, 1].plot(avg_of_l2_losses.keys(), avg_of_l2_losses.values(), label='l2 Losses', marker='o')
    ax[0, 1].set_ylabel('Average Loss')
    ax[0, 1].legend()

    ax[1, 0].plot(avg_of_l3_losses.keys(), avg_of_l3_losses.values(), label='l3 Losses', marker='o')
    ax[1, 0].set_ylabel('Average Loss')
    ax[1, 0].legend()

    ax[1, 1].plot(avg_of_l4_losses.keys(), avg_of_l4_losses.values(), label='l4 Losses', marker='o')
    ax[1, 1].set_ylabel('Average Loss')
    ax[1, 1].legend()


    # plt.title('Comparison of Losses')
    plt.savefig(f'/home/antonxue/foo/arpro/_dump/edit/average_all_losses_plot.png')  # Save the plot
        


def vision_repair(
    x_bad,
    anom_parts,
    ad_model,
    mydiff_model: MyDiffusionModel,
    config,
    noise_level,
    threshold=0.9
):
    ad_out = ad_model(x_bad)
    # anom_parts = (ad_out.alpha > ad_out.alpha.view(x_bad.size(0),-1).quantile(threshold,dim=1).view(-1,1,1,1)).long()
    good_parts = (1 - anom_parts).long()
    average_colors = (x_bad * anom_parts).sum(dim=(-1,-2)) / (anom_parts.sum(dim=(-1,-2)))
    x_bad_masked = (1-anom_parts) * x_bad + anom_parts * (average_colors.view(-1,3,1,1))

    noise = torch.randn_like(x_bad_masked)
    noise_amt = torch.LongTensor([noise_level]).to(x_bad.device)
    x_fix = mydiff_model.add_noise(x_bad_masked, noise, noise_amt)

    num_iters = 0
    cum_prop_loss = 0.0
    cum_L1, cum_L2, cum_L3, cum_L4 = 0.0, 0.0, 0.0, 0.0
    pbar = tqdm(mydiff_model.scheduler.timesteps)
    # guide_scales = torch.linspace(config.guide_scale_start, config.guide_scale_end, 1000)
    guide_scales = torch.linspace(config.guide_scale_end, config.guide_scale_start, 1000) 
    masked_scales = torch.linspace(config.mask_scale_end, config.mask_scale_start, 1000)

    for t in pbar: # This is already reversed from 999, 998, ..., 1, 0
        with torch.no_grad():
            xbm_noised = mydiff_model.add_noise(x_bad_masked, torch.randn_like(x_bad), t)
            # Take a weighted combination only on the good_parts
            
            # x_fix = xbm_noised * good_parts + x_fix * anom_parts
            out = mydiff_model.unet(x_fix, t).sample
            x_fix = mydiff_model.scheduler.step(out, t, x_fix).prev_sample
            x_fix = (1-masked_scales[t]) * xbm_noised * good_parts + x_fix * anom_parts
            

            # x_fix = (masked_scales[t] * xbm_noised + (1-masked_scales[t]) * x_fix) * good_parts + x_fix * anom_parts


        # This is where we enforce our property-based loss
        
        with torch.enable_grad():
            x_fix.requires_grad_(True)
            x_fix_ad_out = ad_model(x_fix)
            L1, L2, L3, L4 = L(x_fix, x_fix_ad_out, x_bad, ad_out, good_parts, anom_parts)
            prop_loss = config.prop1_scale * L1 \
                        + config.prop2_scale * L2 \
                        + config.prop3_scale * L3 \
                        + config.prop4_scale * L4
                        
            
            # prop_loss = config.prop2_scale * L2 \
            #     + config.prop3_scale * L3 \
            #     + config.prop4_scale * L4
            

            if config.guide_scale_end > 0:
                prop_loss.backward(retain_graph=True)
                x_fix.retain_grad()
                # x_fix = x_fix - guide_scales[1000-(t+1)] * x_fix.grad.data
                x_fix = x_fix - guide_scales[t] * x_fix.grad.data
                # x_fix.grad.data.zero_()
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
    # x_fix = x_fix * anom_parts + x_bad * good_parts
    return {"x_fix": x_fix, 
            "prop_loss": prop_loss, 
            "l1": L1,
            "l2": L2,
            "l3": L3,
            "l4": L4,
            }


def L(x_fix, x_fix_ad_out, x, ad_out, good_parts, anom_parts):
    prop1_loss = x_fix_ad_out.score.mean()
    prop2_loss = F.mse_loss(good_parts * x_fix, good_parts * x, reduction="sum") / good_parts.sum()
    # prop2_loss = torch.abs(good_parts * x_fix - good_parts * x).sum() / good_parts.sum()
    # prop2_loss = F.mse_loss(x_fix,  x, reduction="sum") / good_parts.sum()
    prop3_loss = F.relu((x_fix_ad_out.alpha - ad_out.alpha) * anom_parts).sum() / anom_parts.sum()
    prop4_loss = F.relu((x_fix_ad_out.alpha - ad_out.alpha) * good_parts).sum() / good_parts.sum()
    return prop1_loss, prop2_loss, prop3_loss, prop4_loss

# plot("/home/antonxue/foo/arpro/_dump/edit/loss_data_iter4.json")
# prop_loss_plot(num_sample=10)