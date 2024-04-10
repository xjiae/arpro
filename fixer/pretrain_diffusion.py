import sys
import os
from pathlib import Path
from typing import Optional
import torch
import torch.nn.functional as F
from torchvision import transforms

from torch.optim.lr_scheduler import LinearLR, SequentialLR
from dataclasses import dataclass

from tqdm import tqdm

from .models import MyDiffusionModel
from datasets import get_fixer_dataloader


@dataclass
class PretrainMyDiffusionConfig:
    mvtec_category: str
    num_epochs: int
    lr: float
    batch_size: int
    unet2d_ch: int = 224
    device: Optional[str] = None
    do_save: bool = True
    output_dir: Optional[str] = None
    warmup_ratio: float = 0.1


def run_one_epoch(
    diff_model: MyDiffusionModel,
    dataloader,
    train_or_eval: str,
    config: PretrainMyDiffusionConfig,
    optimizer = None
):
    assert train_or_eval in ["train", "eval"]
    device = next(diff_model.parameters()).device

    num_dones, acc_loss = 0, 0.
    pbar = tqdm(dataloader)

    for batch in pbar:
        # Sample a random time step for each image
        x = batch["image"].to(device)
        noise = torch.randn_like(x)

        t = torch.randint(0, diff_model.num_timesteps, (x.size(0),)).to(device)
        noised_images = diff_model.add_noise(x, noise, t)

        with torch.set_grad_enabled(train_or_eval == "train"):
            noise_pred = diff_model.estimate_noise(noised_images, t)
            loss = F.mse_loss(noise_pred, noise)
            if train_or_eval == "train":
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        num_dones += x.size(0)
        acc_loss += loss * x.size(0)

        avg_loss = acc_loss / num_dones
        desc = "[train] " if train_or_eval == "train" else "[eval]  "
        desc += f"N {num_dones}, loss {avg_loss:.4f}"
        pbar.set_description(desc)

    return {
        "diff_model": diff_model,
        "loss": avg_loss,
    }


def pretrain_diffusion(config: PretrainMyDiffusionConfig):
    diff_model = MyDiffusionModel(
        unet2d_block_out_channels = tuple([config.unet2d_ch * i for i in [1,2,3,4]])
    )

    if config.device is not None:
        diff_model.to(config.device)

    train_dataloader = get_fixer_dataloader(
        dataset_name = "mvtec",
        batch_size = config.batch_size,
        category = config.mvtec_category,
        split = "train",
    )

    optimizer = torch.optim.AdamW(
        diff_model.parameters(),
        lr = config.lr
    )

    warmup_epochs = int(config.num_epochs * config.warmup_ratio)
    decay_epochs = config.num_epochs - warmup_epochs

    lr_scheduler = SequentialLR(
        optimizer,
        schedulers = [
            LinearLR(optimizer, start_factor=0.01, end_factor=1.00, total_iters=warmup_epochs),
            LinearLR(optimizer, start_factor=1.00, end_factor=0.01, total_iters=decay_epochs)
        ],
        milestones = [warmup_epochs]
    )

    run_name = f"fixer_diffusion_mvtec_{config.mvtec_category}"

    if config.do_save:
        assert config.output_dir is not None and Path(config.output_dir).is_dir()
        last_saveto = str(Path(config.output_dir, run_name + "_last.pt"))
        best_saveto = str(Path(config.output_dir, run_name + "_best.pt"))
    else:
        print(f"Warning: will NOT save diff_models")


    best_loss = None

    for epoch in range(1, config.num_epochs+1):
        print(f"epochs {epoch}/{config.num_epochs}, lr {lr_scheduler.get_last_lr()[0]:.6f}")
        train_stats = run_one_epoch(diff_model, train_dataloader, "train", config, optimizer=optimizer)
        
        lr_scheduler.step()

        save_dict = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "diff_model_state_dict": {k: v.cpu() for (k,v) in diff_model_state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        }

        if config.do_save:
            torch.save(save_dict, last_saveto)

        this_loss = train_stats["loss"]
        if best_loss is None or this_loss < best_loss:
            best_save_dict = save_dict
            delta = 0. if best_loss is None else (best_loss - this_loss)
            print(f"New best {this_loss:.4f}, delta {delta:.4f}")
            best_loss = this_loss
            if config.do_save:
                torch.save(save_dict, best_saveto)

    return best_save_dict


def init_and_pretrain_diffusion(args):
    assert args.model_name == "diffusion"
    assert args.dataset_name == "mvtec"
    config = PretrainMyDiffusionConfig(
        unet2d_ch = args.unet2d_ch,
        num_epochs = args.num_epochs,
        lr = args.lr,
        mvtec_category = args.mvtec_category,
        batch_size = args.batch_size,
        device = args.device,
        output_dir = args.output_dir,
    )

    train_ret = pretrain_diffusion(config)
    return train_ret
