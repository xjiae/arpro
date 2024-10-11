import sys
import os
import torch
import wandb
from tqdm import tqdm
from pathlib import Path
from typing import Optional
import torch.nn.functional as F
from dataclasses import dataclass
from torchvision import transforms
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from .models import MyDiffusionModel
from mydatasets import get_fixer_dataloader


@dataclass
class PretrainMyDiffusionConfig:
    dataset: str
    category: str
    num_epochs: int
    lr: float
    batch_size: int = 2
    device: Optional[str] = None
    do_save: bool = True
    output_dir: Optional[str] = None
    warmup_ratio: float = 0.1
    image_size: int = 512
    wandb_project: str = "arpro"
    


def run_one_epoch(
    diff_model: MyDiffusionModel,
    dataloader,
    train_or_eval: str,
    config: PretrainMyDiffusionConfig,
    optimizer = None,
    lr_scheduler = None
):
    assert train_or_eval in ["train", "eval"]
    device = next(diff_model.parameters()).device

    num_dones, acc_loss = 0, 0.
    pbar = tqdm(dataloader)

    for batch in pbar:
        # Sample a random time step for each image
        if batch['image'].size(1) == 1:
            batch['image'] = batch['image'].repeat(1, 3, 1, 1)
        x = batch["image"].to(device)
        x = 2*x - 1  # Scale a [0,1] image to [-1,+1]
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
                lr_scheduler.step()

        num_dones += x.size(0)
        acc_loss += loss * x.size(0)

        avg_loss = acc_loss / num_dones

        desc = "[train] " if train_or_eval == "train" else "[eval]  "
        desc += f"N {num_dones}, loss {avg_loss:.4f}, "
        desc += f"lr {lr_scheduler.get_last_lr()[0]:.6f} " if train_or_eval == "train" else ""
        pbar.set_description(desc)
        wandb.log({
            "train_loss": avg_loss,
        })

    return {
        "diff_model": diff_model,
        "loss": avg_loss,
    }


def pretrain_diffusion(config: PretrainMyDiffusionConfig):
    
    diff_model = MyDiffusionModel(image_size=config.image_size)

    if config.device is not None:
        diff_model.to(config.device)

    
    train_dataloader = get_fixer_dataloader(
    dataset_name = config.dataset,
    image_size=config.image_size,
    batch_size = config.batch_size,
    category = config.category,
    split = "train")
    

    optimizer = torch.optim.AdamW(
        diff_model.parameters(),
        lr = config.lr
    )

    num_train_steps = len(train_dataloader) * config.num_epochs
    warmup_steps = int(num_train_steps * config.warmup_ratio)
    decay_steps = num_train_steps - warmup_steps

    lr_scheduler = SequentialLR(
        optimizer,
        schedulers = [
            LinearLR(optimizer, start_factor=0.10, end_factor=1.00, total_iters=warmup_steps),
            LinearLR(optimizer, start_factor=1.00, end_factor=0.01, total_iters=decay_steps)
        ],
        milestones = [warmup_steps]
    )

    run_name = f"fixer_diffusion_{config.dataset}_{config.category}"

    if config.do_save:
        assert config.output_dir is not None and Path(config.output_dir).is_dir()
        last_saveto = str(Path(config.output_dir, run_name + "_last.pt"))
        best_saveto = str(Path(config.output_dir, run_name + "_best.pt"))
    else:
        print(f"Warning: will NOT save diff_models")

    # Setup wandb
    wandb_key = os.getenv("WANDB_ANOMALY_PROJECT_KEY")
    wandb.login(key=wandb_key)
    wandb.init(
        project = config.wandb_project,
        name = run_name,
    )


    best_loss = None

    for epoch in range(1, config.num_epochs+1):
        wandb.log({
            "learning_rate": lr_scheduler.get_last_lr()[0]
        })
        print(f"epochs {epoch}/{config.num_epochs}, start_lr {lr_scheduler.get_last_lr()[0]:.6f}")
        train_stats = run_one_epoch(
            diff_model,
            train_dataloader,
            "train",
            config,
            optimizer = optimizer,
            lr_scheduler = lr_scheduler
        )

        save_dict = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "model_state_dict": {k: v.cpu() for (k,v) in diff_model.state_dict().items()},
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
    wandb.finish()
    return best_save_dict


def init_and_pretrain_diffusion(args):
    assert args.model == "diffusion"
    config = PretrainMyDiffusionConfig(
        dataset = args.dataset,
        num_epochs = args.num_epochs,
        lr = args.lr,
        category = args.category,
        batch_size = args.batch_size,
        device = args.device,
        output_dir = args.output_dir,
        image_size=args.image_size
    )

    train_ret = pretrain_diffusion(config)
    return train_ret
