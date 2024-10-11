import sys
import os
import torch
import wandb
from tqdm import tqdm
from pathlib import Path
from typing import Optional
import torch.nn.functional as F
from dataclasses import dataclass
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from .models import MyTimeDiffusionModel
from mydatasets import get_timeseries_bundle

@dataclass
class PretrainMyTSDiffusionConfig:
    dataset: str
    num_epochs: int
    lr: float
    batch_size: int = 64
    device: Optional[str] = None
    do_save: bool = True
    output_dir: Optional[str] = None
    warmup_ratio: float = 0.1
    window_size: int = 100
    feature_dim: int = 51
    fft_scale: float = 1e-2
    wandb_project: str = "arpro"

def run_one_epoch(
    diff_model: MyTimeDiffusionModel,
    dataloader,
    train_or_eval: str,
    config: PretrainMyTSDiffusionConfig,
    optimizer = None,
    lr_scheduler = None
):
    assert train_or_eval in ["train", "eval"]
    device = next(diff_model.parameters()).device

    num_dones, acc_loss = 0, 0.
    pbar = tqdm(dataloader)

    for batch in pbar:
        with torch.set_grad_enabled(train_or_eval == "train"):
            # Sample a random time step for each image
            x = batch[0].to(device)
            loss = diff_model.compute_loss(x)
        
            """ Older stuff for the GPT2-based diffusion model
            t = torch.randint(0, diff_model.num_timesteps, (x.size(0),)).to(device)
            noise = torch.randn_like(x)
            noised_ts = diff_model.add_noise(x, noise, t)
            noise_pred = diff_model.estimate_noise(noised_ts, t)
            loss = F.mse_loss(noise_pred, noise)

            fft_noise = torch.fft.fft(noise)
            fft_noise_pred = torch.fft.fft(noise_pred)
            fft_loss = F.mse_loss((fft_noise - fft_noise_pred).abs(), torch.zeros_like(noise))
            loss += config.fft_scale * fft_loss
            """

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
        if num_dones > 100000:
            break
        wandb.log({
            "train_loss": avg_loss,
        })

    return {
        "diff_model": diff_model,
        "loss": avg_loss,
    }

def pretrain_ts_diffusion(config: PretrainMyTSDiffusionConfig):
    if config.dataset == "wadi":
        num_features = 127
    elif config.dataset == "swat":
        num_features = 51
    else:
        num_features = 86
    # diff_model = MyTSDiffusionModel(input_dim=config.feature_dim)
    diff_model = MyTimeDiffusionModel(window_size=config.window_size, 
                                      feature_dim=num_features)

    if config.device is not None:
        diff_model.to(config.device)

    train_dataloader = get_timeseries_bundle(
        ds_name = config.dataset,
        stride = 1,
        window_size = config.window_size,
        train_batch_size = config.batch_size, 
        test_batch_size = config.batch_size, 
        train_has_only_goods = True,
        shuffle = False
    )['train_dataloader']

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
            LinearLR(optimizer, start_factor=0.01, end_factor=1.00, total_iters=warmup_steps),
            LinearLR(optimizer, start_factor=1.00, end_factor=0.01, total_iters=decay_steps)
        ],
        milestones = [warmup_steps]
    )

    run_name = f"fixer_ts_diffusion_{config.dataset}"

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

def init_and_pretrain_ts_diffusion(args):
    assert args.model == "ts_diffusion"
    
    config = PretrainMyTSDiffusionConfig(
        dataset = args.dataset,
        num_epochs = args.num_epochs,
        lr = args.lr,
        batch_size = args.batch_size,
        device = args.device,
        output_dir = args.output_dir,
        feature_dim=args.feature_dim
    )

    train_ret = pretrain_ts_diffusion(config)
    return train_ret
