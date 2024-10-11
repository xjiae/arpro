import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from typing import Optional
from dataclasses import dataclass

from tqdm import tqdm

import wandb

from .models import FastflowAdModel
from mydatasets import get_ad_dataloader


@dataclass
class TrainADFastflowConfig:
    dataset_name: str
    category: str
    num_epochs: int
    lr: float
    batch_size: int
    device: Optional[str] = None
    do_save: bool = True
    output_dir: Optional[str] = None
    image_channels: int = 3
    image_size: int = 512
    backbone: str = "wide_resnet50_2"
    warmup_ratio: float = 0.1
    eval_every: int = 5
    wandb_project: str = "arpro"
    aug_noise_scale = 10.


def run_one_epoch(
    model,
    dataloader,
    train_or_eval: str,
    config: TrainADFastflowConfig,
    optimizer = None,
):
    assert train_or_eval in ["train", "eval"]
    _ = model.train() if train_or_eval == "train" else model.eval()
    device = next(model.parameters()).device

    num_dones, acc_loss = 0, 0.
    pbar = tqdm(dataloader)
    for i, batch in enumerate(pbar):
        if batch['image'].size(1) == 1:
            batch['image'] = batch['image'].repeat(1, 3, 1, 1)
        x = batch["image"].to(device)
        x = 2*x - 1 # Scale to [-1,+1]

        # Add some noise
        x += config.aug_noise_scale * torch.randn_like(x) * torch.rand(x.size(0),1,1,1).to(x.device)

        with torch.set_grad_enabled(train_or_eval == "train"):
            out = model(x)
            loss = out.score.mean()
            if train_or_eval == "train":
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            num_dones += x.size(0)
            acc_loss += loss * x.size(0)

        avg_loss = acc_loss / num_dones
        desc = "[train] " if train_or_eval == "train" else "[eval]  "
        desc += f"N {num_dones}, loss {avg_loss:.4f} "
        pbar.set_description(desc)

        wandb.log({
            "train_loss": avg_loss,
        })

    return {
        "model": model,
        "loss": avg_loss,
    }


def train_ad_fastflow(config: TrainADFastflowConfig):
    """ Set up the models, dataloaders, optimizers, etc and start training """
    model = FastflowAdModel(
        image_size = config.image_size,
        backbone = config.backbone
    )
    if config.device is not None:
        model.to(config.device)

    train_dataloader = get_ad_dataloader(
        dataset_name = config.dataset_name,
        batch_size = config.batch_size,
        category = config.category,
        split = "train",
        image_size = config.image_size,
    )

    eval_dataloader = get_ad_dataloader(
        dataset_name = config.dataset_name,
        batch_size = config.batch_size,
        category = config.category,
        split = "test",
        image_size = config.image_size,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = config.lr,
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

    run_name = f"ad_noisy_fast_{config.backbone}_{config.dataset_name}_{config.category}_{config.image_size}"

    if config.do_save:
        assert config.output_dir is not None and Path(config.output_dir).is_dir()
        last_saveto = str(Path(config.output_dir, run_name + "_last.pt"))
        best_saveto = str(Path(config.output_dir, run_name + "_best.pt"))
    else:
        print(f"Warning: will NOT save models")

    best_loss = None

    # Setup wandb
    wandb_key = os.getenv("WANDB_ANOMALY_PROJECT_KEY")
    wandb.login(key=wandb_key)
    wandb.init(
        project = config.wandb_project,
        name = run_name,
    )

    for epoch in range(1, config.num_epochs+1):
        print(f"epoch: {epoch}/{config.num_epochs}, lr {lr_scheduler.get_last_lr()[0]:.6f}")
        train_stats = run_one_epoch(model, train_dataloader, "train", config, optimizer=optimizer)
        if epoch % config.eval_every == 0:
            eval_stats = run_one_epoch(model, eval_dataloader, "eval", config)

        wandb.log({
            "learning_rate": lr_scheduler.get_last_lr()[0]
        })

        lr_scheduler.step()

        save_dict = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "model_state_dict": {k: v.cpu() for (k,v) in model.state_dict().items()},
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


def init_and_train_ad_fastflow(args):
    assert args.model == "fastflow"
    config = TrainADFastflowConfig(
        dataset_name = args.dataset,
        num_epochs = args.epochs,
        lr = args.lr,
        category = args.category,
        batch_size = args.batch_size,
        device = args.device,
        output_dir = args.output_dir,
        backbone = args.backbone,
        image_size=args.image_size
    )

    train_ret = train_ad_fastflow(config)
    return train_ret

