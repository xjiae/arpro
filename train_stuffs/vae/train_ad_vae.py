import sys
from pathlib import Path
import torch
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from typing import Optional
from dataclasses import dataclass

from tqdm import tqdm

import argparse
import wandb

from models import VaeADModel
from datasets import get_ad_dataloader


@dataclass
class VaeADTrainingConfig:
    mvtec_category: str
    num_epochs: int
    lr: float
    batch_size: int
    use_cuda: bool
    do_save: bool = True
    output_dir: Optional[str] = None
    vae_in_channels: int = 3
    vae_ch: int = 3
    vae_latent_channels: int = 256
    warmup: float = 0.1


def run_one_epoch(
    model,
    dataloader,
    train_or_eval = None,
    optimizer = None,
):
    assert train_or_eval in ["train", "eval"]
    _ = model.train() if train_or_eval == "train" else model.eval()
    device = next(model.parameters()).device

    num_dones, running_loss = 0, 0.
    pbar = tqdm(dataloader)

    for i, batch in enumerate(pbar):
        x, m, y = batch
        x, m, y = x.to(device), m.to(device), y.to(device)
        out = model(x)

        xhat, mu, logvar = out.other["xhat"], out.other["mu"], out.other["logvar"]
        loss = torch.norm(x - xhat, p=2) ** 2
        loss += 0.5 * torch.sum(-1 - logvar + torch.exp(logvar) + mu**2)

        if train_or_eval == "train":
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        num_dones += x.size(0)
        running_loss += loss
        avg_loss = running_loss / num_dones
        desc = "[train] " if train_or_eval == "train" else "[eval]  "
        desc += f"num_dones {num_dones}, loss {avg_loss:.4f}"
        pbar.set_description(desc)

    return {
        "model": model,
        "loss": avg_loss
    }


def train_vae_ad(config: VaeADTrainingConfig):
    """ Set up the models, dataloaders, optimizers, etc and start training """
    model = VaeADModel(config.vae_in_channels, config.vae_ch, config.vae_latent_channels)
    if config.use_cuda:
        model.cuda()

    train_dataloader = get_ad_dataloader(
        dataset_name = "mvtec",
        model_name = "vae",
        batch_size = config.batch_size,
        category = config.mvtec_category,
        split = "train"
    )

    eval_dataloader = get_ad_dataloader(
        dataset_name = "mvtec",
        model_name = "vae",
        batch_size = config.batch_size,
        category = config.mvtec_category,
        split = "test"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    warmup_epochs = int(config.num_epochs * config.warmup)
    decay_epochs = config.num_epochs - warmup_epochs

    lr_scheduler = SequentialLR(
        optimizer,
        schedulers = [
            LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs),
            LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=decay_epochs)
        ],
        milestones = [warmup_epochs]
    )

    if config.do_save:
        assert config.output_dir is not None and Path(config.output_dir).is_dir()
        saveto_prefix = f"vae_mvtec_{config.mvtec_category}"
        last_saveto = str(Path(config.output_dir, saveto_prefix + "_last.pt"))
        best_saveto = str(Path(config.output_dir, saveto_prefix + "_best.pt"))
    else:
        print(f"Warning: will NOT save models")

    best_loss = None

    for epoch in range(1, config.num_epochs+1):
        print(f"epoch: {epoch}/{config.num_epochs}, lr: {lr_scheduler.get_last_lr()[0]:.6f}")
        train_stats = run_one_epoch(model, train_dataloader, "train", optimizer)
        eval_stats = run_one_epoch(model, eval_dataloader, "eval", optimizer)
        lr_scheduler.step()

        save_dict = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "eval_loss": eval_stats["loss"],
            "model_state_dict": {k: v.cpu() for (k, v) in model.state_dict().items()},
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

    return None


def init_and_train_vae_ad(args):
    assert args.model_name == "vae"
    assert args.dataset_name == "mvtec"
    config = VaeADTrainingConfig(
        num_epochs = args.num_epochs,
        lr = args.lr,
        mvtec_category = args.mvtec_category,
        batch_size = args.batch_size,
        use_cuda = args.cuda,
        output_dir = args.output_dir
    )

    train_ret = train_vae_ad(config)
    return train_ret

