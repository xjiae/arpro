import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from typing import Optional
from dataclasses import dataclass

from tqdm import tqdm

import wandb

from .models import VaeADModel
from datasets import get_ad_dataloader


@dataclass
class TrainADVaeConfig:
    mvtec_category: str
    num_epochs: int
    lr: float
    batch_size: int
    use_cuda: bool
    do_save: bool = True
    output_dir: Optional[str] = None
    image_channels: int = 3
    warmup_ratio: float = 0.1
    eval_every: int = 5
    recon_scale: float = 10.
    kldiv_scale: float = 1.


def run_one_epoch(
    model,
    dataloader,
    train_or_eval: str,
    config: TrainADVaeConfig,
    optimizer = None,
):
    assert train_or_eval in ["train", "eval"]
    _ = model.train() if train_or_eval == "train" else model.eval()
    device = next(model.parameters()).device

    num_dones, acc_loss, acc_recon_loss, acc_kldiv_loss = 0, 0., 0., 0.
    pbar = tqdm(dataloader)

    for i, batch in enumerate(pbar):
        x = batch["image"].to(device)
        x = 2*x - 1 # Scale to [-1,+1]

        with torch.set_grad_enabled(train_or_eval == "train"):
            out = model(x)
            x_recon, mu, logvar = out.others["x_recon"], out.others["mu"], out.others["logvar"]
            recon_loss = F.mse_loss(x_recon, x, reduction="mean") * config.recon_scale
            kldiv_loss = (-0.5*torch.mean(1 + logvar - (mu**2) - logvar.exp())) * config.kldiv_scale
            loss = recon_loss + kldiv_loss
            if train_or_eval == "train":
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        num_dones += x.size(0)
        acc_loss += loss * x.size(0)
        acc_recon_loss += recon_loss * x.size(0)
        acc_kldiv_loss += kldiv_loss * x.size(0)
        avg_loss = acc_loss / num_dones
        avg_recon_loss = acc_recon_loss / num_dones
        avg_kldiv_loss = acc_kldiv_loss / num_dones
        desc = "[train] " if train_or_eval == "train" else "[eval]  "
        desc += f"N {num_dones}, loss {avg_loss:.4f} "
        desc += f"(recon {avg_recon_loss:.4f}, kldiv {avg_kldiv_loss:.4f})"
        pbar.set_description(desc)

    return {
        "model": model,
        "loss": avg_loss,
        "recon_loss": avg_recon_loss,
        "kldiv_loss": avg_kldiv_loss,
    }


def train_ad_vae(config: TrainADVaeConfig):
    """ Set up the models, dataloaders, optimizers, etc and start training """
    model = VaeADModel(image_channels=config.image_channels)
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = config.lr,
    )

    warmup_epochs = int(config.num_epochs * config.warmup_ratio)
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
        saveto_prefix = f"ad_vae_mvtec_{config.mvtec_category}"
        last_saveto = str(Path(config.output_dir, saveto_prefix + "_last.pt"))
        best_saveto = str(Path(config.output_dir, saveto_prefix + "_best.pt"))
    else:
        print(f"Warning: will NOT save models")

    best_loss = None

    for epoch in range(1, config.num_epochs+1):
        print(f"epoch: {epoch}/{config.num_epochs}, lr: {lr_scheduler.get_last_lr()[0]:.6f}")
        train_stats = run_one_epoch(model, train_dataloader, "train", config, optimizer)
        if epoch % config.eval_every == 0:
            eval_stats = run_one_epoch(model, eval_dataloader, "eval", config)

        lr_scheduler.step()

        save_dict = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
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


def init_and_train_ad_vae(args):
    assert args.model_name == "vae"
    assert args.dataset_name == "mvtec"
    config = TrainADVaeConfig(
        num_epochs = args.num_epochs,
        lr = args.lr,
        mvtec_category = args.mvtec_category,
        batch_size = args.batch_size,
        use_cuda = args.cuda,
        output_dir = args.output_dir
    )

    train_ret = train_ad_vae(config)
    return train_ret

