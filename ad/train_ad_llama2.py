import os
import sys
import torch
import wandb
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from typing import Optional
import torch.nn.functional as F
from dataclasses import dataclass
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from .models import Llama2ADModel
from mydatasets import get_timeseries_bundle

@dataclass
class TrainADLlama2Config:
    num_epochs: int
    lr: float
    batch_size: int
    device: Optional[str] = None    # Use CPU by default
    do_save: bool = True
    output_dir: Optional[str] = None
    warmup_ratio: float = 0.1
    eval_every: int = 5
    wandb_project: str = "arpro"
    dataset: str = "wadi"

def run_one_epoch(
    model,
    dataloader,
    train_or_eval: str,
    config: TrainADLlama2Config,
    optimizer = None,
):
    assert train_or_eval in ["train", "eval"]
    _ = model.train() if train_or_eval == "train" else model.eval()
    device = next(model.parameters()).device

    num_dones, acc_loss = 0, 0.
    pbar = tqdm(dataloader)
    for _, batch in enumerate(pbar):
        x = batch[0].float().to(config.device)

        with torch.set_grad_enabled(train_or_eval == "train"):
            out = model(x)
            x_recon = out.others["x_recon"]
            loss = F.mse_loss(x_recon, x) * x.size(-1) * x.size(-2)
            if train_or_eval == "train":
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

        loss = loss.detach().cpu().item()
        num_dones += x.size(0)
        acc_loss += loss * x.size(0)
        avg_loss = acc_loss / num_dones
        desc = f"N {num_dones}, loss {avg_loss:.5f} "
        pbar.set_description(desc)

        # Log to wandb
        if train_or_eval == "train":
            wandb.log({
                "train_loss": avg_loss
            })
        else:
            wandb.log({
                "eval_loss": avg_loss
            })
       

    return {
        "model": model,
        "loss": avg_loss
    }

def train_ad_llama2(config: TrainADLlama2Config):
    if config.dataset == "wadi":
        num_features = 127
    elif config.dataset == "swat":
        num_features = 51
    else:
        num_features = 86
    model = Llama2ADModel(num_features=num_features)
    if config.device is not None:
        model.to(config.device)
    ret = get_timeseries_bundle(ds_name=config.dataset,
                                stride = 1,
                                window_size=100,
                                train_batch_size=config.batch_size, 
                                test_batch_size=config.batch_size, 
                                train_has_only_goods=True,
                                shuffle=False)
    train_dataloader, eval_dataloader = ret['train_dataloader'], ret['test_dataloader']
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
    run_name = f"ad_llama2_{config.dataset}"
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
        print(f"epoch: {epoch}/{config.num_epochs}, lr: {lr_scheduler.get_last_lr()[0]:.6f}")
        train_stats = run_one_epoch(model, train_dataloader, "train", config, optimizer)
        if epoch % config.eval_every == 0:
            eval_stats = run_one_epoch(model, eval_dataloader, "eval", config)


        wandb.log({
            "learning_rate": lr_scheduler.get_last_lr()[0]
        })

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

    wandb.finish()
    return best_save_dict

def init_and_train_ad_llama2(args):
    assert args.model == "llama2"
    # assert args.dataset == "swat"
    config = TrainADLlama2Config(
        num_epochs = args.epochs,
        lr = args.lr,
        batch_size = args.batch_size,
        device = args.device,
        output_dir = args.output_dir,
        dataset = args.dataset
    )
    train_ret = train_ad_llama2(config)
    return train_ret
