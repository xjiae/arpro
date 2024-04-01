import sys
from pathlib import Path
import torch
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from typing import Optional
from dataclasses import dataclass

from tqdm import tqdm

from .models import VaeFixerModel
from .image_utils import *
from datasets import get_fixer_dataloader

@dataclass
class TrainFixerVaeConfig:
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
    black_q_range: tuple[float, float] = (0.5, 0.9)


def run_one_epoch(
    fixer_model,
    dataloader,
    train_or_eval: str,
    config: TrainFixerVaeConfig,
    optimizer = None,
    ad_model = None,
):
    assert train_or_eval in ["train", "eval"]
    if train_or_eval == "eval":
        print(f"TODO: implement evaluation")

    device = next(fixer_model.parameters()).device

    num_dones = 0
    acc_kldiv_loss = 0. # This part is for actual VAE training
    acc_total_recon_loss = 0. # Recon loss for the entire image
    acc_okay_recon_loss = 0. # How we do for recon on the okay parts
    acc_loss = 0. # The total loss incurred

    pbar = tqdm(dataloader)
    for i, batch in enumerate(pbar):
        x = batch["image"].to(device)

        # Generate some "anomalous" images"
        N, C, H, W = x.shape
        q_lo, q_hi = config.black_q_range
        q = ((q_hi - q_lo) * torch.rand(()) + q_lo).to(x.device)
        anom_parts = make_blobs(N, H, W, q=q, device=x.device)
        okay_parts = 1 - anom_parts
        x_masked = (1 - anom_parts) * x

        with torch.set_grad_enabled(train_or_eval == "train"):
            fixer_out = fixer_model(x_masked, anom_parts)
            x_fix, mu, logvar = fixer_out.x_fix, fixer_out.others["mu"], fixer_out.others["logvar"]
            
            total_recon_err = (x - x_fix).norm(p=2, dim=(1,2,3)) ** 2   # (batch_size,)
            okay_recon_err = ((x - x_fix) * okay_parts).norm(p=2, dim=(1,2,3)) ** 2 # (batch_size,)
            total_recon_loss = total_recon_err.mean()
            okay_recon_loss = okay_recon_err.mean()
            kldiv_loss = -0.5 * torch.mean(1 + logvar - (mu**2) - logvar.exp())
            loss = total_recon_loss + kldiv_loss    # For now, don't include the okay_recon_loss
            if train_or_eval == "train":
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        num_dones += x.size(0)
        acc_loss += loss * x.size(0)
        acc_total_recon_loss += total_recon_loss * x.size(0)
        acc_okay_recon_loss += okay_recon_loss * x.size(0)
        acc_kldiv_loss += kldiv_loss * x.size(0)

        avg_loss = acc_loss / num_dones
        avg_total_recon_loss = acc_total_recon_loss / num_dones
        avg_okay_recon_loss = acc_okay_recon_loss / num_dones
        avg_kldiv_loss = acc_kldiv_loss / num_dones

        desc = "[train] " if train_or_eval == "train" else "[eval]  "
        desc += f"N {num_dones}, loss {avg_loss:.3f} "
        desc += f"(t_recon {avg_total_recon_loss:.3f}, "
        desc += f"o_recon {avg_okay_recon_loss:.3f}, "
        desc += f"kldiv {avg_kldiv_loss:.3f})"
        pbar.set_description(desc)

    return {
        "model": fixer_model,
        "loss": avg_loss,
        "total_recon_loss": avg_total_recon_loss,
        "okay_recon_loss": avg_okay_recon_loss,
        "kldiv_loss": avg_kldiv_loss
    }


def train_fixer_vae(config: TrainFixerVaeConfig):
    """ Set up the models, dataloaders, etc """
    fixer_model = VaeFixerModel(image_channels=config.image_channels)
    if config.use_cuda:
        fixer_model.cuda()

    train_dataloader = get_fixer_dataloader(
        dataset_name = "mvtec",
        model_name = "vae",
        batch_size = config.batch_size,
        category = config.mvtec_category,
        split = "train"
    )

    # TODO: implement the eval portion of training

    optimizer = torch.optim.AdamW(
        fixer_model.parameters(),
        lr = config.lr
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
        saveto_prefix = "fixer_vae_mvtec_{config.mvtec_category}"
        last_saveto = str(Path(config.output_dir, saveto_prefix + "_last.pt"))
        best_saveto = str(Path(config.output_dir, saveto_prefix + "_best.pt"))
    else:
        print(f"Warning: will NOT save models")

    best_loss = None

    for epoch in range(1, config.num_epochs+1):
        print(f"epoch: {epoch}/{config.num_epochs}, lr: {lr_scheduler.get_last_lr()[0]:.6f}")
        train_stats = run_one_epoch(fixer_model, train_dataloader, "train", config, optimizer)
        if epoch % config.eval_every == 0:
            eval_stats = None
            print(f"TODO: implement eval")

        lr_scheduler.step()

        save_dict = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "model_state_dict": {k: v.cpu() for (k, v) in fixer_model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        }

        if config.do_save:
            torch.save(save_dict, last_saveto)

        this_loss = train_stats["loss"]
        if best_loss is None or this_loss < best_loss:
            best_save_dict = save_dict
            delta = 0. if best_loss is None else (best_loss - this_loss)
            print(f"New best {this_loss:.3f}, delta {delta:.3f}")
            best_loss = this_loss
            if config.do_save:
                torch.save(save_dict, best_saveto)

    return None


def init_and_train_fixer_vae(args):
    assert args.model_name == "vae"
    assert args.dataset_name == "mvtec"
    config = TrainFixerVaeConfig(
        num_epochs = args.num_epochs,
        lr = args.lr,
        mvtec_category = args.mvtec_category,
        batch_size = args.batch_size,
        use_cuda = args.cuda,
        output_dir = args.output_dir
    )

    train_ret = train_fixer_vae(config)
    return train_ret

