import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from typing import Optional
from dataclasses import dataclass

from tqdm import tqdm

from .models import VaeFixerModel
from ad.models.vision import VaeADModel

from .image_utils import *
from mydatasets import get_fixer_dataloader

@dataclass
class TrainFixerVaeConfig:
    mvtec_category: str
    num_epochs: int
    lr: float
    batch_size: int
    device: Optional[str] = None
    do_save: bool = True
    output_dir: Optional[str] = None
    image_channels: int = 3
    warmup_ratio: float = 0.1
    eval_every: int = 5
    black_q_range: tuple[float, float] = (0.5, 0.9)
    ad_model_path: Optional[str] = None
    recon_scale: float = 10.
    kldiv_scale: float = 1.
    gamma1: float = 0.1
    gamma2: float = 0.1
    gamma3: float = 0.1
    lambda1: float = 0.1
    lambda2: float = 0.1
    lambda3: float = 0.1


def run_one_epoch(
    fixer_model,
    ad_model,
    dataloader,
    train_or_eval: str,
    config: TrainFixerVaeConfig,
    optimizer = None,
):
    assert train_or_eval in ["train", "eval"]
    device = next(fixer_model.parameters()).device

    num_dones = 0
    acc_kldiv_loss = 0. # This part is for actual VAE training
    acc_total_recon_loss = 0. # Recon loss for the entire image
    acc_good_recon_loss = 0. # How we do for recon on the good parts
    acc_loss = 0. # The total loss incurred

    pbar = tqdm(dataloader)
    for i, batch in enumerate(pbar):
        x = batch["image"].to(device)
        x = 2*x - 1
        N, C, H, W = x.shape

        # Generate x_masked and anom_parts
        if train_or_eval == "train":
            q_lo, q_hi = config.black_q_range
            q = ((q_hi - q_lo) * torch.rand(()) + q_lo).to(x.device)
            anom_parts = make_blobs(N, H, W, q=q, device=x.device)
            good_parts = 1 - anom_parts
            x_masked = good_parts * x

        # Otherwise use the ad model to generate this
        else:
            with torch.set_grad_enabled(False):
                ad_out = ad_model(x)
                alpha = ad_out.alpha.max(dim=1, keepdim=True).values # (N,1,H,W)
                q = (torch.rand(()) * 0.1 + 0.9).to(x.device) # Quantile in [0.9, 1.0]
                thresh = alpha.view(N,-1).quantile(q, dim=1).view(N,1,1,1)
                anom_parts = (alpha > thresh).long() # (N,1,H,W)
                good_parts = 1 - anom_parts
                x_masked = good_parts * x

        # Now run the fixer
        with torch.set_grad_enabled(train_or_eval == "train"):
            fixer_out = fixer_model(x_masked, anom_parts)
            x_fix, mu, logvar = fixer_out.x_fix, fixer_out.others["mu"], fixer_out.others["logvar"]
            
            total_recon_loss = F.mse_loss(x, x_fix) * config.recon_scale
            good_recon_loss = F.mse_loss(x * good_parts, x_fix * good_parts)

            raw_anom_score = ad_model(x).score
            fixed_anom_score = ad_model(x_fix).score
            good_anom_score = ad_model(x_masked).score
            bad_anom_score = ad_model(anom_parts*x).score

            global_loss = fixed_anom_score.sum()
            bad_loss = F.relu(bad_anom_score + config.gamma1).sum() * config.lambda1
            good_loss = F.relu(good_anom_score - config.gamma2).sum() * config.lambda2
            same_loss = F.relu(((good_parts * (x - x_fix))**2) - config.gamma3).sum() * config.lambda2
            kldiv_loss = (-0.5 * torch.mean(1 + logvar - (mu**2) - logvar.exp())) * config.kldiv_scale
            # For now, don't include the good_recon_loss
            loss = total_recon_loss + kldiv_loss  + global_loss + bad_loss + good_loss + same_loss 
            if train_or_eval == "train":
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        num_dones += x.size(0)
        acc_loss += loss * x.size(0)
        acc_total_recon_loss += total_recon_loss * x.size(0)
        acc_good_recon_loss += good_recon_loss * x.size(0)
        acc_kldiv_loss += kldiv_loss * x.size(0)

        avg_loss = acc_loss / num_dones
        avg_total_recon_loss = acc_total_recon_loss / num_dones
        avg_good_recon_loss = acc_good_recon_loss / num_dones
        avg_kldiv_loss = acc_kldiv_loss / num_dones

        desc = "[train] " if train_or_eval == "train" else "[eval]  "
        desc += f"N {num_dones}, loss {avg_loss:.3f} "
        desc += f"(t_recon {avg_total_recon_loss:.3f}, "
        desc += f"o_recon {avg_good_recon_loss:.3f}, "
        desc += f"kldiv {avg_kldiv_loss:.3f})"
        pbar.set_description(desc)

    return {
        "model": fixer_model,
        "loss": avg_loss,
        "total_recon_loss": avg_total_recon_loss,
        "good_recon_loss": avg_good_recon_loss,
        "kldiv_loss": avg_kldiv_loss
    }


def train_fixer_vae(config: TrainFixerVaeConfig):
    """ Set up the models, dataloaders, etc """
    fixer_model = VaeFixerModel(image_channels=config.image_channels)

    # Load the AD Model
    ad_model = VaeADModel(image_channels=config.image_channels)
    ad_model.load_state_dict(torch.load(config.ad_model_path)["model_state_dict"])
    ad_model.eval()

    if config.device is not None:
        fixer_model.to(config.device)
        ad_model.to(config.device)

    train_dataloader = get_fixer_dataloader(
        dataset_name = "mvtec",
        model_name = "vae",
        batch_size = config.batch_size,
        category = config.mvtec_category,
        split = "train"
    )

    eval_dataloader = get_fixer_dataloader(
        dataset_name = "mvtec",
        model_name = "vae",
        batch_size = config.batch_size,
        category = config.mvtec_category,
        split = "test"
    )

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
        saveto_prefix = f"fixer_vae_mvtec_{config.mvtec_category}"
        last_saveto = str(Path(config.output_dir, saveto_prefix + "_last.pt"))
        best_saveto = str(Path(config.output_dir, saveto_prefix + "_best.pt"))
    else:
        print(f"Warning: will NOT save models")

    best_loss = None

    for epoch in range(1, config.num_epochs+1):
        print(f"epoch: {epoch}/{config.num_epochs}, lr: {lr_scheduler.get_last_lr()[0]:.6f}")
        train_stats = run_one_epoch(fixer_model, ad_model, train_dataloader, "train", config, optimizer)
        if epoch % config.eval_every == 0:
            eval_stats = run_one_epoch(fixer_model, ad_model, eval_dataloader, "eval", config)

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
    assert args.ad_model_name == "vae"
    assert args.dataset_name == "mvtec"

    ad_model_path = Path(args.output_dir, f"ad_vae_mvtec_{args.mvtec_category}_best.pt")

    config = TrainFixerVaeConfig(
        num_epochs = args.num_epochs,
        lr = args.lr,
        mvtec_category = args.mvtec_category,
        batch_size = args.batch_size,
        device = args.device,
        output_dir = args.output_dir,
        ad_model_path = ad_model_path,
    )

    train_ret = train_fixer_vae(config)
    return train_ret

