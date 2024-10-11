import sys
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torchvision import transforms
from typing import Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from .models import VaeFixerModel
# from ad.models.vision import FastflowAdModel
from ad.models.vision import EfficientAdADModel
from ad.train_ad_efficientad import *
from .image_utils import *
from mydatasets import get_fixer_dataloader

@dataclass
class TrainFixerVaeConfig:
    dataset: str
    category: str
    num_epochs: int
    lr: float
    batch_size: int
    image_size: int = 512
    device: Optional[str] = None
    do_save: bool = True
    output_dir: Optional[str] = None
    image_channels: int = 3
    warmup_ratio: float = 0.1
    eval_every: int = 5
    black_q_range: tuple[float, float] = (0.5, 0.9)
    ad_model_path: Optional[str] = None
    recon_scale: float = 10.
    kldiv_scale: float = 1.0
    prop1_scale: float = 1e-5
    prop2_scale: float = 1.0
    prop3_scale: float = 0.1
    prop4_scale: float = 0.1
    wandb_project: str = "arpro"



def run_one_epoch(
    fixer_model,
    ad_model,
    dataloader,
    imagenet_iterator,
    imagenet_loader,
    train_or_eval: str,
    config: TrainFixerVaeConfig,
    optimizer = None,
):
    assert train_or_eval in ["train", "eval"]
    device = next(fixer_model.parameters()).device

    num_dones = 0
    acc_kldiv_loss = 0. # This part is for actual VAE training
    acc_total_recon_loss = 0. # Recon loss for the entire image
    acc_loss = 0. # The total loss incurred
    acc_prop1_loss = 0.
    acc_prop2_loss = 0.
    acc_prop3_loss = 0.
    acc_prop4_loss = 0.

    pbar = tqdm(dataloader)
    for idx, batch in enumerate(pbar):
        x = batch["image"].to(device)
        x = 2*x - 1
        N, C, H, W = x.shape

        try:
            # infinite dataloader; [0] getting the image not the label
            batch_imagenet = next(imagenet_iterator)[0].to(config.device)
        except StopIteration:
            imagenet_iterator = iter(imagenet_loader)
            batch_imagenet = next(imagenet_iterator)[0].to(config.device)
        # Either way, we need information about the ad_out(x)
        with torch.no_grad():
            ad_out = ad_model(x, batch_imagenet)

        # Because every training point is "good", we need to artificially break stuff
        if train_or_eval == "train":
            
            q_lo, q_hi = config.black_q_range
            q = ((q_hi - q_lo) * torch.rand(()) + q_lo).to(x.device)
            anom_parts = make_blobs(N, H, W, q=q, device=x.device)
            good_parts = 1 - anom_parts
            x_goods = good_parts * x
            

        # Otherwise we take the output of the ad_model
        else:
            alpha = ad_out.alpha.max(dim=1, keepdim=True).values # (N,1,H,W)
            q = torch.rand(N).to(config.device)
            thresh = alpha.view(N,-1).quantile(q).view(N,1,1,1)
            anom_parts = (alpha > thresh).long() # (N,1,H,W)
            good_parts = 1 - anom_parts
            x_goods = good_parts * x

        # We now run the fixer to figure out a candidate repair (or lack therefore for training)
        with torch.set_grad_enabled(train_or_eval == "train"):
            fixer_out = fixer_model(x_goods, anom_parts)
            # visualize the images
            """
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            inverse_normalize_transform = transforms.Compose([
                transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std]),
            ])
            xr = inverse_normalize_transform(x).clamp(0,1)
            """
            xr = x.clone().detach()
            xr = (xr + 1) * 0.5
            if train_or_eval == "train":
                plt.clf()
                fig, ax = plt.subplots(3,x.size(0), figsize=(12, 8))
                for i in range(x.size(0)):
                    xi = xr[i].detach()
                    xhati = (fixer_out.x_fix[i].detach()*0.5 + 0.5).clamp(0,1)
                    alphai = anom_parts[i].detach()
                    ax[0,i].imshow(xi.cpu().numpy().transpose(1,2,0))
                    ax[1,i].imshow(xhati.cpu().numpy().transpose(1,2,0))
                    ax[2,i].imshow(alphai.cpu().numpy().transpose(1,2,0))
                plt.savefig(f"{config.output_dir}/fixer_train_img/train{idx}.png")
                plt.close()
            x_fix, mu, logvar = fixer_out.x_fix, fixer_out.others["mu"], fixer_out.others["logvar"]
            x_fix_ad_out = ad_model(x_fix, batch_imagenet) # Need to evaluate the ad_model on this thing

            total_recon_loss = F.mse_loss(x, x_fix) * config.recon_scale
            kldiv_loss = (-0.5 * torch.mean(1 + logvar - (mu**2) - logvar.exp())) * config.kldiv_scale

            prop1_loss = x_fix_ad_out.score.mean() * config.prop1_scale
            prop2_loss = F.mse_loss(good_parts * x_fix, good_parts * x)
            prop3_loss = F.relu((x_fix_ad_out.alpha - ad_out.alpha) * anom_parts).mean() * config.prop3_scale
            prop4_loss = F.relu((x_fix_ad_out.alpha - ad_out.alpha) * good_parts).mean() * config.prop4_scale
            
            loss = total_recon_loss + kldiv_loss + prop1_loss + prop2_loss + prop3_loss + prop4_loss
            if train_or_eval == "train":
                loss += F.mse_loss(x_fix * anom_parts, x * anom_parts) * 100
            
            
            if train_or_eval == "train":
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        num_dones += x.size(0)
        acc_loss += loss * x.size(0)
        acc_total_recon_loss += total_recon_loss * x.size(0)
        # acc_good_recon_loss += good_recon_loss * x.size(0)
        acc_kldiv_loss += kldiv_loss * x.size(0)
        acc_prop1_loss += prop1_loss * x.size(0)
        acc_prop2_loss += prop2_loss * x.size(0)
        acc_prop3_loss += prop3_loss * x.size(0)
        acc_prop4_loss += prop3_loss * x.size(0)

        avg_loss = acc_loss / num_dones
        avg_total_recon_loss = acc_total_recon_loss / num_dones
        avg_kldiv_loss = acc_kldiv_loss / num_dones
        avg_prop1_loss = acc_prop1_loss / num_dones
        avg_prop2_loss = acc_prop2_loss / num_dones
        avg_prop3_loss = acc_prop3_loss / num_dones
        avg_prop4_loss = acc_prop4_loss / num_dones

        desc = "[train] " if train_or_eval == "train" else "[eval]  "
        desc += f"N {num_dones}, loss {avg_loss:.3f} "
        desc += f"(trec {avg_total_recon_loss:.3f}, "
        desc += f"kld {avg_kldiv_loss:.3f}, "
        desc += f"P1 {avg_prop1_loss:.3f}, "
        desc += f"P2 {avg_prop2_loss:.3f}, "
        desc += f"P3 {avg_prop3_loss:.3f}, "
        desc += f"P4 {avg_prop4_loss:.3f})"
        pbar.set_description(desc)

        wandb.log({
            "train_loss": avg_loss,
            "train_total_recon_loss": avg_total_recon_loss,
            "train_kldiv_loss": avg_kldiv_loss,
            "train_prop1_loss": avg_prop1_loss,
            "train_prop2_loss": avg_prop2_loss,
            "train_prop3_loss": avg_prop3_loss,
            "train_prop4_loss": avg_prop4_loss
        })

        

    return {
        "model": fixer_model,
        "loss": avg_loss,
        "total_recon_loss": avg_total_recon_loss,
        "kldiv_loss": avg_kldiv_loss
    }


def train_fixer_efficientad(config: TrainFixerVaeConfig):
    """ Set up the models, dataloaders, etc """
    fixer_model = VaeFixerModel(image_channels=config.image_channels)

    # Load the AD Model
    ad_model = EfficientAdADModel(model_size="medium")
    if config.device is not None:
        ad_model.to(config.device)
    teacher_path = prepare_pretrained_model(config.pretrained_dir)
    ad_model.teacher.load_state_dict(torch.load(teacher_path, map_location=torch.device(config.device)))
    imagenet_iterator, imagenet_loader = prepare_imagenette_data(config.image_size, config.imagenette_dir)

    # ad_model = FastflowAdModel()
    ad_model.load_state_dict(torch.load(config.ad_model_path)["model_state_dict"])
    ad_model.eval()

    if config.device is not None:
        fixer_model.to(config.device)
        ad_model.to(config.device)

    train_dataloader = get_fixer_dataloader(
        dataset_name = config.dataset,
        batch_size = config.batch_size,
        category = config.category,
        split = "train"
    )

    eval_dataloader = get_fixer_dataloader(
        dataset_name = config.dataset,
        batch_size = config.batch_size,
        category = config.category,
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

    run_name = f"fixer_eff_{config.dataset}_{config.category}"

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
        train_stats = run_one_epoch(fixer_model, 
                                    ad_model, 
                                    train_dataloader, 
                                    imagenet_iterator,
                                    imagenet_loader,
                                    "train", 
                                    config, 
                                    optimizer)
        wandb.log({
            "learning_rate": lr_scheduler.get_last_lr()[0]
        })
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

    wandb.finish()
    return None


def init_and_train_fixer_efficientad(args):
    assert args.ad_model_name == "efficientad"

    ad_model_path = Path(args.output_dir, f"ad_eff_{args.dataset}_{args.category}_best.pt")

    config = TrainFixerVaeConfig(
        dataset = args.dataset,
        num_epochs = args.num_epochs,
        lr = args.lr,
        category = args.category,
        batch_size = args.batch_size,
        device = args.device,
        output_dir = args.output_dir,
        ad_model_path = ad_model_path,
        image_size=args.image_size
    )

    train_ret = train_fixer_efficientad(config)
    return train_ret

