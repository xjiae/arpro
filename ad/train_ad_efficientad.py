import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torchvision.transforms.v2 import CenterCrop, Compose, Normalize, RandomGrayscale, Resize, ToTensor, Transform
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from typing import Optional
from dataclasses import dataclass
from anomalib.data.utils import DownloadInfo, download_and_extract
from tqdm import tqdm

import wandb

from .models import EfficientAdADModel
from mydatasets import get_ad_dataloader


@dataclass
class TrainADVaeConfig:
    mvtec_category: str
    dataset: str
    num_epochs: int = 50000
    lr: float = 0.0001
    batch_size: int = 1
    device: Optional[str] = None
    output_dir: Optional[str] = None
    image_channels: int = 3
    warmup_ratio: float = 0.1
    eval_every: int = 5
    recon_scale: float = 1.0
    kldiv_scale: float = 1.0
    contrastive_scale: float = 0.01
    wandb_project: str = "arpro"
    imagenette_dir: Optional[str] = None
    pretrained_dir: Optional[str] = None
    image_size: tuple[int, int] = (256, 256)
    do_save: bool=True



## code adapted from https://github.com/openvinotoolkit/anomalib/blob/main/src/anomalib/models/image/efficient_ad/lightning_model.py
def prepare_pretrained_model(pretrained_models_dir: str = "../data/pre_trained/", model_size: str = "medium") -> None:
        WEIGHTS_DOWNLOAD_INFO = DownloadInfo(
            name="efficientad_pretrained_weights.zip",
            url="https://github.com/openvinotoolkit/anomalib/releases/download/efficientad_pretrained_weights/efficientad_pretrained_weights.zip",
            hashsum="c09aeaa2b33f244b3261a5efdaeae8f8284a949470a4c5a526c61275fe62684a",
        )
        """Prepare the pretrained teacher model."""
        pretrained_models_dir = Path(pretrained_models_dir)
        if not (pretrained_models_dir / "efficientad_pretrained_weights").is_dir():
            download_and_extract(pretrained_models_dir, WEIGHTS_DOWNLOAD_INFO)
        teacher_path = (
            pretrained_models_dir / "efficientad_pretrained_weights" / f"pretrained_teacher_{model_size}.pth"
        )
        return teacher_path
        


def prepare_imagenette_data(image_size: tuple[int, int] | torch.Size,
                            imagenet_dir: str = "../data/imagenette",
                            batch_size: int=1) -> None:
    """Prepare ImageNette dataset transformations.

    Args:
        image_size (tuple[int, int] | torch.Size): Image size.
    """
    data_transforms_imagenet = Compose(
        [
            Resize((image_size[0] * 2, image_size[1] * 2)),
            RandomGrayscale(p=0.3),
            CenterCrop((image_size[0], image_size[1])),
            ToTensor(),
        ],
    )
    imagenet_dataset = ImageFolder(imagenet_dir, transform=data_transforms_imagenet)
    imagenet_loader = DataLoader(imagenet_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    imagenet_iterator = iter(imagenet_loader)
    return imagenet_iterator, imagenet_loader,

def run_one_epoch(
    model,
    dataloader,
    imagenet_iterator,
    imagenet_loader,
    train_or_eval: str,
    config: TrainADVaeConfig,
    optimizer = None,
):
    assert train_or_eval in ["train", "eval"]
    _ = model.train() if train_or_eval == "train" else model.eval()
    device = next(model.parameters()).device

    num_dones, acc_loss, acc_st_loss, acc_ae_loss, acc_stae_loss = 0, 0., 0., 0., 0.
    pbar = tqdm(dataloader)
    for i, batch in enumerate(pbar):
        x = batch["image"].to(device)
        x = 2*x - 1 # Scale to [-1,+1]
        
        with torch.set_grad_enabled(train_or_eval == "train"):
            if train_or_eval == "train":
                try:
                    # infinite dataloader; [0] getting the image not the label
                    batch_imagenet = next(imagenet_iterator)[0].to(config.device)
                except StopIteration:
                    imagenet_iterator = iter(imagenet_loader)
                    batch_imagenet = next(imagenet_iterator)[0].to(config.device)
                
                out = model(x, batch_imagenet)                
                st_loss, ae_loss, stae_loss = out.others['loss_st'].mean(), out.others['loss_ae'].mean(), out.others['loss_stae'].mean()
                loss = st_loss + ae_loss + stae_loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                num_dones += x.size(0)
                acc_loss += loss * x.size(0)
                acc_st_loss += st_loss * x.size(0)
                acc_ae_loss += ae_loss * x.size(0)
                acc_stae_loss += stae_loss * x.size(0) 
            else:
                out = model(x)


        avg_loss = acc_loss / num_dones
        avg_st_loss = acc_st_loss / num_dones
        avg_ae_loss = acc_ae_loss / num_dones
        avg_stae_loss = acc_stae_loss / num_dones
        desc = "[train] " if train_or_eval == "train" else "[eval]  "
        desc += f"N {num_dones}, loss {avg_loss:.4f} "
        desc += f"(st{avg_st_loss:.4f}, ae {avg_ae_loss:.4f}, stae {avg_stae_loss:.4f})"
        pbar.set_description(desc)

        wandb.log({
            "train_loss": avg_loss,
            "train_st_loss": avg_st_loss,
            "train_ae_loss": avg_ae_loss,
            "train_stae_loss": avg_stae_loss,
        })

    return {
        "model": model,
        "loss": avg_loss,
        "st_loss": avg_st_loss,
        "ae_loss": avg_ae_loss,
    }


def train_ad_efficient_ad(config: TrainADVaeConfig):
    """ Set up the models, dataloaders, optimizers, etc and start training """
    model = EfficientAdADModel(model_size="medium")
    if config.device is not None:
        model.to(config.device)
    teacher_path = prepare_pretrained_model(config.pretrained_dir)
    model.teacher.load_state_dict(torch.load(teacher_path, map_location=torch.device(config.device)))
    imagenet_iterator, imagenet_loader = prepare_imagenette_data(config.image_size, config.imagenette_dir)

    train_dataloader = get_ad_dataloader(
        dataset_name = config.dataset,
        batch_size = config.batch_size,
        category = config.mvtec_category,
        split = "train"
    )

    eval_dataloader = get_ad_dataloader(
        dataset_name = config.dataset,
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

    run_name = f"ad_eff_{config.dataset}_{config.mvtec_category}"

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
        train_stats = run_one_epoch(model, train_dataloader, imagenet_iterator, imagenet_loader, "train", config, optimizer)
        # if epoch % config.eval_every == 0:
        #     eval_stats = run_one_epoch(model, eval_dataloader, "eval", config)


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


def init_and_train_ad_efficient_ad(args):
    assert args.model == "efficientad"
    # assert args.dataset == "mvtec"
    config = TrainADVaeConfig(
        num_epochs = args.epochs,
        lr = args.lr,
        mvtec_category = args.category,
        batch_size = args.batch_size,
        device = args.device,
        output_dir = args.output_dir,
        recon_scale = args.recon_scale,
        kldiv_scale = args.kldiv_scale,
        contrastive_scale = args.contrast_scale,
        imagenette_dir = args.efficientad_imagenette_dir,
        pretrained_dir= args.efficientad_pretrained_download_dir,
        dataset=args.dataset
    )
    
    train_ret = train_ad_efficient_ad(config)
    return train_ret

