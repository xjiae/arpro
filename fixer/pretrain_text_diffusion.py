import sys
import os
import math
from pathlib import Path
from typing import Optional
import torch
import torch.nn.functional as F
from torchvision import transforms
import wandb
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from dataclasses import dataclass

from tqdm import tqdm

from .models import MyTextDiffusionModel
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
    wandb_project: str = "arpro"
    


def run_one_epoch(
    diff_model: MyTextDiffusionModel,
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
        input_ids = batch[0].long().to(device)
        lengths = batch[1].long().to(device)
        mask = batch[2].to(device)
        loss, loss_diff, loss_reconstruction, accuracy = diff_model.compute_loss(input_ids, lengths, mask)
        '''
        # print(diff_model.tokenizer.decode(input_ids[0]))
        position_ids = diff_model.model.bert.embeddings.position_ids[:, 0 : input_ids.size(1)]
        position_embeddings = diff_model.model.bert.embeddings.position_embeddings(position_ids)
        token_type_ids = torch.zeros(input_ids.size(0),diff_model.max_len).long().to(device)
        token_type_embeddings = diff_model.model.bert.embeddings.token_type_embeddings(token_type_ids)
        
        x = diff_model.model.bert.embeddings.word_embeddings(input_ids)
        noise = torch.randn_like(x)/math.sqrt(diff_model.model.config.hidden_size)
        t = torch.randint(0, diff_model.num_timesteps, (x.size(0),)).to(device)
        time_embedding = diff_model.time_embed(t).unsqueeze(1)
        noised_text = diff_model.add_noise(x, noise, t)
        noised_text += token_type_embeddings + position_embeddings + time_embedding
        noised_text = diff_model.model.bert.embeddings.LayerNorm(noised_text)
        '''
        with torch.set_grad_enabled(train_or_eval == "train"):
            '''
            noise_pred, pred_scores = diff_model.estimate_noise(noised_text, t)
            pred_ids = torch.argmax(pred_scores,-1).long()
            # print("---------")
            # print(diff_model.tokenizer.decode(pred_ids[0]))
            # breakpoint()
            loss = F.cross_entropy(pred_scores.view(-1, diff_model.model.config.vocab_size), \
                                   input_ids.flatten(), ignore_index=0)
            # loss = F.mse_loss(input_ids, pred_ids)
            '''
            if train_or_eval == "train":
                loss.backward()
                torch.nn.utils.clip_grad_norm_(diff_model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

        num_dones += input_ids.size(0)
        acc_loss += loss * input_ids.size(0)

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


def pretrain_text_diffusion(config: PretrainMyDiffusionConfig):
    
    diff_model = MyTextDiffusionModel(num_embeddings=50265, embedding_dim=768)

    if config.device is not None:
        diff_model.to(config.device)

    train_dataloader = get_fixer_dataloader(
                        dataset_name = config.dataset,
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

    run_name = f"fixer_diffusion_{config.dataset}"

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


def init_and_pretrain_text_diffusion(args):
    assert args.model == "textdiffusion"
    config = PretrainMyDiffusionConfig(
        dataset = args.dataset,
        num_epochs = args.num_epochs,
        lr = args.lr,
        category = args.category,
        batch_size = args.batch_size,
        device = args.device,
        output_dir = args.output_dir,
    )

    train_ret = pretrain_text_diffusion(config)
    return train_ret
