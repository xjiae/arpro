import argparse
from pathlib import Path

from mydatasets import *
from fixer import *
from ad.models import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="diffusion")

    # The AD model used to evaluate stuffs with
    parser.add_argument("--ad_model", type=str, default="vae")
    
    # Model-specific parameters
    parser.add_argument("--dataset", type=str, default="mvtec")
    
    # Dataset-specific parameters
    parser.add_argument("--category", type=str, default="transistor")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--feature_dim", type=int, default=51)

    # Training-specific details
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--output_dir",
        default=str(Path(Path(__file__).parent.resolve(), "_dump")))


    # Wandb
    parser.add_argument("--wandb", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()


    if args.model == "diffusion":
        init_and_pretrain_diffusion(args)

    elif args.model == "textdiffusion":
        init_and_pretrain_text_diffusion(args)

    elif args.model == "ts_diffusion":
        init_and_pretrain_ts_diffusion(args)

    elif args.ad_model == "vae":
        init_and_train_fixer_vae(args)

    elif args.ad_model== "fastflow":
        init_and_train_fixer_fastflow(args)

    elif args.ad_model== "efficientad":
        init_and_train_fixer_efficientad(args)

    

