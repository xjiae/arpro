import argparse
from pathlib import Path

from datasets import *
from fixer import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="vae")
    
    # Model-specific parameters
    parser.add_argument("--dataset_name", type=str, default="mvtec")
    
    # Dataset-specific parameters
    parser.add_argument("--mvtec_category", type=str, default="transistor")

    # Training-specific details
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--cuda", default=False, action="store_true")

    parser.add_argument("--output_dir",
        default=str(Path(Path(__file__).parent.resolve(), "_dump")))

    # Wandb
    parser.add_argument("--wandb", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()


    if args.model_name == "vae" and args.dataset_name == "mvtec":
        init_and_train_fixer_vae(args)



