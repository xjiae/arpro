import argparse
from pathlib import Path

from mydatasets import *
from ad import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="fastflow")
    
    # Model-specific parameters
    parser.add_argument("--recon_scale", type=float, default=1.0)
    parser.add_argument("--kldiv_scale", type=float, default=1.0)
    parser.add_argument("--contrast_scale", type=float, default=1.0)

    # Dataset-specific parameters
    parser.add_argument("--dataset", type=str, default="mvtec")
    parser.add_argument("--category", type=str, default="transistor")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2")
    parser.add_argument("--efficientad_imagenette_dir", type=str,
        default=str(Path(Path(__file__).parent.resolve(), "data", "imagenette")))
    
    parser.add_argument("--efficientad_pretrained_download_dir",
        default=str(Path(Path(__file__).parent.resolve(), "data", "efficientad_downloads")))

    # Training-specific details
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--output_dir", type=str,
        default=str(Path(Path(__file__).parent.resolve(), "_dump")))

    # Wandb
    parser.add_argument("--wandb", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()


    if args.model == "vae" and args.dataset == "mvtec":
        init_and_train_ad_vae(args)
    if args.model == "efficientad":
        init_and_train_ad_efficient_ad(args)
    if args.model == "fastflow":
        init_and_train_ad_fastflow(args)
    if args.model == "gpt2":
        init_and_train_ad_gpt2(args)
    if args.model == "llama2":
        init_and_train_ad_llama2(args)


