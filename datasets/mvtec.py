import os
import pwd
import sys
import torch
from typing import Optional
from torch.utils.data import Dataset
from torchvision import transforms

""" Hacky method for setting paths while we work on this """

EXLIB_DIR = "/home/antonxue/foo/exlib/src"
MVTEC_DIR = "/home/antonxue/foo/data/mvtec-ad"

sys.path.append(EXLIB_DIR)
from exlib.datasets.mvtec import MVTec

MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


class MVTecDataset(Dataset):
    """ Wrapper around an mvtec class that we use for vae training. """
    def __init__(
        self,
        category: str,
        mvtec_dir: str = MVTEC_DIR,
        split: str = "train",
        normalize_image: bool = False,
        **kwargs
    ):
        assert os.path.isdir(MVTEC_DIR)
        self.mvtec_dataset = MVTec(mvtec_dir, category, split=split, **kwargs)
        self.normalize_image = normalize_image
        self.normalize_transforms = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.mvtec_dataset)

    def __getitem__(self, idx):
        ret = self.mvtec_dataset[idx]
        if self.normalize_image:
            ret["image"] = self.normalize_transforms(ret["image"])

        return ret
    

