import os
import pwd
import sys
import torch
from typing import Optional
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, \
    random_split

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
        split = "train",
        **kwargs
    ):
        assert os.path.isdir(MVTEC_DIR)
        self.mvtec_dataset = MVTec(mvtec_dir, category, split=split, **kwargs)

    def __len__(self):
        return len(self.mvtec_dataset)

    def __getitem__(self, idx):
        image, mask, y = self.mvtec_dataset[idx]
        return image, mask, y
    
    def get_mask(self, idx):
        _, mask, _ = self.mvtec_dataset[idx]
        return mask
    



