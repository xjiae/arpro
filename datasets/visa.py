import os
import pwd
import sys
import torch
from typing import Optional
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, \
    random_split

""" Hacky method for setting paths while we work on this """

username = pwd.getpwuid(os.getuid()).pw_name

if username == "xjiae":
    EXLIB_PATH = "/home/xjiae/bussie/exlib/src/"
    VISA_DIR = "/home/xjiae/bussie/data/visa/1cls/"
elif username == "antonxue":
    EXLIB_PATH = "/home/antonxue/foo/exlib/src/"
    VISA_DIR = "/home/antonxue/foo/data/visa/1cls/"
else:
    print(f"datasets/mvtec.py: unexpected username {username}, may not set paths correctly")
    EXLIB_PATH = "../exlib/src/"
    VISA_DIR = "../data/visa/1cls"

sys.path.append(EXLIB_PATH)
from exlib.datasets.visa import VisA

VISA_CATEGORIES = ['candle', 
                   'capsules', 
                   'cashew', 
                   'chewinggum', 
                   'fryum', 
                   'macaroni1', 
                   'macaroni2', 
                   'pcb1', 
                   'pcb2',
                   'pcb3', 
                   'pcb4', 
                   'pipe_fryum']

class VisADataset(Dataset):
    """ Wrapper around an mvtec class that we use for diffusion model training.
        Since diffusion training is an unsupervised method, we do not need labels,
        and we modify __getitem__ accordingly.
    """
    def __init__(
        self,
        category: str,
        visa_dir: str = VISA_DIR,
        split = "train",
        **kwargs
    ):
        assert os.path.isdir(VISA_DIR)
        self.visa_dataset = VisA(visa_dir, category, split=split, **kwargs)

    def __len__(self):
        return len(self.visa_dataset)

    def __getitem__(self, idx):
        return self.visa_dataset[idx]
    

