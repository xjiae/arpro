import os
import pwd
import sys
import torch
from typing import Optional
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, \
    random_split

### replace the paths with the directory to the dataset folder

sys.path.insert(0,"/home/antonxue/foo/exlib/src/exlib/datasets")
import visa

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
        split = "train",
        image_size: int = 256,
    ):
        self.visa_dataset = visa.VisADataset(category, split=split, image_size=image_size)

    def __len__(self):
        return len(self.visa_dataset)

    def __getitem__(self, idx):
        return self.visa_dataset[idx]
    

