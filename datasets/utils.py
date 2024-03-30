from torch.utils.data import DataLoader
from .mvtec import *


def get_ad_dataloader(
    dataset_name: str,
    model_name: str,
    batch_size: int,
    **dataset_kwargs
):
    if dataset_name == "mvtec" and model_name == "vae":
        return DataLoader(
            MVTecDataset(**dataset_kwargs),
            batch_size = batch_size,
            shuffle = True
        )
    else:
        raise ValueError(f"Unknown combination of {dataset_name} and {model_name}")


