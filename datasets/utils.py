from torch.utils.data import DataLoader
from .mvtec import *


def get_ad_dataloader(
    dataset_name: str,
    batch_size: int,
    model_name: Optional[str] = None,
    **dataset_kwargs
):
    if dataset_name == "mvtec" and model_name is None:
        return DataLoader(
            MVTecDataset(**dataset_kwargs),
            batch_size = batch_size,
            shuffle = True
        )
    else:
        raise ValueError(f"Unknown combination of {dataset_name} and {model_name}")


def get_fixer_dataloader(
    dataset_name: str,
    batch_size: int,
    model_name: Optional[str] = None,
    **dataset_kwargs
):
    if dataset_name == "mvtec" and model_name is None:
        return DataLoader(
            MVTecDataset(**dataset_kwargs),
            batch_size = batch_size,
            shuffle = True
        )
    else:
        raise ValueError(f"Unknown combination of {dataset_name} and {model_name}")


