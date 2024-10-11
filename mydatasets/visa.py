import torch
from torch.utils.data import Dataset
import torchvision.transforms as tfs
import datasets as hfds


HF_DATA_REPO = "BrachioLab/visa"

class VisADataset(Dataset):

    categories = [
        "candle",
        "capsules",
        "cashew",
        "chewinggum",
        "fryum",
        "macaroni1",
        "macaroni2",
        "pcb1",
        "pcb2",
        "pcb3",
        "pcb4",
        "pipe_fryum"
    ]

    def __init__(
        self,
        category: str,
        split: str,
        image_size: int = 256,
        hf_data_repo = HF_DATA_REPO,
    ):
        self.split = split
        self.dataset = hfds.load_dataset(hf_data_repo, split=(category + "." + split))
        self.dataset.set_format("torch")
        self.preprocess_image = tfs.Compose([
            tfs.Lambda(lambda x: x.float() / 255),
            tfs.Resize((image_size, image_size))
        ])

        self.preprocess_mask = tfs.Compose([
            tfs.Resize((image_size, image_size))
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = self.preprocess_image(item["image"])
        mask = self.preprocess_mask(item["mask"])
        _, H, W = image.shape

        return {
            "image": image,
            "mask": (mask.view(1,H,W) > 0).long(),
            "label": item["label"]
        }

