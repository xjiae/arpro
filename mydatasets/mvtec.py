import torch
from torch.utils.data import Dataset
import torchvision.transforms as tfs
import datasets as hfds

HF_DATA_REPO = "BrachioLab/mvtec-ad"

class MVTecDataset(Dataset):

    categories = [
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

    def __init__(
        self,
        category: str,
        split: str,
        image_size: int = 256,
        hf_data_repo = HF_DATA_REPO,
        normalize_image: bool = False
    ):
        self.split = split
        self.dataset = hfds.load_dataset(hf_data_repo, split=(category + "." + split))
        self.dataset.set_format("torch")
        self.preprocess_image = tfs.Compose([
            tfs.Lambda(lambda x: x.float() / 255),
            tfs.Resize((image_size,image_size))
        ])
        self.normalize_image = normalize_image

        self.preprocess_mask = tfs.Compose([
            tfs.Resize((image_size, image_size))
        ])

        self.normalize_transform = tfs.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = self.preprocess_image(item["image"])

        if self.normalize_image:
            image = self.normalize_transform(image)

        mask = self.preprocess_mask(item["mask"])
        _, H, W = image.shape
        return {
            "image": image,
            "mask": (mask.view(1,H,W) > 0).long(),
            "label": item["label"]
        }

