import math
from enum import Enum
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F  # noqa: N812
from torchvision import transforms
from ..common import *
from anomalib.data.utils import DownloadInfo, download_and_extract
from torchvision.transforms.v2 import CenterCrop, Compose, Normalize, RandomGrayscale, Resize, ToTensor, Transform
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def imagenet_norm_batch(x: torch.Tensor) -> torch.Tensor:
    """Normalize batch of images with ImageNet mean and std.

    Args:
        x (torch.Tensor): Input batch.

    Returns:
        torch.Tensor: Normalized batch using the ImageNet mean and std.
    """
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None].to(x.device)
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None].to(x.device)
    return (x - mean) / std


def reduce_tensor_elems(tensor: torch.Tensor, m: int = 2**24) -> torch.Tensor:
    """Reduce tensor elements.

    This function flatten n-dimensional tensors,  selects m elements from it
    and returns the selected elements as tensor. It is used to select
    at most 2**24 for torch.quantile operation, as it is the maximum
    supported number of elements.
    https://github.com/pytorch/pytorch/blob/b9f81a483a7879cd3709fd26bcec5f1ee33577e6/aten/src/ATen/native/Sorting.cpp#L291.

    Args:
        tensor (torch.Tensor): input tensor from which elements are selected
        m (int): number of maximum tensor elements.
            Defaults to ``2**24``

    Returns:
            Tensor: reduced tensor
    """
    tensor = torch.flatten(tensor)
    if len(tensor) > m:
        # select a random subset with m elements.
        perm = torch.randperm(len(tensor), device=tensor.device)
        idx = perm[:m]
        tensor = tensor[idx]
    return tensor


class EfficientAdModelSize(str, Enum):
    """Supported EfficientAd model sizes."""

    M = "medium"
    S = "small"


class SmallPatchDescriptionNetwork(nn.Module):
    """Patch Description Network small.

    Args:
        out_channels (int): number of convolution output channels
        padding (bool): use padding in convoluional layers
            Defaults to ``False``.
    """

    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__()
        pad_mult = 1 if padding else 0
        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1 * pad_mult)
        self.conv4 = nn.Conv2d(256, out_channels, kernel_size=4, stride=1, padding=0 * pad_mult)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input batch.

        Returns:
            torch.Tensor: Output from the network.
        """
        x = imagenet_norm_batch(x)
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        return self.conv4(x)


class MediumPatchDescriptionNetwork(nn.Module):
    """Patch Description Network medium.

    Args:
        out_channels (int): number of convolution output channels
        padding (bool): use padding in convoluional layers
            Defaults to ``False``.
    """

    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__()
        pad_mult = 1 if padding else 0
        self.conv1 = nn.Conv2d(3, 256, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0 * pad_mult)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1 * pad_mult)
        self.conv5 = nn.Conv2d(512, out_channels, kernel_size=4, stride=1, padding=0 * pad_mult)
        self.conv6 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0 * pad_mult)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input batch.

        Returns:
            torch.Tensor: Output from the network.
        """
        x = imagenet_norm_batch(x)
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return self.conv6(x)


class Encoder(nn.Module):
    """Autoencoder Encoder model."""

    def __init__(self) -> None:
        super().__init__()
        self.enconv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.enconv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.enconv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.enconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv6 = nn.Conv2d(64, 64, kernel_size=8, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass through the network.

        Args:
            x (torch.Tensor): Input batch.

        Returns:
            torch.Tensor: Output from the network.
        """
        x = F.relu(self.enconv1(x))
        x = F.relu(self.enconv2(x))
        x = F.relu(self.enconv3(x))
        x = F.relu(self.enconv4(x))
        x = F.relu(self.enconv5(x))
        return self.enconv6(x)


class Decoder(nn.Module):
    """Autoencoder Decoder model.

    Args:
        out_channels (int): number of convolution output channels
        padding (int): use padding in convoluional layers
    """

    def __init__(self, out_channels: int, padding: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.padding = padding
        # use ceil to match output shape of PDN
        self.deconv1 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv2 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv3 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv6 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv8 = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.dropout4 = nn.Dropout(p=0.2)
        self.dropout5 = nn.Dropout(p=0.2)
        self.dropout6 = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor, image_size: tuple[int, int] | torch.Size) -> torch.Tensor:
        """Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input batch.
            image_size (tuple): size of input images.

        Returns:
            torch.Tensor: Output from the network.
        """
        last_upsample = (
            math.ceil(image_size[0] / 4) if self.padding else math.ceil(image_size[0] / 4) - 8,
            math.ceil(image_size[1] / 4) if self.padding else math.ceil(image_size[1] / 4) - 8,
        )
        x = F.interpolate(x, size=(image_size[0] // 64 - 1, image_size[1] // 64 - 1), mode="bilinear")
        x = F.relu(self.deconv1(x))
        x = self.dropout1(x)
        x = F.interpolate(x, size=(image_size[0] // 32, image_size[1] // 32), mode="bilinear")
        x = F.relu(self.deconv2(x))
        x = self.dropout2(x)
        x = F.interpolate(x, size=(image_size[0] // 16 - 1, image_size[1] // 16 - 1), mode="bilinear")
        x = F.relu(self.deconv3(x))
        x = self.dropout3(x)
        x = F.interpolate(x, size=(image_size[0] // 8, image_size[1] // 8), mode="bilinear")
        x = F.relu(self.deconv4(x))
        x = self.dropout4(x)
        x = F.interpolate(x, size=(image_size[0] // 4 - 1, image_size[1] // 4 - 1), mode="bilinear")
        x = F.relu(self.deconv5(x))
        x = self.dropout5(x)
        x = F.interpolate(x, size=(image_size[0] // 2 - 1, image_size[1] // 2 - 1), mode="bilinear")
        x = F.relu(self.deconv6(x))
        x = self.dropout6(x)
        x = F.interpolate(x, size=last_upsample, mode="bilinear")
        x = F.relu(self.deconv7(x))
        return self.deconv8(x)



class AutoEncoder(nn.Module):
    """EfficientAd Autoencoder.

    Args:
       out_channels (int): number of convolution output channels
       padding (int): use padding in convoluional layers
    """

    def __init__(self, out_channels: int, padding: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = Encoder()
        self.decoder = Decoder(out_channels, padding)

    def forward(self, x: torch.Tensor, image_size: tuple[int, int] | torch.Size) -> torch.Tensor:
        """Perform the forward pass through the network.

        Args:
            x (torch.Tensor): Input batch.
            image_size (tuple): size of input images.

        Returns:
            torch.Tensor: Output from the network.
        """
        x = imagenet_norm_batch(x)
        x = self.encoder(x)
        return self.decoder(x, image_size)

IMAGENETTE_DOWNLOAD_INFO = DownloadInfo(
    name="imagenette2.tgz",
    url="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz",
    hashsum="6cbfac238434d89fe99e651496f0812ebc7a10fa62bd42d6874042bf01de4efd",
)

class EfficientAdModel(nn.Module):
    """EfficientAd model.

    Args:
        teacher_out_channels (int): number of convolution output channels of the pre-trained teacher model
        model_size (str): size of student and teacher model
        padding (bool): use padding in convoluional layers
            Defaults to ``False``.
        pad_maps (bool): relevant if padding is set to False. In this case, pad_maps = True pads the
            output anomaly maps so that their size matches the size in the padding = True case.
            Defaults to ``True``.
    """

    

    def __init__(
        self,
        image_size: tuple[int, int] | torch.Size = (256,256),
        imagenet_dir: Path | str = "./datasets/imagenette",
        teacher_out_channels: int = 384,
        model_size: EfficientAdModelSize = EfficientAdModelSize.S,
        padding: bool = False,
        pad_maps: bool = True,
        batch_size: int = 1,
        device: str = "cuda"
    ) -> None:
        super().__init__()
        self.imagenet_dir = Path(imagenet_dir)
        self.pad_maps = pad_maps
        self.teacher: MediumPatchDescriptionNetwork | SmallPatchDescriptionNetwork
        self.student: MediumPatchDescriptionNetwork | SmallPatchDescriptionNetwork
        self.device = device
        if model_size == EfficientAdModelSize.M:
            self.teacher = MediumPatchDescriptionNetwork(out_channels=teacher_out_channels, padding=padding).eval()
            self.student = MediumPatchDescriptionNetwork(out_channels=teacher_out_channels * 2, padding=padding)

        elif model_size == EfficientAdModelSize.S:
            self.teacher = SmallPatchDescriptionNetwork(out_channels=teacher_out_channels, padding=padding).eval()
            self.student = SmallPatchDescriptionNetwork(out_channels=teacher_out_channels * 2, padding=padding)
        
        else:
            msg = f"Unknown model size {model_size}"
            raise ValueError(msg)
        self.teacher.to(self.device)
        self.student.to(self.device)
        self.ae: AutoEncoder = AutoEncoder(out_channels=teacher_out_channels, padding=padding).to(self.device)
        self.teacher_out_channels: int = teacher_out_channels

        self.mean_std: nn.ParameterDict = nn.ParameterDict(
            {
                "mean": torch.zeros((1, self.teacher_out_channels, 1, 1)),
                "std": torch.zeros((1, self.teacher_out_channels, 1, 1)),
            },
        )

        self.quantiles: nn.ParameterDict = nn.ParameterDict(
            {
                "qa_st": torch.tensor(0.0),
                "qb_st": torch.tensor(0.0),
                "qa_ae": torch.tensor(0.0),
                "qb_ae": torch.tensor(0.0),
            },
        )
        self.batch_size = batch_size
        
        self.prepare_imagenette_data(image_size)
        self.resize = image_size[-1]
        self.score_in_mid_size = int(self.resize*0.9)
        
        

    def prepare_imagenette_data(self, image_size: tuple[int, int] | torch.Size) -> None:
            """Prepare ImageNette dataset transformations.

            Args:
                image_size (tuple[int, int] | torch.Size): Image size.
            """
            self.data_transforms_imagenet = Compose(
                [
                    Resize((image_size[0] * 2, image_size[1] * 2)),
                    RandomGrayscale(p=0.3),
                    CenterCrop((image_size[0], image_size[1])),
                    ToTensor(),
                ],
            )

            if not self.imagenet_dir.is_dir():
                download_and_extract(self.imagenet_dir, IMAGENETTE_DOWNLOAD_INFO)
            imagenet_dataset = ImageFolder(self.imagenet_dir, transform=self.data_transforms_imagenet)
            self.imagenet_loader = DataLoader(imagenet_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
            self.imagenet_iterator = iter(self.imagenet_loader)

    def is_set(self, p_dic: nn.ParameterDict) -> bool:
        """Check if any of the parameters in the parameter dictionary is set.

        Args:
            p_dic (nn.ParameterDict): Parameter dictionary.

        Returns:
            bool: Boolean indicating whether any of the parameters in the parameter dictionary is set.
        """
        return any(value.sum() != 0 for _, value in p_dic.items())

    def choose_random_aug_image(self, image: torch.Tensor) -> torch.Tensor:
        """Choose a random augmentation function and apply it to the input image.

        Args:
            image (torch.Tensor): Input image.

        Returns:
            Tensor: Augmented image.
        """
        transform_functions = [
            transforms.functional.adjust_brightness,
            transforms.functional.adjust_contrast,
            transforms.functional.adjust_saturation,
        ]
        # Sample an augmentation coefficient λ from the uniform distribution U(0.8, 1.2)
        coefficient = np.random.default_rng().uniform(0.8, 1.2)
        transform_function = np.random.default_rng().choice(transform_functions)
        return transform_function(image, coefficient)

    def forward(
        self,
        batch: torch.Tensor,
        # batch_imagenet: torch.Tensor | None = None,
        normalize: bool = True,
    ) -> torch.Tensor | dict:
        """Perform the forward-pass of the EfficientAd models.

        Args:
            batch (torch.Tensor): Input images.
            batch_imagenet (torch.Tensor): ImageNet batch. Defaults to None.
            normalize (bool): Normalize anomaly maps or not

        Returns:
            Tensor: Predictions
        """
        try:
            # infinite dataloader; [0] getting the image not the label
           
            batch_imagenet = next(self.imagenet_iterator)[0].to(self.device)
        except StopIteration:
            self.imagenet_iterator = iter(self.imagenet_loader)
            batch_imagenet = next(self.imagenet_iterator)[0].to(self.device)
        image_size = batch.shape[-2:]
        batch = batch.to(self.device)
        with torch.no_grad():
            teacher_output = self.teacher(batch)
            if self.is_set(self.mean_std):
                teacher_output = (teacher_output - self.mean_std["mean"]) / self.mean_std["std"]

        student_output = self.student(batch)
        distance_st = torch.pow(teacher_output - student_output[:, : self.teacher_out_channels, :, :], 2)

        if self.training:
            # Student loss
            distance_st = reduce_tensor_elems(distance_st)
            d_hard = torch.quantile(distance_st, 0.999)
            loss_hard = torch.mean(distance_st[distance_st >= d_hard])
            student_output_penalty = self.student(batch_imagenet)[:, : self.teacher_out_channels, :, :]
            loss_penalty = torch.mean(student_output_penalty**2)
            loss_st = loss_hard + loss_penalty

            # Autoencoder and Student AE Loss
            aug_img = self.choose_random_aug_image(batch)
            ae_output_aug = self.ae(aug_img, image_size)

            with torch.no_grad():
                teacher_output_aug = self.teacher(aug_img)
                if self.is_set(self.mean_std):
                    teacher_output_aug = (teacher_output_aug - self.mean_std["mean"]) / self.mean_std["std"]

            student_output_ae_aug = self.student(aug_img)[:, self.teacher_out_channels :, :, :]

            distance_ae = torch.pow(teacher_output_aug - ae_output_aug, 2)
            distance_stae = torch.pow(ae_output_aug - student_output_ae_aug, 2)

            loss_ae = torch.mean(distance_ae)
            loss_stae = torch.mean(distance_stae)
            return ADModelOutput(score = loss_st+loss_ae+loss_stae, 
                                 alpha = None,
                                 others= 
                                    {
                                        "loss_st": loss_st,
                                        "loss_ae": loss_ae,
                                        "loss_stae": loss_stae
                                    }
                                 )

        # Eval mode.
        with torch.no_grad():
            ae_output = self.ae(batch, image_size)

            map_st = torch.mean(distance_st, dim=1, keepdim=True)
            map_stae = torch.mean(
                (ae_output - student_output[:, self.teacher_out_channels :]) ** 2,
                dim=1,
                keepdim=True,
            )

        if self.pad_maps:
            map_st = F.pad(map_st, (4, 4, 4, 4))
            map_stae = F.pad(map_stae, (4, 4, 4, 4))
        map_st = F.interpolate(map_st, size=image_size, mode="bilinear")
        map_stae = F.interpolate(map_stae, size=image_size, mode="bilinear")

        if self.is_set(self.quantiles) and normalize:
            map_st = 0.1 * (map_st - self.quantiles["qa_st"]) / (self.quantiles["qb_st"] - self.quantiles["qa_st"])
            map_stae = 0.1 * (map_stae - self.quantiles["qa_ae"]) / (self.quantiles["qb_ae"] - self.quantiles["qa_ae"])

        map_combined = 0.5 * map_st + 0.5 * map_stae
        score_start = (self.resize-self.score_in_mid_size)//2
        image_score = torch.max(map_combined[:,:,
            score_start:score_start+self.score_in_mid_size,
            score_start:score_start+self.score_in_mid_size
        ])
        # return {"anomaly_map": map_combined, "map_st": map_st, "map_ae": map_stae}
        return ADModelOutput(score = image_score, 
                                 alpha = map_combined,
                                 others= 
                                    {
                                    "x_recon": ae_output,
                                     "map_st": map_st,
                                     "map_ae": map_stae
                                    }
                            )


