import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from anomalib.models.image.fastflow.torch_model import FastflowModel
from anomalib.models.image.efficient_ad.torch_model import *
from anomalib.models.image.fastflow.anomaly_map import AnomalyMapGenerator
from timm.models.cait import Cait
from timm.models.vision_transformer import VisionTransformer

from .common import *


class VaeADModel(nn.Module):
    """ A little wrapper around AutoencoderKL Vae from diffusers """
    def __init__(self, image_channels: int = 3, **kwargs):
        super().__init__()
        self.vae = AutoencoderKL(
            in_channels = image_channels,
            out_channels = image_channels,
            **kwargs
        )

    def forward(self, x: torch.FloatTensor):
        enc = self.vae.encode(x)
        mu, logvar = enc.latent_dist.mean, enc.latent_dist.logvar

        if self.training:
            z = mu + (0.5 * logvar).exp() * torch.randn_like(logvar)
        else:
            z = mu

        dec = self.vae.decode(z)
        x_recon = dec.sample
        alpha = (x - x_recon).abs()
        score = alpha.norm(p=2, dim=(1,2,3)) ** 2
        return ADModelOutput(
            score = score,
            alpha = alpha,
            others = {
                "x_recon": x_recon,
                "z": z,
                "mu": mu,
                "logvar": logvar,
                "enc": enc,
                "dec": dec,
            }
        )


class FastflowAdModel(FastflowModel):
    """ We sub-class the FastflowModel because their forward function gives us too little """
    def __init__(
        self,
        image_size: int = 256,
        backbone: str = "wide_resnet50_2",
        **kwargs
    ):
        super().__init__(
            input_size = (image_size, image_size),
            backbone = backbone
        )

    def forward(self, x: torch.FloatTensor):
        """
        Modified from: https://github.com/openvinotoolkit/anomalib/blob/0cd338de51b133dbf60a1e359ff5426ab3afb6d7/src/anomalib/models/image/fastflow/torch_model.py#L173C1-L223C1

        """

        N, C, H, W = x.shape

        self.feature_extractor.eval()
        if isinstance(self.feature_extractor, VisionTransformer):
            features = self._get_vit_features(x)
        elif isinstance(self.feature_extractor, Cait):
            features = self._get_cait_features(x)
        else:
            features = self._get_cnn_features(x)

        # Compute the hidden variable f: X -> Z and log-likelihood of the jacobian
        # (See Section 3.3 in the paper.)
        # NOTE: output variable has z, and jacobian tuple for each fast-flow blocks.
        hidden_vars: list[torch.Tensor] = []
        log_jacs: list[torch.Tensor] = []
        score = torch.zeros(N).to(x.device) # big numbers because it's log prob


        for fast_flow_block, feature in zip(self.fast_flow_blocks, features, strict=True):
            hidden_var, log_jac = fast_flow_block(feature)
            hidden_vars.append(hidden_var)
            log_jacs.append(log_jac)
            score += 0.5 * torch.mean(hidden_var**2, dim=(1,2,3)) - (log_jac/(C*H*W))

        # print(score)

        alpha = self.anomaly_map_generator(hidden_vars)    # (N,1,H,W)

        return ADModelOutput(
            score = score,
            alpha = alpha,
            others = {
                "hidden_variables": hidden_vars,
                "log_jacobians": log_jacs
            }
        )



class EfficientAdADModel(EfficientAdModel):
    def __init__(
            self,
            teacher_out_channels: int = 384,
            **kwargs
        ):
        super().__init__(teacher_out_channels=teacher_out_channels, **kwargs)


    def forward(
        self,
        batch: torch.Tensor,
        batch_imagenet: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ) -> torch.Tensor | dict:
        """
        Adapted from: https://github.com/openvinotoolkit/anomalib/blob/09cbadea3c17d4352b115302ef0c40d426556145/src/anomalib/models/image/efficient_ad/torch_model.py#L353
        """
        N, _, H, W = batch.shape
        image_size = (H,W)
        with torch.no_grad():
            teacher_output = self.teacher(batch)
            if self.is_set(self.mean_std):
                teacher_output = (teacher_output - self.mean_std["mean"]) / self.mean_std["std"]

        student_output = self.student(batch)
        distance_st = torch.pow(teacher_output - student_output[:, : self.teacher_out_channels, :, :], 2)

        loss_st, loss_ae, loss_stae = None, None, None

        # Student loss.
        # This can't be run during eval mode because we have to train with reference to batch_imagenet
        if self.training:
            flat_dist_st = distance_st.view(N,-1)
            d_hard = torch.quantile(flat_dist_st, 0.999, dim=1).view(N,1)
            loss_hard = torch.mean(flat_dist_st[flat_dist_st >= d_hard], dim=-1)
            student_output_penalty = self.student(batch_imagenet)[:, : self.teacher_out_channels, :, :]
            loss_penalty = torch.mean(student_output_penalty**2, dim=(1,2,3))
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

            loss_ae = torch.mean(distance_ae, dim=(1,2,3))
            loss_stae = torch.mean(distance_stae, dim=(1,2,3))


        # Eval mode only
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
        score = map_combined.view(N,-1).max(dim=1).values

        return ADModelOutput(
            score = score,
            alpha = map_combined,
            others = {
                "loss_st": loss_st,
                "loss_ae": loss_ae,
                "loss_stae": loss_stae,
                "anomaly_map": map_combined,
                "map_st": map_st,
                "map_ae": map_stae
            }
        )


