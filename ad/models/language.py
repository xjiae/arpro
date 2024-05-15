import torch
from tqdm import tqdm
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .common import *


class RobertaADModel(nn.Module):
    def __init__(
        self
    ):
        super().__init__()
        model_name = "roberta-base-openai-detector"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).cuda()

    
        # Do classification-based thing
    def attribute(self, 
                  x,
                  x0 = None,
                  num_steps = 32,
                  progress_bar = False):
        """
        Explain a classification model with Integrated Gradients.
        """
        device = next(self.model.parameters()).device 
        x = x.to(device)
        embed_fn = self.model.get_input_embeddings()
        x = embed_fn(x).detach()
        # Default baseline is zeros
        x0 = torch.zeros_like(x) if x0 is None else x0
        x0 = x0.to(device)
        step_size = 1 / num_steps
        intg = torch.zeros_like(x)

        # pbar = tqdm(range(num_steps)) if progress_bar else range(num_steps)

        for k in range(num_steps):
            ak = k * step_size
            xk = x0 + ak * (x - x0)
            xk.requires_grad_()
            loss = self.model(inputs_embeds=xk).logits[:, 0].mean()
            loss.requires_grad_()
            loss.backward(retain_graph=True)
            intg += xk.grad * step_size
        alpha = intg.max(dim=-1).values
        return alpha
    
    # original: real: 1 | fake: 0
    def forward(self, x):
        with torch.no_grad():
            out = self.model(x).logits
        return ADModelOutput(
            score = out[:, 0].detach(),
            # alpha=None,
            alpha = self.attribute(x),
            # flip the pred
            others = {"y_pred": (1-out.argmax(dim=1)).long().detach()}
        )