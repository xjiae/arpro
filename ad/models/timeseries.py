from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from einops import rearrange
from transformers import GPT2ForSequenceClassification
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from .common import *

class GPT2ADModel(nn.Module):
    def __init__(self):
        super(GPT2ADModel, self).__init__()
        self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
        self.gpt2.h = self.gpt2.h[:6]
        self.d_ff = 128
        self.out_layer = nn.Linear(self.d_ff, 51, bias=True)
        self.anomaly_criterion = nn.MSELoss(reduce=False)
    
    def forward(self, x_enc):
        N, L, d = x_enc.shape
        assert L % 4 == 0
        seg_num = L // 4
        x_enc = rearrange(x_enc, 'b (n s) m -> b n s m', s=seg_num)
        means = x_enc.mean(2, keepdim=True).detach()
        x_enc = x_enc - means

        stdev = torch.sqrt(torch.var(x_enc, dim=2, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        x_enc = rearrange(x_enc, 'b n s m -> b (n s) m')
        enc_out = torch.nn.functional.pad(x_enc, (0, 768-x_enc.shape[-1]))
        outputs = self.gpt2(inputs_embeds=enc_out).last_hidden_state
        outputs = outputs[:, :, :self.d_ff]
        dec_out = self.out_layer(outputs)
        dec_out = rearrange(dec_out, 'b (n s) m -> b n s m', s=seg_num)
        dec_out = dec_out * \
                  (stdev[:, :, 0, :].unsqueeze(2).repeat(
                      1, 1, seg_num, 1))
        dec_out = dec_out + \
                  (means[:, :, 0, :].unsqueeze(2).repeat(
                      1, 1, seg_num, 1))
        dec_out = rearrange(dec_out, 'b n s m -> b (n s) m')
        score = torch.mean(self.anomaly_criterion(x_enc, dec_out), dim=-1)
        # return dec_out
        return ADModelOutput(
            score = score,
            alpha = self.anomaly_criterion(x_enc, dec_out),
            others = {"x_recon": dec_out}
        )

