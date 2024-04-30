import math
from typing import Optional, Tuple
from torch.utils.data.dataset import Dataset
import csv
from transformers import AutoModelForPreTraining,AutoModelForMaskedLM
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F 
import random
from tqdm import tqdm
from diffusers import DDPMPipeline,  DDIMScheduler
from transformers import BertTokenizer

class MyTextDiffusionModel(nn.Module):
    def __init__(
            self,
            max_len: int = 130,
            max_step: int = 2000,
            **kwargs
    ):
        super().__init__()
        self.max_len = max_len
        self.max_step = max_step
        self.model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
        freezed_w = [self.model.bert.embeddings.token_type_embeddings.weight, \
                     self.model.bert.embeddings.word_embeddings.weight]
        self.time_embed = nn.Embedding(num_embeddings=max_step,\
                                       embedding_dim=self.model.config.hidden_size)
        for p in  freezed_w:
            p.requires_grad = False
        nn.init.constant_(self.time_embed.weight, 0)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    @property
    def num_timesteps(self):
        return self.max_step
    
    def add_noise(self, 
                  x: torch.FloatTensor, 
                  noise: torch.FloatTensor, 
                  t: Optional[torch.LongTensor]):
        if t is None:
            diffusion_steps = torch.randint(0, 
                                            self.max_step, 
                                            size = (x.size(0),), 
                                            device = x.device)
        else:
            diffusion_steps = t * torch.ones(size = (x.size(0),),
                                         device = x.device).long()
        alpha = 1 - torch.sqrt((diffusion_steps + 1) / self.max_step).view(-1, 1, 1)
        noisy_x = torch.sqrt(alpha) * x + torch.sqrt(1 - alpha) * noise
        return noisy_x
    
    def estimate_noise(self, 
                       xt: torch.FloatTensor, 
                       t: torch.LongTensor):
        attention_mask = torch.ones(xt.size(0),self.max_len).long().to(xt.device)
        extended_attention_mask = self.model.bert.get_extended_attention_mask(attention_mask, 
                                                                              attention_mask.shape)
        encoder_outputs = self.model.bert.encoder(
                xt,
                attention_mask=extended_attention_mask,
                head_mask=[None] * self.model.config.num_hidden_layers)
        sequence_output = encoder_outputs[0]
        prediction_scores = self.model.cls.predictions(sequence_output)
        #clamp
        denoised_word = prediction_scores.softmax(-1) \
            @ self.model.bert.embeddings.word_embeddings.weight.unsqueeze(0)
        #DDIM
        alpha_t = 1 - torch.sqrt((t+1)/self.max_step)+1e-5
        alpha_t = alpha_t[:, None, None]
        noise = (xt - torch.sqrt(alpha_t) * denoised_word)/torch.sqrt(1 - alpha_t)
        return noise, prediction_scores

    def forward(
            self, 
            x: Optional[torch.LongTensor] = None,
            t: Optional[torch.FloatTensor] = None,
            k: int = 10,
            batch_size: Optional[int] = None,
            num_inference_steps: int = 1000,
            progress_bar: bool = False,
            enable_grad: bool = False
            ):
        device = next(self.model.parameters()).device
        if x is None:
            assert batch_size is not None
            noisy_word = torch.normal(0, 1,
                                      (batch_size, self.max_len, self.model.config.hidden_size)).to(device) \
                        / math.sqrt(self.model.config.hidden_size)
            
            position_ids = self.model.bert.embeddings.position_ids[:, 0 : self.max_len]
            position_embeddings = self.model.bert.embeddings.position_embeddings(position_ids)

            token_type_ids = torch.zeros(batch_size,self.max_len).long().to(device)
            token_type_embeddings = self.model.bert.embeddings.token_type_embeddings(token_type_ids)
        else:
            raise NotImplementedError
        with torch.set_grad_enabled(enable_grad):
            pbar = tqdm(range(num_inference_steps- 1, 0, -k)) if progress_bar else range(num_inference_steps - 1, 0, -k)
            
            for t in pbar:
                diffusion_steps = torch.ones(size = (batch_size,),device=device).long() * t
                time_embedding = self.time_embed(diffusion_steps).unsqueeze(1)
                model_input = noisy_word + position_embeddings + token_type_embeddings \
                             + time_embedding 
                model_input = self.model.bert.embeddings.LayerNorm(model_input)
                noise, prediction_scores = self.estimate_noise(model_input, diffusion_steps)
                pred_ids = torch.argmax(prediction_scores, -1).long()
                print(self.tokenizer.decode(pred_ids[0]))
                # DDIM
                alpha_tk = 1 - torch.sqrt((diffusion_steps + 1 - k) / self.max_step)
                alpha_t = 1 - torch.sqrt((diffusion_steps + 1) / self.max_step) + 1e-5
                alpha_tk = alpha_tk[:, None, None]
                alpha_t = alpha_t[:, None, None]
                print(alpha_tk)
                print(alpha)
                noisy_word = torch.sqrt(alpha_tk) * (noisy_word / torch.sqrt(alpha_t) \
                                                   + (torch.sqrt((1 - alpha_tk) / alpha_tk) \
                                                   - torch.sqrt((1 - alpha_t) / alpha_t)) * noise)
                # print(noisy_word)
        pred = torch.argmax(prediction_scores, -1).long()
        return pred