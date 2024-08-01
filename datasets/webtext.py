import json
import numpy as np
from typing import List

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer, RobertaTokenizerFast, BertTokenizer
from .download import download


def load_texts(data_file, expected_size=None):
    texts = []

    for line in tqdm(open(data_file), total=expected_size, desc=f'Loading {data_file}'):
        texts.append(json.loads(line)['text'])

    return texts


class Corpus:
    def __init__(self, name, data_dir='data', skip_train=False):
        # download(name, data_dir=data_dir)
        self.name = name
        self.train = load_texts(f'{data_dir}/{name}.train.jsonl', expected_size=250000) if not skip_train else None
        self.test = load_texts(f'{data_dir}/{name}.test.jsonl', expected_size=5000)
        self.valid = load_texts(f'{data_dir}/{name}.valid.jsonl', expected_size=5000)


class EncodedDataset(Dataset):
    def __init__(self, real_texts: List[str], 
                 fake_texts: List[str], 
                 tokenizer: PreTrainedTokenizer,
                 max_sequence_length: int = None, 
                 min_sequence_length: int = None, 
                 epoch_size: int = None,
                 token_dropout: float = None, 
                 seed: int = None,
                 split: str = "train"):
        self.real_texts = real_texts
        self.fake_texts = fake_texts
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_sequence_length
        self.epoch_size = epoch_size
        self.token_dropout = token_dropout
        self.random = np.random.RandomState(seed)
        self.split = split

    def __len__(self):
        if self.split == "train":
            return len(self.real_texts)

        return self.epoch_size or len(self.real_texts) + len(self.fake_texts)

    def __getitem__(self, index):
        if self.epoch_size is not None:
            label = self.random.randint(2)
            texts = [self.fake_texts, self.real_texts][label]
            text = texts[self.random.randint(len(texts))]
        else:
            if index < len(self.real_texts):
                text = self.real_texts[index]
                label = 0
            else:
                text = self.fake_texts[index - len(self.real_texts)]
                label = 1

        tokens = self.tokenizer.encode(text)

        if self.max_sequence_length is None:
            tokens = tokens[:self.tokenizer.max_len - 2]
        else:
            output_length = min(len(tokens), self.max_sequence_length)
            if self.min_sequence_length:
                output_length = self.random.randint(min(self.min_sequence_length, len(tokens)), output_length + 1)
            start_index = 0 if len(tokens) <= output_length else self.random.randint(0, len(tokens) - output_length + 1)
            end_index = start_index + output_length
            tokens = tokens[start_index:end_index]

        if self.token_dropout:
            dropout_mask = self.random.binomial(1, self.token_dropout, len(tokens)).astype(np.bool)
            tokens = np.array(tokens)
            tokens[dropout_mask] = self.tokenizer.unk_token_id
            tokens = tokens.tolist()
        '''
        if isinstance(self.tokenizer, BertTokenizer):
            if self.max_sequence_length is None or len(tokens) == self.max_sequence_length:
                # BERT uses a [CLS] token at the start
                mask = torch.ones(len(tokens) + 2)
                # Here we use tokenizer.cls_token_id instead of bos_token_id
                return torch.tensor([self.tokenizer.cls_token_id] + tokens + [self.tokenizer.sep_token_id]), mask, label

            # Padding to the max_sequence_length
            padding = [self.tokenizer.pad_token_id] * (self.max_sequence_length - len(tokens))
            # Prepend [CLS] token and append [SEP] token
            tokens = torch.tensor([self.tokenizer.cls_token_id] + tokens + [self.tokenizer.sep_token_id] + padding)
        else:
        '''
        if self.max_sequence_length is None or len(tokens) == self.max_sequence_length:
            mask = torch.ones(len(tokens) + 2)
            # return torch.tensor([self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id])
            return torch.tensor([self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]).tolist(), label

        padding = [self.tokenizer.pad_token_id] * (self.max_sequence_length - len(tokens))
        tokens = torch.tensor([self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id] + padding)
        mask = torch.ones(tokens.shape[0])
        mask[-len(padding):] = 0
        return tokens.tolist(), label
        # return tokens, mask, label
    
def load_text_datasets(
                       data_dir="/home/antonxue/foo/arpro/data/webtext",
                       real_dataset='webtext', 
                       fake_dataset='xl-1542M-nucleus', 
                       max_sequence_length=128, 
                       random_sequence_length=False, 
                       epoch_size=None, 
                       token_dropout=None, 
                       seed=None, 
                       category=None,
                       split = "train"):
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base-openai-detector")
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # if fake_dataset == 'TWO':
    #     download(real_dataset, 'xl-1542M', 'xl-1542M-nucleus', data_dir=data_dir)
    # elif fake_dataset == 'THREE':
    #     download(real_dataset, 'xl-1542M', 'xl-1542M-k40', 'xl-1542M-nucleus', data_dir=data_dir)
    # else:
    #     download(real_dataset, fake_dataset, data_dir=data_dir)

    real_corpus = Corpus(real_dataset, data_dir=data_dir)

    if fake_dataset == "TWO":
        real_train, real_valid = real_corpus.train * 2, real_corpus.valid * 2
        fake_corpora = [Corpus(name, data_dir=data_dir) for name in ['xl-1542M', 'xl-1542M-nucleus']]
        fake_train = sum([corpus.train for corpus in fake_corpora], [])
        fake_valid = sum([corpus.valid for corpus in fake_corpora], [])
    elif fake_dataset == "THREE":
        real_train, real_valid = real_corpus.train * 3, real_corpus.valid * 3
        fake_corpora = [Corpus(name, data_dir=data_dir) for name in
                        ['xl-1542M', 'xl-1542M-k40', 'xl-1542M-nucleus']]
        fake_train = sum([corpus.train for corpus in fake_corpora], [])
        fake_valid = sum([corpus.valid for corpus in fake_corpora], [])
    else:
        fake_corpus = Corpus(fake_dataset, data_dir=data_dir)

        real_train, real_valid = real_corpus.train, real_corpus.valid
        fake_train, fake_valid = fake_corpus.train, fake_corpus.valid

    min_sequence_length = 10 if random_sequence_length else None
    if split == "train":
        # return EncodedDataset(real_train, fake_train, tokenizer, max_sequence_length, min_sequence_length, epoch_size, token_dropout, seed)
        return EncodedDataset(real_train, None, tokenizer, max_sequence_length, min_sequence_length, epoch_size, token_dropout, seed, split=split)
    else: 
        return EncodedDataset(real_valid, fake_valid, tokenizer, max_sequence_length=max_sequence_length, split=split)
