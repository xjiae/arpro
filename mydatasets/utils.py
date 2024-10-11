import random
from torch.utils.data import DataLoader
import torch.utils.data as tud
from typing import Optional
from .mvtec import *
from .visa import *
from .webtext import *

# code source https://github.com/thorinf/simple-diffusion-lm/blob/main/pytorch/train.py#L55
class Collate:
    def __init__(self, crop_length=-1, eos_id=-1, pad_id=-1, length_includes_pad=False, fold_size=None):
        assert not (pad_id < 0 and length_includes_pad)
        assert not (pad_id < 0 and fold_size)
        self.crop_length = crop_length
        self.fold_size = fold_size
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.pad_insert_rate = 0.0
        self.length_includes_pad = length_includes_pad

    def fold(self, ids):
        # pad the list for folding
        remainder = len(ids) % self.fold_size
        if remainder != 0:
            ids += [self.pad_id] * (self.fold_size - remainder)
        # fold the list
        ids = [ids[i:i + self.fold_size] for i in range(0, len(ids), self.fold_size)]
        return ids

    def generate_mask(self, length):
        conditional_mask = [False] * length
        mask_span_length = random.randint(0, length - 1)
        start_index = random.randint(0, length - mask_span_length)
        conditional_mask[start_index:start_index + mask_span_length] = [True] * mask_span_length
        # half of the masks will be completely random
        if random.random() < 0.5:
            random.shuffle(conditional_mask)
        return conditional_mask

    def process_ids(self, ids):
        # and the eos token
        if self.eos_id >= 0:
            ids.append(self.eos_id)
        # randomly insert pads into ids
        if self.pad_id >= 0 and self.pad_insert_rate > 0:
            pad_count = int(len(ids) * self.pad_insert_rate)
            pad_indices = random.sample(range(len(ids)), pad_count)
            for index in pad_indices:
                ids.insert(index, self.pad_id)
        if self.fold_size is not None:
            ids = self.fold(ids)
        # crops the length
        if 0 < self.crop_length < len(ids):
            ids = ids[:self.crop_length]
        # create a conditional mask
        conditional_mask = self.generate_mask(len(ids))
        return ids, len(ids), conditional_mask

    def __call__(self, batch):
        tokens = [b[0] for b in batch]
        labels = torch.tensor([b[1] for b in batch])
        batch = tokens
        processed = list(map(self.process_ids, batch))
        ids, lengths, conditional_mask = zip(*processed)

        # sample a random amount of padding
        padded_lengths = [random.randint(length, max(lengths)) for length in lengths]
        lengths = torch.tensor(padded_lengths) if self.length_includes_pad else torch.tensor(lengths)

        ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x, dtype=torch.int64) for x in ids],
            batch_first=True,
            padding_value=self.pad_id
        )
        conditional_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x, dtype=torch.bool) for x in conditional_mask],
            batch_first=True,
            padding_value=False
        )

        return ids, lengths, conditional_mask, labels

def get_binary_balanced_subset(dataset, pos_mask, seed=None):
  assert len(dataset) == len(pos_mask)
  if seed is not None: torch.manual_seed(seed)
  neg_mask = 1 - pos_mask
  npos = pos_mask.sum()
  nneg = pos_mask.numel() - npos

  neg_keep_frac = npos / nneg
  neg_keep_mask = neg_mask * (torch.rand(pos_mask.shape) < neg_keep_frac)
  keep_mask = (pos_mask + neg_keep_mask).clamp(0,1)
  keep_inds = keep_mask.nonzero()
  subset = tud.Subset(dataset, indices=keep_inds)
  return subset

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
    elif dataset_name == "visa":
        return DataLoader(
            VisADataset(**dataset_kwargs),
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
    elif dataset_name == "visa":
        return DataLoader(
            VisADataset(**dataset_kwargs),
            batch_size = batch_size,
            shuffle = True
        )
    elif dataset_name == "webtext":
        dataset = load_text_datasets(**dataset_kwargs)
        tokenizer = dataset.tokenizer
        collate = Collate(
                    crop_length=64,
                    eos_id=tokenizer.eos_token_id,
                    pad_id=tokenizer.pad_token_id,
                    length_includes_pad=True
                )
        return DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle = True,
            collate_fn=collate
        )
    else:
        raise ValueError(f"Unknown combination of {dataset_name} and {model_name}")


