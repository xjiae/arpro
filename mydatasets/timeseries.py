import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import os

from datetime import datetime

from .utils import *

class TimeSeriesDataset(Dataset):
    def __init__(self, ds_name, window_size,
                 stride = 1,
                 root = None,
                 contents = "all",
                 label_choice = "last",
                 overwrite_cache = False):
        ds_name = ds_name.lower()
        assert ds_name in ["hai", "swat", "wadi"]
        assert contents in ["all", "train", "test", "raw"]
        assert label_choice in ["all", "last", "exist"]
        print(f"root: {root}")
        print(f"ds_name: {ds_name}")
        self.data_dir = os.path.join(root, ds_name)
        self.train_file = os.path.join(self.data_dir, "train_processed.csv")
        self.test_file = os.path.join(self.data_dir, "test_processed.csv")
        self.test_explanations_file = os.path.join(self.data_dir, "test_gt_exp.csv")
        self.raw_file = os.path.join(self.data_dir, "raw.csv")
        self.contents = contents
        self.window_size = window_size
        self.stride = stride
        self.label_choice = label_choice
        self.num_feats = 0

        # Set up cache if not exists
        self.cache_dir = os.path.join(self.data_dir, "cache")
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        # Get cache file if exists
        self.cache_file = os.path.join(self.cache_dir, \
            f"{ds_name}_W{window_size}_C{contents}_L{label_choice}.cache")
        if os.path.exists(self.cache_file) and not overwrite_cache:
            print(f"loading cache from {self.cache_file}")
            processed_dict = torch.load(self.cache_file)
        else:
            processed_dict = self.make_processed_dict()
            torch.save(processed_dict, self.cache_file)
            print(f"cached to {self.cache_file}")

        self.ts = processed_dict["ts"]
        self.explanations = processed_dict["explanations"]
        self.y = processed_dict["y"]
        self.n_idxs = processed_dict["n_idxs"]
        self.test_idxs = processed_dict["test_idxs"]
        self.tag_values = processed_dict["tag_values"]


    def make_processed_dict(self):
        if self.contents == "train":
            df =  pd.read_csv(self.train_file, index_col=0)
            explanation = pd.DataFrame(np.zeros((df.shape[0], df.shape[1]-2))) 
        elif self.contents == "test":
            df = pd.read_csv(self.test_file, index_col=0)
            explanation = pd.read_csv(self.test_explanations_file)
        elif self.contents == "all":
            train_data = pd.read_csv(self.train_file, index_col=0)
            test_data = pd.read_csv(self.test_file, index_col=0)
            df = pd.concat([train_data, test_data])
            train_exp = pd.DataFrame(np.zeros((train_data.shape[0], train_data.shape[1]-2)))
            test_exp = pd.read_csv(self.test_explanations_file)
            explanation = pd.DataFrame(np.vstack([train_exp.values, test_exp.values]), columns=test_exp.columns)
        elif self.contents == "raw":
            df =  pd.read_csv(self.raw_file, index_col=0)
            train_data = pd.read_csv(self.train_file, index_col=0)
            train_exp = pd.DataFrame(np.zeros((train_data.shape[0], train_data.shape[1]-2)))
            test_exp = pd.read_csv(self.test_explanations_file)
            explanation = pd.DataFrame(np.vstack([train_exp.values, test_exp.values]), columns=test_exp.columns)
        else:
            raise NotImplementedError()
        num_features = df.shape[1] - 2
        self.num_feats = num_features
        ts = df['epoch'].values
        labels = df['label']
        df_tag = df.drop(columns=['epoch', 'label'])
        tag_values = np.array(df_tag, dtype=np.float32)
        test_idxs = []
        y = []
        for L in range(0, len(ts) - self.window_size + 1, self.stride):
            R = L + self.window_size - 1
            if (ts[R]-ts[L]) == self.window_size - 1:
                test_idxs.append(L)
                if self.label_choice == "last":
                    y.append(labels.values[R])
                elif self.label_choice == "all":
                    y.append(labels.values[L:R+1])
                elif self.label_choice == "exist":
                    y.append(max(labels.values[L:R+1]))
                else:
                    raise NotImplementedError()

        y = torch.tensor(np.array(y)).long()
        if self.label_choice == "all":
          y = y.view(-1, self.window_size)
        else:
          y = y.view(-1)

        test_idxs = np.array(test_idxs, dtype=np.int32)[::self.stride]
        n_idxs = len(test_idxs)
        print(f"# of {self.contents} windows: {n_idxs}")

        return {
            "ts" : ts,
            "explanations" : explanation,
            "y" : y,
            "n_idxs" : n_idxs,
            "test_idxs" : test_idxs,
            "tag_values" : tag_values,
        }

         
    def __len__(self):
        return self.n_idxs

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        i = self.test_idxs[idx]
        y = self.y[idx]
        x = torch.from_numpy(self.tag_values[i : i + self.window_size])
        w = torch.from_numpy(self.explanations.iloc[i : i + self.window_size].values)
        return x, y, w
    
    def get_ts(self):
        return self.ts


def get_timeseries_bundle(
    ds_name,
    window_size = 100,
    label_choice = "last",
    stride = 1,
    train_batch_size = 32,
    test_batch_size = 32,
    train_has_only_goods = False,
    train_frac = 0.7,
    shuffle = True,
    root = "mydatasets/data",
    seed = 1234
):

    good_dataset = TimeSeriesDataset(
        ds_name,
        window_size,
        stride = stride,
        contents = "train",
        label_choice = label_choice,
        root = root
    )
    anom_dataset = TimeSeriesDataset(ds_name,
        window_size,
        stride = stride,
        contents = "test",
        label_choice = label_choice,                                
        root = root
    )

    torch.manual_seed(seed)
    if train_has_only_goods:
        trains, tests = good_dataset, anom_dataset
    else:
        concats = tud.ConcatDataset([good_dataset, anom_dataset])
        good_y = good_dataset.y
        anom_y = anom_dataset.y
        # In this case, label is (N,window_size)
        if label_choice == "all":
          good_y = good_y.max(dim=1).values
          anom_y = anom_y.max(dim=1).values

        pos_mask = torch.cat([good_y, anom_y])
        balanced = get_binary_balanced_subset(concats, pos_mask)

        total = len(balanced)
        num_train = int(total * train_frac)
        trains, tests = tud.random_split(balanced, [num_train, total - num_train])

    train_dataloader = tud.DataLoader(trains, batch_size=train_batch_size, shuffle=shuffle)
    test_dataloader = tud.DataLoader(tests, batch_size=test_batch_size, shuffle=shuffle)
    return { "train_dataset" : trains,
             "test_dataset" : tests,
             "train_dataloader" : train_dataloader,
             "test_dataloader" : test_dataloader
           }
