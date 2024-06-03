import numpy as np
import pandas as pd 
import torch 
from torch.utils.data import Dataset, DataLoader
from utils.timefeatures import time_features

"""
This script is used to load the datasets
"""
# some tools for datasets
class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean

# EET datasets
class Dataset_ETT(Dataset):
    def __init__(self, dataset_path, size=None, dataset_type="train", features="S", target="OT", \
                        scale=True, inverse=False, time_enc=0, freq="h"):
        # identify the input size which should include: sequence length, label length and prediction length
        self.seq_len = 24*4*4 if size == None else size[0]
        self.label_len = 24*4 if size == None else size[1]
        self.pred_len = 24*4 if size == None else size[2]
        # to check the type of the dataset 
        self.dataset_type = dataset_type
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.time_enc = time_enc
        self.freq = freq
        self.dataset_path = dataset_path
        self._read_data()

    # read data from csv files
    def _read_data(self):
        # this is used to normalize the data
        self.scaler = StandardScaler()
        df_data_raw = pd.read_csv(self.dataset_path)
        if "ETTh" in self.dataset_path: 
            # how should we slice the data into train (the first year), val (4 months next) and test (4 months next)
            border1s = {"train": 0, "val": 12*30*24 - self.seq_len, "test": 12*30*24+4*30*24 - self.seq_len}
            border2s = {"train": 12*30*24, "val": 12*30*24+4*30*24, "test": 12*30*24+8*30*24}
        elif "ETTm" in self.dataset_path:
            border1s = {"train": 0, "val": 12*30*24*4 - self.seq_len, "test": 12*30*24*4+4*30*24*4 - self.seq_len}
            border2s = {"train": 12*30*24*4, "val": 12*30*24*4+4*30*24*4, "test": 12*30*24*4+8*30*24*4}
        else:
            raise NotImplementedError
        border1, border2 = border1s[self.dataset_type], border2s[self.dataset_type]
        # in this code I only consider the single output
        if self.features == "S":
            df_data = df_data_raw[[self.target]]
        elif self.features in ["MS", "M"]:
            cols_data = df_data_raw.columns[1:]
            df_data = df_data_raw[cols_data]
        else:
            raise NotImplementedError
        # noramlize the data
        if self.scale:
            # use training data's statistics
            train_data = df_data[border1s["train"]:border2s["train"]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        # timestamp
        df_stamp = df_data_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        # get the timestep features
        data_stamp = time_features(df_stamp, timeenc=self.time_enc, freq=self.freq)        
        # get the final data
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        # decoder's input 
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        # sequence
        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        # sequence's timestamp
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

if __name__ == "__main__":
    dataset = Dataset_ETT("../data/ETT/ETTh1.csv")
    print(len(dataset))