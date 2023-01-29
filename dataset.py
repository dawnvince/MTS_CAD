from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch
import torch.utils.data
import os
import logging
import csv
import numpy as np

class MTSDataset(torch.utils.data.Dataset):
    
    def __init__(self,
                window,
                horizon,
                data_name,
                set_type='train',
                dataset="SMD"
    ):
        assert type(set_type) == type('str')
        self.window = window
        self.horizon = horizon
        self.set_type = set_type
        
        file_path = os.path.join("./data", "%s_%s.npy"%(data_name, set_type))
        rawdata = np.load(file_path)
        
        self.len, self.var_num = rawdata.shape
        self.sample_num = max(self.len - self.window - self.horizon + 1, 0)
        self.samples, self.labels = self.__getsamples(rawdata)

    def __getsamples(self, data):
        X = torch.zeros((self.sample_num, self.window, self.var_num))
        Y = torch.zeros((self.sample_num, 1, self.var_num))

        for i in range(self.sample_num):
            start = i
            end = i + self.window
            X[i, :, :] = torch.from_numpy(data[start:end, :])
            Y[i, :, :] = torch.from_numpy(data[end+self.horizon-1, :])

        return (X, Y)

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        sample = [self.samples[idx, :, :], self.labels[idx, :, :]]

        return sample
    
