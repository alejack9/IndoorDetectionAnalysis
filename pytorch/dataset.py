import numpy as np
import torch
from torch.utils.data import Dataset


# dataset stored as set of tensors
class IndoorDataset(Dataset):
    def __init__(self, X, y):
        self.num_classes = len(np.unique(y))
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        '''Number of rows'''
        return self.X.shape[0]

    def __getitem__(self, idx):
        '''Iterates on observations'''
        return self.X[idx, :], self.y[idx]
