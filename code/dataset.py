import numpy as np

import torch
from torch.utils.data import Dataset


class SNEMI3D_Dataset(Dataset):
    """
    Dummy dataset.
    """
    def __init__(self, sampler, size):
        super(SNEMI3D_Dataset, self).__init__()
        self.sampler = sampler
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.sampler(imgs=['input'])
