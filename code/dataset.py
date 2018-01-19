from __future__ import print_function

import numpy as np

import torch
from torch.utils.data import Dataset


class SNEMI3D_Dataset(Dataset):
    """
    SNEMI3D dataset.
    """
    def __init__(self, sampler, size, margin):
        super(SNEMI3D_Dataset, self).__init__()
        self.sampler = sampler
        self.size = size
        self.seeded = False
        self.margin = margin

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if not self.seeded:
            assert idx < self.margin
            high = 2**32 - self.margin
            seed = np.random.randint(high) + idx
            print("idx = {}, seed = {}".format(idx, seed))
            np.random.seed(seed)
            self.seeded = True
        return self.sampler(imgs=['input'])
