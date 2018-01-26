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
        # Seeding.
        self.rng = np.random.RandomState()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if not self.seeded:
            assert idx < self.margin
            high = 2**32 - self.margin
            seed = self.rng.randint(high) + idx
            print("idx = {}, seed = {}".format(idx, seed))
            np.random.seed(seed)
            self.seeded = True
        return self.sampler(imgs=['input'])


####################################################################
## Shared memory experiments.
####################################################################

from ctypes import c_float
import multiprocessing as mp


class SharedMemoryPool(object):
    """
    Shared memory pool.
    """
    def __init__(self, spec, shared=True):
        self.pool = dict()
        for key, shape in spec:
            data = np.random.rand(*shape)
            if shared:
                shm_arr = mp.Array(c_float, data.flatten().tolist())
                self.pool[key] = (shm_arr, shape)
            else:
                arr = data.flatten().tolist()
                self.pool[key] = (arr, shape)
            del data

    def get(self, key):
        data = None
        if key in self.pool:
            shm_arr, shape = self.pool[key]
            flat = np.frombuffer(shm_arr.get_obj(), dtype='float')
            data = flat.reshape(shape)
        return data


class SharedMemoryDataset(Dataset):
    """
    Shared memory for multiprocessing.
    """
    def __init__(self, spec, pool):
        self.data = dict()
        for key in spec:
            self.data[key] = pool.get(key)
        self.size = max([v.size for v in self.data.values()])
        # self.i = 0

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # self.i = idx % len(self.data)
        # v = self.data.values()[self.i]
        i = idx % len(self.data)
        v = self.data.values()[i]
        return v[np.unravel_index(idx, v.shape)]
