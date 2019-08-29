import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms
import torch


def add_return_index(Dataset):
    getitem_old = Dataset.__getitem__
    def getitem(self, i):
        ret = getitem_old(self, i)
        if type(ret) is tuple:
            return (i, *ret)
        return i, ret
    Dataset.__getitem__ = getitem


def get_batch_suite(batch_size, train=True):
    x_ent = nn.CrossEntropyLoss()
    add_return_index(datasets.MNIST)
    d_set = datasets.MNIST('../data', train=train, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))
    d_set = torch.utils.data.Subset(d_set, np.arange(10))
    loader = torch.utils.data.DataLoader(d_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    return x_ent, loader
