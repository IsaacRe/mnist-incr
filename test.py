import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms
import torch
from main import Net


def get_batch_suite(batch_size, dset_size=None, train=True):
    net = Net()
    x_ent = nn.CrossEntropyLoss()
    d_set = datasets.MNIST('../data', train=train, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))
    if dset_size is not None:
        d_set = torch.utils.data.Subset(d_set, np.arange(dset_size))
    loader = torch.utils.data.DataLoader(d_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    return net, x_ent, loader
