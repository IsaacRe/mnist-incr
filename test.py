import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms
import torch
from main import Net


def add_return_index(Dataset):
    getitem_old = Dataset.__getitem__
    def getitem(self, i):
        ret = getitem_old(self, i)
        if type(ret) is tuple:
            return (i, *ret)
        return i, ret
    Dataset.__getitem__ = getitem
add_return_index(datasets.MNIST)


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


def get_incr_suite(batch_size, dset_size=None, train=True):
    net = Net()
    x_ent = nn.CrossEntropyLoss()

    d_set = datasets.MNIST('../data', train=train, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))

    def make_sampler(label):
        select = np.where(d_set.train_labels == label)[0]
        indices = np.random.choice(select, dset_size if dset_size is not None else len(select), replace=False)
        return torch.utils.data.SubsetRandomSampler(indices)
    loaders = [torch.utils.data.DataLoader(d_set, batch_size=batch_size, shuffle=False, sampler=make_sampler(l),
                                           num_workers=1, pin_memory=True) for l in range(10)]

    return net, x_ent, loaders
