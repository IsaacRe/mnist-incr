import numpy as np
import torch.nn as nn
import torch
from torchvision import datasets, transforms
from stats_tracking import ActivationTracker
from main import Net


def test_layer():
    layer = nn.Linear(10, 10)
    # check tracking setup
    a = ActivationTracker(layer)
    x_ent = nn.CrossEntropyLoss()

    data_x, data_y = zip(*[(np.random.rand(100, 10), np.random.choice(10, 100)) for i in range(10)])
    for i, (x, y) in enumerate(zip(data_x, data_y)):
        logits = layer(torch.Tensor(x))

        # check forward stats update
        loss = x_ent(logits, torch.LongTensor(y))
        for p in layer.parameters():
            if p.grad is not None:
                p.grad.zero_()

        # check backward stats update
        loss.backward()
    return


def test_net():
    net = Net()
    tracker = ActivationTracker(net)
    x_ent = nn.CrossEntropyLoss()
    d_set = datasets.MNIST('../data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))
    d_set = torch.utils.data.Subset(d_set, np.arange(10))
    loader = torch.utils.data.DataLoader(d_set, batch_size=10, shuffle=False, num_workers=1, pin_memory=True)
    for i, (x, y) in enumerate(loader):
        logits = net(x)
        loss = x_ent(logits, y)
        for p in net.parameters():
            if p.grad is not None:
                p.grad.zero_()
        loss.backward()
    return


if __name__ == '__main__':
    test_net()