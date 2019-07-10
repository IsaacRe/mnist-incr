import numpy as np
import torch.nn as nn
import torch
from stats_tracking import ActivationTracker
from test import get_batch_suite


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
    net, x_ent, loader = get_batch_suite()
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