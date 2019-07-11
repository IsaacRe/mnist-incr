import numpy as np
import torch
import torch.nn as nn
from test import get_batch_suite
from prune import ActivationPrune
from stats_tracking import ActivationTracker


batch_size = 10
alpha = 0.1


def test_prune():
    net, x_ent, loader = get_batch_suite(batch_size)
    tracker = ActivationTracker(net)

    # get mean activations
    for i, (x, y) in enumerate(loader):
        # test forwward hooks
        out = net(x)

        loss = x_ent(out, y)

        # test backward pass
        loss.backward()

    # prune based off the collected mean activations
    stat_name = 'out'
    prune = ActivationPrune(net, stat_name, alpha=alpha, prune_method='by_value')
    for i, (x, y) in enumerate(loader):
        # test forwward hooks
        out = net(x)

        loss = x_ent(out, y)

        # test backward pass
        loss.backward()


if __name__ == '__main__':
    test_prune()