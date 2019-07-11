import numpy as np
import torch
import torch.nn as nn
from test import get_batch_suite
from prune import ActivationPrune
from stats_tracking import ActivationTracker


dataset_size = 100
batch_size = 10
alpha = 0.1


def test_prune():
    net, x_ent, loader = get_batch_suite(batch_size, dataset_size)
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
    prune = ActivationPrune(net, stat_name, alpha=alpha)
    for i, (x, y) in enumerate(loader):
        if i == 1:
            # test stop prune
            prune.stop_masking()
        elif i == 2:
            # test resume prune
            prune.resume_masking()
        # test forward hooks
        out = net(x)

        loss = x_ent(out, y)

        # test backward pass
        loss.backward()


if __name__ == '__main__':
    test_prune()