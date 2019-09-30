import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from test import get_batch_suite
from os.path import exists


def save_img(x, savefile=None):
    plt.imshow(x)
    if savefile:
        plt.imsave(savefile, x)


# TODO use numpy instead
def map_weights(w, device, filter_by_label=True):
    _, loader = get_batch_suite(100)
    w = torch.from_numpy(w).to(device)
    map = [torch.zeros(w.shape[1]).to(device) for i in range(10)]
    total = [0] * 10
    for _, x, y in loader:
        shape = x.shape[2:]
        x = x.view(-1, 28 * 28).to(device)
        for i in range(10):
            total[i] += len(np.where(y == i))
            if filter_by_label:
                map[i] += torch.sum(x[y == i] * w[i], dim=0)
            else:
                map[i] += torch.sum(x * w[i], dim=0)
    for i in range(10):
        map[i] = map[i].view(*shape).data.cpu().numpy() / total[i]
    return map


def map_weights_approx(w, device, filter_by_label=True):
    if not exists('mnist-class-means.npy'):
        _, loader = get_batch_suite(100)
        means = {}
        total = [0] * 10
        for _, x, y in loader:
            x = x.to(device)
            for i in range(10):
                total[i] += len(np.where(y == i))
                sum = torch.sum(x[y == i], dim=0)
                if i in means:
                    means[i] += sum
                else:
                    means[i] = sum
        for i in range(10):
            means[i] = (means[i] / total[i]).data.cpu().numpy()
        means = np.concatenate(list(means.values()), axis=0)
        np.save('mnist-class-means.npy', means)
    else:
        means = np.load('mnist-class-means.npy')
    means = means.reshape((10, 28 * 28))
    if not filter_by_label:
        means = np.mean(means, axis=0)[None]
    return (means * w).reshape((10, 28, 28))
