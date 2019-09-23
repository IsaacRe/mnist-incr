import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def input2img(x, savefile=None):
    plt.imshow(x)
    if savefile:
        plt.imsave(savefile, x)


def map_weights(w, loader, device, savefile=None):
    map = [torch.zeros(w.shape[1]).to(device) for i in range(10)]
    for _, x, _ in loader:
        shape = x.shape[2:]
        x = x.view(-1, 28 * 28).to(device)
        for i in range(10):
            map[i] += torch.sum(x * w[i], dim=0)
    length = len(loader.dataset)
    for i in range(10):
        x = map[i].view(*shape).data.cpu().numpy() / length
        input2img(x, savefile='%s-%d.png' % (savefile, i) if savefile else None)


def vis_w(w, savefile=None):
    w = w.data.cpu().view(10, 28, 28).numpy()
    for i in range(10):
        input2img(w[i], savefile='%s-%d.png' % (savefile, i))
