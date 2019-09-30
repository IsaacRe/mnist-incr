import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.view = View((-1, 4 * 4 * 50))
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.features = [self.conv1, F.relu, torch.nn.MaxPool2d(2, 2), self.conv2, F.relu, torch.nn.MaxPool2d(2, 2),
                         self.view, self.fc1, F.relu, self.fc2]

    def feature_map(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.view(x)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x):
        x = self.feature_map(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ShortNet(nn.Module):
    def __init__(self):
        super(ShortNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.view = View((-1, 4 * 4 * 50))
        self.fc1 = nn.Linear(4 * 4 * 50, 10)
        self.features = [self.conv1, F.relu, torch.nn.MaxPool2d(2, 2), self.conv2, F.relu, torch.nn.MaxPool2d(2, 2),
                         self.view, self.fc1]

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.view(x)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)
