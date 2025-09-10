import torch
import torch.nn as nn
import torch.nn.functional as F


class HelocNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(23, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class SpamBaseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(57, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class GenericNet(nn.Module):
    def __init__(self, layers, activation="relu"):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(inp, out) for inp, out in zip(layers[:-1], layers[1:])]
        )
        if activation == "relu":
            activation = nn.ReLU
        elif activation == "sigmoid":
            activation = nn.Sigmoid
        elif activation == "tanh":
            activation = nn.Tanh
        else:
            raise ValueError("Activation must be: relu | sigmoid | tanh")

        self.activation = activation()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # if not the last layer -> apply activation
            if i != len(self.layers) - 1:
                x = self.activation(x)
        return x


class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 16 * 4 * 4)  # flatten
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.fc3(x)
        return x


class IrisNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class BloodTransfusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 256)

        self.fc15 = nn.Linear(256, 128)
        self.fc16 = nn.Linear(128, 64)

        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc15(x)
        x = F.relu(x)
        x = self.fc16(x)
        x = F.relu(x)

        x = self.fc2(x)
        return x
