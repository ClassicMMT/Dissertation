"""
This file plots the probability gap landscape
"""

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from src.datasets import create_loaders, make_chessboard
from src.utils import calculate_probability_gap, plot_probability_gap_landscape, set_all_seeds, train_model
from src.models import GenericNet
import matplotlib.pyplot as plt


random_state = 123
g = set_all_seeds(random_state)
device = torch.device("mps")
batch_size = 128

# simulate data
x, y = make_chessboard(n_blocks=4, n_points_in_block=100, random_state=random_state)

# split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=random_state)
train_loader, _ = create_loaders(x_train, y_train, batch_size=batch_size, generator=g)
test_loader, _ = create_loaders(x_test, y_test, batch_size=batch_size, generator=g)

model = GenericNet(layers=[2, 1024, 512, 256, 2], activation="relu", random_state=random_state)

model = train_model(
    model, train_loader, torch.nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters()), n_epochs=30, verbose=False
)


fig, axs = plt.subplots()
plot_probability_gap_landscape(axs, model, x, y, device)
plt.show()


# calcualte max and min
