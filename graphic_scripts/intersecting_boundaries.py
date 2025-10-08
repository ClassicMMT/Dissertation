"""
This script trains the chessboard model and shows, for every 10 epochs:
    * the decision boundary
    * the loss landscape
    * the entropy landscape
    * the information content landscape

This script differs from chessboard.py because it has different classes
"""

import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from src.datasets import create_loaders, load_chessboard, make_chessboard
from src.utils import (
    evaluate_model,
    plot_boundary,
    plot_probability_gap_landscape,
    set_all_seeds,
    train_model,
    plot_information_content_landscape,
    plot_loss_landscape,
    plot_entropy_landscape,
)
from src.models import GenericNet
import matplotlib.pyplot as plt


# Setup
random_state = 123
g = set_all_seeds(random_state)
device = torch.device("mps")
batch_size = 128

# Data
x, y = make_chessboard(n_blocks=2, random_state=random_state, all_different_classes=True)
n = len(y.unique())  # number of classes
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=random_state)
train_loader, _ = create_loaders(x_train, y_train, batch_size, generator=g)
test_loader, _ = create_loaders(x_test, y_test, batch_size, generator=g)


############# Train models #############


# Model
model = GenericNet(layers=[2, 512, 256, n], activation="relu", random_state=random_state).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Training the model
model = train_model(model, train_loader, criterion, optimizer, n_epochs=10, device=device, verbose=False)


############# Entropy vs. Information content #############

if True:
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    test_accuracy = evaluate_model(model, test_loader, device=device)
    # plot_boundary(
    #     axs[0],
    #     model,
    #     x,
    #     y,
    #     device=device,
    #     title=f"Boundary",
    # )

    # plot loss
    plot_loss_landscape(axs[0], model, x, y, device=device, title=f"Loss", no_scatter=True)

    # plot the decision boundary
    plot_entropy_landscape(
        axs[1],
        model,
        x,
        y,
        device=device,
        title=f"Entropy",
        plot_x=x_test,
        plot_y=y_test,
        no_scatter=True,
    )
    plot_information_content_landscape(
        axs[2],
        model,
        x,
        y,
        device=device,
        title="Information Content",
        plot_x=x_test,
        plot_y=y_test,
        no_scatter=True,
    )
    plot_probability_gap_landscape(
        axs[3], model, x, y, device=device, title="Probability Gaps", plot_x=x_test, plot_y=y_test, no_scatter=True
    )

    # fig.text(0.005, 0.8, "Loss", va="center", rotation="vertical", fontsize=14)
    # fig.text(0.005, 0.75, "Entropy", va="center", rotation="vertical", fontsize=14)
    # fig.text(0.005, 0.5, "Information content", va="center", rotation="vertical", fontsize=14)
    # fig.text(0.005, 0.25, "Probability gaps", va="center", rotation="vertical", fontsize=14)
    plt.tight_layout(rect=[0.01, 0, 1, 1])
    plt.show()

# Single boundary plot
if True:
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))

    plot_boundary(
        axs,
        model,
        x,
        y,
        device=device,
        title=f"Boundary",
    )
    plt.tight_layout()
    plt.show()
