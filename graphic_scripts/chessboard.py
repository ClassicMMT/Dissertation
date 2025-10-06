"""
This script trains the chessboard model and shows, for every 10 epochs:
    * the decision boundary
    * the loss landscape
    * the entropy landscape
    * the information content landscape
"""

import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from src.datasets import create_loaders, load_chessboard, make_chessboard
from src.utils import (
    evaluate_model,
    plot_boundary,
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
x, y = make_chessboard(n_blocks=4, random_state=random_state)
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=random_state)
train_loader, _ = create_loaders(x_train, y_train, batch_size, generator=g)
test_loader, _ = create_loaders(x_test, y_test, batch_size, generator=g)

############# Train models #############

epochs = np.arange(10, 101, 10)
models = []

for n_epoch in epochs:
    print(f"Training model with n_epochs: {n_epoch}")

    # Model
    model = GenericNet(layers=[2, 1024, 512, 256, 2], activation="relu", random_state=random_state).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Training the model
    model = train_model(model, train_loader, criterion, optimizer, n_epochs=n_epoch, device=device, verbose=False)

    # save model
    models.append(model)


############# Decision boundary or loss landscape #############

fig, axs = plt.subplots(2, 5, figsize=(20, 8))

for i, (n_epoch, model) in enumerate(zip(epochs, models)):
    row, col = i // 5, i % 5

    test_accuracy = evaluate_model(model, test_loader, device=device)

    # plot the decision boundary
    # plot_boundary(
    #     axs[row, col],
    #     model,
    #     x,
    #     y,
    #     device=device,
    #     title=f"Epochs: {n_epoch}, Acc: {test_accuracy*100:.2f}%",
    #     plot_x=x_test,
    #     plot_y=y_test,
    # )
    plot_loss_landscape(
        axs[row, col],
        model,
        x,
        y,
        device=device,
        title=f"Epochs: {n_epoch}, Acc: {test_accuracy*100:.2f}%",
        plot_x=x_test,
        plot_y=y_test,
    )

plt.tight_layout()
plt.show()


############# Entropy vs. Information content #############
model_subset = models[:5]
epoch_subset = epochs[:5]

fig, axs = plt.subplots(2, 5, figsize=(20, 8))

for i, (n_epoch, model) in enumerate(zip(epoch_subset, model_subset)):

    test_accuracy = evaluate_model(model, test_loader, device=device)

    # plot the decision boundary
    plot_entropy_landscape(
        axs[0, i],
        model,
        x,
        y,
        device=device,
        title=f"Epochs: {n_epoch}, Acc: {test_accuracy*100:.2f}%",
        plot_x=x_test,
        plot_y=y_test,
    )
    plot_information_content_landscape(
        axs[1, i],
        model,
        x,
        y,
        device=device,
        title=None,
        plot_x=x_test,
        plot_y=y_test,
    )

fig.text(0.01, 0.75, "Entropy", va="center", rotation="vertical", fontsize=14)
fig.text(0.01, 0.25, "Information content", va="center", rotation="vertical", fontsize=14)
plt.tight_layout(rect=[0.01, 0, 1, 1])
plt.show()
