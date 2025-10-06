import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils import (
    plot_boundary,
    plot_entropy_landscape,
    plot_information_content_landscape,
    plot_loss_landscape,
    set_all_seeds,
    train_model,
    evaluate_model,
    save_model,
)
from src.datasets import create_loaders, make_chessboard
from src.models import GenericNet
import matplotlib.pyplot as plt


# Set variables and random seeds
random_state = 123
device = torch.device("mps")
g = set_all_seeds(random_state)
batch_size = 128

# Load data
x, y = make_chessboard(random_state=random_state)
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=random_state)
train_loader, _ = create_loaders(x_train, y_train, batch_size=batch_size, generator=g)
test_loader, _ = create_loaders(x_test, y_test, batch_size=batch_size, generator=g)

training_accuracies = []
test_accuracies = []

# epochs = [30, 50]
epochs = np.arange(1, 102, 10)

for epoch in epochs:
    model = GenericNet(layers=[2, 1024, 512, 256, 2], activation="relu", random_state=random_state)

    ########### SET NAME HERE ###########
    save_name = "chessboard_" + str(epoch)
    #####################################

    # Initialise model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    model, train_accuracy = train_model(
        model,
        loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        n_epochs=epoch,
        return_accuracy=True,
        verbose=False,
    )

    # Evaluate model
    train_accuracy = evaluate_model(model, train_loader, device=device)
    test_accuracy = evaluate_model(model, test_loader, device=device)
    training_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)
    print(f"Chessboard Model Test Accuracy: {test_accuracy:.4f}")

    # Save model
    # save_model(model, save_name)

# plot the training graph
if True:
    plt.plot(epochs - 1, training_accuracies, label="Train")
    plt.plot(epochs - 1, test_accuracies, label="Test")
    plt.axvline(30, color="red", linestyle="dashed")
    plt.xticks(epochs - 1)
    plt.title("Chessboard Training")
    plt.ylabel("Accuracy %")
    plt.xlabel("Epochs")
    plt.tight_layout()
    plt.legend(loc="lower right")
    plt.show()

# if True:
#     fig, axs = plt.subplots(1, 4, figsize=(20, 5))
#     plot_boundary(axs[0], model, x_train, y_train, device=device)
#     plot_loss_landscape(axs[1], model, x=x_train, y=y_train, device=device)
#     plot_entropy_landscape(axs[2], model, x=x_train, y=y_train, device=device)
#     plot_information_content_landscape(axs[3], model, x=x_train, y=y_train)
#     plt.tight_layout()
#     plt.show()
