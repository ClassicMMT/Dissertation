import torch
import torch.nn as nn
import torch.optim as optim
from src.utils import (
    set_all_seeds,
    train_model,
    evaluate_model,
    save_model,
)
from src.datasets import load_spambase
from src.models import GenericNet
import matplotlib.pyplot as plt


# Set variables and random seeds
random_state = 123
device = torch.device("mps")
set_all_seeds(random_state)

# Load data
(train_loader, test_loader), (train_dataset, test_dataset) = load_spambase(
    random_state=random_state
)

training_accuracies = []
test_accuracies = []

# epochs = [60, 100, 200, 300, 400]
epochs = range(1000)
model = GenericNet(layers=[57, 256, 128, 64, 32, 2], activation="tanh")

train_accuracies = []
test_accuracies = []

for epoch in epochs:

    ########### SET NAME HERE ###########
    save_name = "spambase_generic_" + str(epoch + 1)
    #####################################

    # Initialise model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train model
    model, train_accuracy = train_model(
        model,
        loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        n_epochs=1,
        return_accuracy=True,
    )

    # Evaluate model
    test_accuracy = evaluate_model(model, test_loader, device=device)

    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    if (epoch + 1) in [25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        # Save model
        save_model(model, save_name)

plt.plot(epochs, train_accuracies, c="red")
plt.plot(epochs, test_accuracies, c="orange")
plt.legend(["Train", "Test"])
plt.vlines(x=199, ymin=0.85, ymax=1)
plt.show()
