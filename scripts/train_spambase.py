"""
This file trains SpamBaseNet on the SpamBase dataset and saves the models.
"""

import numpy as np
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
from src.models import SpamBaseNet


# Set variables and random seeds
random_state = 123
device = torch.device("mps")
set_all_seeds(random_state)

# Load data
(train_loader, test_loader), (train_dataset, test_dataset) = load_spambase(random_state=random_state)

training_accuracies = []
test_accuracies = []

epochs = [6, 10, 60, 100, 200, 300, 400]

for epoch in epochs:
    model = SpamBaseNet()

    ########### SET NAME HERE ###########
    save_name = "spambase_" + str(epoch)
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
        n_epochs=epoch,
        return_accuracy=True,
    )

    # Evaluate model
    print(f"SpamBase Model Test Accuracy: {evaluate_model(model ,test_loader, device=device):.4f}")

    # Save model
    save_model(model, save_name)
