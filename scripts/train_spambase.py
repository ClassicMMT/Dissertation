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
from src.datasets import create_loaders, load_spambase, scale_datasets
from src.models import SpamBaseNet


# Set variables and random seeds
random_state = 123
device = torch.device("mps")
g = set_all_seeds(random_state)
batch_size = 128

# Load data
(train_loader, calib_loader, test_loader), (train_dataset, calib_dataset, test_dataset) = load_spambase(
    batch_size=batch_size, test_size=0.25, return_train_val_test=True, random_state=random_state
)


# Model
model = SpamBaseNet().to(device)
n_epochs = 30

########### SET NAME HERE ###########
save_name = "spambase_" + "final"
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
    n_epochs=n_epochs,
    return_accuracy=True,
)

# Evaluate model
print(f"SpamBase Model Test Accuracy: {evaluate_model(model ,test_loader, device=device):.4f}")

# Save model
save_model(model, save_name)
