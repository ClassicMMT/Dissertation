import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from src.utils import (
    set_all_seeds,
    train_model,
    evaluate_model,
    save_model,
)
from src.datasets import load_mnist
from src.models import MNISTNet

########### SET NAME HERE ###########
save_name = "mnist"
#####################################

# Set variables and random seeds
random_state = 123
device = torch.device("mps")
set_all_seeds(random_state)

# Load data
(train_loader, test_loader), (train_dataset, test_dataset) = load_mnist(batch_size=512)

# Initialise model
model = MNISTNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model
model = train_model(
    model,
    loader=train_loader,
    criterion=criterion,
    optimizer=optimizer,
    n_epochs=5,
    device=device,
)

# Evaluate model
print(
    f"MNIST Model Test Accuracy: {
        evaluate_model(model ,test_loader, device=device):.4f
    }"
)

# Save model
save_model(model, save_name)
