import torch
import torch.nn as nn
from src.datasets import load_heloc
from src.models import HelocNet
from src.utils import (
    set_all_seeds,
    train_model,
    evaluate_model,
    save_model,
)


device = torch.device("mps")
(train_loader, test_loader), (train_dataset, test_dataset) = load_heloc()
random_state = 123
set_all_seeds(random_state)


# Normal Heloc
# Achieves 70.69% Training and 76.15% test accuracy

model = HelocNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
n_epochs = 10

model = train_model(
    model,
    train_loader,
    criterion,
    optimizer,
    n_epochs=n_epochs,
    device=device,
)

train_accuracy = evaluate_model(model, train_loader, device=device)
test_accuracy = evaluate_model(model, test_loader, device=device)
print(
    f"Trained for {n_epochs} epochs."
    + f"Train accuracy: {train_accuracy*100:.2f}%, "
    + f"Test accuracy: {test_accuracy*100:.2f}%"
)

save_model(model, "heloc")

# OVERFIT Heloc
# Achieves 90.3% training accuracyand 52.49% test accuracy

set_all_seeds(random_state)

model = HelocNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
n_epochs = 1000

model = train_model(
    model,
    train_loader,
    criterion,
    optimizer,
    n_epochs=n_epochs,
    device=device,
)

train_accuracy = evaluate_model(model, train_loader, device=device)
test_accuracy = evaluate_model(model, test_loader, device=device)
print(
    f"Trained for {n_epochs} epochs."
    + f"Train accuracy: {train_accuracy*100:.2f}%, "
    + f"Test accuracy: {test_accuracy*100:.2f}%"
)

save_model(model, "heloc_overfit")
