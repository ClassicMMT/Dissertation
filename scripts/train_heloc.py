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
import matplotlib.pyplot as plt


device = torch.device("mps")
random_state = 123
set_all_seeds(random_state)


(train_loader, test_loader), (train_dataset, test_dataset) = load_heloc()

# Normal Heloc
# Achieves 70.69% Training and 76.15% test accuracy

model = HelocNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

train_accuracies = []
test_accuracies = []
epochs = range(1, 101)

for epoch in epochs:
    print(f"Progess: {epoch}")

    model, train_accuracy = train_model(
        model,
        train_loader,
        criterion,
        optimizer,
        n_epochs=1,
        device=device,
        verbose=False,
        return_accuracy=True,
    )

    test_accuracy = evaluate_model(model, test_loader, device=device)

    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

plt.plot(epochs, train_accuracies, c="red")
plt.plot(epochs, test_accuracies, c="orange")
# plt.vlines(epochs, train_accuracies, c="orange")
plt.legend(["Train", "Test"])
plt.show()

# save_model(model, "heloc")
