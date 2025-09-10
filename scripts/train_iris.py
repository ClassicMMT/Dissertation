import torch
import torch.nn as nn
from src.datasets import load_iris
from src.utils import evaluate_model, save_model, set_all_seeds, train_model
from src.models import IrisNet
import matplotlib.pyplot as plt

device = torch.device("mps")
random_state = 123
set_all_seeds(random_state)


# Load data
(train_loader, test_loader), (train_dataset, test_dataset) = load_iris()

# make model


model = IrisNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

train_accuracies = []
test_accuracies = []

epochs = range(1500)
for epoch in epochs:
    model, train_accuracy = train_model(
        model,
        train_loader,
        criterion,
        optimizer,
        n_epochs=1,
        device=device,
        return_accuracy=True,
    )

    test_accuracy = evaluate_model(model, test_loader, device=device)

    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    if (epoch + 1) in [200, 400, 600, 800, 1000, 1500]:
        save_model(model, f"iris_{epoch+1}")


plt.plot(epochs, train_accuracies, c="red")
plt.plot(epochs, test_accuracies, c="blue")
plt.show()
