import torch
import torch.nn as nn
from src.datasets import load_blood_transfusion
from src.utils import evaluate_model, set_all_seeds, train_model
from src.models import BloodTransfusionNet
import matplotlib.pyplot as plt

device = torch.device("mps")
random_state = 123
set_all_seeds(random_state)


# Load data
(train_loader, test_loader), (train_dataset, test_dataset) = load_blood_transfusion()

# make model

f, l = next(iter(train_loader))


model = BloodTransfusionNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

train_accuracies = []
test_accuracies = []

epochs = range(10000)
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


plt.plot(epochs, train_accuracies, c="red")
plt.plot(epochs, test_accuracies, c="blue")
plt.legend(["Train", "Test"])
plt.show()
