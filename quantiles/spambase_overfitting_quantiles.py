"""
This file plots a graph of uncertain point accuracy and
certain point accuracy by model overfitting using the quantile method.

This file plots accuracies of:
    * uncertain points
    * certain points
    * test accuracy
    * train accuracy

by epoch trained.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.datasets import create_loaders, load_spambase, scale_datasets
from src.models import SpamBaseNet
from src.utils import calculate_entropies_from_loader, evaluate_model, set_all_seeds, train_model
import matplotlib.pyplot as plt


random_state = 123
g = set_all_seeds(random_state)
device = torch.device("mps")
batch_size = 128

# load data
x_train, x_test, y_train, y_test = load_spambase(return_raw=True, random_state=random_state)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, stratify=y_train, random_state=random_state)
x_train, x_val, x_test = scale_datasets(x_train, x_val, x_test)
train_loader, train_dataset = create_loaders(x_train, y_train, batch_size=batch_size, generator=g)
val_loader, val_dataset = create_loaders(x_val, y_val, batch_size=batch_size, generator=g)
test_loader, test_dataset = create_loaders(x_test, y_test, batch_size=batch_size, generator=g)

# load model etc
model = SpamBaseNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
alpha = 0.05
epochs = [1, 2, 3, 4, 5, 10, 15, 20, 50, 100]  # the epochs we want to see results for

results = {
    "epochs": epochs,
    "uncertain_accuracy": [],
    "certain_accuracy": [],
    "train_accuracy": [],
    "test_accuracy": [],
}

for epoch in range(max(epochs)):
    print(f"Epoch:{epoch+1}/{max(epochs)}")
    model = train_model(model, train_loader, criterion=criterion, optimizer=optimizer, n_epochs=1, verbose=False)

    if (epoch + 1) in epochs:
        with torch.no_grad():
            model.eval()

            # calculate validation entropies
            entropies, _ = calculate_entropies_from_loader(model, loader=val_loader, device=device)

            # calculate q_hat
            q_hat = torch.quantile(entropies, 1 - alpha, interpolation="higher")

            # calculate entropies and whether model is correct for the test loader
            entropies, is_correct = calculate_entropies_from_loader(model, loader=test_loader, device=device)

            # calculate which are uncertain
            uncertain = entropies >= q_hat

            # calculate accuracies based on q_hat
            uncertain_accuracy = is_correct[uncertain].float().mean().item()
            certain_accuracy = is_correct[~uncertain].float().mean().item()

            # also get train + test accuracies
            train_accuracy = evaluate_model(model, loader=train_loader, device=device)
            test_accuracy = evaluate_model(model, loader=test_loader, device=device)

            # save results
            results["certain_accuracy"].append(certain_accuracy)
            results["uncertain_accuracy"].append(uncertain_accuracy)
            results["test_accuracy"].append(test_accuracy)
            results["train_accuracy"].append(train_accuracy)

if True:
    x_ticks = range(len(epochs))
    plt.plot(x_ticks, results["certain_accuracy"], color="blue", label="Certain Accuracy")
    plt.plot(x_ticks, results["uncertain_accuracy"], color="red", label="Uncertain Accuracy")
    plt.plot(x_ticks, results["train_accuracy"], color="purple", label="Train Accuracy", linestyle="dotted")
    plt.plot(x_ticks, results["test_accuracy"], color="orange", label="Test Accuracy", linestyle="dotted")
    plt.xticks(x_ticks, labels=epochs)
    plt.axhline(0.5, label="Random Chance", linestyle="dashed")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy by Entropy >= q_hat")
    plt.tight_layout()
    plt.legend()
    plt.show()
