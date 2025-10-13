"""
This file plots a graph of uncertain point accuracy and
certain point accuracy by model overfitting using the quantile method.
The curves are from each layer of logits.

This file plots accuracies of:
    * uncertain points
    * certain points

by epoch trained.

Note: This experiment is the one where entropy is calculated from the hidden layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.datasets import create_loaders, load_spambase, scale_datasets
from src.utils import calculate_entropy, evaluate_model, set_all_seeds, train_model
import matplotlib.pyplot as plt

################## Modified functions just for this experiment ######################


class SpamBaseNetModified(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(57, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x, layer_returned: int = 3):
        x = self.fc1(x)
        x = F.relu(x)
        if layer_returned == 1:
            return x
        x = self.fc2(x)
        x = F.relu(x)
        if layer_returned == 2:
            return x
        x = self.fc3(x)
        return x


def calculate_entropies_from_loader(model, loader, extraction_layer, device="mps", verbose=False):
    """
    Computes the entropy for all examples using the given model and dataloader.

    Returns: (entropies, is_correct)
    """
    with torch.no_grad():
        model.eval()

        result_entropies = []
        result_is_correct = []

        for i, (features, labels) in enumerate(loader):
            if verbose:
                print(f"Batch: {i+1}/{len(loader)}")
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features, extraction_layer)
            entropies = calculate_entropy(logits, detach=True)
            is_correct = model(features).argmax(dim=-1) == labels

            result_entropies.append(entropies)
            result_is_correct.append(is_correct)

        result_entropies = torch.cat(result_entropies)
        result_is_correct = torch.cat(result_is_correct)

    return result_entropies, result_is_correct


################## END ######################

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
model = SpamBaseNetModified().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
alpha = 0.05
epochs = [1, 2, 3, 4, 5, 10, 15, 20, 50, 100]  # the epochs we want to see results for

results = {
    "epochs": epochs,
    "uncertain_accuracy_1": [],
    "certain_accuracy_1": [],
    "uncertain_accuracy_2": [],
    "certain_accuracy_2": [],
    "uncertain_accuracy_3": [],
    "certain_accuracy_3": [],
}

for epoch in range(max(epochs)):
    print(f"Epoch:{epoch+1}/{max(epochs)}")
    model = train_model(model, train_loader, criterion=criterion, optimizer=optimizer, n_epochs=1, verbose=False)

    if (epoch + 1) in epochs:
        with torch.no_grad():
            model.eval()

            for layer in [1, 2, 3]:

                # calculate validation entropies
                entropies, _ = calculate_entropies_from_loader(
                    model, extraction_layer=layer, loader=val_loader, device=device
                )

                # calculate q_hat
                q_hat = torch.quantile(entropies, 1 - alpha, interpolation="higher")

                # calculate entropies and whether model is correct for the test loader
                entropies, is_correct = calculate_entropies_from_loader(
                    model, extraction_layer=layer, loader=test_loader, device=device
                )

                # calculate which are uncertain
                uncertain = entropies >= q_hat

                # calculate accuracies based on q_hat
                uncertain_accuracy = is_correct[uncertain].float().mean().item()
                certain_accuracy = is_correct[~uncertain].float().mean().item()

                # save results
                results[f"certain_accuracy_{layer}"].append(certain_accuracy)
                results[f"uncertain_accuracy_{layer}"].append(uncertain_accuracy)

if True:
    x_ticks = range(len(epochs))
    plt.figure(figsize=(14, 8))
    for layer, colour in zip([1, 2, 3], ["red", "blue", "orange"]):
        plt.plot(
            x_ticks,
            results[f"certain_accuracy_{layer}"],
            color=colour,
            label=f"Certain Accuracy from Layer: {layer}",
        )
        plt.plot(
            x_ticks,
            results[f"uncertain_accuracy_{layer}"],
            color=colour,
            label=f"Uncertain Accuracy from Layer: {layer}",
            linestyle="--",
        )
    plt.xticks(x_ticks, labels=epochs)
    plt.axhline(0.5, label="Random Chance", linestyle="dotted")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy by Entropy >= q_hat")
    plt.tight_layout()
    plt.legend(loc="lower right")
    plt.show()
