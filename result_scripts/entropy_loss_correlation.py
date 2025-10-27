"""
This script computes and plots the correlation between loss, information_content, and entropy on the spambase dataset.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from src.datasets import load_spambase
from src.models import SpamBaseNet
from src.utils import (
    calculate_entropy,
    calculate_information_content,
    calculate_probability_gap,
    load_model,
    set_all_seeds,
)

random_state = 123
set_all_seeds(random_state)
device = torch.device("mps")

# load data and model
(train_loader, test_loader), (train_dataset, test_dataset) = load_spambase(random_state)
model = load_model(SpamBaseNet(), "spambase").to(device).eval()
criterion = nn.CrossEntropyLoss(reduction="none")


# calculate entropy, information_content and loss
with torch.no_grad():

    losses = []
    entropies = []
    information_contents = []
    probability_gaps = []

    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)

        # calculate loss, entropy, information_content
        logits = model(features)
        loss = criterion(logits, labels)
        entropy = calculate_entropy(logits)
        information_content = calculate_information_content(logits)
        gaps = calculate_probability_gap(logits)

        losses.append(loss)
        entropies.append(entropy)
        information_contents.append(information_content)
        probability_gaps.append(gaps)

    losses = torch.cat(losses).cpu().numpy()
    entropies = torch.cat(entropies).cpu().numpy()
    information_contents = torch.cat(information_contents).cpu().numpy()
    probability_gaps = torch.cat(probability_gaps).cpu().numpy()


if True:
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    axs[0, 0].scatter(losses, entropies)
    axs[0, 0].set_xlabel("Loss")
    axs[0, 0].set_ylabel("Entropy")

    axs[0, 1].scatter(losses, information_contents)
    axs[0, 1].set_xlabel("Loss")
    axs[0, 1].set_ylabel("Information Content")

    axs[0, 2].scatter(entropies, information_contents)
    axs[0, 2].set_xlabel("Entropy")
    axs[0, 2].set_ylabel("Information Content")

    axs[1, 0].scatter(losses, probability_gaps)
    axs[1, 0].set_xlabel("Loss")
    axs[1, 0].set_ylabel("Probability Gap")

    axs[1, 1].scatter(probability_gaps, information_contents)
    axs[1, 1].set_xlabel("Probability Gap")
    axs[1, 1].set_ylabel("Information Content")

    axs[1, 2].scatter(probability_gaps, entropies)
    axs[1, 2].set_xlabel("Probability Gap")
    axs[1, 2].set_ylabel("Entropy")

    plt.tight_layout()
    plt.show()
