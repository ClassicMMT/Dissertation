"""
This script computes and plots the correlation between loss, information_content, and entropy on the spambase dataset.
"""

import numpy as np
import torch
import torch.nn as nn
import foolbox as fb
import matplotlib.pyplot as plt
from src.attacks import attack_model, load_adversarial_examples
from src.datasets import load_heloc, load_mnist, load_spambase
from src.models import HelocNet, MNISTNet, SpamBaseNet
from src.utils import calculate_entropy, calculate_information_content, load_model, set_all_seeds, drop_duplicates

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

    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)

        # calculate loss, entropy, information_content
        logits = model(features)
        loss = criterion(logits, labels)
        entropy = calculate_entropy(logits)
        information_content = calculate_information_content(logits)

        losses.append(loss)
        entropies.append(entropy)
        information_contents.append(information_content)

    losses = torch.cat(losses).cpu().numpy()
    entropies = torch.cat(entropies).cpu().numpy()
    information_contents = torch.cat(information_contents).cpu().numpy()


if True:
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].scatter(losses, entropies)
    axs[0].set_xlabel("Loss")
    axs[0].set_ylabel("Entropy")

    axs[1].scatter(losses, information_contents)
    axs[1].set_xlabel("Loss")
    axs[1].set_ylabel("Information Content")

    axs[2].scatter(entropies, information_contents)
    axs[2].set_xlabel("Entropy")
    axs[2].set_ylabel("Information Content")

    plt.tight_layout()
    plt.show()
