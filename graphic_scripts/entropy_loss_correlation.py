"""
This script computes and plots the correlation between loss and entropy
on the spambase dataset.
"""

import numpy as np
import torch
import torch.nn as nn
import foolbox as fb
import matplotlib.pyplot as plt
from src.attacks import attack_model, load_adversarial_examples
from src.datasets import load_heloc, load_mnist, load_spambase
from src.models import HelocNet, MNISTNet, SpamBaseNet
from src.utils import calculate_entropy, load_model, set_all_seeds, drop_duplicates

random_state = 123
set_all_seeds(random_state)
device = torch.device("mps")

# load data and model
(train_loader, test_loader), (train_dataset, test_dataset) = load_spambase(random_state)
model = load_model(SpamBaseNet(), "spambase").to(device).eval()

adv_examples, original_labels = load_adversarial_examples("spambase", attack_name="fgsm", epsilon=0.01)
adv_examples_subset = adv_examples[:128].to(device)
original_labels_subset = original_labels[:128]


features, labels = next(iter(train_loader))
features = features.to(device)
labels = labels.to(device)

criterion = nn.CrossEntropyLoss(reduction="none")

# calculate entropy and loss
with torch.no_grad():
    # for normal examples
    logits = model(features)
    loss = criterion(logits, labels).cpu().numpy()
    entropy = calculate_entropy(logits).cpu().numpy()

    # same for adversarial
    adv_logits = model(adv_examples_subset)
    adv_loss = criterion(adv_logits, original_labels_subset).cpu().numpy()
    adv_entropy = calculate_entropy(adv_logits).cpu().numpy()

if True:
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].scatter(loss, entropy)
    axs[0].set_xlabel("Loss")
    axs[0].set_ylabel("Entropy")
    axs[0].set_title("Normal Points")
    axs[1].scatter(adv_loss, adv_entropy)
    axs[1].set_xlabel("Loss")
    axs[1].set_ylabel("Entropy")
    axs[1].set_title("Adversarial Points")
    plt.tight_layout()
    plt.show()
