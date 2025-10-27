"""
The point of this file is to make a plot about how
the entropy distribution of the test set changes
for different levels of overfitting
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from src.models import SpamBaseNet
from src.datasets import load_spambase
from src.utils import ApproxConvexHull, calculate_entropy, evaluate_model, set_all_seeds, train_model, BoundingBox

random_state = 123
g = set_all_seeds(random_state)
device = torch.device("mps")

# loading data and training model
(train_loader, test_loader), (train_dataset, test_dataset) = load_spambase(scale="standard", random_state=random_state)
model = SpamBaseNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs_of_interest = [1, 5, 10, 20, 50, 200]

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

for n_epochs in range(1, max(epochs_of_interest) + 1):
    model = train_model(model, loader=train_loader, criterion=criterion, optimizer=optimizer, n_epochs=1, device=device)

    if n_epochs in epochs_of_interest:

        # calculate test entropies
        with torch.no_grad():
            test_entropies = []
            model.eval()
            for f, l in test_loader:
                f = f.to(device)
                l = l.to(device)

                logits = model(f)
                test_entropies += [calculate_entropy(logits)]

            test_entropies = torch.cat(test_entropies)

        i = epochs_of_interest.index(n_epochs)
        row = i // 3
        col = i % 3
        sns.histplot(test_entropies, bins=30, ax=axs[row, col], kde=True)
        # sns.kdeplot(test_entropies, fill=False, ax=axs[row, col])
        axs[row, col].set_title(f"Epoch {n_epochs}")

plt.tight_layout()
plt.show()
