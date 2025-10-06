"""
This experiment calculates the densities for the training data
and then plots the densities of the training and test set for
the spambase dataset.
"""

import numpy as np
import torch
from src.utils import set_all_seeds, KNNDensity, evaluate_model
from src.datasets import load_spambase
from src.models import SpamBaseNet


random_state = 123
g = set_all_seeds(random_state)
device = torch.device("mps")

(train_loader, test_loader), _ = load_spambase(random_state=random_state)
model = SpamBaseNet().to(device)
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss()

knndensity = KNNDensity(n_neighbours=5)
for epoch in range(10):
    for f, l in train_loader:
        # only update on the first round
        # no point doing it for all epochs
        if epoch == 0:
            knndensity = knndensity.update(f)

        f = f.to(device)
        l = l.to(device)

        logits = model(f)
        loss = criterion(logits, l)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

with torch.no_grad():
    model.eval()

    distances = []
    correct = []

    for features, labels in test_loader:
        features = features.to(device)
        labels = labels.to(device)

        # get correct examples
        logits = model(features)
        is_correct = logits.argmax(dim=-1) == labels

        # calculate distances
        densities = knndensity.predict(features)

        # save results
        distances.append(densities)
        correct.append(is_correct)

    distances = torch.cat(distances).cpu()
    correct = torch.cat(correct).cpu()


import seaborn as sns
import matplotlib.pyplot as plt

sns.kdeplot(distances[correct], label="Correct")
sns.kdeplot(distances[~correct], label="Incorrect")
plt.legend()
plt.show()
