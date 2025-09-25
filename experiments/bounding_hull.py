"""
The point of this experiment is to get a bounding box or approximate convex hull and
detect whether points will be outside of the box or hull.

We can then look at their entropies or whether the points are correct to see some properties
of these points.
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
n_epochs = 200  # 6 is a good number of epochs for these data
model = train_model(
    model, loader=train_loader, criterion=criterion, optimizer=optimizer, n_epochs=n_epochs, device=device
)
x, y = train_dataset.tensors
xx, yy = test_dataset.tensors

# Bounding box
box = BoundingBox()
box.fit(x)
is_outside = box.predict(xx)

# get the test points which are outside the bounding box
outside_test_x = xx[is_outside].to(device)
outside_test_y = yy[is_outside].to(device)

# get all entropies from the whole test set
with torch.no_grad():
    test_entropies = []
    model.eval()
    for f, l in test_loader:
        f = f.to(device)
        l = l.to(device)
        logits = model(f)
        preds = logits.argmax(dim=-1)
        correct = preds == l

        entropies = calculate_entropy(logits)
        test_entropies += [entropies]

    # combine  all entropies
    test_entropies = torch.cat(test_entropies)

    # calculate entropies for the "outside" points
    logits = model(outside_test_x)
    preds = logits.argmax(dim=-1)
    correct = preds == outside_test_y
    outside_points_entropies = calculate_entropy(logits)

# make a plot of the density and where the points lie
if True:
    colors = plt.cm.tab10.colors
    sns.kdeplot(test_entropies, fill=True)
    for i, outside_entropy in enumerate(outside_points_entropies):
        plt.axvline(
            outside_entropy.item(), color=colors[i % len(colors)], linestyle="--", linewidth=2, label=f"Point {i+1}"
        )
    plt.legend()
    plt.show()

plt.hist(test_entropies, bins=30, density=True, alpha=0.6, color="g")
plt.show()

###### Convex hull approximation

ch = ApproxConvexHull()
ch.fit(x)
is_outside = ch.predict(xx)
outside_test_x = xx[is_outside].to(device)
outside_test_y = yy[is_outside].to(device)

# get logits and entropy of the outside points
logits = model(outside_test_x)
preds = logits.argmax(dim=-1)
correct = preds == outside_test_y
outside_points_entropies = calculate_entropy(logits=logits)


# plot
if True:
    colors = plt.cm.tab10.colors
    sns.kdeplot(test_entropies, fill=True)
    for i, outside_entropy in enumerate(outside_points_entropies):
        plt.axvline(
            outside_entropy.item(), color=colors[i % len(colors)], linestyle="--", linewidth=2, label=f"Point {i+1}"
        )
    plt.legend()
    plt.show()
