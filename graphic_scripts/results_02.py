"""
This script creates some of the results for the section "Quantile Method".

The script creates agreement matrices, for given alphas on the chessboard dataset,
computing the relevant threshold for the uncertainty metric and putting the agreement between
the methods into a matrix.
"""

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from src.utils import (
    calculate_entropy,
    calculate_information_content,
    calculate_probability_gap,
    evaluate_model,
    plot_boundary,
    plot_entropy_landscape,
    set_all_seeds,
    train_model,
)
from src.models import GenericNet
from src.datasets import create_loaders, make_chessboard
import matplotlib.pyplot as plt

random_state = 123
g = set_all_seeds(random_state)
device = torch.device("mps")
batch_size = 128


x, y = make_chessboard(n_blocks=4, n_points_in_block=128, random_state=random_state)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=random_state)
x_train, x_calib, y_train, y_calib = train_test_split(
    x_train, y_train, test_size=0.25, stratify=y_train, random_state=random_state
)
train_loader, _ = create_loaders(x_train, y_train, batch_size=batch_size, generator=g)
calib_loader, _ = create_loaders(x_calib, y_calib, batch_size=batch_size, generator=g)
test_loader, _ = create_loaders(x_test, y_test, batch_size=batch_size, generator=g)
n = len(y.unique())


model = GenericNet(layers=[2, 1024, 512, 256, n], random_state=random_state)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model = train_model(model, train_loader, criterion, optimizer, n_epochs=30, verbose=False, device=device)

# fig, axs = plt.subplots(1, 1)
# plot_boundary(axs, model, x_test, y_test, device)
# plt.show()

alphas = [0.05, 0.1, 0.2, 0.3]


# compute thresholds
with torch.no_grad():
    model.eval()

    entropies = []
    information_contents = []
    probability_gaps = []

    for features, labels in calib_loader:
        features = features.to(device)
        lables = labels.to(device)
        logits = model(features)

        entropy = calculate_entropy(logits)
        info_content = calculate_information_content(logits)
        gaps = calculate_probability_gap(logits)

        entropies.append(entropy)
        information_contents.append(info_content)
        probability_gaps.append(gaps)

    entropies = torch.cat(entropies)
    information_contents = torch.cat(information_contents)
    probability_gaps = torch.cat(probability_gaps)

    # compute thresholds
    entropy_thresholds = [torch.quantile(entropies, 1 - alpha, interpolation="higher") for alpha in alphas]
    information_thresholds = [
        torch.quantile(information_contents, 1 - alpha, interpolation="higher") for alpha in alphas
    ]
    gap_thresholds = [torch.quantile(probability_gaps, alpha, interpolation="lower") for alpha in alphas]

# compute test uncertainties
with torch.no_grad():
    model.eval()

    logits = model(x_test.to(device))

    test_entropies = calculate_entropy(logits)
    test_info_content = calculate_information_content(logits)
    test_gaps = calculate_probability_gap(logits)


# Agreement Matrices

if True:
    i = 0
    print(f"Alpha: {alphas[i]}")
    entropy_threshold = entropy_thresholds[i]
    information_threshold = information_thresholds[i]
    gap_threshold = gap_thresholds[i]

    uncertain_entropy = test_entropies >= entropy_threshold
    uncertain_information = test_info_content >= information_threshold
    uncertain_gap = test_gaps <= gap_threshold

    agreement_matrix = torch.zeros((3, 3))
    agreement_matrix[1, 0] = (uncertain_entropy == uncertain_information).float().mean()
    agreement_matrix[2, 0] = (uncertain_entropy == uncertain_gap).float().mean()
    agreement_matrix[2, 1] = (uncertain_information == uncertain_gap).float().mean()

    print(agreement_matrix)
