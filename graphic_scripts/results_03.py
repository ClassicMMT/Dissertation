"""
This script creates some of the first plots for the section "Quantile Method".

The script creates four plots, for alphas [0.05, 0.1, 0.2, 0.3] on the chessboard dataset,
computes the relevant threshold for the uncertainty metric and plots them.

NOTE: This script is for MANY classes. Not two.

Also calculates the agreement matrix
"""

from sklearn.model_selection import train_test_split
import torch
from src.utils import (
    calculate_entropy,
    calculate_information_content,
    calculate_probability_gap,
    evaluate_model,
    plot_boundary,
    plot_entropy_landscape,
    plot_information_content_landscape,
    plot_probability_gap_landscape,
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
epochs = 10


x, y = make_chessboard(n_blocks=4, n_points_in_block=200, random_state=random_state, all_different_classes=True)
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

model = train_model(model, train_loader, criterion, optimizer, n_epochs=epochs, verbose=False, device=device)

# fig, axs = plt.subplots(1, 1)
# plot_boundary(axs, model, x_test, y_test, device)
# plt.show()

alpha = 0.1

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

    entropy_threshold = torch.quantile(entropies, 1 - alpha, interpolation="higher")
    information_threshold = torch.quantile(information_contents, 1 - alpha, interpolation="higher")
    gap_threshold = torch.quantile(probability_gaps, alpha, interpolation="lower")

# compute test entropies
with torch.no_grad():
    model.eval()

    logits = model(x_test.to(device))

    test_entropies = calculate_entropy(logits)
    test_info_content = calculate_information_content(logits)
    test_gaps = calculate_probability_gap(logits)

    uncertain_entropy = test_entropies >= entropy_threshold
    uncertain_information = test_info_content >= information_threshold
    uncertain_gaps = test_gaps <= gap_threshold
    uncertain_either = uncertain_entropy | uncertain_information | uncertain_gaps
    uncertain_all = uncertain_entropy & uncertain_information & uncertain_gaps


# agreement matrix
if True:
    agreement_matrix = torch.zeros((3, 3))
    agreement_matrix[1, 0] = (uncertain_entropy == uncertain_information).float().mean()
    agreement_matrix[2, 0] = (uncertain_entropy == uncertain_gaps).float().mean()
    agreement_matrix[2, 1] = (uncertain_information == uncertain_gaps).float().mean()

    print(agreement_matrix)


# plots
if True:
    x_entropy, y_entropy = x_test[uncertain_entropy], y_test[uncertain_entropy]
    x_info, y_info = x_test[uncertain_information], y_test[uncertain_information]
    x_gap, y_gap = x_test[uncertain_gaps], y_test[uncertain_gaps]
    x_either, y_either = x_test[uncertain_either], y_test[uncertain_either]

    fig, axs = plt.subplots(1, 5, figsize=(25, 5))
    colour_map = {0: "blue", 1: "red"}

    plot_boundary(axs[0], model, x_test, y_test, device=device, title="Test Data - Coloured by Class")
    plot_entropy_landscape(
        axs[1], model, x_test, y_test, device=device, title="Entropy", plot_x=x_entropy, plot_y=y_entropy
    )
    plot_information_content_landscape(
        axs[2], model, x_test, y_test, device=device, title="Information Content", plot_x=x_info, plot_y=y_info
    )
    plot_probability_gap_landscape(
        axs[3], model, x_test, y_test, device=device, title="Probability Gap", plot_x=x_gap, plot_y=y_gap
    )
    plot_boundary(
        axs[4],
        model,
        x_test,
        y_test,
        device=device,
        title="Test Data - Coloured by High Uncertainty",
        plot_x=x_test,
        plot_y=uncertain_all,
        alpha=[1 if x else 0.1 for x in uncertain_either],
    )
    plt.tight_layout()
    plt.show()
