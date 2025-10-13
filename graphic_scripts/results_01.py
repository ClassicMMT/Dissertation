"""
This script creates some of the first plots for the section "Quantile Method".

The script creates four plots, for alphas [0.05, 0.1, 0.2, 0.3] on the chessboard dataset,
computes the relevant threshold for the uncertainty metric and plots them.
"""

from sklearn.model_selection import train_test_split
import torch
from src.utils import (
    calculate_entropy,
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

    uncertainties = []
    for features, labels in calib_loader:
        features = features.to(device)
        lables = labels.to(device)
        logits = model(features)
        uncertainty = calculate_entropy(logits)
        uncertainties.append(uncertainty)

    uncertainties = torch.cat(uncertainties)
    thresholds = [torch.quantile(uncertainties, 1 - alpha, interpolation="higher") for alpha in alphas]

# compute test entropies
with torch.no_grad():
    model.eval()

    logits = model(x_test.to(device))
    test_uncertainties = calculate_entropy(logits)


# plots
if True:
    fig, axs = plt.subplots(1, len(thresholds), figsize=(5 * len(thresholds), 5))
    colour_map = {0: "blue", 1: "red"}
    for i, (alpha, threshold, ax) in enumerate(zip(alphas, thresholds, axs)):
        entropies_over_threshold = (test_uncertainties >= threshold).int().numpy()
        plot_boundary(
            ax,
            model,
            x_test,
            entropies_over_threshold,
            device,
            title=f"Alpha={alpha}          Threshold:{threshold:.4f}",
        )
        # plot_entropy_landscape(ax, model, x_test, entropies_over_threshold, device, title=f"Alpha={alpha}          Threshold:{threshold:.4f}")
    plt.tight_layout()
    plt.show()
