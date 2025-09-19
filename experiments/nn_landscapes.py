"""
The purpose of this file is to explore the decision boundaries, loss landscape, and entropy landscape of a model.
"""

import numpy as np
from sklearn.datasets import make_circles
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.ticker import LinearLocator
import copy
from src.datasets import create_loaders, make_chessboard
from src.utils import (
    calculate_entropy,
    plot_entropy_landscape,
    plot_loss_landscape,
    set_all_seeds,
    train_model,
    plot_boundary,
)
from src.models import GenericNet

device = torch.device("mps")
random_state = 123
g = set_all_seeds(random_state)

############# Simulate Data #############

# x1 = torch.randn((200, 1), generator=g) * (5**0.5)
# x2 = torch.cat(
#     (torch.randn((100, 1), generator=g) + 5, torch.randn((100, 1), generator=g))
# )
# x = torch.cat((x1, x2), dim=1)
# y = torch.cat((torch.zeros(100), torch.ones(100)))
# loader, dataset = create_loaders(x, y, batch_size=50, generator=g)

# x1 = torch.randn((200, 1), generator=g) * (5**0.5)
# x2 = torch.cat(
#     (torch.randn((100, 1), generator=g) + 1, torch.randn((100, 1), generator=g))
# )
# x = torch.cat((x1, x2), dim=1)
# y = torch.cat((torch.zeros(100), torch.ones(100)))
# # x = scale_datasets(x)
# # x = torch.tensor(x, dtype=torch.float32)
# loader, dataset = create_loaders(x, y, batch_size=50, generator=g)


# x, y = make_circles(n_samples=500, noise=0.05, factor=0.5, random_state=42)
# x = torch.tensor(x, dtype=torch.float32)
# y = torch.tensor(y, dtype=torch.long)
# loader, dataset = create_loaders(x, y, batch_size=50, generator=g)


# x, y = make_chessboard(n_blocks=2, random_state=random_state)
x, y = make_chessboard(n_blocks=4, random_state=random_state)
# x, y = make_chessboard(n_blocks=2, all_different_classes=True, random_state=random_state)
loader, dataset = create_loaders(x, y, batch_size=128, generator=g)

############# Stuff that's required for later #############

# required to pre-calculate the global loss and entropy colour bar limits
xx, yy = np.meshgrid(
    np.linspace(x[:, 0].min() - 1, x[:, 0].max() + 1, 200),
    np.linspace(x[:, 1].min() - 1, x[:, 1].max() + 1, 200),
)
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)

############# Model + Training #############


# model = GenericNet(layers=[2, 16, 2], activation="relu")
# model = GenericNet(layers=[2, 256, 2], activation="relu")
# model = GenericNet(layers=[2, 1024, 256, 2], activation="relu")
model = GenericNet(layers=[2, 1024, 512, 256, 128, 2], activation="relu")

# this one is for when there is a chessboard with four classes
# model = GenericNet(layers=[2, 256, 4], activation="relu")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())


############# Train + loop #############

# epochs_to_plot = [10, 20, 30, 40, 50, 100, 200, 500, 750, 1000]
epochs_to_plot = [10, 20, 30, 40, 50, 100, 500, 1000]
epochs = max(epochs_to_plot)

models = []
accuracies = []
loss_min, loss_max = float("inf"), -float("inf")
entropy_min, entropy_max = float("inf"), -float("inf")
log_entropy = True

# train models
for epoch in range(epochs):
    print(f"Epoch: {epoch+1}")
    model, accuracy = train_model(
        model,
        loader,
        criterion,
        optimizer,
        n_epochs=1,
        device=device,
        verbose=False,
        return_accuracy=True,
    )

    # Pre-computing the loss and entropy values for consistent colour bars later
    if (epoch + 1) in epochs_to_plot:
        models.append(copy.deepcopy(model))
        accuracies.append(accuracy)
        with torch.no_grad():
            logits = model(grid)
            predicted = logits.argmax(dim=-1)
            losses = F.cross_entropy(logits, predicted, reduction="none").cpu()
            entropy = calculate_entropy(logits).cpu()
            if log_entropy:
                entropy = torch.log(entropy + 1e-12)

            # save global results
            loss_min = min(loss_min, losses.min())
            loss_max = max(loss_max, losses.max())
            entropy_min = min(entropy_min, entropy.min())
            entropy_max = max(entropy_max, entropy.max())


fig, axs = plt.subplots(nrows=3, ncols=len(epochs_to_plot), figsize=(20, 8))
loss_contours = []
entropy_contours = []
point_size = 10

for index, (epoch, accuracy) in enumerate(zip(epochs_to_plot, accuracies)):
    current_model = models[index]
    plot_boundary(
        axs[0, index],
        current_model,
        x,
        y,
        device,
        f"Epochs: {epoch},\nAcc: {accuracy*100:.2f}",
        point_size=point_size,
    )
    loss_contour = plot_loss_landscape(
        axs[1, index],
        current_model,
        x,
        y,
        device,
        title=f"Loss",
        colour_limits=(loss_min, loss_max),
        point_size=point_size,
    )
    entropy_contour = plot_entropy_landscape(
        axs[2, index],
        current_model,
        x,
        y,
        device,
        title=f"Entropy",
        colour_limits=(entropy_min, entropy_max),
        point_size=point_size,
        log_entropy=log_entropy,
    )
    loss_contours.append(loss_contour)
    entropy_contours.append(entropy_contour)

# Making the colour bar


# For loss colorbar
loss_norm = Normalize(vmin=loss_min.item(), vmax=loss_max.item())
loss_sm = cm.ScalarMappable(norm=loss_norm, cmap="viridis")  # use same cmap as in plot_loss_landscape
loss_sm.set_array([])
fig.colorbar(loss_sm, ax=axs[1, -1], label="Loss", shrink=0.8)

# For entropy colorbar
entropy_norm = Normalize(vmin=entropy_min.item(), vmax=entropy_max.item())
entropy_sm = cm.ScalarMappable(norm=entropy_norm, cmap="plasma")  # use same cmap as in plot_entropy_landscape
entropy_sm.set_array([])
label = "Entropy (log)" if log_entropy else "Entropy"
cbar = fig.colorbar(entropy_sm, ax=axs[2, -1], label=label, shrink=0.8)
cbar.locator = LinearLocator(numticks=4)
cbar.update_ticks()


plt.tight_layout()
for temp in axs:
    for ax in temp:
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
plt.show()
