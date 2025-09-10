"""
The purpose of this file is to explore the decision boundaries learned by neural networks on a simple (200, 2) dataset.
"""

import numpy as np
from sklearn.datasets import make_circles
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from src.datasets import create_loaders, scale_datasets
from src.utils import set_all_seeds, train_model
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
# # x = scale_datasets(x)
# # x = torch.tensor(x, dtype=torch.float32)
# loader, dataset = create_loaders(x, y, batch_size=50, generator=g)

x, y = make_circles(n_samples=500, noise=0.05, factor=0.5, random_state=42)
x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
loader, dataset = create_loaders(x, y, batch_size=50, generator=g)


############# Model + Training #############


model = GenericNet(layers=[2, 16, 2], activation="relu")
# model = GenericNet(layers=[2, 1024, 256, 2], activation="relu")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

############# Train + loop #############

fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))
epochs_to_plot = [10, 20, 30, 40, 50, 100, 200, 500, 750, 1000]
epochs = max(epochs_to_plot)


def plot_boundary(axs, model, x, y, device, title):
    """
    Function to plot the boundary of a network.

    Data must be 2 column tabular data.
    """
    xx, yy = np.meshgrid(
        np.linspace(x[:, 0].min() - 1, x[:, 0].max() + 1, 200),
        np.linspace(x[:, 1].min() - 1, x[:, 1].max() + 1, 200),
    )
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)
    with torch.no_grad():
        model.eval()
        labels = model(grid).argmax(dim=-1).cpu()
    Z = labels.reshape(xx.shape)
    axs.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    axs.scatter(x[:, 0], x[:, 1], c=y, edgecolors="k", cmap=plt.cm.coolwarm)
    axs.set_title(title)


for epoch in range(epochs):
    print(f"Epoch: {epoch+1}")
    model = train_model(
        model, loader, criterion, optimizer, n_epochs=1, device=device, verbose=False
    )
    if (epoch + 1) in epochs_to_plot:
        index = epochs_to_plot.index(epoch + 1)
        j, i = index % 5, index // 5
        plot_boundary(axs[i, j], model, x, y, device, f"Epochs Trained: {epoch+1}")

plt.tight_layout()
plt.show()
