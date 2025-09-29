import torch
import torch.nn as nn
from src.datasets import create_loaders, make_chessboard
from src.models import GenericNet
from src.utils import *
import matplotlib.pyplot as plt

random_state = 123
g = set_all_seeds(random_state)
device = torch.device("mps")

x, y = make_chessboard(n_blocks=4, random_state=random_state)
loader, dataset = create_loaders(x, y, batch_size=128, generator=g)

model = GenericNet(layers=[2, 1024, 512, 256, 2], activation="relu").to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
n_epochs = 50

model = train_model(model, loader, criterion, optimizer, n_epochs=n_epochs)


if True:
    no_scatter = True
    fig, axs = plt.subplots(1, 5, figsize=(16, 4))
    plot_boundary(axs[0], model, x, y, device=device)
    plot_loss_landscape(axs[1], model, x, y, device=device)
    plot_entropy_landscape(axs[2], model, x, y, device=device)
    plot_density_landscape(axs[3], x, y)
    plot_information_content_landscape(axs[4], model, x, y, no_scatter=no_scatter)
    plt.tight_layout()
    plt.show()
