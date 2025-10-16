"""
This script creates an image to demonstrate the CIFAR-10 dataset.
"""

import torch
from src.datasets import load_cifar
from src.utils import set_all_seeds
import matplotlib.pyplot as plt


random_state = 123
g = set_all_seeds(random_state)
batch_size = 128
device = torch.device("mps")


def load_cifar(batch_size=128, generator=None, download=True):
    """
    Function to load the CIFAR data.

    Returns (val_loader, test_loader), (val_dataset, test_dataset)
    """

    from torch.utils.data import DataLoader, random_split
    import torchvision.transforms as transforms
    from torchvision.datasets import CIFAR10

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CIFAR10(root="data/", transform=transform, download=download, train=False)
    val_dataset, test_dataset = random_split(dataset, lengths=[0.5, 0.5], generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return (val_loader, test_loader), (val_dataset, test_dataset)


(val_loader, test_loader), _ = load_cifar(batch_size, generator=g)

features, labels = next(iter(val_loader))

if True:
    start = 0
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    for i in range(3):

        label = labels[start + i]
        image = features[start + i].permute(1, 2, 0)
        axs[i].imshow(image)
    plt.tight_layout()
    plt.show()
