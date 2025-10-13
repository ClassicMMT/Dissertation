from src.datasets import load_imagenet
from src.utils import set_all_seeds
import matplotlib.pyplot as plt

random_state = 123
g = set_all_seeds(random_state)

(val_loader, test_loader), _ = load_imagenet(generator=g)
features, labels = next(iter(val_loader))

if True:
    start = 5
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    for i in range(3):

        label = labels[start + i]
        image = features[start + i].permute(1, 2, 0)
        axs[i].imshow(image)
    plt.tight_layout()
    plt.show()
