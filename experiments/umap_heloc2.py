import numpy as np
import torch
import umap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from src.attacks import load_all_adversarial_examples
from src.datasets import load_heloc
from src.models import HelocNet
from src.utils import sample, set_all_seeds, load_model

random_state = 123
set_all_seeds(random_state)

(train_loader, test_loader), (train_dataset, test_dataset) = load_heloc()


reducer = umap.UMAP(n_components=2, random_state=random_state, n_jobs=1)
train_projection = reducer.fit_transform(train_dataset.tensors[0])
test_projection = reducer.transform(test_dataset.tensors[0])

# Original Data
plt.scatter(
    train_projection[:, 0], train_projection[:, 1], c="blue", label="Train", alpha=0.1
)
plt.scatter(
    test_projection[:, 0], test_projection[:, 1], c="red", label="Test", alpha=0.1
)
plt.legend()
plt.title("Point in Train/Test")
plt.show()
