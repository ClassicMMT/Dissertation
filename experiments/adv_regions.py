import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import load_model, set_all_seeds
from src.datasets import load_spambase
from src.models import SpamBaseNet
from src.attacks import attack_model
import foolbox as fb
import umap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

device = torch.device("mps")
random_state = 123
set_all_seeds(random_state)

# Load model
model = load_model(empty_instance=SpamBaseNet(), load_name="spambase")
(train_loader, test_loader), (train_dataset, test_dataset) = load_spambase()

# Attacks and projections
epsilons = [0.01, 0.05, 0.1, 0.15, 0.2]
projections, adversarial_indicators, true_class, predicted_labels = [], [], [], []

for epsilon in epsilons:

    # Generate attacks
    attack = fb.attacks.basic_iterative_method.LinfAdamBasicIterativeAttack()
    adversarial_examples, true_labels = attack_model(
        model, attack=attack, loader=train_loader, epsilons=epsilon
    )

    # move examples to cpu
    adversarial_examples = adversarial_examples.to("cpu")
    true_labels = true_labels.to("cpu")

    # Extract the original data
    examples = train_dataset.tensors[0]
    labels = train_dataset.tensors[1]

    # combine original data with the adversarial examples
    all_examples = torch.cat((examples, adversarial_examples)).numpy()
    is_adv = torch.cat(
        (torch.zeros(len(examples)), torch.ones(len(adversarial_examples)))
    ).numpy()
    all_labels = torch.cat((labels, true_labels))

    # Predict on all examples
    predictions = model(torch.tensor(all_examples).to(device)).argmax(dim=-1).to("cpu")

    umap_reducer = umap.UMAP(n_components=2, n_jobs=1)
    umap_projection = umap_reducer.fit_transform(all_examples)

    projections.append(umap_projection)
    adversarial_indicators.append(is_adv)
    true_class.append(all_labels)
    predicted_labels.append(predictions)


# Plots of projections coloured by adversarial or normal
fig, axes = plt.subplots(3, 5, figsize=(25, 15))
for ax, projection, indicator, epsilon in zip(
    axes[0], projections, adversarial_indicators, epsilons
):

    sc = ax.scatter(
        projection[:, 0],
        projection[:, 1],
        c=indicator,
        alpha=0.01,
        cmap="bwr",
    )
    ax.set_title(f"Epsilon={epsilon}")
legend_handles = [
    mpatches.Patch(color="blue", label="Original"),
    mpatches.Patch(color="red", label="Adversarial"),
]
fig.legend(handles=legend_handles)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Plots of projections coloured by true_labels
for ax, projection, classes, epsilon in zip(axes[1], projections, true_class, epsilons):

    sc = ax.scatter(
        projection[:, 0],
        projection[:, 1],
        c=classes,
        alpha=0.01,
        cmap="bwr",
    )
    ax.set_title(f"Epsilon={epsilon}")
legend_handles = [
    mpatches.Patch(color="blue", label="True Class 0"),
    mpatches.Patch(color="red", label="True Class 1"),
]
fig.legend(handles=legend_handles, loc="center right")
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Plots of projections coloured by predicted labels
for ax, projection, predicted_class, epsilon in zip(
    axes[2], projections, predicted_labels, epsilons
):

    sc = ax.scatter(
        projection[:, 0],
        projection[:, 1],
        c=predicted_class,
        alpha=0.01,
        cmap="bwr",
    )
    ax.set_title(f"Epsilon={epsilon}")
legend_handles = [
    mpatches.Patch(color="blue", label="Predicted Class 0"),
    mpatches.Patch(color="red", label="Predicted Class 1"),
]
fig.legend(handles=legend_handles, loc="lower right")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
