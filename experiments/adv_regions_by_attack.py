"""
This experiment generates adversarial examples for a given attacks
and generates UMAP projection plots for every epsilon.

Each row shows:
    * The original examples vs. adversarial
    * The true class labels
    * Points coloured by attack
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import foolbox as fb
import umap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from src.utils import load_model, set_all_seeds, sample
from src.datasets import load_spambase
from src.models import SpamBaseNet
from src.attacks import attack_model, run_multiple_attacks, load_adversarial_examples

device = torch.device("mps")
random_state = 123
set_all_seeds(random_state)

# Load model
model_name = "spambase"
model = load_model(empty_instance=SpamBaseNet(), load_name=model_name).to(device)
(train_loader, test_loader), (train_dataset, test_dataset) = load_spambase()

attacks = [
    "bim_linf",
    "cw_l2",
    "df_l2",
    # "df_linf",
    "fgsm",
]

attack_index_to_name = {i: (["original"] + attacks)[i] for i in range(len(attacks) + 1)}

projections = []
adversarial_indicators = []
true_class = []
predicted_labels = []
attack_types = []
epsilons = [0.01, 0.05, 0.1, 0.15, 0.2]

for epsilon in epsilons:

    adversarial_examples = []
    true_labels = []
    attack_names = []

    # Load pre-generated attacks
    for i, attack_name in enumerate(attacks):

        adv_examples, original_labels = load_adversarial_examples(
            model_name, attack_name, epsilon, load_directory="saved_tensors/"
        )
        attack_identifier = torch.zeros(len(adv_examples)) + i
        adversarial_examples.append(adv_examples)
        true_labels.append(original_labels)
        attack_names.append(attack_identifier)

    # Combine all examples
    adversarial_examples = torch.cat(adversarial_examples).to("cpu")
    true_labels = torch.cat(true_labels).to("cpu")
    attack_names = torch.cat(attack_names).to("cpu")

    # Extract the original data
    examples = train_dataset.tensors[0]
    labels = train_dataset.tensors[1]

    # sample 10% of points
    examples, labels = sample(examples, labels, size=0.2)
    adversarial_examples, true_labels, attack_names = sample(adversarial_examples, true_labels, attack_names, size=0.2)

    # combine original data with the adversarial examples
    all_examples = torch.cat((examples, adversarial_examples)).numpy()
    is_adv = torch.cat((torch.zeros(len(examples)), torch.ones(len(adversarial_examples)))).numpy()
    attack_names = torch.cat((torch.zeros(len(labels)), attack_names + 1))  # 0 corresponds to original point
    labels = torch.cat((labels, true_labels))

    # Predict on all examples
    predictions = model(torch.tensor(all_examples).to(device)).argmax(dim=-1).to("cpu")

    # get the projection
    umap_reducer = umap.UMAP(n_components=2, n_jobs=-1)
    umap_projection = umap_reducer.fit_transform(all_examples)

    projections.append(umap_projection)
    adversarial_indicators.append(is_adv)
    true_class.append(labels)
    predicted_labels.append(predictions)
    attack_types.append(attack_names)


# Plots of projections coloured by adversarial or normal
fig, axes = plt.subplots(3, 5, figsize=(20, 12))
for ax, projection, indicator, epsilon in zip(axes[0], projections, adversarial_indicators, epsilons):

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


# define colour map
# colour_map = {0: "blue", 1: "red", 2: "green", 3: "purple"}
colour_map = {
    0: "#B79F00",
    1: "#F8766D",
    2: "#00BA38",
    3: "#F564E3",
    4: "#00BFC4",
    5: "#619CFF",
}

colours = [pd.Series(attack_type.int()).apply(lambda x: colour_map[x]).to_numpy() for attack_type in attack_types]

# Plots of projections coloured by attack_type
for ax, projection, attack_type, epsilon in zip(axes[2], projections, colours, epsilons):

    sc = ax.scatter(
        projection[:, 0],
        projection[:, 1],
        c=attack_type,
        alpha=0.01,
    )
    ax.set_title(f"Epsilon={epsilon}")
legend_handles = [mpatches.Patch(color=colour_map[i], label=attack_index_to_name[i]) for i in range(len(attacks) + 1)]
# legend_handles = [
#     mpatches.Patch(color=colour_map[0], label="Original"),
#     mpatches.Patch(color=colour_map[1], label="BIM"),
#     mpatches.Patch(color=colour_map[2], label="FGSM"),
#     mpatches.Patch(color=colour_map[3], label="PGD"),
# ]
fig.legend(handles=legend_handles, loc="lower right")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
