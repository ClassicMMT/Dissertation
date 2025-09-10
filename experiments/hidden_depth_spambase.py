"""
The idea behind this experiment is to make a neural network and increase the hidden layer's size.
The idea was to see if the model would start "overfitting" at some point, and so we could see the
normalised MSE start to increase across classes, although this could not be confirmed.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import foolbox as fb
from scipy.stats import wasserstein_distance_nd
from src.attacks import attack_model
from src.datasets import load_spambase
from src.models import GenericNet
from src.utils import normalised_mse, set_all_seeds, train_model

random_state = 123
set_all_seeds(random_state)
device = torch.device("mps")

######### DATA #########

(train_loader, test_loader), (train_dataset, test_dataset) = load_spambase()
train_data = train_dataset.tensors[0]
variances = train_data.var(dim=0, unbiased=False)

######### MODEL #########

n_epochs = 50
criterion = nn.CrossEntropyLoss()
hidden_layer_sizes = np.arange(5, 201)

######### ATTACKS #########

attack = fb.attacks.basic_iterative_method.LinfAdamBasicIterativeAttack()
epsilon = 0.01

results = {
    "hidden_size": [],
    "epochs": [],
    "attack": [],
    "epsilon": [],
    "distance_class0": [],
    "distance_class1": [],
    # "wasserstein_class0": [],
    # "wasserstein_class1": [],
}

for hidden_size in hidden_layer_sizes:

    # model related stuff
    layers = [57, hidden_size, 2]
    model = GenericNet(layers=layers, activation="sigmoid")
    optimizer = torch.optim.Adam(model.parameters())

    print(f"Training model with hidden size: {hidden_size}")
    model, train_accuracy = train_model(
        model,
        train_loader,
        criterion,
        optimizer,
        n_epochs,
        verbose=False,
        return_accuracy=True,
        device=device,
    )

    adv_examples, labels, is_correct_and_adv = attack_model(
        model,
        loader=train_loader,
        attack=attack,
        device=device,
        epsilons=epsilon,
        return_correct_and_adversarial=True,
        verbose=False,
    )

    adv_examples, labels, is_correct_and_adv = (
        adv_examples.to("cpu"),
        labels.to("cpu"),
        is_correct_and_adv.to("cpu"),
    )
    # Get only those examples which are correct and adversarial
    train_data_subset = train_data[is_correct_and_adv]
    adv_examples_subset = adv_examples[is_correct_and_adv]
    labels = labels[is_correct_and_adv]

    # compute normalised MSE across classes
    distance_class0 = normalised_mse(
        train_data_subset[labels == 0],
        adv_examples_subset[labels == 0],
        variances=variances,
    )
    distance_class1 = normalised_mse(
        train_data_subset[labels == 1],
        adv_examples_subset[labels == 1],
        variances=variances,
    )
    # print(f"Computing Wasserstein class 0: n_rows: {len(train_data_subset[labels==0])}")
    # wasserstein_class0 = wasserstein_distance_nd(
    #     train_data_subset[labels == 0], adv_examples_subset[labels == 0]
    # )
    # print(f"Computing Wasserstein class 1: n_rows: {len(train_data_subset[labels==1])}")
    # wasserstein_class1 = wasserstein_distance_nd(
    #     train_data_subset[labels == 1], adv_examples_subset[labels == 1]
    # )

    results["hidden_size"].append(hidden_size)
    results["epochs"].append(n_epochs)
    results["attack"].append("BIM_linf")
    results["epsilon"].append(epsilon)
    results["distance_class0"].append(distance_class0)
    results["distance_class1"].append(distance_class1)
    # results["wasserstein_class0"].append(wasserstein_class0)
    # results["wasserstein_class1"].append(wasserstein_class1)

data = pd.DataFrame(results)
data.to_csv("saved_results/hidden_depth_spambase.csv", index=False)
