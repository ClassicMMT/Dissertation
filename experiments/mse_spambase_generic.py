"""
This, and similar files, compute the MSE on the dataset.

The idea here was to get adversarial examples and check the
normalised MSE distance between the original points and adversarial examples.


Note that: "spambase_generic_<n_epochs>.pt"
with n_epochs in [25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

are all trained with the architecture:

layers = [57, 256, 128, 64, 32, 2]
activation = "tanh"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance_nd
import foolbox as fb
import torch
from src.attacks import attack_model, load_all_adversarial_examples
from src.datasets import load_spambase
from src.models import GenericNet
from src.utils import load_model, normalised_mse, set_all_seeds

# Initialisation
device = torch.device("mps")
random_state = 123
set_all_seeds(random_state)

# Data
(train_loader, test_loader), (train_dataset, test_dataset) = load_spambase()
train_data = train_dataset.tensors[0]
variances = train_data.var(dim=0, unbiased=False)

########### MODELS ###########

epochs_trained = [25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
models = [f"spambase_generic_{epoch}" for epoch in epochs_trained]

# NOTE THE ARCHITECTURE
layers = [57, 256, 128, 64, 32, 2]
activation = "tanh"

########### ATTACKS ###########

attacks = {
    "FGSM": fb.attacks.fast_gradient_method.LinfFastGradientAttack(),
    "DF_L2": fb.attacks.deepfool.L2DeepFoolAttack(),
    "DF_Linf": fb.attacks.deepfool.LinfDeepFoolAttack(),
    # "CW_L2": fb.attacks.carlini_wagner.L2CarliniWagnerAttack(),
    "BIM_Linf": fb.attacks.basic_iterative_method.LinfAdamBasicIterativeAttack(),
}
epsilons = [0.01, 0.05, 0.1]

########### Results ###########

results = {
    "model": [],
    "attack": [],
    "epsilon": [],
    "distance_class_0": [],
    "distance_class_1": [],
    "wasserstein_class0": [],
    "wasserstein_class1": [],
}

########### LOOP ###########

for model_name in models:
    for attack_name in attacks:
        for epsilon in epsilons:
            print(model_name, attack_name, epsilon)

            attack = attacks[attack_name]

            model = load_model(
                GenericNet(layers=layers, activation=activation), model_name
            )

            adv_examples, labels, is_correct_and_adv = attack_model(
                model,
                attack,
                train_loader,
                device=device,
                verbose=False,
                return_correct_and_adversarial=True,
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
            print(
                f"Computing Wasserstein class 0: n_rows: {len(train_data_subset[labels==0])}"
            )
            wasserstein_class0 = wasserstein_distance_nd(
                train_data_subset[labels == 0], adv_examples_subset[labels == 0]
            )
            print(
                f"Computing Wasserstein class 1: n_rows: {len(train_data_subset[labels==1])}"
            )
            wasserstein_class1 = wasserstein_distance_nd(
                train_data_subset[labels == 1], adv_examples_subset[labels == 1]
            )

            results["model"].append(model_name)
            results["attack"].append(attack_name)
            results["epsilon"].append(epsilon)
            results["distance_class_0"].append(distance_class0)
            results["distance_class_1"].append(distance_class1)
            results["wasserstein_class0"].append(wasserstein_class0)
            results["wasserstein_class1"].append(wasserstein_class1)


# RUN THE SAME EXPERIMENT AS THIS BUT WITH INCREASING THE NUMBER OF NEURONS IN THE HIDDEN LAYER (keep epochs same)

data = pd.DataFrame(results)
data.to_csv("saved_results/spambase_generic_mse_by_attack_and_epsilon.csv", index=False)
