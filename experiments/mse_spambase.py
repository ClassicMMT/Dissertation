"""
This, and similar files, compute the MSE on the dataset.

The idea here was to get adversarial examples and check the
normalised MSE distance between the original points and adversarial examples.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance_nd
import foolbox as fb
import torch
from src.attacks import attack_model, load_all_adversarial_examples
from src.datasets import load_spambase
from src.models import SpamBaseNet
from src.utils import load_model, normalised_mse, set_all_seeds

device = torch.device("mps")
random_state = 123
set_all_seeds(random_state)
(train_loader, test_loader), (train_dataset, test_dataset) = load_spambase()
train_data = train_dataset.tensors[0]
variances = train_data.var(dim=0, unbiased=False)

models = ["spambase_60", "spambase_100", "spambase_200", "spambase_300", "spambase_400"]
attacks = {
    "FGSM": fb.attacks.fast_gradient_method.LinfFastGradientAttack(),
    "DF_L2": fb.attacks.deepfool.L2DeepFoolAttack(),
    "DF_Linf": fb.attacks.deepfool.LinfDeepFoolAttack(),
    # "CW_L2": fb.attacks.carlini_wagner.L2CarliniWagnerAttack(),
    "BIM_Linf": fb.attacks.basic_iterative_method.LinfAdamBasicIterativeAttack(),
}
epsilons = [0.01, 0.05, 0.1]

results = {
    "model": [],
    "attack": [],
    "epsilon": [],
    "distance_class_0": [],
    "distance_class_1": [],
}


for model_name in models:
    for attack_name in attacks:
        for epsilon in epsilons:
            print(model_name, attack_name, epsilon)

            attack = attacks[attack_name]

            model = load_model(SpamBaseNet(), model_name)

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

            results["model"].append(model_name)
            results["attack"].append(attack_name)
            results["epsilon"].append(epsilon)
            results["distance_class_0"].append(distance_class0)
            results["distance_class_1"].append(distance_class1)


data = pd.DataFrame(results)
data.to_csv("saved_results/spambase_mse_by_attack_and_epsilon.csv", index=False)
