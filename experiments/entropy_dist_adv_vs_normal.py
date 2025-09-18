"""
This file calculates entropy figures for spambase. The points are split into:
    * Adversarial points - points which were originally correct but are now adversarial
    * correct points - original points which are classified correctly
    * incorrect points - original points which are not classified correctly
    * non-adversarial points - points which the attack modified, but did not manage to make adversarial
    * correct_test_entropies - points which are in the test set and are correct
    * incorrect_test_entropies - points which are in the training set and are incorrect

The distributions of the entropies of these group are then plotted.

IDEAS:
    * worthwhile to try this across multiple attacks
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import foolbox as fb
from src.attacks import attack_model
from src.datasets import load_spambase
from src.models import SpamBaseNet
from src.utils import load_model, set_all_seeds, calculate_entropy

random_state = 123
set_all_seeds(random_state)
device = torch.device("mps")

# Data
(train_loader, test_loader), (train_dataset, test_dataset) = load_spambase(
    induce_test_covariate_shift=True
)
model = load_model(SpamBaseNet(), "spambase").to(device).eval()

adversarial_examples, original_labels, is_correct_and_adversarial = attack_model(
    model,
    # fb.attacks.FGSM(),
    fb.attacks.basic_iterative_method.LinfAdamBasicIterativeAttack(),
    # fb.attacks.deepfool.LinfDeepFoolAttack(),
    train_loader,
    epsilons=0.01,
    verbose=False,
    device=device,
    return_correct_and_adversarial=True,
)


with torch.no_grad():
    model.eval()

    correct_entropies = []
    incorrect_entropies = []
    adversarial_entropies = []
    non_adversarial_entropies = []
    correct_test_entropies = []
    incorrect_test_entropies = []

    # train entropies
    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)

        # get logits and predictions from model
        logits = model(features)
        is_correct = (logits.argmax(dim=-1) == labels).cpu()

        # calculate entropy for each point
        entropies = calculate_entropy(logits)

        # separate entropies
        correct_entropy = entropies[is_correct]
        incorrect_entropy = entropies[~is_correct]

        correct_entropies.append(correct_entropy)
        incorrect_entropies.append(incorrect_entropy)

    # test entropies
    for features, labels in test_loader:
        features = features.to(device)
        labels = labels.to(device)

        # get logits and predictions from model
        logits = model(features)
        is_correct = (logits.argmax(dim=-1) == labels).cpu()

        # calculate entropy for each point
        entropies = calculate_entropy(logits)

        # separate entropies
        correct_entropy = entropies[is_correct]
        incorrect_entropy = entropies[~is_correct]

        correct_test_entropies.append(correct_entropy)
        incorrect_test_entropies.append(incorrect_entropy)

    # combine
    correct_entropies = torch.cat(correct_entropies)
    incorrect_entropies = torch.cat(incorrect_entropies)
    correct_test_entropies = torch.cat(correct_test_entropies)
    incorrect_test_entropies = torch.cat(incorrect_test_entropies)

    # adversarial entropies
    adv_examples = adversarial_examples[is_correct_and_adversarial]
    logits = model(adv_examples)
    entropies = calculate_entropy(logits)
    adversarial_entropies = entropies

    # non_adversarial_entropies - these points are not adversarial, but are also not their original points anymore
    # these points should all be correctly classified (since they're not adversarial)
    other_examples = adversarial_examples[~is_correct_and_adversarial]
    logits = model(other_examples)
    entropies = calculate_entropy(logits)
    non_adversarial_entropies = entropies


results = {
    "entropy_type": np.concat(
        (
            np.repeat("correct train", len(correct_entropies)),
            np.repeat("incorrect train", len(incorrect_entropies)),
            np.repeat("adversarial", len(adversarial_entropies)),
            np.repeat("non-adversarial", len(non_adversarial_entropies)),
            np.repeat("correct test", len(correct_test_entropies)),
            np.repeat("incorrect test", len(incorrect_test_entropies)),
        )
    ),
    "entropy": np.concat(
        (
            correct_entropies.numpy(),
            incorrect_entropies.numpy(),
            adversarial_entropies.numpy(),
            non_adversarial_entropies.numpy(),
            correct_test_entropies.numpy(),
            incorrect_test_entropies.numpy(),
        )
    ),
}

import pandas as pd

pd_data = pd.DataFrame(results)
pd_data.to_csv("saved_results/entropy_distributions.csv", index=False)
