"""
This script varies alpha and computes the following metric:
    * Accuracy on uncertain points
    * Accuracy on certain points
    * Proportion of points incorrectly labelled uncertain (false positives)
        - points which would have been classified correctly
    * Proportion of points incorrectly labelled certain (false negatives)
        - points which would have been classified incorrectly
    * Adversarial "catchment" - proportion of attacks identified as adversarial
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import foolbox as fb
from src.models import GenericNet, SpamBaseNet
from src.attacks import attack_model
from src.utils import calculate_entropies_from_loader, calculate_entropy, evaluate_model, set_all_seeds, train_model
from src.datasets import create_loaders, load_spambase, make_chessboard, scale_datasets


# setting up
random_state = 123
g = set_all_seeds(random_state)
batch_size = 128
device = torch.device("mps")

# make data
x_train, x_test, y_train, y_test = load_spambase(return_raw=True, random_state=random_state)
x_train, x_calib, y_train, y_calib = train_test_split(
    x_train, y_train, test_size=0.1, random_state=random_state, stratify=y_train
)
x_train, x_calib, x_test = scale_datasets(x_train, x_calib, x_test)
train_loader, train_dataset = create_loaders(x_train, y_train, batch_size=batch_size, generator=g)
calib_loader, calib_dataset = create_loaders(x_calib, y_calib, batch_size=batch_size, generator=g)
test_loader, test_dataset = create_loaders(x_test, y_test, batch_size=batch_size, generator=g)


# train model
model = SpamBaseNet().train().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
n_epochs = 50
model = train_model(model, train_loader, criterion, optimizer, n_epochs=n_epochs, device=device, verbose=False)

# set alphas
alphas = np.linspace(0.05, 0.25, num=9)

# pre-compute the entropies
calib_entropies, _ = calculate_entropies_from_loader(model, calib_loader, device=device)
test_entropies, is_correct = calculate_entropies_from_loader(model, test_loader, device=device)

# pre-compute adversarial examples
attack = fb.attacks.deepfool.L2DeepFoolAttack()
adversarial_examples, original_labels = attack_model(
    model, attack, train_loader, device=device, epsilons=0.01, verbose=False
)
adversarial_examples2, original_labels2 = attack_model(
    model, attack, train_loader, device=device, epsilons=0.05, verbose=False
)


# get the entropies of the adversarial examples
with torch.no_grad():
    model.eval()
    logits = model(adversarial_examples)
    logits2 = model(adversarial_examples2)
    adversarial_entropies = calculate_entropy(logits)
    adversarial_entropies2 = calculate_entropy(logits2)

# calculate true test accuracy
true_test_accuracy = evaluate_model(model, test_loader, device=device)

results = {
    "alpha": alphas,
    "q_hat": [],
    "certain_accuracy": [],
    "uncertain_accuracy": [],
    # a bit crude, I know
    "proportion_adv_caught_0.01": [],
    "proportion_adv_caught_0.05": [],
    "false_positive_proportion": [],
    "false_negative_proportion": [],
    "true_test_accuracy": [],
}

for alpha in alphas:
    # calculate q_hat
    q_hat = torch.quantile(calib_entropies, 1 - alpha, dim=-1, interpolation="higher")

    # calculate certain and uncertain points
    is_uncertain = test_entropies >= q_hat
    uncertain_is_correct = is_correct[is_uncertain]
    certain_is_correct = is_correct[~is_uncertain]

    # calculate certain/uncertain accuracies
    certain_accuracy = certain_is_correct.float().mean()
    uncertain_accuracy = uncertain_is_correct.float().mean()

    # adversarial
    proportion_of_adversarial_examples_caught = (adversarial_entropies >= q_hat).sum() / len(adversarial_entropies)
    proportion_of_adversarial_examples_caught2 = (adversarial_entropies2 >= q_hat).sum() / len(adversarial_entropies2)

    # compute false positives and false negatives
    false_positives = is_correct & is_uncertain
    false_negatives = (~is_correct) & (~is_uncertain)

    # compute false positive and false negative proportions
    false_positive_proportion = false_positives.float().mean()
    false_negative_proportion = false_negatives.float().mean()

    # save results
    results["q_hat"].append(q_hat.item())
    results["certain_accuracy"].append(certain_accuracy.item())
    results["uncertain_accuracy"].append(uncertain_accuracy.item())
    results["false_positive_proportion"].append(false_positive_proportion.item())
    results["false_negative_proportion"].append(false_negative_proportion.item())
    results["true_test_accuracy"].append(true_test_accuracy)
    results["proportion_adv_caught_0.01"].append(proportion_of_adversarial_examples_caught.item())
    results["proportion_adv_caught_0.05"].append(proportion_of_adversarial_examples_caught2.item())


pd.DataFrame(results).to_csv("saved_results/quantiles_alpha_spambase.csv", index=False)
