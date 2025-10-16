"""
This script calculates three plots:
    1. The proportion of points identified as adversarial

on the CIFAR-10 dataset with the pre-trained ResNet56 model.
"""

import torch
from src.utils import calculate_entropy, set_all_seeds, load_cifar_model, drop_duplicates
from src.datasets import load_cifar
from src.method import compute_threshold_all
import matplotlib.pyplot as plt

# import matplotlib.lines as mlines
import foolbox as fb
import pandas as pd


random_state = 123
g = set_all_seeds(random_state)
device = torch.device("mps")

# load model and data
(calib_loader, test_loader), (calib_dataset, test_dataset) = load_cifar(batch_size=128, generator=g)
model = load_cifar_model(model_name="cifar10_resnet56", device=device)


adversarial_identification_results = {
    "entropy": [],
    "information_content": [],
    "probability_gap": [],
    "any": [],
    "all": [],
    "attack": [],
    "alpha": [],
}

attacks = {
    "fgsm": fb.attacks.fast_gradient_method.L2FastGradientAttack(),
    # "fgsm": fb.attacks.fast_gradient_method.L1FastGradientAttack(),
    "bim": fb.attacks.basic_iterative_method.L2AdamBasicIterativeAttack(),
    "deepfool": fb.attacks.deepfool.L2DeepFoolAttack(),
}

# calculate bounds
global_min = 10
global_max = -10
for features, _ in calib_loader:
    min_ = features.min().item()
    if min_ < global_min:
        global_min = min_
    max_ = features.max().item()
    if max_ > global_max:
        global_max = max_

# Foolbox model
model.eval()
fmodel = fb.PyTorchModel(model, bounds=(global_min, global_max), device=device)

# Get results
# alphas = torch.linspace(0.01, 0.25, 25)
alphas = torch.linspace(0.01, 0.1, 10)
for alpha in alphas:
    print(f"Alpha: {alpha:.4f}")

    # get thresholds
    entropy_threshold, info_threshold, gap_threshold = compute_threshold_all(model, calib_loader, alpha, device=device)

    # go through all attacks
    for attack_name, attack in attacks.items():

        entropy_n_identified = 0
        info_n_identified = 0
        gap_n_identified = 0
        any_n_identified = 0
        all_n_identified = 0
        n_total = 0

        # go through the test set
        for i, (features, labels) in enumerate(test_loader):
            features, labels = features.to(device), labels.to(device)
            is_correct_prediction = model(features).argmax(dim=-1) == labels

            print(f"Alpha: {alpha:.4f}, batch:{i+1}/{len(test_loader)}, attack:{attack_name}")

            # get adversarial examples
            _, clipped, is_adv = attack(fmodel, features, labels, epsilons=0.03)
            is_correct_and_adversarial = is_correct_prediction & is_adv
            adversarial_examples = drop_duplicates(clipped[is_correct_and_adversarial])

            # no examples found
            if len(adversarial_examples) == 0:
                continue

            # compute identification of these examples
            with torch.no_grad():
                logits = model(adversarial_examples)

                entropy = calculate_entropy(logits, apply_softmax=True)
                info = calculate_entropy(logits, apply_softmax=True)
                gap = calculate_entropy(logits, apply_softmax=True)

                # compute whether examples are highly-uncertain
                entropy_mask = entropy >= entropy_threshold
                info_mask = info >= info_threshold
                gap_mask = gap <= gap_threshold
                any_mask = entropy_mask | info_mask | gap_mask
                all_mask = entropy_mask & info_mask & gap_mask

                entropy_n_identified += entropy_mask.sum().item()
                info_n_identified += info_mask.sum().item()
                gap_n_identified += gap_mask.sum().item()
                any_n_identified += any_mask.sum().item()
                all_n_identified += all_mask.sum().item()
                n_total += len(adversarial_examples)

        # compute identification rates
        entropy_rate = entropy_n_identified / n_total
        info_rate = info_n_identified / n_total
        gap_rate = gap_n_identified / n_total
        any_rate = any_n_identified / n_total
        all_rate = all_n_identified / n_total

        # save results
        adversarial_identification_results["entropy"].append(entropy_rate)
        adversarial_identification_results["information_content"].append(info_rate)
        adversarial_identification_results["probability_gap"].append(gap_rate)
        adversarial_identification_results["any"].append(any_rate)
        adversarial_identification_results["all"].append(all_rate)
        adversarial_identification_results["attack"].append(attack_name)
        adversarial_identification_results["alpha"].append(alpha.item())


# data = pd.read_csv("saved_results/adversarial_case_cifar_results.csv")
data = pd.DataFrame(adversarial_identification_results)
# data.to_csv("saved_results/adversarial_case_cifar_results.csv", index=False)

# Plots
if True:
    fig, axs = plt.subplots(1, 5, figsize=(25, 5))

    for i, method in enumerate(["entropy", "information_content", "probability_gap", "any", "all"]):

        for attack_name, _ in attacks.items():
            if attack_name in ["fgsm", "bim"]:
                label = attack_name.upper()
            elif attack_name == "deepfool":
                label = "DeepFool"
            subset = data[data["attack"] == attack_name]
            axs[i].plot(subset.alpha, subset[method], label=label)

        axs[i].set_ylim(bottom=0, top=1.05)
        if method == "any":
            method = "At Least One"
        elif method == "all":
            method = "All Three"
        axs[i].set_title(method.replace("_", " ").title())
        axs[i].set_xlabel("Alpha")
        axs[i].legend(loc="lower right")
        axs[i].grid(alpha=0.3)

    axs[0].set_ylabel("Proportion")
    # fig.suptitle("Proportion of Adversarial Points Identified")
    plt.tight_layout()
    plt.show()
