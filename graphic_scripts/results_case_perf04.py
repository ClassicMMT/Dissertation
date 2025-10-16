"""
This script calculates three plots:
    1. The proportion of points selected as uncertain by alpha
    2. Model accuracy of low and highly uncertain points
    3. The false positive and false negatives by alpha

on the CIFAR-10 dataset with the pre-trained ResNet56 model.
"""

import torch
import pandas as pd
from src.utils import set_all_seeds, load_cifar_model
from src.datasets import load_cifar
from src.method import identify_uncertain_points_all
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


random_state = 123
g = set_all_seeds(random_state)
device = torch.device("mps")

# load model and data
(calib_loader, test_loader), _ = load_cifar(batch_size=128, generator=g)
model = load_cifar_model(model_name="cifar10_resnet56", device=device)

rejection_proportion_results = {
    "entropy": [],
    "information_content": [],
    "probability_gap": [],
    "any": [],
    "all": [],
}

low_uncertainty_accuracy_results = {key: [] for key in rejection_proportion_results}
high_uncertainty_accuracy_results = {key: [] for key in rejection_proportion_results}
false_positive_results = {key: [] for key in rejection_proportion_results}
false_negative_results = {key: [] for key in rejection_proportion_results}


# get uncertain points
alphas = torch.linspace(0.01, 0.25, 25)
for alpha in alphas:
    print(f"Alpha: {alpha:.4f}")
    uncertain_results = identify_uncertain_points_all(model, calib_loader, test_loader, alpha=alpha, device=device)

    # unpack results
    entropy_reject = uncertain_results["highly_uncertain_entropy"]
    info_content_reject = uncertain_results["highly_uncertain_information_content"]
    probability_gap_reject = uncertain_results["highly_uncertain_probability_gap"]
    any_reject = uncertain_results["highly_uncertain_any"]
    all_reject = uncertain_results["highly_uncertain_all"]
    is_correct = uncertain_results["is_correct_predictions"].cpu()

    # calculate proportions rejected by each method
    entropy_prop = entropy_reject.float().mean().item()
    info_prop = info_content_reject.float().mean().item()
    gap_prop = probability_gap_reject.float().mean().item()
    any_prop = any_reject.float().mean().item()
    all_prop = all_reject.float().mean().item()

    # calculate accuracy by low and high certainty points
    entropy_low = is_correct[~entropy_reject].float().mean().item()
    entropy_high = is_correct[entropy_reject].float().mean().item()
    info_low = is_correct[~info_content_reject].float().mean().item()
    info_high = is_correct[info_content_reject].float().mean().item()
    gap_low = is_correct[~probability_gap_reject].float().mean().item()
    gap_high = is_correct[probability_gap_reject].float().mean().item()
    any_low = is_correct[~any_reject].float().mean().item()
    any_high = is_correct[any_reject].float().mean().item()
    all_low = is_correct[~all_reject].float().mean().item()
    all_high = is_correct[all_reject].float().mean().item()

    # calculate false positives
    # entropy_fp = (is_correct & entropy_reject).float().mean().item()
    # info_fp = (is_correct & info_content_reject).float().mean().item()
    # gap_fp = (is_correct & probability_gap_reject).float().mean().item()
    # any_fp = (is_correct & any_reject).float().mean().item()
    # all_fp = (is_correct & all_reject).float().mean().item()
    #
    # # calculate false negatives
    # entropy_fn = (~is_correct & ~entropy_reject).float().mean().item()
    # info_fn = (~is_correct & ~info_content_reject).float().mean().item()
    # gap_fn = (~is_correct & ~probability_gap_reject).float().mean().item()
    # any_fn = (~is_correct & ~any_reject).float().mean().item()
    # all_fn = (~is_correct & ~all_reject).float().mean().item()

    # calculate false positives
    entropy_fp = (is_correct & entropy_reject).float().sum() / is_correct.sum()
    info_fp = (is_correct & info_content_reject).float().sum() / is_correct.sum()
    gap_fp = (is_correct & probability_gap_reject).float().sum() / is_correct.sum()
    any_fp = (is_correct & any_reject).float().sum() / is_correct.sum()
    all_fp = (is_correct & all_reject).float().sum() / is_correct.sum()
    # # calculate false negatives
    entropy_fn = (~is_correct & ~entropy_reject).float().sum() / (~is_correct).sum()
    info_fn = (~is_correct & ~info_content_reject).float().sum() / (~is_correct).sum()
    gap_fn = (~is_correct & ~probability_gap_reject).float().sum() / (~is_correct).sum()
    any_fn = (~is_correct & ~any_reject).float().sum() / (~is_correct).sum()
    all_fn = (~is_correct & ~all_reject).float().sum() / (~is_correct).sum()

    # save results
    rejection_proportion_results["entropy"].append(entropy_prop)
    rejection_proportion_results["information_content"].append(info_prop)
    rejection_proportion_results["probability_gap"].append(gap_prop)
    rejection_proportion_results["any"].append(any_prop)
    rejection_proportion_results["all"].append(all_prop)

    low_uncertainty_accuracy_results["entropy"].append(entropy_low)
    low_uncertainty_accuracy_results["information_content"].append(info_low)
    low_uncertainty_accuracy_results["probability_gap"].append(gap_low)
    low_uncertainty_accuracy_results["any"].append(any_low)
    low_uncertainty_accuracy_results["all"].append(all_low)

    high_uncertainty_accuracy_results["entropy"].append(entropy_high)
    high_uncertainty_accuracy_results["information_content"].append(info_high)
    high_uncertainty_accuracy_results["probability_gap"].append(gap_high)
    high_uncertainty_accuracy_results["any"].append(any_high)
    high_uncertainty_accuracy_results["all"].append(all_high)

    false_positive_results["entropy"].append(entropy_fp)
    false_positive_results["information_content"].append(info_fp)
    false_positive_results["probability_gap"].append(gap_fp)
    false_positive_results["any"].append(any_fp)
    false_positive_results["all"].append(all_fp)

    false_negative_results["entropy"].append(entropy_fn)
    false_negative_results["information_content"].append(info_fn)
    false_negative_results["probability_gap"].append(gap_fn)
    false_negative_results["any"].append(any_fn)
    false_negative_results["all"].append(all_fn)

if True:
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    colors = {
        "entropy": "C0",
        "information_content": "C1",
        "probability_gap": "C2",
        "any": "C3",
        "all": "C4",
    }

    for key, proportions in rejection_proportion_results.items():
        label = key.replace("_", " ").replace("any", "At least One").replace("all", "All Three").title()
        # if label == "Any":
        #     label = "At Least One"
        # elif label == "All":
        #     label = "A"
        axs[0].plot(alphas, proportions, label=label, color=colors[key])

    for key, accuracies in low_uncertainty_accuracy_results.items():
        label = "Low " + key.replace("_", " ").capitalize()
        axs[1].plot(alphas, accuracies, label=label, linestyle="dashed", color=colors[key])

    for key, accuracies in high_uncertainty_accuracy_results.items():
        label = "High " + key.replace("_", " ").capitalize()
        axs[1].plot(alphas, accuracies, label=label, color=colors[key])

    for key, fp in false_positive_results.items():
        label = "FP " + key.replace("_", " ").capitalize()
        axs[2].plot(alphas, fp, label=label, linestyle="dashed", color=colors[key])

    for key, fn in false_negative_results.items():
        label = "FN " + key.replace("_", " ").capitalize()
        axs[2].plot(alphas, fn, label=label, color=colors[key])

    axs[0].set_xlabel("Alpha")
    axs[0].set_ylabel("Proportion")
    axs[1].set_xlabel("Alpha")
    axs[1].set_ylabel("Accuracy")
    axs[2].set_xlabel("Alpha")
    axs[2].set_ylabel("Proportion")

    axs[0].legend(title="Methods", loc="best")
    axs[1].legend(
        handles=[
            mlines.Line2D([], [], color="black", linestyle="solid", label="High"),
            mlines.Line2D([], [], color="black", linestyle="dashed", label="Low"),
        ],
        title="Uncertainty",
        loc="lower right",
    )
    axs[2].legend(
        handles=[
            mlines.Line2D([], [], color="black", linestyle="solid", label="False Negatives"),
            mlines.Line2D([], [], color="black", linestyle="dashed", label="False Positives"),
        ],
        title="Error",
        loc="upper right",
    )

    axs[0].set_title("Proportion Uncertain")
    axs[1].set_title("Accuracy by Uncertainty Class")
    axs[2].set_title("FP and FN Proportions")

    # axs[1].set_ylim(0, 1)
    # axs[2].set_ylim(0, 0.3)

    axs[0].grid(alpha=0.3)
    axs[1].grid(alpha=0.3)
    axs[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
