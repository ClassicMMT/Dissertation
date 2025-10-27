"""
The purpose of this script is to compare the different selective classification methods:
    * entropy
    * information content
    * probability gaps

on the CIFAR10 dataset.
"""

import torch
from src.datasets import load_cifar
from src.utils import (
    evaluate_model,
    load_cifar_model,
    set_all_seeds,
)
from src.method import identify_uncertain_points_all


random_state = 123
g = set_all_seeds(random_state)
batch_size = 128
device = torch.device("mps")

# make train, calibration, and test splits
(calib_loader, test_loader), _ = load_cifar(batch_size, generator=g)

# train model
model = load_cifar_model(model_name="cifar10_resnet56", device=device)

results = identify_uncertain_points_all(model, calib_loader, test_loader, alpha=0.1, device=device)
entropy_reject = results["highly_uncertain_entropy"]
info_content_reject = results["highly_uncertain_information_content"]
probability_gap_reject = results["highly_uncertain_probability_gap"]
either_reject = results["highly_uncertain_any"]
all_reject = results["highly_uncertain_all"]
is_correct = results["is_correct_predictions"]


certain_entropy = is_correct[~entropy_reject].float()
uncertain_entropy = is_correct[entropy_reject].float()

certain_info_content = is_correct[~info_content_reject].float()
uncertain_info_content = is_correct[info_content_reject].float()

certain_probability_gap = is_correct[~probability_gap_reject].float()
uncertain_probability_gap = is_correct[probability_gap_reject].float()

certain_either = is_correct[~either_reject].float()
uncertain_either = is_correct[either_reject].float()

certain_all = is_correct[~all_reject].float()
uncertain_all = is_correct[all_reject].float()

# print results
print(f"Model Test Accuracy: {evaluate_model(model, test_loader, device=device)*100:.2f}")
print(
    f"ENTROPY: Certain Acc: {certain_entropy.mean()*100:.2f}%, n: {len(certain_entropy)}, Uncertain Acc: {uncertain_entropy.mean()*100:.2f}%, n: {len(uncertain_entropy)}"
)
print(
    f"INFORMATION CONTENT: Certain Acc: {certain_info_content.mean()*100:.2f}%, n: {len(certain_info_content)}, Uncertain Acc: {uncertain_info_content.mean()*100:.2f}%, n: {len(uncertain_info_content)}"
)
print(
    f"PROBABILITY GAP: Certain Acc: {certain_probability_gap.mean()*100:.2f}%, n: {len(certain_probability_gap)}, Uncertain Acc: {uncertain_probability_gap.mean()*100:.2f}%, n: {len(uncertain_probability_gap)}"
)
print(
    f"EITHER: Certain Acc: {certain_either.mean()*100:.2f}%, n: {len(certain_either)}, Uncertain Acc: {uncertain_either.mean()*100:.2f}%, n: {len(uncertain_either)}"
)
print(
    f"ALL: Certain Acc: {certain_all.mean()*100:.2f}%, n: {len(certain_all)}, Uncertain Acc: {uncertain_all.mean()*100:.2f}%, n: {len(uncertain_all)}"
)
