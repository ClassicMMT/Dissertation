"""
This script experiments with the YOLO model and ImageNet dataset.

This file then should be equivalent to just the ImageNet with ResNet-50.

This model gets ~66% test accuracy on the test loader.
"""

import torch
from src.datasets import load_imagenet
from src.utils import (
    calculate_entropy,
    calculate_information_content,
    calculate_probability_gap,
    load_yolo,
    set_all_seeds,
    yolo_to_probs,
    evaluate_yolo,
)

random_state = 123
g = set_all_seeds(random_state)
device = torch.device("mps")
batch_size = 128

# Load model and data
model = load_yolo(device="mps")
(calib_loader, test_loader), _ = load_imagenet(batch_size=batch_size, generator=g)

# Model evaluation
evaluate_yolo(model, test_loader, device)

# Calculate the per class uncertainties on the calibration set
with torch.no_grad():
    model.eval()

    entropies = []
    information_contents = []
    probability_gaps = []

    for i, (features, labels) in enumerate(calib_loader):
        print(f"Calibration Batch: {i+1}/{len(calib_loader)}")
        features = features.to(device)
        labels = labels.to(device)

        output = model(features, verbose=False)
        probs = yolo_to_probs(output)  # (batch_size, n_classes)

        entropy = calculate_entropy(probs, apply_softmax=False)
        info_content = calculate_information_content(probs, apply_softmax=False)
        gaps = calculate_probability_gap(probs, apply_softmax=False)

        entropies.append(entropy)
        information_contents.append(info_content)
        probability_gaps.append(gaps)

    entropies = torch.cat(entropies)
    information_contents = torch.cat(information_contents)
    probability_gaps = torch.cat(probability_gaps)

    # calculate thresholds
    alpha = 0.1
    entropy_threshold = torch.quantile(entropies, 1 - alpha, dim=0, interpolation="higher")
    information_threshold = torch.quantile(information_contents, 1 - alpha, dim=0, interpolation="higher")
    gap_threshold = torch.quantile(probability_gaps, alpha, dim=0, interpolation="lower")


# Compute the same on the test set
with torch.no_grad():
    model.eval()

    entropies = []
    information_contents = []
    probability_gaps = []
    is_correct = []

    for i, (features, labels) in enumerate(test_loader):
        print(f"Test Batch: {i+1}/{len(test_loader)}")
        features = features.to(device)
        labels = labels.to(device)

        output = model(features, verbose=False)
        probs = yolo_to_probs(output)  # (batch_size, n_classes)
        preds = probs.argmax(dim=-1)
        correct = preds == labels

        entropy = calculate_entropy(probs, apply_softmax=False)
        info_content = calculate_information_content(probs, apply_softmax=False)
        gaps = calculate_probability_gap(probs, apply_softmax=False)

        entropies.append(entropy)
        information_contents.append(info_content)
        probability_gaps.append(gaps)
        is_correct.append(correct)

    entropies = torch.cat(entropies)
    information_contents = torch.cat(information_contents)
    probability_gaps = torch.cat(probability_gaps)
    is_correct = torch.cat(is_correct)

    uncertain_entropy = entropies >= entropy_threshold
    uncertain_information = information_contents >= information_threshold
    uncertain_gaps = probability_gaps <= gap_threshold
    uncertain_either = uncertain_entropy | uncertain_information | uncertain_gaps
    uncertain_all = uncertain_entropy & uncertain_information & uncertain_gaps

# Agreement matrix
if True:
    agreement_matrix = torch.zeros((3, 3))
    agreement_matrix[1, 0] = (uncertain_entropy == uncertain_information).float().mean()
    agreement_matrix[2, 0] = (uncertain_entropy == uncertain_gaps).float().mean()
    agreement_matrix[2, 1] = (uncertain_information == uncertain_gaps).float().mean()

    print(agreement_matrix)

# Model performance
is_correct[uncertain_entropy].float().mean()
is_correct[~uncertain_entropy].float().mean()

is_correct[uncertain_information].float().mean()
is_correct[~uncertain_information].float().mean()

is_correct[uncertain_gaps].float().mean()
is_correct[~uncertain_gaps].float().mean()

is_correct[uncertain_either].float().mean()
is_correct[~uncertain_either].float().mean()

is_correct[uncertain_all].float().mean()
is_correct[~uncertain_all].float().mean()
