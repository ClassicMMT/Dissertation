"""
This script calculates the agreeableness between the three methods on the ImageNet dataset.
"""

import torch
from src.utils import (
    calculate_entropy,
    calculate_information_content,
    calculate_probability_gap,
    load_resnet50,
    set_all_seeds,
)
from src.datasets import load_imagenet
import matplotlib.pyplot as plt

random_state = 123
g = set_all_seeds(random_state)
device = torch.device("mps")
batch_size = 128

(calib_loader, test_loader), _ = load_imagenet(batch_size=batch_size, generator=g)
model = load_resnet50(device=device)


alpha = 0.1

# compute thresholds
with torch.no_grad():
    model.eval()

    entropies = []
    information_contents = []
    probability_gaps = []

    for i, (features, labels) in enumerate(calib_loader):
        print(f"Batch: {i+1}/{len(calib_loader)}")
        features = features.to(device)
        lables = labels.to(device)
        logits = model(features)

        entropy = calculate_entropy(logits)
        info_content = calculate_information_content(logits)
        gaps = calculate_probability_gap(logits)

        entropies.append(entropy)
        information_contents.append(info_content)
        probability_gaps.append(gaps)

    entropies = torch.cat(entropies)
    information_contents = torch.cat(information_contents)
    probability_gaps = torch.cat(probability_gaps)

    entropy_threshold = torch.quantile(entropies, 1 - alpha, interpolation="higher")
    information_threshold = torch.quantile(information_contents, 1 - alpha, interpolation="higher")
    gap_threshold = torch.quantile(probability_gaps, alpha, interpolation="lower")

# compute test entropies
with torch.no_grad():
    model.eval()

    test_entropies = []
    test_info_content = []
    test_gaps = []

    for i, (features, _) in enumerate(test_loader):
        print(f"Batch: {i+1}/{len(test_loader)}")
        features = features.to(device)

        logits = model(features)

        entropy = calculate_entropy(logits)
        info_content = calculate_information_content(logits)
        gaps = calculate_probability_gap(logits)

        test_entropies.append(entropy)
        test_info_content.append(info_content)
        test_gaps.append(gaps)

    test_entropies = torch.cat(test_entropies)
    test_info_content = torch.cat(test_info_content)
    test_gaps = torch.cat(test_gaps)

    uncertain_entropy = test_entropies >= entropy_threshold
    uncertain_information = test_info_content >= information_threshold
    uncertain_gaps = test_gaps <= gap_threshold
    uncertain_either = uncertain_entropy | uncertain_information | uncertain_gaps
    uncertain_all = uncertain_entropy & uncertain_information & uncertain_gaps


# agreement matrix
if True:
    agreement_matrix = torch.zeros((3, 3))
    agreement_matrix[1, 0] = (uncertain_entropy == uncertain_information).float().mean()
    agreement_matrix[2, 0] = (uncertain_entropy == uncertain_gaps).float().mean()
    agreement_matrix[2, 1] = (uncertain_information == uncertain_gaps).float().mean()

    print(agreement_matrix)
