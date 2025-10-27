"""
This script uses a pre-trained MNIST model to see how many fashion MNIST examples are flagged.
"""

import torch
from src.models import MNISTNet
from src.datasets import load_fashion_mnist, load_mnist
from src.utils import (
    calculate_entropy,
    calculate_information_content,
    calculate_probability_gap,
    load_model,
    set_all_seeds,
)
from src.method import compute_threshold_all


random_state = 123
g = set_all_seeds(random_state)
device = torch.device("mps")
batch_size = 128

model = load_model(MNISTNet(), "mnist.pt").to(device)

# Load original MNIST
(_, mnist_calib_loader, _), _ = load_mnist(batch_size, generator=g, return_val=True)

# Compute thresholds
entropy_threshold, information_threshold, gap_threshold = compute_threshold_all(
    model, mnist_calib_loader, alpha=0.05, device=device
)

# Load Fashion MNIST
(_, fashion_test_loader), _ = load_fashion_mnist(batch_size, generator=g)


# Compute if any of the points are OOD
with torch.no_grad():
    model.eval()

    uncertain_entropy = []
    uncertain_info = []
    uncertain_gaps = []
    uncertain_any = []
    uncertain_all = []

    for features, _ in fashion_test_loader:
        features = features.to(device)

        # get logits
        logits = model(features)

        # get uncertainty metrics
        entropy = calculate_entropy(logits)
        information = calculate_information_content(logits)
        gaps = calculate_probability_gap(logits)

        # compute masks
        entropy_mask = entropy >= entropy_threshold
        information_mask = information >= information_threshold
        gap_mask = gaps <= gap_threshold
        any_mask = entropy_mask | information_mask | gap_mask
        all_mask = entropy_mask & information_mask & gap_mask

        # save results
        uncertain_entropy.append(entropy_mask)
        uncertain_info.append(information_mask)
        uncertain_gaps.append(gap_mask)
        uncertain_any.append(any_mask)
        uncertain_all.append(all_mask)

    uncertain_entropy = torch.cat(uncertain_entropy)
    uncertain_info = torch.cat(uncertain_info)
    uncertain_gaps = torch.cat(uncertain_gaps)
    uncertain_any = torch.cat(uncertain_any)
    uncertain_all = torch.cat(uncertain_all)

uncertain_entropy.float().mean()
uncertain_info.float().mean()
uncertain_gaps.float().mean()
uncertain_any.float().mean()
uncertain_all.float().mean()
