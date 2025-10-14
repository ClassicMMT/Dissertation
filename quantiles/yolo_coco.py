"""
This script experiments with the YOLO model and COCO dataset.

This example isn't finished because the yolo model yolov8n-cls was trained on
ImageNet and not on the COCO dataset.
"""

import torch
from src.datasets import load_coco
from src.utils import load_yolo, set_all_seeds

random_state = 123
g = set_all_seeds(random_state)
device = torch.device("mps")

# Load model and data
model = load_yolo(device="mps")
(calib_loader, test_loader), _ = load_coco(batch_size=128, generator=g)


# Utility function
def yolo_to_probs(output):
    return torch.stack([x.probs.data for x in output])


def yolo_entropy(p, eps=1e-12):
    """
    Calculates entropy for matrix of size (batch_size, n_classes).
    This is for the multi-label classification case.
    """
    return -p * torch.log(p + eps) - (1 - p) * torch.log(1 - p + eps)


def yolo_information_content(p, eps=1e-12):
    """
    Calculates the information content for matrix of size (batch_size, n_classes).
    This is for the multi-label classification case.
    """
    return -torch.log(p + eps)


def yolo_probability_gaps(p):
    """
    Calculates probability gaps for matrix of size (batch_size, n_classes).
    This is for the multi-label classification case.
    """
    return torch.abs(p - 0.5)


# Calculate the per class uncertainties on the calibration set
with torch.no_grad():
    model.eval()

    entropies = []
    information_contents = []
    probability_gaps = []

    for features, labels in calib_loader:
        features = features.to(device)
        labels = labels.to(device)

        output = model(features, verbose=False)
        per_class_probs = yolo_to_probs(output)  # (batch_size, n_classes)

        entropy = yolo_entropy(per_class_probs)
        info_content = yolo_information_content(per_class_probs)
        gaps = yolo_probability_gaps(per_class_probs)

        entropies.append(entropy)
        information_contents.append(info_content)
        probability_gaps.append(gaps)

    entropies = torch.cat(entropies)
    information_contents = torch.cat(information_contents)
    probability_gaps = torch.cat(probability_gaps)

# thresholds here are calculated PER CLASS
alpha = 0.05
entropy_thresholds = torch.quantile(entropies, 1 - alpha, dim=0, interpolation="higher")  # dim=0 is important!
info_content_thresholds = torch.quantile(information_contents, 1 - alpha, dim=0, interpolation="higher")
gap_thresholds = torch.quantile(probability_gaps, alpha, dim=0, interpolation="lower")


# Compute the same on the test set
with torch.no_grad():
    model.eval()

    entropies = []
    information_contents = []
    probability_gaps = []

    for features, labels in test_loader:
        features = features.to(device)
        labels = labels.to(device)

        output = model(features, verbose=False)
        per_class_probs = yolo_to_probs(output)  # (batch_size, n_classes)

        entropy = yolo_entropy(per_class_probs)
        info_content = yolo_information_content(per_class_probs)
        gaps = yolo_probability_gaps(per_class_probs)

        entropies.append(entropy)
        information_contents.append(info_content)
        probability_gaps.append(gaps)

    entropies = torch.cat(entropies)
    information_contents = torch.cat(information_contents)
    probability_gaps = torch.cat(probability_gaps)


per_class_probs.sort(descending=True).values
per_class_probs.max(dim=1).values

f, l = next(iter(test_loader))
f = f.to(device)
l = l.to(device)
output = model(f, verbose=False)
per_class_probs = yolo_to_probs(output)  # (batch_size, n_classes)
entropy = yolo_entropy(per_class_probs)
info_content = yolo_information_content(per_class_probs)
gaps = yolo_probability_gaps(per_class_probs)

l.shape

(per_class_probs > 0.5).float().shape
labels.shape
