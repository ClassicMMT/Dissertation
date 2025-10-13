"""
The purpose of this script is to compare the different selective classification methods:
    * entropy
    * information content
    * probability gaps
    * conformal prediction

on the chessboard dataset
"""

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src.datasets import create_loaders, make_chessboard
from src.utils import (
    calculate_entropy,
    calculate_information_content,
    calculate_probability_gap,
    compute_acceptance_threshold,
    evaluate_model,
    set_all_seeds,
    train_model,
)
from src.models import GenericNet
import torch
import torch.nn as nn
import torch.nn.functional as F


random_state = 123
g = set_all_seeds(random_state)
batch_size = 128
device = torch.device("mps")

# make train, calibration, and test splits
# x, y = make_chessboard(n_blocks=4, n_points_in_block=100, random_state=random_state, all_different_classes=True)
x, y = make_chessboard(n_blocks=4, n_points_in_block=100, random_state=random_state)
n = len(y.unique())
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=random_state)
x_train, x_calib, y_train, y_calib = train_test_split(
    x_train, y_train, test_size=0.25, stratify=y_train, random_state=random_state + 1
)
train_loader, train_dataset = create_loaders(x_train, y_train, batch_size=batch_size, generator=g)
calib_loader, calib_dataset = create_loaders(x_calib, y_calib, batch_size=batch_size, generator=g)
test_loader, test_dataset = create_loaders(x_test, y_test, batch_size=batch_size, generator=g)


# train model
model = GenericNet(layers=[2, 1024, 512, 256, n]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
n_epochs = 30

# train model
model = train_model(model, train_loader, criterion, optimizer, n_epochs, verbose=False, device=device)


# calibration part
with torch.no_grad():
    model.eval()

    # for conformal prediction
    calibration_errors = []
    calibration_confidences = []

    entropies = []
    information_contents = []
    probability_gaps = []

    for features, labels in calib_loader:
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)
        predictions = logits.argmax(dim=-1)
        probs = F.softmax(logits, dim=-1)

        # For conformal prediction
        model_confidences = probs.max(dim=-1).values
        prediction_is_incorrect = predictions != labels

        # My methods
        entropy = calculate_entropy(logits)
        info_content = calculate_information_content(logits)
        gaps = calculate_probability_gap(logits)

        # save
        calibration_errors.append(prediction_is_incorrect)
        calibration_confidences.append(model_confidences)
        entropies.append(entropy)
        information_contents.append(info_content)
        probability_gaps.append(gaps)

    # combine
    calibration_errors = torch.cat(calibration_errors)
    calibration_confidences = torch.cat(calibration_confidences)
    entropies = torch.cat(entropies)
    information_contents = torch.cat(information_contents)
    probability_gaps = torch.cat(probability_gaps)

# Compute Thresholds
alpha = 0.05
delta = 0.05

entropy_threshold = entropies.quantile(1 - alpha, dim=0, interpolation="higher")
information_content_threshold = information_contents.quantile(1 - alpha, dim=0, interpolation="higher")
probability_gap_threshold = probability_gaps.quantile(alpha, dim=0, interpolation="lower")
conformal_threshold = compute_acceptance_threshold(
    calibration_errors, calibration_confidences, alpha=alpha, delta=delta
)


# Compute test results
with torch.no_grad():
    is_correct = []
    conformal_results = []
    entropy_results = []
    info_content_results = []
    probability_gap_results = []
    combined_results = []

    for features, labels in test_loader:
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)
        probs = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)

        # compute what we need
        entropy = calculate_entropy(logits)
        info_content = calculate_information_content(logits)
        gaps = calculate_probability_gap(logits)
        model_confidences = probs.max(dim=-1).values

        # compute masks
        conformal_mask = model_confidences < conformal_threshold
        entropy_mask = entropy > entropy_threshold
        info_content_mask = info_content > information_content_threshold
        probability_gap_mask = gaps < probability_gap_threshold  # IMPORTANT
        all_mask = entropy_mask | info_content_mask | probability_gap_mask

        prediction_is_correct = predictions == labels

        # save
        is_correct.append(prediction_is_correct)
        conformal_results.append(conformal_mask)
        entropy_results.append(entropy_mask)
        info_content_results.append(info_content_mask)
        probability_gap_results.append(probability_gap_mask)
        combined_results.append(all_mask)

    is_correct = torch.cat(is_correct)
    conformal_reject = torch.cat(conformal_results)
    entropy_reject = torch.cat(entropy_results)
    info_content_reject = torch.cat(info_content_results)
    probability_gap_reject = torch.cat(probability_gap_results)
    combined_reject = torch.cat(combined_results)


# evaluation
certain_conformal = is_correct[~conformal_reject].float()
uncertain_conformal = is_correct[conformal_reject].float()

certain_entropy = is_correct[~entropy_reject].float()
uncertain_entropy = is_correct[entropy_reject].float()

certain_info_content = is_correct[~info_content_reject].float()
uncertain_info_content = is_correct[info_content_reject].float()

certain_probability_gap = is_correct[~probability_gap_reject].float()
uncertain_probability_gap = is_correct[probability_gap_reject].float()

certain_combined = is_correct[~combined_reject].float()
uncertain_combined = is_correct[combined_reject].float()

# print results
print(f"Model Test Accuracy: {evaluate_model(model, test_loader, device=device)*100:.2f}")
print(
    f"CONFORMAL: Certain Acc: {certain_conformal.mean()*100:.2f}%, n: {len(certain_conformal)}, Uncertain Acc: {uncertain_conformal.mean()*100:.2f}%, n: {len(uncertain_conformal)}"
)
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
    f"COMBINED: Certain Acc: {certain_combined.mean()*100:.2f}%, n: {len(certain_combined)}, Uncertain Acc: {uncertain_combined.mean()*100:.2f}%, n: {len(uncertain_combined)}"
)
