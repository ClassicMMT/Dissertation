"""
This experiment calculates the alpha-accuracy curves for each epoch trained on the spambase dataset.

To be plotted in R using the script with the same name.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from src.utils import (
    calculate_entropy,
    calculate_probability_gap,
    set_all_seeds,
    train_model,
    calculate_information_content,
)
from src.models import SpamBaseNet
from src.datasets import create_loaders, load_spambase, scale_datasets
import matplotlib.pyplot as plt

random_state = 123
g = set_all_seeds(random_state)
device = torch.device("mps")
batch_size = 128

# get data
x_train, x_test, y_train, y_test = load_spambase(batch_size=batch_size, random_state=random_state, return_raw=True)
x_train, x_calib, y_train, y_calib = train_test_split(x_train, y_train, test_size=0.2, random_state=random_state + 1)
x_train, x_calib, x_test = scale_datasets(x_train, x_calib, x_test)
train_loader, _ = create_loaders(x_train, y_train, batch_size=batch_size, generator=g)
calib_loader, _ = create_loaders(x_calib, y_calib, batch_size=batch_size, generator=g)
test_loader, _ = create_loaders(x_test, y_test, batch_size=batch_size, generator=g)

# model
model = SpamBaseNet().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
max_epochs = 20

results = {
    "epoch": [],
    "alpha": [],
    "method": [],
    "certain_accuracy": [],
    "uncertain_accuracy": [],
    "n_certain": [],
    "n_uncertain": [],
}

for epoch in range(max_epochs):
    print(f"Running epoch {epoch+1}/{max_epochs}")
    model = train_model(model, train_loader, criterion, optimizer, n_epochs=1, device=device, verbose=False)

    model.eval()

    # Find thresholds from the calibration data
    with torch.no_grad():
        probability_gaps = []
        entropies = []
        info_contents = []
        correct_predictions = []

        for features, labels in calib_loader:
            features = features.to(device)
            labels = labels.to(device)

            # get logits and predictions
            logits = model(features)

            # compute what we need
            gaps = calculate_probability_gap(logits)
            entropy = calculate_entropy(logits)
            information_content = calculate_information_content(logits)

            # save
            probability_gaps.append(gaps)
            entropies.append(entropy)
            info_contents.append(information_content)

        # combine
        probability_gaps = torch.cat(probability_gaps)
        entropies = torch.cat(entropies)
        info_contents = torch.cat(info_contents)

    # Get test metrics
    with torch.no_grad():
        probability_gaps = []
        entropies = []
        info_contents = []
        correct_predictions = []

        for features, labels in calib_loader:
            features = features.to(device)
            labels = labels.to(device)

            # get logits and predictions
            logits = model(features)
            preds = logits.argmax(dim=-1)

            # compute what we need
            gaps = calculate_probability_gap(logits)
            entropy = calculate_entropy(logits)
            information_content = calculate_information_content(logits)
            is_correct = preds == labels

            # save
            probability_gaps.append(gaps)
            entropies.append(entropy)
            info_contents.append(information_content)
            correct_predictions.append(is_correct)

        # Combine
        probability_gaps = torch.cat(probability_gaps)
        entropies = torch.cat(entropies)
        info_contents = torch.cat(info_contents)
        correct_predictions = torch.cat(correct_predictions)

    # Get results for all alphas
    alphas = torch.linspace(0, 1, 101)
    for alpha in alphas:

        # Compute thresholds
        entropy_threshold = entropies.quantile(1 - alpha, dim=0, interpolation="higher")
        information_content_threshold = info_contents.quantile(1 - alpha, dim=0, interpolation="higher")
        probability_gap_threshold = probability_gaps.quantile(alpha, dim=0, interpolation="lower")

        # compute rejection vectors
        entropy_reject = entropies > entropy_threshold
        info_content_reject = info_contents > information_content_threshold
        probability_gap_reject = probability_gaps < probability_gap_threshold
        combined_reject = entropy_reject | info_content_reject | probability_gap_reject

        # compute results
        certain_entropy = correct_predictions[~entropy_reject].float()
        uncertain_entropy = correct_predictions[entropy_reject].float()

        certain_info_content = correct_predictions[~info_content_reject].float()
        uncertain_info_content = correct_predictions[info_content_reject].float()

        certain_probability_gap = correct_predictions[~probability_gap_reject].float()
        uncertain_probability_gap = correct_predictions[probability_gap_reject].float()

        certain_combined = correct_predictions[~combined_reject].float()
        uncertain_combined = correct_predictions[combined_reject].float()

        # Save results
        results["epoch"] += [epoch + 1] * 4
        results["alpha"] += [alpha.item()] * 4

        results["method"].append("entropy")
        results["certain_accuracy"].append(certain_entropy.mean().item())
        results["uncertain_accuracy"].append(uncertain_entropy.mean().item())
        results["n_certain"].append(len(certain_entropy))
        results["n_uncertain"].append(len(uncertain_entropy))

        results["method"].append("information_content")
        results["certain_accuracy"].append(certain_info_content.mean().item())
        results["uncertain_accuracy"].append(uncertain_info_content.mean().item())
        results["n_certain"].append(len(certain_info_content))
        results["n_uncertain"].append(len(uncertain_info_content))

        results["method"].append("probability_gap")
        results["certain_accuracy"].append(certain_probability_gap.mean().item())
        results["uncertain_accuracy"].append(uncertain_probability_gap.mean().item())
        results["n_certain"].append(len(certain_probability_gap))
        results["n_uncertain"].append(len(uncertain_probability_gap))

        results["method"].append("combined")
        results["certain_accuracy"].append(certain_combined.mean().item())
        results["uncertain_accuracy"].append(uncertain_combined.mean().item())
        results["n_certain"].append(len(certain_combined))
        results["n_uncertain"].append(len(uncertain_combined))


data = pd.DataFrame(results)
data.to_csv("saved_results/quantile_accuracy_overfitting_spambase.py", index=False)
