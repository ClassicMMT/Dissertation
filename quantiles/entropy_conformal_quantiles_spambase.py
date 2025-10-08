"""
This experiment follows the tutorial on conformal prediction for section 5.5 on selective classification. However, the found lambda_hat ends up being too small; i.e. 0.

So this method is not helpful.

Same experiment as entropy_conformal_quantiles.py but using the SpamBase dataset
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src.datasets import create_loaders, load_spambase, scale_datasets
from src.utils import (
    compute_acceptance_threshold,
    set_all_seeds,
    train_model,
)
from src.attacks import attack_model
from src.models import SpamBaseNet
import foolbox as fb
import torch
import torch.nn as nn


random_state = 123
g = set_all_seeds(random_state)
batch_size = 128
device = torch.device("mps")

# make train, calibration, and test splits
x_train, x_test, y_train, y_test = load_spambase(return_raw=True, random_state=random_state)
x_train, x_calib, y_train, y_calib = train_test_split(x_train, y_train, stratify=y_train, random_state=random_state + 1)
x_train, x_calib, x_test = scale_datasets(x_train, x_calib, x_test)
train_loader, train_dataset = create_loaders(x_train, y_train, batch_size=batch_size, generator=g)
calib_loader, calib_dataset = create_loaders(x_calib, y_calib, batch_size=batch_size, generator=g)
test_loader, test_dataset = create_loaders(x_test, y_test, batch_size=batch_size, generator=g)


# instantiate model
model = SpamBaseNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
n_epochs = 50

# train model
model = train_model(model, train_loader, criterion, optimizer, n_epochs, verbose=False, device=device)


# pre-compute calibration confidences and calibration errors
with torch.no_grad():
    model.eval()

    calib_confidences = []
    calib_correct = []

    for features, labels in calib_loader:
        features = features.to(device)
        labels = labels.to(device)

        # get model predictions
        logits = model(features)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

        # get model confidence
        confidences = probs.max(dim=-1).values

        calib_confidences.append(confidences.cpu())
        calib_correct.append((preds == labels).cpu())

    # combine
    calib_confidences = torch.cat(calib_confidences)
    calib_correct = torch.cat(calib_correct)
    calib_errors = ~calib_correct

# find lambda_hat from the calibration data
lambda_hat = compute_acceptance_threshold(calib_errors, calib_confidences, alpha=0.05, delta=0.05)

# experiment on x_test
with torch.no_grad():
    """
    certain_predictions here means predictions where the the predicted probability >= lambda_hat
    uncertain_predictions means predictions where predicted probability < lamda_hat
    """
    model.eval()

    certain_accuracy = []
    uncertain_accuracy = []

    uncertain_points = []
    uncertain_labels = []
    certain_points = []
    certain_labels = []

    all_confidences = []

    for features, labels in test_loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)

        # figure out which test points are in uncertain regions
        probs = torch.nn.functional.softmax(logits, dim=-1)
        model_confidence = probs.max(dim=-1).values
        confidence_over_lambda = model_confidence >= lambda_hat

        # get predictions
        preds = logits.argmax(dim=-1)
        is_correct_pred = preds == labels

        # accuracies
        uncertain_preds = is_correct_pred[~confidence_over_lambda]
        uncertain_accuracy.append(uncertain_preds)
        certain_preds = is_correct_pred[confidence_over_lambda]
        certain_accuracy.append(certain_preds)

        # points
        uncertain_points.append(features[~confidence_over_lambda].cpu())
        uncertain_labels.append(labels[~confidence_over_lambda].cpu())
        certain_points.append(features[confidence_over_lambda].cpu())
        certain_labels.append(labels[confidence_over_lambda].cpu())

        all_confidences.append(model_confidence)

    # combine
    certain_accuracy = torch.cat(certain_accuracy)
    uncertain_accuracy = torch.cat(uncertain_accuracy)

    uncertain_points = torch.cat(uncertain_points)
    certain_points = torch.cat(certain_points)
    uncertain_labels = torch.cat(uncertain_labels)
    certain_labels = torch.cat(certain_labels)

    points = torch.cat((certain_points, uncertain_points))
    labels = torch.cat(
        (
            torch.zeros(len(certain_points)),
            torch.ones(len(uncertain_points)),
        )
    )

    all_confidences = torch.cat(all_confidences)


# accuracy of points with predicted_probability >= lambda_hat
certain_accuracy.float().mean()
# accuracy of points with predicted_probability >= lambda_hat
uncertain_accuracy.float().mean()


################ The relationship between lambda_hat and adversarial examples

attacks = {
    # "fgsm": fb.attacks.FGSM(),
    "fgsm": fb.attacks.fast_gradient_method.L1FastGradientAttack(),
    "bim": fb.attacks.basic_iterative_method.L2AdamBasicIterativeAttack(),
    "deepfool": fb.attacks.deepfool.L2DeepFoolAttack(),
}
results = {}
examples = {}
for name, attack in attacks.items():
    adversarial_examples, original_labels = attack_model(
        model, attack=attack, loader=train_loader, verbose=False, epsilons=0.01
    )
    examples[name] = adversarial_examples.cpu()

    with torch.no_grad():
        model.eval()
        logits = model(adversarial_examples)
        adversarial_probs = torch.nn.functional.softmax(logits, dim=-1)
        adversarial_confidences = adversarial_probs.max(dim=-1).values
        results[name] = adversarial_confidences.cpu()


for name, adversarial_entropies in results.items():
    percent_below_lambda_hat = (adversarial_confidences < lambda_hat).float().mean()
    print(f"{name}: {percent_below_lambda_hat.item():.4f}")
