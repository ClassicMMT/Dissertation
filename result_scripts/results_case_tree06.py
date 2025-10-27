"""
This script computes the PR-AUC curves for every model that's not a chessboard model.
"""

import torch
from src.datasets import load_cifar, load_imagenet, load_spambase
from src.utils import YoloWrapper, load_model, set_all_seeds, load_resnet50, load_cifar_model, load_yolo
from src.models import SpamBaseNet
from src.method import get_uncertainty_features
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

random_state = 123
g = set_all_seeds(random_state)
device = torch.device("mps")
batch_size = 128

# Set up
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

################################## SpamBase ##################################

# Load all data
(_, calib_loader, test_loader), _ = load_spambase(
    batch_size=batch_size, test_size=0.25, random_state=random_state, return_train_val_test=True
)

# Load model
model = load_model(SpamBaseNet(), "spambase_final.pt").to(device)

# Get uncertainty features and labels
features, labels = get_uncertainty_features(model, calib_loader, verbose=True, device=device)
test_features, test_labels = get_uncertainty_features(model, test_loader, verbose=True, device=device)

# Train classifier

clf = BalancedRandomForestClassifier(n_jobs=-1, random_state=random_state)
clf.fit(features, labels)

# Get predicted probabilities for the positive class
probs_pos = clf.predict_proba(test_features)[:, 1]

# Compute precision-recall curve and AUC
precision, recall, thresholds = precision_recall_curve(test_labels, probs_pos, pos_label=1)
pr_auc = average_precision_score(test_labels, probs_pos, pos_label=1)

# Plot Precision-Recall curve
axs[0].plot(recall, precision, label=f"PR curve (AUC = {pr_auc:.3f})", linewidth=2)
axs[0].set_xlabel("Recall")
axs[0].set_ylabel("Precision")
axs[0].set_title("Precision–Recall Curve – SpamBase")
axs[0].legend(loc="lower left")
axs[0].grid(True)

################################## CIFAR 10 ##################################

# Load Data
(calib_loader, test_loader), _ = load_cifar(batch_size, generator=g)

# Load model
model = load_cifar_model("cifar10_resnet56", device=device)

# Get uncertainty features and labels
features, labels = get_uncertainty_features(model, calib_loader, verbose=True, device=device)
test_features, test_labels = get_uncertainty_features(model, test_loader, verbose=True, device=device)

# Train classifier
clf = BalancedRandomForestClassifier(n_jobs=-1, random_state=random_state)
clf.fit(features, labels)

# Get predicted probabilities for the positive class
probs_pos = clf.predict_proba(test_features)[:, 1]

# Compute precision-recall curve and AUC
precision, recall, thresholds = precision_recall_curve(test_labels, probs_pos, pos_label=1)
pr_auc = average_precision_score(test_labels, probs_pos, pos_label=1)

# Plot Precision-Recall curve
axs[1].plot(recall, precision, label=f"PR curve (AUC = {pr_auc:.3f})", linewidth=2)
axs[1].set_xlabel("Recall")
axs[1].set_ylabel("Precision")
axs[1].set_title("Precision–Recall Curve – CIFAR-10 (ResNet56)")
axs[1].legend(loc="lower left")
axs[1].grid(True)

################################## ImageNet ResNet-50 ##################################

# Load data
(calib_loader, test_loader), _ = load_imagenet(batch_size, generator=g)

# Load model
model = load_resnet50(device=device)

# Get uncertainty features and labels
features, labels = get_uncertainty_features(model, calib_loader, verbose=True, device=device)
test_features, test_labels = get_uncertainty_features(model, test_loader, verbose=True, device=device)

# Train classifier
clf = BalancedRandomForestClassifier(n_jobs=-1, random_state=random_state)
clf.fit(features, labels)

# Get predicted probabilities for the positive class
probs_pos = clf.predict_proba(test_features)[:, 1]

# Compute precision-recall curve and AUC
precision, recall, thresholds = precision_recall_curve(test_labels, probs_pos, pos_label=1)
pr_auc = average_precision_score(test_labels, probs_pos, pos_label=1)

# Plot Precision-Recall curve
axs[2].plot(recall, precision, label=f"PR curve (AUC = {pr_auc:.3f})", linewidth=2)
axs[2].set_xlabel("Recall")
axs[2].set_ylabel("Precision")
axs[2].set_title("Precision–Recall Curve – ImageNet (Resnet50)")
axs[2].legend(loc="lower left")
axs[2].grid(True)

################################## ImageNet YOLO ##################################

# Load model
model = YoloWrapper(load_yolo(device=device))

# Get uncertainty features and labels
features, labels = get_uncertainty_features(model, calib_loader, apply_softmax=False, verbose=True, device=device)
test_features, test_labels = get_uncertainty_features(
    model, test_loader, apply_softmax=False, verbose=True, device=device
)

# Train classifier
clf = BalancedRandomForestClassifier(n_jobs=-1, random_state=random_state)
clf.fit(features, labels)

# Get predicted probabilities for the positive class
probs_pos = clf.predict_proba(test_features)[:, 1]

# Compute precision-recall curve and AUC
precision, recall, thresholds = precision_recall_curve(test_labels, probs_pos, pos_label=1)
pr_auc = average_precision_score(test_labels, probs_pos, pos_label=1)

# Plot Precision-Recall curve
axs[3].plot(recall, precision, label=f"PR curve (AUC = {pr_auc:.3f})", linewidth=2)
axs[3].set_xlabel("Recall")
axs[3].set_ylabel("Precision")
axs[3].set_title("Precision–Recall Curve – ImageNet (Resnet50)")
axs[3].legend(loc="lower left")
axs[3].grid(True)


# Show plots
plt.tight_layout()
plt.show()
