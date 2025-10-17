"""
This script calculates features:
    * entropy
    * information content
    * probability gaps

and uses them to train another classifier which will
predict whether the original model will misclassify
the examples.

On the ImageNet Dataset
"""

import torch
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, recall_score, precision_score
from src.method import get_uncertainty_features
from src.utils import load_resnet50, load_yolo, set_all_seeds, load_cifar_model, YoloWrapper
from src.datasets import load_cifar, load_imagenet
import matplotlib.pyplot as plt

random_state = 123
g = set_all_seeds(random_state)
device = torch.device("mps")
batch_size = 128

# get data and model
(calib_loader, test_loader), _ = load_imagenet(batch_size=batch_size, generator=g)
model = load_resnet50(device=device)
# model = YoloWrapper(load_yolo(device=device))

# get features
features, labels = get_uncertainty_features(model, calib_loader, device=device, verbose=True)

# Train model on the features
clf = BalancedRandomForestClassifier(n_jobs=-1, random_state=random_state)
clf.fit(features, labels)

# evaluate performance
train_preds = clf.predict(features)
confusion_matrix(labels, train_preds)

# get test features
test_features, test_labels = get_uncertainty_features(model, test_loader, device=device, verbose=True)

# Predict on the test set
test_preds = clf.predict(test_features)
confusion_matrix(test_labels, test_preds)

accuracy_score(test_labels, test_preds)
recall_score(test_labels, test_preds)
precision_score(test_labels, test_preds)

# Recall: tp / (tp + fn)
# 241 / (241 + 47)

# Precision: tp / (tp + fp)
# 241 / (241 + 727)


# ROC AUC
probs_neg = clf.predict_proba(test_features)[:, 0]
fpr, tpr, thresholds = roc_curve(test_labels, probs_neg, pos_label=0)
roc_auc = auc(fpr, tpr)

if True:
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – Incorrect Predictions")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# feature importances
for feature, value in zip(
    [
        "entropy",
        "information content",
        "probability gaps",
    ],
    clf.feature_importances_,
):
    print(f"Feature: '{feature}' with importance: {value:.4f}")
