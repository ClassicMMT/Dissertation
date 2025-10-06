"""
This experiment calculates:
    * density
    * information content
    * entropy
    * probability gaps

and trains a decision tree classifier to predict points which will be misclassified.
"""

import torch
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
from src.utils import (
    calculate_entropy,
    calculate_probability_gap,
    set_all_seeds,
    train_model,
    calculate_information_content,
    KNNDensity,
)
from src.models import GenericNet
from src.datasets import create_loaders, make_chessboard
import matplotlib.pyplot as plt

random_state = 123
g = set_all_seeds(random_state)
device = torch.device("mps")
batch_size = 128

# make data
x, y = make_chessboard(n_blocks=4, n_points_in_block=100, random_state=random_state)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=random_state)
train_loader, _ = create_loaders(x_train, y_train, batch_size=batch_size)
test_loader, _ = create_loaders(x_test, y_test, batch_size=batch_size)

# model
model = GenericNet(layers=[2, 1024, 512, 256, 2], activation="relu", random_state=random_state).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
model = train_model(model, train_loader, criterion, optimizer, n_epochs=30, device=device, verbose=False)


# do a fun for density normalisation
knndensity = KNNDensity()
for features, _ in train_loader:
    knndensity.update(features)

# get features
with torch.no_grad():
    tree_features = []
    tree_labels = []

    model.eval()
    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)

        # get logits and predictions
        logits = model(features)
        preds = logits.argmax(dim=-1)
        is_correct = preds == labels

        # calculate features
        gaps = calculate_probability_gap(logits).unsqueeze(dim=0)
        entropies = calculate_entropy(logits).unsqueeze(dim=0)
        information_content = calculate_information_content(logits).unsqueeze(dim=0)
        densities = knndensity.predict(features).unsqueeze(dim=0)

        batch_data = torch.stack(
            [
                calculate_probability_gap(logits),
                calculate_entropy(logits),
                calculate_information_content(logits),
                knndensity.predict(features),
            ]
        )

        tree_features.append(batch_data)
        tree_labels.append(is_correct.int())

    tree_features = torch.cat(tree_features, dim=-1).cpu().T.numpy()
    tree_labels = torch.cat(tree_labels, dim=0).cpu().numpy()

# train decision tree
# clf = DecisionTreeClassifier(
#     max_depth=6, min_samples_leaf=3, ccp_alpha=0.001, class_weight="balanced", random_state=random_state
# )
# clf = RandomForestClassifier(n_jobs=-1, random_state=random_state, class_weight={0: 1000000, 1: 1})
clf = BalancedRandomForestClassifier(n_jobs=-1, random_state=random_state)
clf.fit(tree_features, tree_labels)
train_preds = clf.predict(tree_features)
confusion_matrix(tree_labels, train_preds)

# get test features
with torch.no_grad():
    tree_test_features = []
    tree_test_labels = []

    model.eval()
    for features, labels in test_loader:
        features = features.to(device)
        labels = labels.to(device)

        # get logits and predictions
        logits = model(features)
        preds = logits.argmax(dim=-1)
        is_correct = preds == labels

        # calculate features
        gaps = calculate_probability_gap(logits).unsqueeze(dim=0)
        entropies = calculate_entropy(logits).unsqueeze(dim=0)
        information_content = calculate_information_content(logits).unsqueeze(dim=0)
        densities = knndensity.predict(features).unsqueeze(dim=0)

        batch_data = torch.stack(
            [
                calculate_probability_gap(logits),
                calculate_entropy(logits),
                calculate_information_content(logits),
                knndensity.predict(features),
            ]
        )

        tree_test_features.append(batch_data)
        tree_test_labels.append(is_correct.int())

    tree_test_features = torch.cat(tree_test_features, dim=-1).cpu().T.numpy()
    tree_test_labels = torch.cat(tree_test_labels, dim=0).cpu().numpy()

test_preds = clf.predict(tree_test_features)
(test_preds == tree_test_labels).mean()
confusion_matrix(tree_test_labels, test_preds)

# ROC AUC
probs_neg = clf.predict_proba(tree_test_features)[:, 0]
fpr, tpr, thresholds = roc_curve(tree_test_labels, probs_neg, pos_label=0)
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
    plt.show()
