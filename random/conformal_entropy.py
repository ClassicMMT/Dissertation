"""
This experiment takes inspiration from conformal prediction.

The process is:
    1. Calculate entropies on the calibration set
    2. Define an alpha-quantile based on entropy
    3. Use this cut-off to abstain from prediction on the test set
"""

from sklearn.model_selection import train_test_split
import torch
from src.datasets import create_loaders, load_spambase, scale_datasets
from src.models import SpamBaseNet
from src.utils import evaluate_model, set_all_seeds, train_model


random_state = 123
g = set_all_seeds(random_state)
device = torch.device("mps")
batch_size = 128

# load data
x_train, x_test, y_train, y_test = load_spambase(return_raw=True, random_state=random_state)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, stratify=y_train, random_state=random_state)
x_train, x_val, x_test = scale_datasets(x_train, x_val, x_test)
train_loader, train_dataset = create_loaders(x_train, y_train, batch_size=batch_size, generator=g)
calib_loader, calib_dataset = create_loaders(x_val, y_val, batch_size=batch_size, generator=g)
test_loader, test_dataset = create_loaders(x_test, y_test, batch_size=batch_size, generator=g)

# make and train model
model = SpamBaseNet()
model = train_model(
    model,
    loader=train_loader,
    criterion=torch.nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters()),
    n_epochs=10,
    device=device,
)


def label_nonconformity_scores(logits, detach=True):
    """
    Higher score = less likely label
    """
    if detach:
        logits = logits.clone().detach()
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return 1 - probs


with torch.no_grad():
    model.eval()

    calib_scores = []
    true_label_scores = []

    for features, labels in calib_loader:
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)
        # compute non conformity scores
        scores = label_nonconformity_scores(logits)

        # compute true label scores
        label_scores = scores[torch.arange(len(features)), labels]

        # save
        calib_scores.append(scores)
        true_label_scores.append(label_scores)

    true_label_scores = torch.cat(true_label_scores)


# conformal threshold
alpha = 0.05
threshold = torch.quantile(true_label_scores, 1 - alpha, interpolation="higher")


# construct prediction sets
with torch.no_grad():
    model.eval()

    for features, labels in test_loader:
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)
        # compute non conformity scores
        scores = label_nonconformity_scores(logits)

        # compute true label scores
        label_scores = scores[torch.arange(len(features)), labels]

        # save
        calib_scores.append(scores)
        true_label_scores.append(label_scores)

    true_label_scores = torch.cat(true_label_scores)
