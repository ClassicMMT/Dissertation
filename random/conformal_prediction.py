"""
This file is trying the algorithm on conformal prediction found here:
https://www.youtube.com/watch?v=nql000Lu_iE


The proper tutorial is here:
https://people.eecs.berkeley.edu/~angelopoulos/publications/downloads/gentle_intro_conformal_dfuq.pdf

The repo is here:
https://github.com/aangelopoulos/conformal-prediction?tab=readme-ov-file
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.models import SpamBaseNet
from src.datasets import create_loaders, load_spambase, scale_datasets
from src.utils import evaluate_model, load_model, set_all_seeds, train_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

random_state = 123
g = set_all_seeds(random_state)
device = torch.device("mps")


# create training, calibration, and test sets
x_train, x_test, y_train, y_test = load_spambase(return_raw=True, random_state=random_state)
x_train, x_calib, y_train, y_calib = train_test_split(x_train, y_train, test_size=0.1, random_state=random_state)
x_train, x_calib, x_test = scale_datasets(x_train, x_calib, x_test)
train_loader, train_dataset = create_loaders(x_train, y_train, batch_size=128, generator=g)
calib_loader, calib_dataset = create_loaders(x_calib, y_calib, batch_size=128, generator=g)
test_loader, test_dataset = create_loaders(x_test, y_test, batch_size=128, generator=g)


# train model
model = SpamBaseNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

model = train_model(model, train_loader, criterion=criterion, optimizer=optimizer, n_epochs=50)

################### NON-ADAPTIVE PREDICTION SET METHOD

# This method involves:
#   1. smallest average size of prediction sets
#   2. not very adaptive
#   3. only uses output of true class


######## PART 1: get scores of correct class for all examples in the calibration set
E = []
size_of_calibration_set = 0
with torch.no_grad():
    model.eval()
    for features, labels in calib_loader:
        features = features.to(device)
        labels = labels.to(device)

        output = model(features)
        probabilities = F.softmax(output, dim=-1)

        # get score of correct class
        scores_of_correct_class = (
            torch.gather(probabilities, dim=-1, index=torch.unsqueeze(labels, 1)).squeeze(dim=-1).cpu()
        )
        E.append(scores_of_correct_class)

        size_of_calibration_set += len(labels)

# E is the scores of the correct classes from the calibration set
E = torch.cat(E)

####### PART 2: Take the ~= 10% quantile
# at least 90% of examples have a true class score above q_hat
alpha = 0.1
q_hat = E.quantile(alpha, dim=0, interpolation="lower")

if True:
    # plots the distribution of
    plt.hist(E)
    plt.vlines(1 - alpha, ymin=0, ymax=350, color="red")
    plt.show()


####### PART 3: Form prediction sets
# these are the sets of all classes whose score exceeds q_hat
# I will run this on the test set

prediction_sets = []
test_labels = []
with torch.no_grad():
    for features, labels in test_loader:
        features = features.to(device)
        labels = labels.to(device)

        scores = model(features)
        probabilities = F.softmax(scores, dim=-1)
        scores_exceeding_q_hat = (probabilities >= q_hat).cpu()

        prediction_sets += [torch.nonzero(score, as_tuple=True)[0] for score in scores_exceeding_q_hat]

        test_labels += labels.cpu().tolist()

        # the coverage theorem states
        # 1-alpha <= P[Y is in the prediction set] <= 1-alpha + 1/(size_of_calibration_set + 1)
        # this is saying that the probability that the true class label falls into the prediction
        # set is inside those bounds
        # this is guaranteed for any algorithm, dataset, alpha, or size_of_calibration_set


# therefore the probability that the correct label is in the prediction set is:
lower = 1 - alpha
upper = 1 - alpha + 1 / (size_of_calibration_set + 1)
print(f"Theoretical coverage guarantee: [{lower:.3f}, {upper:.3f}]")

# compute the coverage (whether the true label is in the prediction set)
covered = [(test_labels[i] in prediction_sets[i].tolist()) for i in range(len(test_labels))]

empirical_coverage = sum(covered) / len(covered)
print(f"Empirical coverage: {empirical_coverage:.3f}")


################### ADAPTIVE PREDICTION SET METHOD (top down CP)

# This method involves:
#   1. usually larger size (compared to above method) of prediction sets
#   2. designed to be adaptive
#   3. uses output of all classes

######## PART 1: get scores of correct class for all examples in the calibration set

# the procedure here is to "sort" the softmaxed outputs and sum up all of the scores that are greater or equal to the true class

E = []
size_of_calibration_set = 0
with torch.no_grad():
    model.eval()
    for features, labels in calib_loader:
        features, labels = next(iter(calib_loader))
        features = features.to(device)
        labels = labels.to(device)

        output = model(features)
        probabilities = F.softmax(output, dim=-1)

        # get score of correct class
        scores_of_correct_class = torch.gather(probabilities, dim=-1, index=torch.unsqueeze(labels, 1))

        scores_greater_or_equal_to_true_class = output >= scores_of_correct_class
        E += [(probabilities * scores_greater_or_equal_to_true_class).sum(dim=-1)]

        size_of_calibration_set += len(labels)


# E is the "total mass" of the sorted softmax outputs up to and including the true class
# i.e. "amount of mass" needed to include the true label
E = torch.cat(E)

####### PART 2: Take the ~= 10% quantile
alpha = 0.1
q_hat = torch.quantile(E, 1 - alpha, dim=0, interpolation="higher")
q_hat

# idea: when the "predicted mass" exceeds q_hat, then we're pretty sure
#   we've gotten the right answer

####### PART 3: Form prediction sets

# take the k most likely classes where the sum of their probabilities >= q_hat

with torch.no_grad():
    model.eval()
    prediction_sets = []
    test_labels = []

    for features, labels in test_loader:
        features = features.to(device)

        output = model(features)
        probabilities = F.softmax(output, dim=-1)

        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True, dim=-1)

        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

        batch_prediction_sets = []

        for i in range(features.shape[0]):
            # find minimal set of classes where cumulative sum >= q_hat
            k = torch.searchsorted(cumsum_probs[i], q_hat)

            pred_set = sorted_indices[i, :k]
            batch_prediction_sets.append(pred_set)

        prediction_sets.extend(batch_prediction_sets)

        test_labels.append(labels)

test_labels = torch.cat(test_labels).tolist()

# compute empirical coverage
covered = [(test_labels[i] in prediction_sets[i].tolist()) for i in range(len(test_labels))]

empirical_coverage = sum(covered) / len(covered)
print(f"Empirical coverage: {empirical_coverage:.3f}")
