"""
This file finds information about the distributions of the test data.
Specifically the points which are misclassified.

This file labels the training data as:
    * correct
    * incorrect - misclassified points
    * uncertain - points which are correctly classified, but have an "easy" adversarial example.
        - The idea is that "easy" points (with small epsilon) are close to the decision boundary

These points are then combined with the adversarial examples and a knn is trained to predict which distribution the point is from:
    * correct
    * incorrect
    * uncertain
    * adversarial

A neural network is also trained, although the results of this are worse than the knn

"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import foolbox as fb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from src.utils import load_model, sample, set_all_seeds, train_model, calculate_entropy
from src.datasets import load_spambase, create_loaders
from src.attacks import attack_model
from src.models import GenericNet, SpamBaseNet

device = torch.device("mps")
random_state = 123
g = set_all_seeds(random_state)

# dataset
(train_loader, test_loader), (train_dataset, test_dataset) = load_spambase()

# model
model = load_model(SpamBaseNet(), "spambase").to(device)

# attacks
attacks = {
    "FGSM": fb.attacks.fast_gradient_method.LinfFastGradientAttack(),
    "DF_L2": fb.attacks.deepfool.L2DeepFoolAttack(),
    "DF_Linf": fb.attacks.deepfool.LinfDeepFoolAttack(),
    # "CW_L2": fb.attacks.carlini_wagner.L2CarliniWagnerAttack(),
    "BIM_Linf": fb.attacks.basic_iterative_method.LinfAdamBasicIterativeAttack(),
}
attack = attacks["DF_L2"]
epsilon = 0.05

uncertain_points = []
incorrect_points = []
correct_points = []
adversarial_examples = []


for features, labels in train_loader:
    features = features.to(device)
    labels = labels.to(device)

    # model predictions
    with torch.no_grad():
        model.eval()
        predictions = model(features).argmax(dim=-1)

    # attacks
    fmodel = fb.models.pytorch.PyTorchModel(model, bounds=(0, 1), device=device)
    _, clipped, is_adv = attack(fmodel, features, labels, epsilons=epsilon)

    # incorrect - points which the model did not classify
    incorrect = predictions != labels

    # uncertain - points which have easy adversarial examples
    uncertain = is_adv & ~incorrect

    # correct - everything else
    correct = ~incorrect & ~uncertain

    # adversarial examples
    clipped = clipped[is_adv]

    # save results
    uncertain_points.append(uncertain.to("cpu"))
    incorrect_points.append(incorrect.to("cpu"))
    correct_points.append(correct.to("cpu"))
    adversarial_examples.append(clipped.to("cpu"))


# Combine all points
uncertain_points = torch.cat(uncertain_points)
incorrect_points = torch.cat(incorrect_points)
correct_points = torch.cat(correct_points)
adversarial_examples = torch.cat(adversarial_examples)


############# Make Dataset #############

train_data = train_dataset.tensors[0]
correct_data = train_data[correct_points]
# sampling the data so there are not as many true examples
correct_data = sample(correct_data, size=len(adversarial_examples), random_state=random_state)
uncertain_data = train_data[uncertain_points]
incorrect_data = train_data[incorrect_points]
labels = torch.cat(
    (
        torch.zeros(len(correct_data)),
        torch.ones(len(uncertain_data)),
        torch.ones(len(incorrect_data)) * 2,
        torch.ones(len(adversarial_examples)) * 3,
    )
)
data = torch.cat((correct_data, uncertain_data, incorrect_data, adversarial_examples))

# get test data
test_data, test_labels = test_dataset.tensors

# fit KNN
knn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
knn.fit(data, labels)

# predict on test data
test_preds = model(test_data.to(device)).to("cpu").argmax(dim=-1)
knn_preds = knn.predict(test_data)
test_incorrect = (test_preds != test_labels).int()

# fit nn
model = GenericNet(layers=[57, 1024, 256, 64, 32, 4], activation="relu")
loader, dataset = create_loaders(data, labels, batch_size=256, generator=g)
_, counts = labels.unique(return_counts=True)
weights = 1.0 / counts * 10000
model, accuracy = train_model(
    model,
    loader,
    nn.CrossEntropyLoss(weights.to(device)),
    torch.optim.Adam(model.parameters(), lr=0.005),
    n_epochs=300,
    return_accuracy=True,
    device=device,
)
model.eval()
with torch.no_grad():
    nn_preds = model(test_data.to(device)).argmax(dim=-1).to("cpu")
    p = model(data.to(device)).argmax(dim=-1).to("cpu")


# test_incorrect -> 0 means it was correct and 1 is incorrect
m = confusion_matrix(test_incorrect, knn_preds)
m[:2, :]

m = confusion_matrix(test_incorrect, nn_preds)
m[:2, :]

confusion_matrix(labels, p)
