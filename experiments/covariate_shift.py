"""
This experiment introduces a covariate shift in the test data and assesses
if adversarial examples can be used to predict whether the incorrectly classified points
can be predicted using these examples.
"""

import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from src.attacks import load_all_adversarial_examples
from src.models import SpamBaseNet
from src.utils import load_model, set_all_seeds
from src.datasets import induce_covariate_shift, load_spambase, scale_datasets

random_state = 123
g = set_all_seeds(random_state)

# load data with no scaling
(train_loader, test_loader), (train_dataset, test_dataset) = load_spambase(
    scale=False, random_state=random_state
)
model = load_model(SpamBaseNet(), "spambase")

# Induce a covariate shift
shifted_test_loader, shifted_test_dataset = induce_covariate_shift(
    test_dataset, n_features_to_shift=57, intensity=0.1, random_state=random_state
)

# extract tensors
shifted_x, shifted_y = shifted_test_dataset.tensors
train_x, train_y = train_dataset.tensors

# load adversarial examples
adversarial_examples, _ = load_all_adversarial_examples("spambase")

# create 0: original, 1: adversarial labels
labels = torch.cat((torch.zeros(len(train_x)), torch.ones(len(adversarial_examples))))

# combine all examples
examples = torch.cat((train_x, adversarial_examples))


# predict which points have had a distribution shift
knn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
knn.fit(examples, labels)

predictions = knn.predict(shifted_x)

"""
The shifted points here are possibly shifted too far. Maybe the intensity is too high or something like that?
"""

# find points which are misclassified
with torch.no_grad():
    model.eval()

    incorrectly_classified = (model(shifted_x).argmax(dim=-1) != shifted_y).int()

confusion_matrix(incorrectly_classified, predictions)
