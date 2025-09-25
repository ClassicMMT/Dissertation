"""
This experiment varies intensity and the number of features to induce a covariate shift in
to see the impact of this covariate shift
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from src.models import SpamBaseNet
from src.utils import calculate_entropy, evaluate_model, load_model, set_all_seeds
from src.datasets import induce_covariate_shift, load_spambase, scale_datasets, create_loaders

random_state = 123
g = set_all_seeds(random_state)
device = torch.device("mps")

# load model
model = load_model(SpamBaseNet(), "spambase").to(device)
model.eval()

# load data
X_train, X_test, y_train, y_test = load_spambase(return_raw=True)

# scale the training set
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

# parameters
intensities = [0.5, 1, 2, 3, 4, 5, 10, 20]
n_features = [5, 10, 20, 30, 40, 50, 57]
# intensity = 2.0
# n_features_to_shift = 57


results = {"intensity": [], "features_shifted": [], "is_correct": [], "entropy": []}

for intensity in intensities:
    for n_features_to_shift in n_features:

        # induce covariate shift
        X_test_shifted = induce_covariate_shift(X_test, n_features_to_shift=n_features_to_shift, intensity=intensity)

        # scale the data according to the training data
        X_test_shifted = scaler.transform(X_test_shifted)

        test_loader, test_dataset = create_loaders(X_test_shifted, y_test, batch_size=128, generator=g)

        for features, labels in test_loader:
            features = features.to(device)

            # get logits
            logits = model(features).cpu()

            # calculate entropies for each point
            entropy = calculate_entropy(logits)

            # calculate which points are correct
            is_correct = logits.argmax(dim=-1) == labels

            # save results
            results["intensity"].append(np.repeat(intensity, len(labels)))
            results["features_shifted"].append(np.repeat(n_features_to_shift, len(labels)))
            results["entropy"].append(entropy)
            results["is_correct"].append(is_correct)

# combine results
results["intensity"] = np.concat(results["intensity"])
results["features_shifted"] = np.concat(results["features_shifted"])
results["entropy"] = torch.cat(results["entropy"]).numpy()
results["is_correct"] = torch.cat(results["is_correct"]).numpy()

# save results
data = pd.DataFrame(results)
data.to_csv("saved_results/covariate_shift_entropy_exploration.csv", index=False)
