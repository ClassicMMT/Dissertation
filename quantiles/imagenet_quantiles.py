"""
This file explores the entropy quantile potential on the imagenet dataset.

Note:
    * Since imagenet takes a long time to run, the results of this file are saved
        - see the block below for loading the data
"""

import torch
import numpy as np
import pandas as pd
from src.utils import calculate_entropy, load_resnet50, set_all_seeds
from src.datasets import load_imagenet
import matplotlib.pyplot as plt


random_state = 123
g = set_all_seeds(random_state)
device = torch.device("mps")

# load model and data
model = load_resnet50(device=device)
(val_loader, test_loader), _ = load_imagenet(batch_size=128, generator=g)


# calculate entropies on the validation set
with torch.no_grad():
    model.eval()

    val_entropies = []
    val_is_correct = []

    for i, (features, labels) in enumerate(val_loader):
        print(f"Batch: {i+1}/{len(val_loader)}")
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)
        entropies = calculate_entropy(logits)

        is_correct = model(features).argmax(dim=-1) == labels

        val_entropies.append(entropies)
        val_is_correct.append(is_correct)

    val_entropies = torch.cat(val_entropies)
    val_is_correct = torch.cat(val_is_correct)


# plot entropy distribution
plt.hist(val_entropies)
plt.show()


# calculate q_hat
alpha = 0.05
q_hat = torch.quantile(val_entropies, 1 - alpha, interpolation="higher")


# calculate accuracy of examples split by entropy > q_hat

with torch.no_grad():
    model.eval()

    test_is_correct = []
    test_entropies = []

    for i, (features, labels) in enumerate(val_loader):
        print(f"Batch: {i+1}/{len(val_loader)}")
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)

        entropies = calculate_entropy(logits)
        is_correct = model(features).argmax(dim=-1) == labels

        test_is_correct.append(is_correct)
        test_entropies.append(entropies)

    test_is_correct = torch.cat(test_is_correct)
    test_entropies = torch.cat(test_entropies)


############################### Saving/Loading results ###############################

# note that the ordering of the data may be different each time due to the data loader
pd.DataFrame(
    {
        # entropies are the entropies calculated from the validation set
        "entropy": val_entropies.cpu().numpy(),
        # whether the observation was correct
        "is_correct": val_is_correct.cpu().numpy(),
        # q_hat is here so we can save it
        "q_hat": np.repeat(q_hat.item(), len(val_entropies)),
    }
).to_csv("saved_results/imagenet_quantiles_val.csv", index=False)

pd.DataFrame(
    {
        # entropies are the entropies calculated from the validation set
        "entropy": test_entropies.cpu().numpy(),
        # whether the observation was correct
        "is_correct": test_is_correct.cpu().numpy(),
        # q_hat is here so we can save it
        "q_hat": np.repeat(q_hat.item(), len(val_entropies)),
    }
).to_csv("saved_results/imagenet_quantiles_test.csv", index=False)

##### val data
if True:
    data = pd.read_csv("saved_results/imagenet_quantiles_val.csv")
    plt.hist(data["entropy"], bins=50)
    plt.axvline(data["q_hat"].unique(), color="red")
    plt.show()

###### test data
data = pd.read_csv("saved_results/imagenet_quantiles_test.csv")

test_entropies = data["entropy"]
is_correct = data["is_correct"]
q_hat = data["q_hat"].iloc[0]
uncertain = test_entropies >= q_hat

# results
is_correct[uncertain].mean()
is_correct[~uncertain].mean()

if True:
    plt.hist(data["entropy"], bins=50)
    plt.axvline(data["q_hat"].unique(), color="red")
    plt.show()
