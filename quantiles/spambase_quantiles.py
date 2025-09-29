"""
This file explores the entropy quantile potential on the spambase dataset.
"""

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.attacks import attack_model
from src.datasets import create_loaders, load_spambase, scale_datasets
from src.models import SpamBaseNet
from src.utils import calculate_entropy, load_model, set_all_seeds
import matplotlib.pyplot as plt
import foolbox as fb


random_state = 123
g = set_all_seeds(random_state)
device = torch.device("mps")
batch_size = 128

# load model and data
model = load_model(SpamBaseNet(), "spambase").to(device)
x_train, x_test, y_train, y_test = load_spambase(return_raw=True, random_state=random_state)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, stratify=y_train, random_state=random_state)
x_train, x_val, x_test = scale_datasets(x_train, x_val, x_test)
train_loader, train_dataset = create_loaders(x_train, y_train, batch_size=batch_size, generator=g)
val_loader, val_dataset = create_loaders(x_val, y_val, batch_size=batch_size, generator=g)
test_loader, test_dataset = create_loaders(x_test, y_test, batch_size=batch_size, generator=g)

# calculate entropies on the validation set
with torch.no_grad():
    model.eval()

    val_entropies = []
    val_is_correct = []

    for i, (features, labels) in enumerate(val_loader):
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)
        entropies = calculate_entropy(logits)

        is_correct = model(features).argmax(dim=-1) == labels

        val_entropies.append(entropies)
        val_is_correct.append(is_correct)

    val_entropies = torch.cat(val_entropies)
    val_is_correct = torch.cat(val_is_correct)

# calculate q_hat
alpha = 0.05
q_hat = torch.quantile(val_entropies, 1 - alpha, interpolation="higher")

# plot entropy distribution
if True:
    plt.hist(val_entropies, bins=50)
    plt.axvline(q_hat, color="red")
    plt.show()

# calculate accuracy of examples split by entropy >= q_hat

with torch.no_grad():
    model.eval()

    test_is_correct = []
    test_entropies = []

    for i, (features, labels) in enumerate(val_loader):
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)

        entropies = calculate_entropy(logits)
        is_correct = model(features).argmax(dim=-1) == labels

        test_is_correct.append(is_correct)
        test_entropies.append(entropies)

    test_is_correct = torch.cat(test_is_correct)
    test_entropies = torch.cat(test_entropies)


uncertain = test_entropies >= q_hat
test_is_correct[uncertain].float().mean()
test_is_correct[~uncertain].float().mean()

test_is_correct.float().mean()


################ The relationship between q_hat and adversarial examples

attacks = {
    # "fgsm": fb.attacks.FGSM(),
    "fgsm": fb.attacks.fast_gradient_method.L1FastGradientAttack(),
    "bim": fb.attacks.basic_iterative_method.L2AdamBasicIterativeAttack(),
    "deepfool": fb.attacks.deepfool.L2DeepFoolAttack(),
}
results = {}
examples = {}
labels = {}
for name, attack in attacks.items():
    adversarial_examples, original_labels = attack_model(
        model, attack=attack, loader=train_loader, verbose=False, epsilons=0.01
    )
    examples[name] = adversarial_examples.cpu()

    with torch.no_grad():
        model.eval()
        logits = model(adversarial_examples)
        adversarial_entropies = calculate_entropy(logits)
        results[name] = adversarial_entropies.cpu()
        labels[name] = (adversarial_entropies >= q_hat).cpu()


for name, adversarial_entropies in results.items():
    percent_over_q_hat = (adversarial_entropies > q_hat).float().mean()
    print(f"{name}: {percent_over_q_hat.item():.4f}")
