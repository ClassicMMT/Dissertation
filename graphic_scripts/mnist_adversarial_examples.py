"""
This script generates adversarial examples and creates a graphic including:

    * original examples
    * adversarial perturbations
    * adversarial examples
"""

import numpy as np
import torch
import foolbox as fb
import matplotlib.pyplot as plt
from src.attacks import attack_model
from src.datasets import load_heloc, load_mnist, load_spambase
from src.models import HelocNet, MNISTNet, SpamBaseNet
from src.utils import load_model, set_all_seeds, drop_duplicates

device = torch.device("mps")
random_state = 123
set_all_seeds(random_state)

# For MNIST
(train_loader, test_loader), (train_dataset, test_dataset) = load_mnist()
model_name = "mnist"
model = load_model(MNISTNet(), model_name)

attacks = {
    "FGSM": fb.attacks.fast_gradient_method.LinfFastGradientAttack(),
    # "FGSM": fb.attacks.fast_gradient_method.L2FastGradientAttack(),
    # "DeepFool": fb.attacks.deepfool.L2DeepFoolAttack(),
    "DeepFool": fb.attacks.deepfool.LinfDeepFoolAttack(),
    # "CW_L2": fb.attacks.carlini_wagner.L2CarliniWagnerAttack(),
    "BIM": fb.attacks.basic_iterative_method.LinfAdamBasicIterativeAttack(),
    # "BIM": fb.attacks.basic_iterative_method.L2AdamBasicIterativeAttack(),
    # "BIM_L1": fb.attacks.basic_iterative_method.L1AdamBasicIterativeAttack(),
}

epsilons = [0.03, 0.05, 0.1]

model.eval()
fmodel = fb.PyTorchModel(model, bounds=(0, 1), device=device)

results = {
    "FGSM": {0.03: [], 0.05: [], 0.1: []},
    "DeepFool": {0.03: [], 0.05: [], 0.1: []},
    "BIM": {0.03: [], 0.05: [], 0.1: []},
}

features, labels = next(iter(train_loader))
features, labels = features.to(device), labels.to(device)

for attack_name, attack in attacks.items():
    for epsilon in epsilons:

        raw, clipped, is_adv = attack(fmodel, features, labels, epsilons=epsilon)

        with torch.no_grad():
            original_preds = model(features).argmax(dim=-1)
            adv_preds = model(clipped).argmax(dim=-1)

        is_originally_correct = original_preds == labels
        is_true_attack = is_originally_correct & is_adv

        results[attack_name][epsilon] = (
            features[is_true_attack].cpu(),
            clipped[is_true_attack].cpu(),
            labels[is_true_attack].cpu(),
            adv_preds[is_true_attack].cpu(),
        )


original_example, adversarial_example, original_label, adversarial_label = results["FGSM"][0.10]
original_example.shape

epsilon = 0.03
indices = [1, 2, 0]


fig, axs = plt.subplots(3, 3, figsize=(10, 10))
for i, (attack, index) in enumerate(zip(results, indices)):
    original_example, adversarial_example, original_label, adversarial_label = results[attack][epsilon]
    noise = original_example - adversarial_example
    axs[0, i].imshow(original_example[index].view(28, 28), vmin=0, vmax=1)
    axs[1, i].imshow(noise[index].view(28, 28), vmin=-epsilon, vmax=epsilon)
    axs[2, i].imshow(adversarial_example[index].view(28, 28), vmin=0, vmax=1)

    axs[0, i].set_title(f"Original Label: {original_label[index]}")
    axs[1, i].set_title(f"{attack}")
    axs[2, i].set_title(f"Adversarial Label: {adversarial_label[index]}")

row_labels = ["Original Example", "Noise", "Adversarial Example"]
for row, label in enumerate(row_labels):
    axs[row, 0].set_ylabel(label, fontsize=12, rotation=90, labelpad=10)


plt.tight_layout()
plt.show()
