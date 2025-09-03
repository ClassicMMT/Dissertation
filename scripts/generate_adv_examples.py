import numpy as np
import torch
import foolbox as fb
from src.attacks import attack_model
from src.datasets import load_spambase
from src.models import SpamBaseNet
from src.utils import load_model, set_all_seeds, drop_duplicates

device = torch.device("mps")
random_state = 123
set_all_seeds(random_state)

# For SpamBase
model_name = "spambase"
model = load_model(SpamBaseNet(), load_name=model_name)
(train_loader, test_loader), (train_dataset, test_dataset) = load_spambase()

attacks = {
    "FGSM": fb.attacks.fast_gradient_method.LinfFastGradientAttack(),
    "DF_L2": fb.attacks.deepfool.L2DeepFoolAttack(),
    "DF_Linf": fb.attacks.deepfool.LinfDeepFoolAttack(),
    "CW_L2": fb.attacks.carlini_wagner.L2CarliniWagnerAttack(),
    "BIM_Linf": fb.attacks.basic_iterative_method.LinfAdamBasicIterativeAttack(),
}

# epsilons = [0.01, 0.05, 0.1, 0.15, 0.2]
epsilons = np.linspace(0.001, 0.2, 50).round(3)

for attack_name, attack in attacks.items():
    for epsilon in epsilons:
        print(f"Generating examples for {attack_name} with epsilon={epsilon}")

        adversarial_examples, original_labels = attack_model(
            model,
            attack=attack,
            loader=train_loader,
            device=device,
            epsilons=epsilon,
            verbose=False,
        )

        # remove duplicates if there are any
        adversarial_examples, original_labels = drop_duplicates(
            adversarial_examples, original_labels
        )

        tensors = {
            "adversarial_examples": adversarial_examples,
            "original_labels": original_labels,
        }
        torch.save(
            tensors,
            f"saved_tensors/{model_name}-{attack_name}-epsilon={epsilon}-adv_examples.pt".lower(),
        )
