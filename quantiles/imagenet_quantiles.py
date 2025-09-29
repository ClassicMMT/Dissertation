"""
This file explores the entropy quantile potential on the imagenet dataset.

Note:
    * Since imagenet takes a long time to run, the results of this file are saved
        - see the block below for loading the data
"""

import torch
import numpy as np
import pandas as pd
from src.utils import calculate_entropies_from_loader, load_resnet50, set_all_seeds, calculate_entropy
from src.datasets import load_imagenet
from src.attacks import attack_model
import matplotlib.pyplot as plt
import foolbox as fb


random_state = 123
g = set_all_seeds(random_state)
device = torch.device("mps")

# load model and data
model = load_resnet50(device=device)
(val_loader, test_loader), _ = load_imagenet(batch_size=128, generator=g)

# calculate entropies on the validation set
val_entropies, val_is_correct = calculate_entropies_from_loader(model, val_loader, device=device, verbose=True)

# plot entropy distribution
plt.hist(val_entropies)
plt.show()

# calculate q_hat
alpha = 0.1
q_hat = torch.quantile(val_entropies, 1 - alpha, interpolation="higher")

# calculate accuracy of examples split by entropy > q_hat
test_entropies, test_is_correct = calculate_entropies_from_loader(model, test_loader, device=device, verbose=True)


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
    q_hat = data["q_hat"].iloc[0]
    plt.hist(data["entropy"], bins=50)
    plt.axvline(q_hat, color="red")
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

################ The relationship between q_hat and adversarial examples

attacks = {
    # "fgsm": fb.attacks.FGSM(),
    "fgsm": fb.attacks.fast_gradient_method.L2FastGradientAttack(),
    "bim": fb.attacks.basic_iterative_method.L2AdamBasicIterativeAttack(),
    # "deepfool": fb.attacks.deepfool.L2DeepFoolAttack(),
}
results = {}
examples = {}
original_examples = {}
original_labels = {}
entropy_flags = {}

features, labels = next(iter(test_loader))
fmodel = fb.models.pytorch.PyTorchModel(model.eval(), bounds=(0, 1), device=device)

model = model.eval()
for name, attack in attacks.items():
    print(name)

    features, labels = features.to(device), labels.to(device)

    # see if the predictions were originally correct
    predictions = model(features).argmax(dim=-1)
    is_originally_correct = predictions == labels

    _, clipped, is_adv = attack(fmodel, features, labels, epsilons=0.1)

    # extract only those points which were initially predicted successfully
    # but now are adversarial
    adversarial_examples = clipped[is_originally_correct & is_adv]

    if adversarial_examples.shape[0] > 0:
        # calculate entropy
        logits = model(adversarial_examples)
        adversarial_entropies = calculate_entropy(logits)

        # save results
        examples[name] = adversarial_examples.cpu()
        results[name] = adversarial_entropies.cpu()
        entropy_flags[name] = (adversarial_entropies >= q_hat).cpu()
        original_examples[name] = features[is_originally_correct & is_adv].cpu()
        original_labels[name] = labels[is_originally_correct & is_adv].cpu()


for name, flag in entropy_flags.items():
    percent_over_q_hat = flag.float().mean()
    print(f"{name}: {percent_over_q_hat.item():.4f}")

if True:
    attack = "bim"
    index = 1
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(examples[attack][index].permute(1, 2, 0).cpu().numpy())
    axs[1].imshow(original_examples[attack][index].permute(1, 2, 0).cpu().numpy())
    axs[2].hist(results[attack])
    axs[2].axvline(q_hat, color="red")
    axs[2].set_title("Entropy Distribution (Adversarial)")
    axs[3].hist(data["entropy"])
    axs[3].axvline(q_hat, color="red")
    axs[3].set_title("Entropy Distribution (Validation)")
    plt.show()
