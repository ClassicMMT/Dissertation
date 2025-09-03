import foolbox as fb
import torch
import os


def attack_model(
    model, attack, loader, verbose=True, device="mps", epsilons=0.01, bounds=(0, 1)
):
    """
    Returns a tuple containing the adversarial examples and their original labels
    """
    model.eval()
    model = fb.PyTorchModel(model, bounds=bounds, device=device)
    adversarial_examples, original_labels = [], []
    for i, (features, labels) in enumerate(loader):
        features, labels = features.to(device), labels.to(device)

        predictions = model(features).argmax(dim=-1)
        is_originally_correct = predictions == labels

        raw, clipped, is_adv = attack(model, features, labels, epsilons=epsilons)

        # extract only those points which were initially predicted successfully
        # but now are adversarial
        clipped = clipped[is_originally_correct & is_adv]
        labels = labels[is_originally_correct & is_adv]

        adversarial_examples.append(clipped)
        original_labels.append(labels)

        if verbose and i % 10 == 0:
            print(f"Attack Progress: {i/len(loader)*100:.1f}%")

    return torch.cat(adversarial_examples), torch.cat(original_labels)


def run_multiple_attacks(model, attacks, loader, epsilon, verbose=True, device="mps"):
    """
    Function to run multiple attacks.

    model: PyTorch model
    attacks: list | tuple containing foolbox attacks
    loader: PyTorch data loader
    epsilon: attack strength

    returns three tensors:
        1. Adversarial examples
        2. Original class labels (NOT adversarial)
        3. Index of the attack the example is from

    Note: Only returns successful adversarial examples.
    Also note: returns tensors on CPU
    """
    all_examples = []
    all_labels = []
    attack_index = []

    for i, attack in enumerate(attacks):
        if verbose:
            print(f"Running Attack {i+1}/{len(attacks)} with Epsilon={epsilon}")

        examples, original_labels = attack_model(
            model, attack, loader, epsilons=epsilon, verbose=False, device=device
        )
        index = torch.zeros(len(original_labels)) + i

        all_examples.append(examples)
        all_labels.append(original_labels)
        attack_index.append(index)

    return (
        torch.cat(all_examples).to("cpu"),
        torch.cat(all_labels).to("cpu"),
        torch.cat(attack_index).int().to("cpu"),
    )


def load_adversarial_examples(
    model_name: str = "",
    attack_name: str = "",
    epsilon: float = 0.0,
    load_directory="saved_tensors/",
):
    """
    Loads pre-generated adversarial examples.

    Returns a tuple containing the adversarial examples and their original labels
    """

    load_name = f"{model_name}-{attack_name}-epsilon={epsilon}-adv_examples.pt".lower()
    if load_name not in os.listdir(load_directory):
        pt_filenames = [
            filename
            for filename in os.listdir(load_directory)
            if filename[-3:] == ".pt"
        ]
        if len(pt_filenames) > 0:
            options = "\n\nAvailable Options:\n"
            for filename in sorted(pt_filenames):
                model_name, attack_name, epsilon, _ = filename.split("-")
                epsilon = epsilon.split("=")[1]
                options += f"{model_name}, {attack_name}, {epsilon}\n"
            raise FileNotFoundError("Examples Not Found." + options)
        else:
            raise FileNotFoundError("No Examples In This Directory.")
    tensors = torch.load(load_directory + load_name)
    return tensors["adversarial_examples"], tensors["original_labels"]
