def attack_model(
    model, attack, loader, verbose=True, device="mps", epsilons=0.01, bounds=(0, 1)
):
    """
    Returns a tuple containing the adversarial examples and their original labels
    """

    import foolbox as fb
    import torch

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

    import torch

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

    import torch
    import os

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


def load_all_adversarial_examples(
    model_name: str = "",
    return_attack_and_epsilon: bool = False,
    include_attacks=None,
    exclude_attacks=None,
    include_epsilons=None,
    exclude_epsilons=None,
    device="cpu",
    load_directory: str = "saved_tensors/",
):
    """
    Loads all pre-generated adversarial examples for a single model.

    Returns (adversarial_examples, original_labels)

    If return_attack_and_epsilon=True:
        returns (adversarial_examples, original_labels, attack_names, epsilons)
    """

    import os
    import numpy as np
    from src.utils import drop_duplicates
    import torch

    relevant_filenames = [
        filename.split("-")[:3]
        for filename in os.listdir(load_directory)
        if filename.split("-")[0] == model_name
    ]

    attacks_to_include = np.unique(
        np.array([attack_name for _, attack_name, _ in relevant_filenames])
    )
    epsilons_to_include = np.unique(
        np.array([float(epsilon.split("=")[1]) for _, _, epsilon in relevant_filenames])
    )

    # Filter
    if include_attacks is not None:
        attacks_to_include = include_attacks

    if exclude_attacks is not None:
        attacks_to_include = np.setdiff1d(attacks_to_include, exclude_attacks)

    if include_epsilons is not None:
        epsilons_to_include = include_epsilons

    if exclude_epsilons is not None:
        epsilons_to_include = np.setdiff1d(epsilons_to_include, exclude_epsilons)

    adversarial_examples = []
    original_labels = []

    if return_attack_and_epsilon:
        attack_identifiers = []
        epsilons = []

    for model_name, attack_name, epsilon in relevant_filenames:
        epsilon = float(epsilon.split("=")[1])

        if epsilon in epsilons_to_include and attack_name in attacks_to_include:

            adv_examples, labels = load_adversarial_examples(
                model_name, attack_name, epsilon, load_directory=load_directory
            )
            adversarial_examples.append(adv_examples)
            original_labels.append(labels)

            if return_attack_and_epsilon:
                attack_name_vector = np.repeat(attack_name, len(adv_examples))
                epsilon_vector = np.repeat(epsilon, len(adv_examples))
                attack_identifiers.append(attack_name_vector)
                epsilons.append(epsilon_vector)

    adversarial_examples = torch.cat(adversarial_examples).to(device)
    original_labels = torch.cat(original_labels).to(device)

    if return_attack_and_epsilon:
        attack_identifiers = np.concat(attack_identifiers)
        epsilons = np.concat(epsilons)
        return drop_duplicates(
            adversarial_examples, original_labels, attack_identifiers, epsilons
        )

    return drop_duplicates(adversarial_examples, original_labels)
