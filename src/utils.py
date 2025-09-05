import numpy as np
import random
import torch
import os


######################### General Utility Functions #########################


def save_model(model, name):
    if name[-3:] != ".pt":
        name += ".pt"
    if name in os.listdir("trained_models"):
        overwrite = input(f"Model '{name}' already exists. Overwrite? [y/n]: ") in [
            "y",
            "yes",
        ]
        if not overwrite:
            print("Model not saved.")
            return
    torch.save(model.state_dict(), "trained_models/" + name)


def load_model(empty_instance, load_name):
    """
    Function to load a saved model.

    empty_instance: initialised object of the relevant class
    load_name: the name of the model to load
    """
    if load_name[-3:] != ".pt":
        load_name += ".pt"
    state_dict = torch.load("trained_models/" + load_name)
    empty_instance.load_state_dict(state_dict)
    return empty_instance


def set_all_seeds(random_state=123):
    """
    Function to set all possible random seeds.
    """
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.use_deterministic_algorithms(True)
    g = torch.Generator().manual_seed(random_state)
    return g


def sample(*args, size: float | int = 0.2, random_state=123):
    """
    Samples a `size` proportion of points and returns the sampled points.
    """
    g = torch.Generator().manual_seed(random_state)
    n = len(args[0])
    assert all(
        n == len(arg) for arg in args
    ), "Provided args don't have the same length."
    if 0 < size < 1:
        size *= n
    size = int(size)
    indices = torch.randperm(n, generator=g)[:size]
    if len(args) == 1:
        return args[0][indices]
    return [arg[indices] for arg in args]


def drop_duplicates(data, *args, dim=0, preserve_order=True):
    """
    Drops duplicates across the given dimension.

    Supports multiple tensors.
    """
    n = len(data)
    assert all(len(arg) == n for arg in args), "All tensors must have the same length."

    np_data = data.to("cpu").numpy()

    _, indices = np.unique(np_data, axis=dim, return_index=True)
    if preserve_order:
        indices = np.sort(indices)

    data = data[indices]
    args = [arg[indices] for arg in args]

    if not args:
        return data
    return [data] + args


######################### Model Related Utility Functions #########################


def train_model(
    model, loader, criterion, optimizer, n_epochs, verbose=True, device="mps"
):
    model.train()
    model.to(device)
    for epoch in range(n_epochs):
        epoch_loss = 0
        n_correct, n_total = 0, 0
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

            # accuracy
            predictions = outputs.argmax(dim=1)
            n_correct += sum(predictions == labels)
            n_total += len(labels)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if verbose:
            print(
                f"Epoch: {epoch+1}, "
                + f"Epoch loss: {epoch_loss:.4f}, "
                + f"Avg loss: {epoch_loss/len(loader):.4f}, "
                + f"Accuracy: {n_correct / n_total:.4f}"
            )

    return model


def evaluate_model(model, loader, device="mps") -> float:
    with torch.no_grad():
        model.eval()
        model.to(device)

        n_correct, n_total = 0, 0
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            n_correct += (model(features).argmax(dim=-1) == labels).sum().item()
            n_total += len(labels)

        return n_correct / n_total
