######################### General Utility Functions #########################


def save_model(model, name):
    """
    Function to save a trained model.

    model: a PyTorch model inheriting from nn.Module
    name: the name to save the model as
    """

    import os
    import torch

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

    import torch

    if load_name[-3:] != ".pt":
        load_name += ".pt"
    state_dict = torch.load("trained_models/" + load_name)
    empty_instance.load_state_dict(state_dict)
    return empty_instance


def set_all_seeds(random_state=123):
    """
    Function to set all possible random seeds.
    """
    import random
    import numpy as np
    import torch

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

    import torch

    g = set_all_seeds(random_state)

    n = len(args[0])
    assert all(n == len(arg) for arg in args), "Provided args don't have the same length."
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

    import numpy as np

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


def normalised_mse(X, X2, variances):
    """
    Function to compute the normalised MSE.

    X should be the original examples since the variances are calculated from it.

    Note that variances are required to be passed in
    because the variances need to be computed from the training data
    """
    if len(X.shape) > 2:
        X = X.view(X.shape[0], -1)
        X2 = X2.view(X2.shape[0], -1)
    # variances = X.var(dim=0, unbiased=False)
    n_dim = X.shape[1]
    mse = (((X - X2) ** 2) / variances).mean(dim=1).mean()
    return mse.item()


def calculate_entropy(logits):
    """
    Computes the entropy for each example.

    Requires logits from the model.
    """
    import torch
    import torch.nn.functional as F

    logits = logits.clone().detach().cpu()
    if len(logits.shape) > 2:
        logits = logits.reshape(logits.shape[0], -1)
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
    return entropy


######################### Model Related Utility Functions #########################


def train_model(
    model,
    loader,
    criterion,
    optimizer,
    n_epochs,
    verbose=True,
    return_accuracy=False,
    device="mps",
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

    if return_accuracy:
        return model, (n_correct / n_total).item()
    return model


def evaluate_model(model, loader, device="mps") -> float:
    import torch

    with torch.no_grad():
        model.eval()
        model.to(device)

        n_correct, n_total = 0, 0
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            n_correct += (model(features).argmax(dim=-1) == labels).sum().item()
            n_total += len(labels)

        return n_correct / n_total


def plot_boundary(axs, model, x, y, device, title, point_size=None):
    """
    Function to plot the decision boundary of a network.

    Data must be 2 column tabular data.
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    xx, yy = np.meshgrid(
        np.linspace(x[:, 0].min() - 1, x[:, 0].max() + 1, 200),
        np.linspace(x[:, 1].min() - 1, x[:, 1].max() + 1, 200),
    )
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)
    with torch.no_grad():
        model.eval()
        labels = model(grid).argmax(dim=-1).cpu()
    Z = labels.reshape(xx.shape)
    axs.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    axs.scatter(x[:, 0], x[:, 1], c=y, edgecolors="k", cmap=plt.cm.coolwarm, s=point_size)
    axs.set_title(title)


def plot_loss_landscape(
    axs,
    model,
    x,
    y,
    device,
    title,
    loss_fn=None,
    add_colour_bar=False,
    colour_limits=(None, None),
    point_size=None,
):
    """
    Function to plot the loss landscape of a network.

    Data must be 2 column tabular data.

    colour_limits is a tuple of (vmin, vmax) for consistency across multiple plots

    Notes:
        * loss_fn needs to have reduction="none" so loss is returned per point
        * set add_colour_bar=True if you want colour bars on the plot
        * colour_limits is for making the colourbar consistent across multiple plots
            - however, the colour bar must be created separately.
            - see "nn_landscapes.py" for how this is done.
    """
    import torch
    import torch.nn.functional as F
    import numpy as np
    import matplotlib.pyplot as plt

    # Create a mesh grid
    xx, yy = np.meshgrid(
        np.linspace(x[:, 0].min() - 1, x[:, 0].max() + 1, 200),
        np.linspace(x[:, 1].min() - 1, x[:, 1].max() + 1, 200),
    )
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)

    with torch.no_grad():
        model.eval()
        outputs = model(grid)

        # Get predicted classes for the grid points
        predicted_classes = outputs.argmax(dim=-1)

        # Calculate loss for each point
        # Use the predicted class as the "true" label - shows confidence
        if loss_fn is None:
            # Default to cross-entropy
            losses = F.cross_entropy(outputs, predicted_classes, reduction="none")
        else:
            losses = loss_fn(outputs, predicted_classes)

    Z = losses.cpu().numpy().reshape(xx.shape)

    # Plot the loss landscape as a heatmap
    contour = axs.contourf(
        xx,
        yy,
        Z,
        levels=20,
        cmap="viridis",
        alpha=0.7,
        vmin=colour_limits[0],
        vmax=colour_limits[1],
    )

    axs.contour(xx, yy, Z, levels=10, colors="white", alpha=0.3, linewidths=0.5)

    # Plot the actual data points
    axs.scatter(
        x[:, 0],
        x[:, 1],
        c=y,
        edgecolors="k",
        cmap=plt.cm.coolwarm,
        s=point_size,
        zorder=5,
    )

    axs.set_title(title)

    if add_colour_bar:
        # Add a colorbar to show loss values
        plt.colorbar(contour, ax=axs, label="Loss")

    return contour


def plot_entropy_landscape(
    axs,
    model,
    x,
    y,
    device,
    title,
    add_colour_bar=False,
    colour_limits=(None, None),
    point_size=None,
    log_entropy=False,
):
    """
    Plot the entropy of predictions across the input space.
    High entropy = high uncertainty

    colour_limits is a tuple of (vmin, vmax) for consistency across multiple plots

    Notes:
        * loss_fn needs to have reduction="none" so loss is returned per point
        * set add_colour_bar=True if you want colour bars on the plot
        * colour_limits is for making the colourbar consistent across multiple plots
            - however, the colour bar must be created separately.
            - see "nn_landscapes.py" for how this is done.
        * log_entropy will compute the log entropies
            - useful if the entropies are all very small
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    xx, yy = np.meshgrid(
        np.linspace(x[:, 0].min() - 1, x[:, 0].max() + 1, 200),
        np.linspace(x[:, 1].min() - 1, x[:, 1].max() + 1, 200),
    )
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)

    with torch.no_grad():
        model.eval()
        outputs = model(grid)
        entropy = calculate_entropy(outputs)

    Z = entropy.cpu().numpy().reshape(xx.shape)
    if log_entropy:
        Z = np.log(Z + 1e-12)

    # Plot entropy landscape
    contour = axs.contourf(
        xx,
        yy,
        Z,
        levels=20,
        cmap="plasma",
        alpha=0.7,
        vmin=colour_limits[0],
        vmax=colour_limits[1],
    )
    axs.contour(xx, yy, Z, levels=10, colors="white", alpha=0.3, linewidths=0.5)

    # Plot data points
    axs.scatter(
        x[:, 0],
        x[:, 1],
        c=y,
        edgecolors="k",
        cmap=plt.cm.coolwarm,
        s=point_size,
        zorder=5,
    )

    axs.set_title(title)
    if add_colour_bar:
        plt.colorbar(contour, ax=axs, label="Entropy (Uncertainty)")

    return contour
