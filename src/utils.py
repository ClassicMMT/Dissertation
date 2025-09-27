######################### Utility Classes #########################
import torch


class KNNDensity:
    """
    Computes the normalised density for the given points.

    The density is computed as follows.
        1. The distance between every point and its nearest n_neighbours is computed to find the scaling range
        2. Given a new (test) point, computes the distance and scales it

    The scaling is done using min-max scaling
    """

    def __init__(self, n_neighbours: int = 5):
        self.n_neighbours = n_neighbours

    def fit(self, x: torch.Tensor):
        """
        Computes the minimum and maximum scaling values for future scaling.
        """
        self.x = x
        distances = torch.tensor([self.compute_average_distance_to_neighbours(x_index=i) for i in range(len(self.x))])
        self.min_d = distances.min()
        self.max_d = distances.max()
        return self

    def predict(self, x: torch.Tensor):
        """
        Returns a tensor of shape (x.shape[0],) containing the scaled distances.

        x: a tensor of shape (batch_size, n_features)
        """

        assert x.shape[1] == self.x.shape[1], "x must have the same n_features as the training data"

        distances = torch.tensor([self.compute_average_distance_to_neighbours(point=point) for point in x])
        return 1 - (distances - self.min_d) / (self.max_d - self.min_d)

    def compute_average_distance_to_neighbours(self, x_index: int | None = None, point: torch.Tensor | None = None):
        """
        Computes the average distance to the nearest n_neighbours.

        Can compute the average for a given x_index or for a new point
        """
        if point is None and x_index is None or x_index is not None and point is not None:
            raise ValueError("Must pass one of x_index or point")

        if point is None:  # then use x_index
            point = self.x[x_index]
            k = self.n_neighbours + 1
        else:
            k = self.n_neighbours

        # compute all distances
        distances = ((self.x - point) ** 2).sum(dim=1)

        # compute largest k
        values, indices = torch.topk(distances, k=k, largest=False, sorted=True)

        if x_index is not None:
            # remove the point itself (if using x_index)
            values = values[indices != x_index]
            indices = indices[indices != x_index]

        # calculate average distance
        return values.mean().item()


class ApproxConvexHull:
    """
    Approximates a convex hull using random projections.
    Works well in high dimensions when the exact convex hull is infeasible.

    Code taken from chatgpt. Relevant citations:
        * ε-kernels for convex hull approximation
        * support function approximation via random directions
        * coresets for convex hulls
    """

    def __init__(self, n_directions=100):
        self.n_directions = n_directions
        self.max_proj = None
        self.min_proj = None
        self.directions = None

    def fit(self, x):
        """
        Fits the approximate convex hull on the given data.
        Args:
            x: (n_samples, n_features) tensor
        """
        import torch

        n_features = x.shape[1]
        # sample random directions (normalize for unit length)
        directions = torch.randn(self.n_directions, n_features, device=x.device)
        directions = directions / directions.norm(dim=1, keepdim=True)

        # project points onto directions
        projections = x @ directions.T

        # record min/max along each direction
        self.max_proj = projections.max(dim=0).values
        self.min_proj = projections.min(dim=0).values
        self.directions = directions

    def predict(self, x):
        """
        Returns True if the instance is outside the convex hull approximation,
        False otherwise.
        Args:
            x: (n_samples, n_features) tensor
        Returns:
            (n_samples,) boolean tensor
        """
        projections = x @ self.directions.T
        outside_min = projections < self.min_proj
        outside_max = projections > self.max_proj
        is_outside = outside_min | outside_max
        return is_outside.sum(dim=-1) > 0


class BoundingBox:
    """
    Fits a bounding box on the given data.
    The bounding box should be fitted on the training data,
    and predicted on the test/validation data.
    """

    def __init__(self):
        self.max = None
        self.min = None

    def fit(self, x):
        """
        Fits the bounding box on the given data.
        """
        self.max = x.max(dim=0).values
        self.min = x.min(dim=0).values

    def predict(self, x):
        """
        Returns True if the instance is outside the bounding box
        or returns False otherwise.
        returns a tensor of shape (x.shape[0],)
        """
        is_outside = (x < self.min) | (x > self.max)
        return is_outside.sum(dim=-1) > 0


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


def load_resnet50(device="mps"):
    """
    Loads the pre-trained ResNet50 model.
    """
    from torchvision.models import resnet50, ResNet50_Weights

    return resnet50(weights=ResNet50_Weights.DEFAULT).to(device)


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
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(random_state)
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


def calculate_entropy(logits, detach=True):
    """
    Computes the entropy for each example.

    Requires logits from the model.

    detach=False will allow backpropogation through this calculation.
    """
    import torch
    import torch.nn.functional as F

    if detach:
        logits = logits.clone().detach().cpu()
    if len(logits.shape) > 2:
        logits = logits.reshape(logits.shape[0], -1)
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
    return entropy


def calculate_entropies_from_loader(model, loader, device="mps", verbose=False):
    """
    Computes the entropy for all examples using the given model and dataloader.

    Returns: (entropies, is_correct)
    """
    with torch.no_grad():
        model.eval()

        result_entropies = []
        result_is_correct = []

        for i, (features, labels) in enumerate(loader):
            if verbose:
                print(f"Batch: {i+1}/{len(loader)}")
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            entropies = calculate_entropy(logits, detach=True)
            is_correct = model(features).argmax(dim=-1) == labels

            result_entropies.append(entropies)
            result_is_correct.append(is_correct)

        result_entropies = torch.cat(result_entropies)
        result_is_correct = torch.cat(result_is_correct)

    return result_entropies, result_is_correct


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


def evaluate_model(model, loader, device="mps", verbose=False) -> float:
    """
    Evaluates a model on the given dataloader.

    Returns the accuracy.
    """
    import torch

    with torch.no_grad():
        model.eval()
        model.to(device)

        n_correct, n_total = 0, 0
        n = len(loader)
        for i, (features, labels) in enumerate(loader):
            if verbose:
                print(f"Batch: {i+1}/{n}")
            features, labels = features.to(device), labels.to(device)
            n_correct += (model(features).argmax(dim=-1) == labels).sum().item()
            n_total += len(labels)

        return n_correct / n_total


######################### Landscape Plotting Utility Functions #########################


def plot_boundary(
    axs, model, x, y, device="mps", title="Boundary", point_size=None, plot_x=None, plot_y=None, colour_map: dict = None
):
    """
    Function to plot the decision boundary of a network.

    Data must be 2 column tabular data.

    Args:
        plot_x: points to plot
        plot_y: labels to plot (colours)
        colour_map: dict object (e.g. {0: "red", 1: "blue"}) including a mapping for all labels
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    # calculate padding
    x0_range = x[:, 0].max() - x[:, 0].min()
    x1_range = x[:, 1].max() - x[:, 1].min()
    x0_padding = x0_range * 0.05
    x1_padding = x1_range * 0.05

    # Create a mesh grid
    xx, yy = np.meshgrid(
        np.linspace(x[:, 0].min() - x0_padding, x[:, 0].max() + x0_padding, 200),
        np.linspace(x[:, 1].min() - x1_padding, x[:, 1].max() + x1_padding, 200),
    )
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)
    with torch.no_grad():
        model.eval()
        labels = model(grid).argmax(dim=-1).cpu()
    Z = labels.reshape(xx.shape)
    axs.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    if plot_x is None and plot_y is None:
        plot_x = x
        plot_y = y

    if isinstance(colour_map, dict):
        c = [colour_map[int(x)] for x in plot_y]
        cmap = None
    else:
        c = plot_y
        cmap = plt.cm.coolwarm
    axs.scatter(plot_x[:, 0], plot_x[:, 1], c=c, edgecolors="k", cmap=cmap, s=point_size)
    axs.set_title(title)


def plot_loss_landscape(
    axs,
    model,
    x,
    y,
    device="mps",
    title="Loss",
    loss_fn=None,
    add_colour_bar=False,
    colour_limits=(None, None),
    point_size=None,
    plot_x=None,
    plot_y=None,
    colour_map=None,
):
    """
    Function to plot the loss landscape of a network.

    Data must be 2 column tabular data.

    colour_limits is a tuple of (vmin, vmax) for consistency across multiple plots

    Args:
        plot_x: points to plot
        plot_y: labels to plot (colours)
        colour_map: dict object (e.g. {0: "red", 1: "blue"}) including a mapping for all labels

    Notes:
        * loss_fn needs to have reduction="none" so loss is returned per point
        * set add_colour_bar=True if you want colour bars on the plot
        * colour_limits is for making the colourbar consistent across multiple plots
            - however, the colour bar must be created separately.
            - see "nn_landscapes.py" for how this is done.
        * if not None, plot_x, plot_y are used for the scatterplot.
    """
    import torch
    import torch.nn.functional as F
    import numpy as np
    import matplotlib.pyplot as plt

    # calculate padding
    x0_range = x[:, 0].max() - x[:, 0].min()
    x1_range = x[:, 1].max() - x[:, 1].min()

    x0_padding = x0_range * 0.05
    x1_padding = x1_range * 0.05

    # Create a mesh grid
    xx, yy = np.meshgrid(
        np.linspace(x[:, 0].min() - x0_padding, x[:, 0].max() + x0_padding, 200),
        np.linspace(x[:, 1].min() - x1_padding, x[:, 1].max() + x1_padding, 200),
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
    if plot_x is None and plot_y is None:
        plot_x = x
        plot_y = y

    if isinstance(colour_map, dict):
        c = [colour_map[int(x)] for x in plot_y]
        cmap = None
    else:
        c = plot_y
        cmap = plt.cm.coolwarm

    axs.scatter(
        plot_x[:, 0],
        plot_x[:, 1],
        c=c,
        edgecolors="k",
        cmap=cmap,
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
    device="mps",
    title="Entropy",
    add_colour_bar=False,
    colour_limits=(None, None),
    point_size=None,
    log_entropy=False,
    plot_x=None,
    plot_y=None,
    colour_map=None,
):
    """
    Plot the entropy of predictions across the input space.
    High entropy = high uncertainty

    Args:
        colour_limits: a tuple of (vmin, vmax) for consistency across multiple plots
        plot_x: points to plot
        plot_y: labels to plot (colours)
        colour_map: dict object (e.g. {0: "red", 1: "blue"}) including a mapping for all labels

    Notes:
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

    x0_range = x[:, 0].max() - x[:, 0].min()
    x1_range = x[:, 1].max() - x[:, 1].min()

    x0_padding = x0_range * 0.05
    x1_padding = x1_range * 0.05

    # Create a mesh grid
    xx, yy = np.meshgrid(
        np.linspace(x[:, 0].min() - x0_padding, x[:, 0].max() + x0_padding, 200),
        np.linspace(x[:, 1].min() - x1_padding, x[:, 1].max() + x1_padding, 200),
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
    if plot_x is None and plot_y is None:
        plot_x = x
        plot_y = y

    if isinstance(colour_map, dict):
        c = [colour_map[int(x)] for x in plot_y]
        cmap = None
    else:
        c = plot_y
        cmap = plt.cm.coolwarm

    axs.scatter(
        plot_x[:, 0],
        plot_x[:, 1],
        c=c,
        edgecolors="k",
        cmap=cmap,
        s=point_size,
        zorder=5,
    )

    axs.set_title(title)
    if add_colour_bar:
        plt.colorbar(contour, ax=axs, label="Entropy (Uncertainty)")

    return contour


def plot_density_landscape(
    axs,
    x,
    y,
    title="Density",
    n_neighbours=5,
    add_colour_bar=False,
    colour_limits=(0, 1),
    point_size=None,
    plot_x=None,
    plot_y=None,
    colour_map=None,
):
    """
    Plot the density of points across the training space.
    high values = more density

    Args:
        colour_limits: a tuple of (vmin, vmax) for consistency across multiple plots
        plot_x: points to plot
        plot_y: labels to plot (colours)
        colour_map: dict object (e.g. {0: "red", 1: "blue"}) including a mapping for all labels

    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    x0_range = x[:, 0].max() - x[:, 0].min()
    x1_range = x[:, 1].max() - x[:, 1].min()

    x0_padding = x0_range * 0.05
    x1_padding = x1_range * 0.05

    # Create a mesh grid
    xx, yy = np.meshgrid(
        np.linspace(x[:, 0].min() - x0_padding, x[:, 0].max() + x0_padding, 200),
        np.linspace(x[:, 1].min() - x1_padding, x[:, 1].max() + x1_padding, 200),
    )
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    knndensity = KNNDensity(n_neighbours=n_neighbours)
    knndensity = knndensity.fit(x)
    densities = knndensity.predict(grid)

    Z = densities.numpy().reshape(xx.shape)

    # Plot density landscape
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
    if plot_x is None and plot_y is None:
        plot_x = x
        plot_y = y

    if isinstance(colour_map, dict):
        c = [colour_map[int(x)] for x in plot_y]
        cmap = None
    else:
        c = plot_y
        cmap = plt.cm.coolwarm

    axs.scatter(
        plot_x[:, 0],
        plot_x[:, 1],
        c=c,
        edgecolors="k",
        cmap=cmap,
        s=point_size,
        zorder=5,
    )

    axs.set_title(title)
    if add_colour_bar:
        plt.colorbar(contour, ax=axs, label="Density")

    return contour
