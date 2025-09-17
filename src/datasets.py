######################### Dataset Loading Utility Functions #########################


def load_mnist(batch_size=128):

    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    from torchvision.datasets import MNIST

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = MNIST(root="data/", transform=transform, download=False)
    test_dataset = MNIST(root="data/", transform=transform, download=False, train=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return (train_loader, test_loader), (train_dataset, test_dataset)


def create_loaders(x, y, batch_size, shuffle=True, generator=None):
    """
    Creates a PyTorch dataloader and dataset from the provided data.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if isinstance(x, torch.Tensor):
        x = x.clone().detach()
    else:
        x = torch.tensor(x, dtype=torch.float32)

    if isinstance(y, torch.Tensor):
        y = y.clone().detach()
    else:
        y = torch.tensor(y)

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)

    return loader, dataset


def load_spambase(
    batch_size: int = 128,
    test_size: float = 0.1,
    scale: bool = True,
    # covariate shift arguments
    induce_test_covariate_shift: bool = False,
    covariate_shift_intensity: float = 2.0,
    covariate_shift_n_features: int = 57,
    random_state: int = 123,
):
    """
    Function to load the spambase data.

    See: https://archive.ics.uci.edu/dataset/94/spambase

    Notes:
        * covariate_shift_intensity and covariate_shift_n_features do not do anything unless
            induce_test_covariate_shift=True.
        * covariate_shift_intensity controls the the intensity of the covariate shift
        * covariate_shift_n_features controls the number of features to induce shift in
    """

    import torch
    from sklearn.model_selection import train_test_split
    from ucimlrepo import fetch_ucirepo
    from src.utils import set_all_seeds

    g = set_all_seeds(random_state)

    spambase = fetch_ucirepo(id=94)
    X, y = spambase.data.features.to_numpy(), spambase.data.targets["Class"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if induce_test_covariate_shift:
        X_test = induce_covariate_shift(
            X_test,
            n_features_to_shift=covariate_shift_n_features,
            intensity=covariate_shift_intensity,
            random_state=random_state,
        )

    if scale:
        X_train, X_test = scale_datasets(X_train, X_test)

    train_loader, train_dataset = create_loaders(X_train, y_train, batch_size=batch_size, generator=g)
    test_loader, test_dataset = create_loaders(X_test, y_test, batch_size=batch_size, generator=g)

    return (train_loader, test_loader), (train_dataset, test_dataset)


def load_heloc(
    batch_size=128,
    scale=True,
    directory="data/",
    random_split=False,
    random_state=123,
    size=0.2,
):
    """
    Function to load the HELOC dataset.

    The if random_split=False, there is a distribution shift as outlined here:
        https://tableshift.org/datasets.html

    If random_split=True, then a random sample is taken based on the random_state and size
    """

    import torch
    from torch.utils.data import TensorDataset, DataLoader
    import os
    import pandas as pd
    from src.utils import set_all_seeds

    assert "heloc_dataset_v1.csv" in os.listdir(directory), (
        "Dataset not found. Download here:\n"
        + "https://www.kaggle.com/datasets/averkiyoliabev/home-equity-line-of-creditheloc"
    )

    data = pd.read_csv(directory + "heloc_dataset_v1.csv")

    if not random_split:
        # Train/test split done the same way as in the tableshift repo.
        # see this link: https://tableshift.org/datasets.html
        train_indicator = data["ExternalRiskEstimate"] > 63

        train = data[train_indicator]
        test = data[~train_indicator]
    else:
        # random split
        g = set_all_seeds(random_state)
        n = len(data)
        if 0 < size < 1:  # proportion
            size = int(size * n)
        test_indices = torch.randperm(n, generator=g)[:size]
        train = data.drop(test_indices)
        test = data.iloc[test_indices]

    X_train = train.drop("RiskPerformance", axis=1)
    X_test = test.drop("RiskPerformance", axis=1)

    if scale:
        X_train, X_test = scale_datasets(X_train, X_test)

    # convert response to numeric
    y_train = train["RiskPerformance"]
    y_train = y_train.apply(lambda risk: 0 if risk == "Bad" else 1).to_numpy()
    y_test = test["RiskPerformance"]
    y_test = y_test.apply(lambda risk: 0 if risk == "Bad" else 1).to_numpy()

    # The data was initially integer data, so may need to take that into consideration
    train_loader, train_dataset = create_loaders(X_train, y_train, batch_size=batch_size)
    test_loader, test_dataset = create_loaders(X_test, y_test, batch_size=batch_size)

    return (train_loader, test_loader), (train_dataset, test_dataset)


def load_iris(batch_size=120, scale=True, test_size=0.2, random_state=123):
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from src.utils import set_all_seeds

    set_all_seeds(random_state)

    data = load_iris()
    features, labels = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )

    if scale:
        X_train, X_test = scale_datasets(X_train, X_test)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return (train_loader, test_loader), (train_dataset, test_dataset)


def load_blood_transfusion(batch_size=100, scale=True, test_size=0.2, random_state=123):
    """
    https://archive.ics.uci.edu/dataset/176/blood+transfusion+service+center
    """

    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.model_selection import train_test_split
    from src.utils import set_all_seeds
    from ucimlrepo import fetch_ucirepo

    set_all_seeds(random_state)

    blood_transfusion = fetch_ucirepo(id=176)
    X = blood_transfusion.data.features.to_numpy()
    y = blood_transfusion.data.targets.to_numpy().reshape(-1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    if scale:
        X_train, X_test = scale_datasets(X_train, X_test)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return (train_loader, test_loader), (train_dataset, test_dataset)


######################### Dataset Manipulation Utility Functions #########################


def induce_covariate_shift(
    X,
    n_features_to_shift: int = None,
    intensity: float = 0.5,
    random_state: int = 123,
):
    """
    Induces a covariate shift on a test_dataset.

    Note:
        * if using outside of a loading function, remember to set scale=False in the loading function.
        * X must be a PyTorch tensor
    """

    import torch
    from src.utils import set_all_seeds

    g = set_all_seeds(random_state)

    X_shifted = X.clone().detach()
    n_features = X.shape[1]

    if n_features_to_shift is None:
        n_features_to_shift = n_features

    # can't shift more features than we have
    n_features_to_shift = min(n_features, n_features_to_shift)

    # randomly select the features to shift
    features_to_shift = torch.randperm(n_features, generator=g)[:n_features_to_shift]

    # randomly select which features will have additive or multiplicative shift induced
    additive_multiplicative_indicator = torch.randint(0, 2, (n_features_to_shift,), generator=g)

    for feature_index, add_or_mul in zip(features_to_shift, additive_multiplicative_indicator):
        # additive shift
        if add_or_mul == 0:
            shift_amount = intensity * torch.std(X[:, feature_index])
            X_shifted[:, feature_index] += shift_amount

        # multiplicative shift
        else:
            X_shifted[:, feature_index] *= 1 + intensity

    return X_shifted


def scale_datasets(data, *args):
    """
    Scales one or more datasets.

    The first argument is used to fit the scaler.
    """

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    args = [scaler.transform(arg) for arg in args]
    if len(args) == 0:
        return data
    return [data] + args


######################### Dataset Creating Functions #########################


def make_chessboard(n_blocks=5, n_points_in_block=100, variance=0.05, scale=True, random_state=123):
    from src.utils import set_all_seeds
    from src.datasets import scale_datasets
    import torch

    g = set_all_seeds(random_state)

    x = []
    y = []

    for row in range(n_blocks):
        y_center = row + 0.5
        for col in range(n_blocks):
            x_center = col + 0.5

            # generate random data
            x_temp = torch.randn((n_points_in_block, 2), generator=g)

            # change variance and make x-mean x_center
            x_temp[:, 0] = x_temp[:, 0] * (variance**0.5) + x_center
            # change variance and make y-mean y_center
            x_temp[:, 1] = x_temp[:, 1] * (variance**0.5) + y_center

            # class label
            y_temp = torch.ones(n_points_in_block) * ((row + col) % 2)

            x.append(x_temp)
            y.append(y_temp)

    x, y = torch.cat(x), torch.cat(y)

    if scale:
        x = torch.tensor(scale_datasets(x), dtype=torch.float32)

    return x, y
