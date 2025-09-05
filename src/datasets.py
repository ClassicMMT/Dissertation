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


def load_spambase(
    batch_size: int = 128,
    test_size: float = 0.1,
    scale: bool = True,
    induce_test_covariate_shift: bool = False,
    random_state: int = 123,
):
    """
    Function to load the spambase data.

    See: https://archive.ics.uci.edu/dataset/94/spambase
    """

    import torch
    from sklearn.model_selection import train_test_split
    from torch.utils.data import TensorDataset, DataLoader
    from ucimlrepo import fetch_ucirepo

    spambase = fetch_ucirepo(id=94)
    X, y = spambase.data.features.to_numpy(), spambase.data.targets["Class"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if induce_test_covariate_shift:
        X_test = induce_covariate_shift(X_test, return_tensor_only=True)

    if scale:
        X_train, X_test = scale_datasets(X_train, X_test)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return (train_loader, test_loader), (train_dataset, test_dataset)


def load_heloc(batch_size=128, scale=True, directory="data/"):

    import torch
    from torch.utils.data import TensorDataset, DataLoader
    import os
    import pandas as pd

    assert "heloc_dataset_v1.csv" in os.listdir(directory), (
        "Dataset not found. Download here:\n"
        + "https://www.kaggle.com/datasets/averkiyoliabev/home-equity-line-of-creditheloc"
    )

    data = pd.read_csv(directory + "heloc_dataset_v1.csv")

    # Train/test split done the same way as in the tableshift repo.
    # see this link: https://tableshift.org/datasets.html
    train_indicator = data["ExternalRiskEstimate"] > 63

    train = data[train_indicator]
    test = data[~train_indicator]

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
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return (train_loader, test_loader), (train_dataset, test_dataset)


######################### Dataset Manipulation Utility Functions #########################


def induce_covariate_shift(
    test_dataset,
    n_features_to_shift: int = 10,
    intensity: float = 0.5,
    return_tensor_only: bool = False,
    batch_size: int = 128,
    random_state: int = 123,
):
    """
    Induces a covariate shift on a test_dataset.

    Note: if using outside of a loading function, remember to set scale=False in the loading function.
    """

    import torch
    from src.utils import set_all_seeds

    g = set_all_seeds(random_state)

    X, y = test_dataset.tensors
    X_shifted = X.clone()
    n_features = X.shape[1]

    n_features_to_shift = min(n_features, n_features_to_shift)

    features_to_shift = torch.randperm(n_features, generator=g)[:n_features_to_shift]
    additive_multiplicative_indicator = (
        torch.randperm(n_features, generator=g)[:n_features_to_shift] % 2
    )

    for feature_index, add_or_mul in zip(
        features_to_shift, additive_multiplicative_indicator
    ):
        # additive shift
        if add_or_mul == 0:
            shift_amount = intensity * torch.std(X[:, feature_index])
            X_shifted[:, feature_index] += shift_amount

        # multiplicative shift
        else:
            X_shifted[:, feature_index] *= 1 + intensity

    if return_tensor_only:
        return X_shifted

    from torch.utils.data import TensorDataset, DataLoader

    test_dataset = TensorDataset(X_shifted, y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return test_loader, test_dataset


def scale_datasets(data, *args):
    """
    Scales one or more datasets.

    The first argument is used to fit the scaler.
    """

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    args = [scaler.transform(arg) for arg in args]
    return [data] + args
