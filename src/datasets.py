######################### Dataset Loading Utility Functions #########################


def load_coco(batch_size=128, generator=None, val_fraction=0.5):
    """
    Loads COCO 2017 validation images and splits them into a validation and test set.
    Each image is represented by one or more category labels (multi-label classification).

    Args:
        val_fraction: Fraction of the data to use for validation (rest for test).

    Returns:
        (val_loader, test_loader), (val_dataset, test_dataset)

    Notes:
        * requires pycocotools (pip install pycocotools)
    """
    import torch
    from torchvision import transforms
    from torchvision.datasets import CocoDetection
    from torch.utils.data import DataLoader, random_split

    root = "data/coco/val2017"
    annFile = "data/coco/annotations/instances_val2017.json"

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    class CocoClassification(CocoDetection):
        """Treats COCO images as multi-label classification samples."""

        def __init__(self, root, annFile, transform=None):
            super().__init__(root, annFile)
            self.transform = transform
            self.cat_ids = sorted(self.coco.getCatIds())
            self.cat_id_to_idx = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        def __getitem__(self, index):
            img, target = super().__getitem__(index)
            if self.transform is not None:
                img = self.transform(img)

            # Build binary multi-label vector for this image
            labels = torch.zeros(len(self.cat_ids), dtype=torch.float32)
            for ann in target:
                cat_id = ann["category_id"]
                labels[self.cat_id_to_idx[cat_id]] = 1.0

            return img, labels

    full_dataset = CocoClassification(root, annFile, transform=transform)

    n_total = len(full_dataset)
    n_val = int(val_fraction * n_total)
    n_test = n_total - n_val
    val_dataset, test_dataset = random_split(full_dataset, [n_val, n_test], generator=generator)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, generator=generator)

    return (val_loader, test_loader), (val_dataset, test_dataset)


def load_imagenet(batch_size=128, generator=None):
    """
    Loads the validation and test set.
    Returns: (val_loader, test_loader), (val_dataset, test_dataset)

    Notes:
        * Training data is NOT downloaded.
        * The validation split is split into validation and test sets
            so both are labelled.

    """
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    val_dataset = datasets.ImageFolder("data/imagenet/val_split", transform=transform)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    test_dataset = datasets.ImageFolder("data/imagenet/test_split", transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    return (val_loader, test_loader), (val_dataset, test_dataset)


def load_mnist(batch_size=128, generator=None):
    """
    Function to load the mnist data.

    Returns (train_loader, test_loader), (train_dataset, test_dataset)
    """

    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    from torchvision.datasets import MNIST

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = MNIST(root="data/", transform=transform, download=False)
    test_dataset = MNIST(root="data/", transform=transform, download=False, train=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return (train_loader, test_loader), (train_dataset, test_dataset)


def load_cifar(batch_size=128, generator=None, download=True):
    """
    Function to load the CIFAR data.

    Returns (val_loader, test_loader), (val_dataset, test_dataset)
    """

    from torch.utils.data import DataLoader, random_split
    import torchvision.transforms as transforms
    from torchvision.datasets import CIFAR10

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])]
    )
    dataset = CIFAR10(root="data/", transform=transform, download=download, train=False)
    val_dataset, test_dataset = random_split(dataset, lengths=[0.5, 0.5], generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return (val_loader, test_loader), (val_dataset, test_dataset)


def create_loaders(x, y, batch_size, shuffle=True, generator=None):
    """
    Creates a PyTorch dataloader and dataset from the provided data.

    Returns (loader, dataset)
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
    scale: str = "minmax",
    # covariate shift arguments
    induce_test_covariate_shift: bool = False,
    covariate_shift_intensity: float = 2.0,
    covariate_shift_n_features: int = 57,
    # return (X_train, X_test, y_train, y_test)
    return_raw: bool = False,
    random_state: int = 123,
):
    """
    Function to load the spambase data.

    Returns (train_loader, test_loader), (train_dataset, test_dataset)

    If return_raw:
        Returns X_train, X_test, y_train, y_test

    See: https://archive.ics.uci.edu/dataset/94/spambase

    Notes:
        * covariate_shift_intensity and covariate_shift_n_features do not do anything unless
            induce_test_covariate_shift=True.
        * covariate_shift_intensity controls the the intensity of the covariate shift
        * covariate_shift_n_features controls the number of features to induce shift in
        * setting return_raw=True will return (X_train, X_test, y_train, y_test)
        * scale = "minmax" | "standard" | "robust"
    """

    import torch
    from sklearn.model_selection import train_test_split
    from ucimlrepo import fetch_ucirepo
    from src.utils import set_all_seeds

    g = set_all_seeds(random_state)

    spambase = fetch_ucirepo(id=94)
    X, y = spambase.data.features.to_numpy(), spambase.data.targets["Class"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if return_raw:
        return X_train, X_test, y_train, y_test

    if induce_test_covariate_shift:
        X_test = induce_covariate_shift(
            X_test,
            n_features_to_shift=covariate_shift_n_features,
            intensity=covariate_shift_intensity,
            random_state=random_state,
        )

    if scale:
        X_train, X_test = scale_datasets(X_train, X_test, scaler=scale)

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

    Returns (train_loader, test_loader), (train_dataset, test_dataset)

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

    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)

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


def scale_datasets(data, *args, scaler="minmax", return_scaler=False):
    """
    Scales one or more datasets.

    The first argument is used to fit the scaler.

    scaler = "minmax" | "standard" | "robust"
    """
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

    assert scaler in ["minmax", "standard", "robust"], "scaler must be one of ('minmax', 'standard', 'robust')"

    if scaler == "minmax":
        scaler = MinMaxScaler()
    elif scaler == "standard":
        scaler = StandardScaler()
    elif scaler == "robust":
        scaler = RobustScaler()
    data = scaler.fit_transform(data)
    args = [scaler.transform(arg) for arg in args]
    if len(args) == 0:
        if return_scaler:
            return data, scaler
        return data
    if return_scaler:
        return [data] + args, scaler
    return [data] + args


######################### Dataset Creating Functions #########################


def make_chessboard(
    n_blocks: int = 4,
    n_points_in_block: int = 100,
    variance: float = 0.05,
    scale: str = "minmax",
    all_different_classes: bool = False,
    random_state: int = 123,
):
    """
    Returns two tensors (x, y)

    Note that the use of scaler here doesn't matter since we just want the data to inside the 0, 1 range.

    Args:
        * n_blocks: the number of blocks/blobs. n_blocks=4 gives a 4x4 board.
        * n_points_in_block: the number of points in each of the blocks.
            - n_points = n_blocks ** 2 * n_points_in_block
        * variance: changes the variance of the blocks. Is not really necessary.
        * scale: which scaler to use. One of: ["minmax", "standard", "robust"]
        * all_different_classes: whether each block will have a different class.
            - n_blocks=4 will result in: 4**2 = 16 classes
    """
    from src.utils import set_all_seeds
    from src.datasets import scale_datasets
    import torch

    g = set_all_seeds(random_state)

    x = []
    y = []

    if all_different_classes:
        class_ = 0

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
            if all_different_classes:
                y_temp = torch.ones(n_points_in_block) * class_
                class_ += 1
            else:
                y_temp = torch.ones(n_points_in_block) * ((row + col) % 2)

            x.append(x_temp)
            y.append(y_temp)

    x, y = torch.cat(x), torch.cat(y)

    if scale:
        x = torch.tensor(scale_datasets(x, scaler=scale), dtype=torch.float32)

    return x, y


def load_chessboard(
    n_blocks=4,
    n_points_in_block=100,
    batch_size=128,
    variance=0.05,
    scale="minmax",
    all_different_classes=False,
    random_state=123,
    generator=None,
):
    """
    Returns (train_loader, test_loader), (train_dataset, test_dataset)

    Note that the use of scaler here doesn't matter since we just want the data to inside the 0, 1 range.
    """
    from sklearn.model_selection import train_test_split
    from src.datasets import create_loaders

    x, y = make_chessboard(n_blocks, n_points_in_block, variance, scale, all_different_classes, random_state)
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=random_state)
    train_loader, train_dataset = create_loaders(x_train, y_train, batch_size=batch_size, generator=generator)
    test_loader, test_dataset = create_loaders(x_test, y_test, batch_size=batch_size, generator=generator)

    return (train_loader, test_loader), (train_dataset, test_dataset)
