import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import os

######################### Dataset Loading Utility Functions #########################


def load_mnist(batch_size=128):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = MNIST(root="data/", transform=transform, download=False)
    test_dataset = MNIST(root="data/", transform=transform, download=False, train=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return (train_loader, test_loader), (train_dataset, test_dataset)


def load_spambase(batch_size=128, test_size=0.1, scale=True, random_state=123):
    """
    Function to load the spambase data.

    See: https://archive.ics.uci.edu/dataset/94/spambase
    """

    spambase = fetch_ucirepo(id=94)
    X, y = spambase.data.features.to_numpy(), spambase.data.targets["Class"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if scale:
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

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
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

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
