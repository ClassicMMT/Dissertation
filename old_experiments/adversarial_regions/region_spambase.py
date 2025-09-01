import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import umap
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import foolbox as fb


random_state = 123
np.random.seed(random_state)
torch.manual_seed(random_state)
random.seed(random_state)
torch.use_deterministic_algorithms(True)

batch_size = 128
# device = torch.device("mps")
device = torch.device("cpu")

# Data and Preprocessing

spambase = fetch_ucirepo(id=94)
X = spambase.data.features
y = spambase.data.targets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    stratify=y,
    random_state=random_state,
)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

y_train = y_train["Class"].to_numpy()
y_test = y_test["Class"].to_numpy()


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


# Network


class SpamBaseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(57, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


# Training
model = SpamBaseNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    epoch_loss = 0
    n_correct, n_total = 0, 0

    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)

        # forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)
        epoch_loss += loss.item()

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accuracy
        n_correct += (outputs.argmax(dim=-1) == labels).sum().item()
        n_total += len(labels)

    print(
        f"Epoch: {epoch+1}, "
        + f"Epoch loss: {epoch_loss:.4f}, "
        + f"Avg loss: {epoch_loss/len(train_loader):.4f}, "
        + f"Accuracy: {n_correct / n_total:.4f}"
    )

# Testing

with torch.no_grad():
    model.eval()
    n_correct, n_total = 0, 0
    for features, labels in test_loader:
        features, labels = features.to(device), labels.to(device)

        n_correct += (model(features).argmax(dim=-1) == labels).sum().item()
        n_total += len(labels)

    print(f"Test Accuracy: {n_correct / n_total:.4f}")


# Attacks
fmodel = fb.PyTorchModel(model, bounds=(0, 1), device=device)
# attack = fb.attacks.FGSM()
# attack = fb.attacks.deepfool.L2DeepFoolAttack(steps=100)
attack = fb.attacks.basic_iterative_method.LinfAdamBasicIterativeAttack()
adversarial, adv_success, all_labels, original_success = [], [], [], []
for i, (features, labels) in enumerate(train_loader):
    features, labels = features.to(device), labels.to(device)
    raw, clipped, is_adv = attack(fmodel, features, labels, epsilons=(0.1,))
    is_adv = is_adv.reshape(-1)
    adversarial += clipped
    adv_success += [is_adv]
    original_success += [model(features).argmax(dim=-1) == labels]
    all_labels += [labels]
    if not i % 10:
        print(f"{i+1}/{len(train_loader)}")

# combine all examples into single tensors
all_clipped = torch.cat(adversarial, dim=0)
all_is_adv_bool = torch.cat(adv_success, dim=0)
all_is_correct_bool = torch.cat(original_success, dim=0)
all_is_correct_and_adv_bool = all_is_adv_bool & all_is_correct_bool

# indices of successful attacks
all_is_adv_indices = all_is_correct_and_adv_bool.nonzero(as_tuple=True)[0]
all_labels = torch.cat(all_labels, dim=0)

# extract only successful attacks
n_successful_attacks = len(all_is_adv_indices)
successful_clipped = (
    all_clipped[all_is_adv_indices].view(n_successful_attacks, -1).to("cpu")
)
true_labels = all_labels[all_is_adv_indices].to("cpu")

# t-SNE
tsne_reducer = TSNE(
    n_components=2, perplexity=100.0, early_exaggeration=500, n_jobs=-1, max_iter=1000
)
tsne_projection = tsne_reducer.fit_transform(successful_clipped)
plt.scatter(tsne_projection[:, 0], tsne_projection[:, 1], alpha=0.1)
plt.show()


# UMAP
umap_reducer = umap.UMAP(n_components=2)
umap_projection = umap_reducer.fit_transform(successful_clipped)
plt.scatter(umap_projection[:, 0], umap_projection[:, 1], alpha=0.01)
plt.show()


# clustering
projection = umap_projection
projection = tsne_projection
clusterer = KMeans(n_clusters=2, random_state=random_state)
# clusterer = DBSCAN(eps=0.31, n_jobs=-1)
projection_clusters = clusterer.fit_predict(projection)
plt.scatter(projection[:, 0], projection[:, 1], c=projection_clusters, alpha=0.05)
plt.show()


# can use knn to find which test examples are in adversarial clusters - can test this using the projection or the raw values
knn1 = KNeighborsClassifier(n_neighbors=7)

# combine the train and adversarial datasets first
original_examples = train_dataset.tensors[0]
adversarial_examples = successful_clipped
original_and_adversarial = torch.cat((original_examples, adversarial_examples), dim=0)
original_and_adversarial_y = torch.cat(
    (torch.zeros(len(original_examples)), torch.ones(len(adversarial_examples)))
)

# knn on original data
knn1 = knn1.fit(original_and_adversarial, original_and_adversarial_y)
test_set_predictions = knn1.predict(test_dataset.tensors[0])


# knn on projection
knn2 = KNeighborsClassifier(n_neighbors=7)
knn2 = knn2.fit(umap_projection)
test_umap_projection = umap_reducer.transform(test_dataset.tensors[0])


# digression - DO THIS FIRST - SEE IF WE CAN SEPARATE THE ADVERSARIAL EXAMPLES FROM THE TRAINING DATA AND SEE CLUSTERS
umap_reducer_digression = umap.UMAP(n_components=2)
umap_projection_digression = umap_reducer_digression.fit_transform(
    original_and_adversarial
)
plt.scatter(
    umap_projection_digression[:, 0],
    umap_projection_digression[:, 1],
    alpha=0.1,
    c=original_and_adversarial_y,
    cmap="bwr",
)
import matplotlib.patches as mpatches

legend_handles = [
    mpatches.Patch(color="blue", label="Original"),
    mpatches.Patch(color="red", label="Adversarial"),
]
plt.legend(handles=legend_handles)
plt.show()
