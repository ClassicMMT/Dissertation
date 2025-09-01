import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_iris
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN, KMeans
import foolbox as fb
import umap
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import TensorDataset, DataLoader


device = torch.device("mps")

# reproducibility
np.random.seed(123)
torch.manual_seed(123)
random.seed(123)
torch.use_deterministic_algorithms(True)

# DATA

x, y = make_classification(
    n_samples=4000,
    n_features=5,
    n_informative=5,
    n_redundant=0,
    n_classes=2,
    shuffle=True,
)
x_train, x_test = x[:3000,], x[3000:,]
y_train, y_test = y[:3000], y[3000:]


# convert to tensors
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# normalisation
x_min, _ = x_train.min(dim=0, keepdim=True)
x_max, _ = x_train.max(dim=0, keepdim=True)
x_train = (x_train - x_min) / (x_max - x_min)
x_test = (x_test - x_min) / (x_max - x_min)


# mean = x_train.mean(axis=0)
# std = x_train.std(axis=0)
# x_train = (x_train - mean) / std
# x_test = (x_test - mean) / std

# dataloaders
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=512)
test_loader = DataLoader(test_dataset, batch_size=512)


# NETWORK


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


net = Net().to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    epoch_loss = 0
    n_correct, n_total = 0, 0
    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = net(features)
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

    print(
        f"Epoch: {epoch+1}, "
        + f"Epoch loss: {epoch_loss:.4f}, "
        + f"Avg loss: {epoch_loss/len(train_loader):.4f}, "
        + f"Accuracy: {n_correct / n_total:.4f}"
    )

net.eval()

# FGSM attack
fmodel = fb.PyTorchModel(net, bounds=(0, 1))
attack = fb.attacks.FGSM()
adversarial, adv_success, all_labels, original_success = [], [], [], []
for features, labels in train_loader:
    raw, clipped, is_adv = attack(fmodel, features, labels, epsilons=(0.1,))
    is_adv = is_adv.reshape(-1)
    adversarial += clipped
    adv_success += [is_adv]
    original_success += [net(features).argmax(dim=-1) == labels]
    all_labels += [labels]


# combine all examples into single tensors
all_clipped = torch.cat(adversarial, dim=0)
all_is_adv_bool = torch.cat(adv_success, dim=0)
all_is_correct_bool = torch.cat(original_success, dim=0)
all_is_correct_and_adv_bool = all_is_adv_bool & all_is_correct_bool

# indices of successful attacks
all_is_adv_indices = all_is_correct_and_adv_bool.nonzero(as_tuple=True)[0]
all_labels = torch.cat(all_labels, dim=0)

# extract only successful attacks
successful_clipped = all_clipped[all_is_adv_indices]
true_labels = all_labels[all_is_adv_indices]

# t-SNE
tsne_reducer = TSNE(
    n_components=2, perplexity=100.0, early_exaggeration=500, n_jobs=-1, max_iter=1000
)
tsne_projection = tsne_reducer.fit_transform(successful_clipped)
plt.scatter(tsne_projection[:, 0], tsne_projection[:, 1], alpha=0.1)
plt.show()

# PCA (not good for non-linear data)
pca_reducer = PCA(n_components=2)
pca_projection = pca_reducer.fit_transform(successful_clipped)
plt.scatter(pca_projection[:, 0], pca_projection[:, 1])
plt.show()

# UMAP
umap_reducer = umap.UMAP(n_components=2)
umap_projection = umap_reducer.fit_transform(successful_clipped)
plt.scatter(umap_projection[:, 0], umap_projection[:, 1])
plt.show()


# Clustering
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
clusters = []
for i, (projection, title) in enumerate(
    zip([tsne_projection, umap_projection], ["t-SNE", "UMAP"])
):
    clusterer = KMeans(n_clusters=2)
    projection_clusters = clusterer.fit_predict(projection)
    clusters += [projection_clusters]
    scatter = axs[i].scatter(
        projection[:, 0],
        projection[:, 1],
        alpha=0.5,
        c=projection_clusters,
        cmap="tab10",
    )
    axs[i].set_title(title)

    # Create automatic legend
    handles, labels = scatter.legend_elements()
    axs[i].legend(handles, labels, title="Cluster")
plt.tight_layout()
plt.show()


# combining results
tsne_clusters = clusters[0]
umap_clusters = clusters[1]
# each entry in these lists corresponds to a cluster
uncertain_points, uncertain_points_adv = [], []
for cluster_id in np.unique(umap_clusters):
    cluster_indices = all_is_adv_indices[umap_clusters == cluster_id]
    points_in_cluster = x_train[cluster_indices]
    points_in_cluster_adv = successful_clipped[umap_clusters == cluster_id]
    uncertain_points += [points_in_cluster]
    uncertain_points_adv += [points_in_cluster_adv]


umap_clusters
# defining the adversarial regions
mins = cluster_adv_examples.min(dim=0).values
maxs = cluster_adv_examples.max(dim=0).values
