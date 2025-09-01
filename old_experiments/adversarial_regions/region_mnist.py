import numpy as np
import random
from sklearn.datasets import make_classification, load_iris
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN, KMeans, DBSCAN
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import foolbox as fb
import matplotlib.pyplot as plt
import umap
import torchvision.transforms as transforms
from torchvision.datasets import MNIST


# make utils imports work
import os

os.chdir("..")

device = torch.device("mps")


# reproducibility
random_state = 123
np.random.seed(random_state)
torch.manual_seed(random_state)
random.seed(random_state)
torch.use_deterministic_algorithms(True)

# DATA
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = MNIST(root="data/", transform=transform, download=False)
test_dataset = MNIST(root="data/", transform=transform, download=False, train=False)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 16 * 4 * 4)  # flatten
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.fc3(x)
        return x


net = Net().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    net.train()
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

# ATTACKS
fmodel = fb.PyTorchModel(net, bounds=(0, 1), device=device)
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
    original_success += [net(features).argmax(dim=-1) == labels]
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
plt.scatter(tsne_projection[:, 0], tsne_projection[:, 1], alpha=0.01)
plt.show()

# PCA (not good for non-linear data)
pca_reducer = PCA(n_components=2)
pca_projection = pca_reducer.fit_transform(successful_clipped)
plt.scatter(pca_projection[:, 0], pca_projection[:, 1])
plt.show()

# UMAP
umap_reducer = umap.UMAP(n_components=2)
umap_projection = umap_reducer.fit_transform(successful_clipped)
plt.scatter(umap_projection[:, 0], umap_projection[:, 1], alpha=0.01)
plt.show()


# Clustering - one projection
projection = umap_projection
clusterer = KMeans(n_clusters=6, random_state=random_state)
# clusterer = DBSCAN(eps=0.31, n_jobs=-1)
projection_clusters = clusterer.fit_predict(projection)
np.unique(projection_clusters, return_counts=True)
plt.scatter(projection[:, 0], projection[:, 1], c=projection_clusters, alpha=0.05)
plt.show()


# Clustering - multiple projections
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
clusters = []
for i, (projection, title) in enumerate(
    zip(
        [tsne_projection, umap_projection],
        ["t-SNE", "UMAP"],
    )
):
    clusterer = KMeans(n_clusters=6)
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


# # combining results
# tsne_clusters = clusters[0]
# umap_clusters = clusters[1]
# # each entry in these lists corresponds to a cluster
# uncertain_points, uncertain_points_adv = [], []
# for cluster_id in np.unique(umap_clusters):
#     cluster_indices = all_is_adv_indices[umap_clusters == cluster_id]
#     points_in_cluster = x_train[cluster_indices]
#     points_in_cluster_adv = successful_clipped[umap_clusters == cluster_id]
#     uncertain_points += [points_in_cluster]
#     uncertain_points_adv += [points_in_cluster_adv]
#
#
# # defining the adversarial regions
# mins = cluster_adv_examples.min(dim=0).values
# maxs = cluster_adv_examples.max(dim=0).values

# train model to predict adversarial region
adv_region_dataset = TensorDataset(
    successful_clipped.reshape((-1, 1, 28, 28)),
    torch.full((n_successful_attacks,), 1, dtype=torch.long),
)

original_train = TensorDataset(
    train_dataset.data.unsqueeze(1).float(), torch.zeros(60000).long()
)

real_and_adv_dataset = torch.utils.data.ConcatDataset(
    (adv_region_dataset, original_train)
)
adv_region_loader = DataLoader(adv_region_dataset, batch_size=512, shuffle=True)
real_and_adv_loader = DataLoader(real_and_adv_dataset, batch_size=512, shuffle=True)


# network to learn adversarial regions
class AdversarialRegionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)
        self.bn2 = nn.BatchNorm1d(2)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        return x


adv_net = AdversarialRegionNet().to(device)
optimizer = torch.optim.Adam(adv_net.parameters())
criterion = nn.CrossEntropyLoss()

# train adversarial region net
for epoch in range(1):
    adv_net.train()
    epoch_loss = 0
    n_correct, n_total = 0, 0
    for i, (features, labels) in enumerate(real_and_adv_loader):
        features, labels = features.to(device), labels.to(device)

        # forward pass
        outputs = adv_net(features)
        loss = criterion(outputs, labels)
        epoch_loss += loss.item()

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # evaluate
        n_correct += (outputs.argmax(dim=-1) == labels).sum().item()
        n_total += len(labels)

    print(
        f"Epoch: {epoch+1}, "
        + f"Epoch loss: {epoch_loss:.4f}, "
        + f"Avg loss: {epoch_loss/len(real_and_adv_loader):.4f}, "
        + f"Accuracy: {n_correct / n_total:.4f}"
    )

adv_net.eval()
net.eval()

n_correct, n_total = 0, 0
n_adversarial = 0
for features, labels in test_loader:
    features, labels = features.to(device), labels.to(device)

    # predict if inside adversarial region
    adv_preds = adv_net(features).argmax(dim=-1)
    n_adv = adv_preds.sum().item()
    n_adversarial += n_adv

    # get normal predictions
    predictions = net(features).argmax(dim=-1)

    n_correct += (predictions == labels).sum().item()
    n_total += len(labels)

it = iter(real_and_adv_loader)
f, l = next(it)
adv_net(f.to(device)).argmax(dim=-1).unique()


l = []

n_normal, n_adv = 0, 0
for features, labels in test_loader:
    features, labels = features.to(device), labels.to(device)

    preds = adv_net(features).argmax(dim=-1)
    n_adv += (preds == 1).sum().item()
    n_normal += (preds == 0).sum().item()


# Logistic regression
x = torch.vstack((successful_clipped, train_dataset.data.view(60000, -1)))
y = torch.concat((torch.ones(len(successful_clipped)), torch.zeros(60000)))
lr = LogisticRegression(n_jobs=-1)
lr.fit(x, y)
(lr.predict(x) == y).float().mean()
test_preds = lr.predict(test_dataset.data.reshape(10000, 784))
