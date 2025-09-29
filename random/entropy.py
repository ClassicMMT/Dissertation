import torch
from src.utils import calculate_entropy, load_model, set_all_seeds
from src.models import SpamBaseNet
from src.datasets import load_spambase

random_state = 123
g = set_all_seeds(random_state)
device = torch.device("mps")

model = load_model(SpamBaseNet(), "spambase").eval().to(device)
(train_loader, test_loader), (train_dataset, test_dataset) = load_spambase()

max_entropy = 0
with torch.no_grad():
    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)

        entropy = calculate_entropy(logits).max().item()
        # print(entropy)

        max_entropy = max(entropy, max_entropy)

# using optimisation
x = torch.randn_like(features).requires_grad_(True)
optimizer = torch.optim.Adam([x], lr=1e-2)
for _ in range(100):
    logits = model(x)
    entropy = calculate_entropy(logits, detach=False)
    loss = -entropy.sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

logits = model(x)
calculate_entropy(logits).max()
