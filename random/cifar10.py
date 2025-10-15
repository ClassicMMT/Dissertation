import torch
from src.utils import evaluate_model, load_cifar_model, set_all_seeds
from src.datasets import load_cifar


random_state = 123
g = set_all_seeds(random_state)
batch_size = 128
device = torch.device("mps")

(val_loader, test_loader), _ = load_cifar(batch_size, generator=g)

# ResNet
model = load_cifar_model()

evaluate_model(model, val_loader, device=device)
