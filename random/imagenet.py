import torch
from torchvision.models import resnet50, ResNet50_Weights
from src.datasets import load_imagenet
from src.utils import evaluate_model, set_all_seeds, load_resnet50

random_state = 123
g = set_all_seeds(random_state)
device = torch.device("mps")

(val_loader, test_loader), _ = load_imagenet(batch_size=128, generator=g)
model = load_resnet50(device=device)

f, l = next(iter(val_loader))
f = f.to(device)
l = l.to(device)


(model(f).argmax(dim=-1) == l).float().mean()
