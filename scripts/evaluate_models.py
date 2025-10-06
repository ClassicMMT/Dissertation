from sklearn.model_selection import train_test_split
import torch
from src.utils import set_all_seeds, evaluate_model, load_model, load_resnet50
from src.models import SpamBaseNet, HelocNet, GenericNet, MNISTNet
from src.datasets import create_loaders, load_spambase, load_heloc, load_imagenet, load_mnist, make_chessboard

random_state = 123
g = set_all_seeds(random_state)
batch_size = 128
device = torch.device("mps")

############## Chessboard ##############

x, y = make_chessboard(random_state=random_state)
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=random_state)
train_loader, _ = create_loaders(x_train, y_train, batch_size=batch_size, generator=g)
test_loader, _ = create_loaders(x_test, y_test, batch_size=batch_size, generator=g)
model = load_model(
    GenericNet(layers=[2, 1024, 512, 256, 2], activation="relu", random_state=random_state), "chessboard_30.pt"
)
train_accuracy = evaluate_model(model, train_loader, device=device)
test_accuracy = evaluate_model(model, test_loader, device=device)
print(f"Chessboard Train: {train_accuracy:.4f}, Test: {test_accuracy:.4f}")

############## Spambase ##############

(train_loader, test_loader), _ = load_spambase(batch_size=batch_size, random_state=random_state)
model = load_model(SpamBaseNet(), "spambase_6.pt")
accuracy = evaluate_model(model, test_loader, device=device)
train_accuracy = evaluate_model(model, train_loader, device=device)
test_accuracy = evaluate_model(model, test_loader, device=device)
print(f"SpamBase Train: {train_accuracy:.4f}, Test: {test_accuracy:.4f}")

############## Heloc ##############

(train_loader, test_loader), _ = load_heloc(batch_size=batch_size, random_state=random_state)
model = load_model(HelocNet(), "heloc.pt")
train_accuracy = evaluate_model(model, train_loader, device=device)
test_accuracy = evaluate_model(model, test_loader, device=device)
print(f"Heloc Train: {train_accuracy:.4f}, Test: {test_accuracy:.4f}")

############## MNIST ##############

(train_loader, test_loader), _ = load_mnist(batch_size=batch_size, generator=g)
model = load_model(MNISTNet(), "mnist.pt")
accuracy = evaluate_model(model, test_loader, device=device)
train_accuracy = evaluate_model(model, train_loader, device=device)
test_accuracy = evaluate_model(model, test_loader, device=device)
print(f"MNIST Train: {train_accuracy:.4f}, Test: {test_accuracy:.4f}")

############## ImageNet ##############

(_, test_loader), (_, test_dataset) = load_imagenet(batch_size=batch_size, generator=g)
model = load_resnet50(device=device)
accuracy = evaluate_model(model, test_loader, device=device)
print(f"ImageNet Accuracy: {accuracy:.4f}")
