import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=5, kernel_size=5),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Flatten(),
            nn.Linear(5 * 52 * 52, 256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(256, 10),
            nn.Softmax(),
        )

        self.large = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=20)
        )

    def forward(self, x):

        return self.layers(x)


device = "mps"


transform = transforms.Compose([transforms.ToTensor()])
dataset = MNIST(root="data/", transform=transform, download=True)
test_dataset = MNIST(root="data/", transform=transform, download=True, train=False)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

model = Net().to(device)
model.train()
optimizer = optim.Adam(model.parameters())
lr = 3e-4
criterion = nn.CrossEntropyLoss()

# images, labels = next(iter(loader))

epochs = 10

for epoch in range(epochs):
    epoch_loss = 0
    correct, total = 0, 0
    for i, (images, target) in enumerate(loader):
        large_images = torch.zeros((32, 1, 56, 56))
        large_images[:, :, 0:28, 0:28] = images

        large_images = large_images.to(device)
        target = target.to(device)

        preds = model(large_images)

        loss = criterion(preds, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss
        correct += (preds.argmax(dim=-1) == target).sum().item()
        total += len(target)

        # print(f"[{i+1}/{len(loader)}], epoch={epoch+1}, loss={epoch_loss.item():.4f}, Accuracy={(correct / total):.4f}")
    print(
        f"epoch={epoch+1}, loss={epoch_loss.item():.4f}, Accuracy={(correct / total):.4f}"
    )


model.eval()

with torch.no_grad():
    correct, total = 0, 0
    for i, (images, target) in enumerate(loader):
        large_images = torch.zeros((32, 1, 56, 56))
        large_images[:, :, 0:28, 0:28] = images

        large_images = large_images.to(device)
        target = target.to(device)

        preds = model(large_images)

        correct += (preds.argmax(dim=-1) == target).sum().item()
        total += len(target)
    print(f"Test Accuracy: {correct / total}")

with torch.no_grad():
    correct, total = 0, 0
    for i, (images, target) in enumerate(loader):
        large_images = torch.zeros((32, 1, 56, 56))
        large_images[:, :, 28:, 28:] = images

        large_images = large_images.to(device)
        target = target.to(device)

        preds = model(large_images)

        correct += (preds.argmax(dim=-1) == target).sum().item()
        total += len(target)
    print(f"Test Accuracy (OOD): {correct / total}")


x = 5
