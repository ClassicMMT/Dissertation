"""
The idea is that a machine learning model's entropy landscape changes as the model trains.
Since adversarial points want to cross the decision boundary, they typically find
the closest point on the boundary they can cross and stop.

This means that adversarial examples could be identified using entropy, because entropy
is highest where the loss is also highest.

This also means that we can determine an "applicability domain" - the areas where the entropy is not extremely high.

We can do this by, in the final part of the training loop, calculate the entropy values, take the 95th quantile and see how the adversarial points compare.

Ideas:
    * Could also use adversarial points to figure out the points of high density
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src.datasets import create_loaders, load_spambase, make_chessboard
from src.utils import (
    calculate_entropy,
    evaluate_model,
    plot_boundary,
    plot_density_landscape,
    plot_entropy_landscape,
    plot_information_content_landscape,
    plot_loss_landscape,
    set_all_seeds,
    train_model,
)
from src.attacks import attack_model
from src.models import GenericNet
import foolbox as fb
import torch
import torch.nn as nn

random_state = 123
g = set_all_seeds(random_state)
batch_size = 128
device = torch.device("mps")

x, y = make_chessboard(n_blocks=4, n_points_in_block=100, random_state=random_state)

# make train, calibration, and test splits
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=random_state)
x_train, x_calib, y_train, y_calib = train_test_split(x_train, y_train, stratify=y_train, random_state=random_state + 1)
train_loader, train_dataset = create_loaders(x_train, y_train, batch_size=batch_size, generator=g)
calib_loader, calib_dataset = create_loaders(x_calib, y_calib, batch_size=batch_size, generator=g)
test_loader, test_dataset = create_loaders(x_test, y_test, batch_size=batch_size, generator=g)


# instantiate model
model = GenericNet(layers=[2, 1024, 512, 256, 2], activation="relu").train().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
n_epochs = 50

# train model
train_entropies = []
for epoch in range(n_epochs):
    print(f"Training epoch: {epoch+1}")
    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # calculate entropies for the final epoch
        if epoch + 1 == n_epochs:
            entropies = calculate_entropy(logits)
            train_entropies.append(entropies)

train_entropies = torch.cat(train_entropies)
train_q_hat = torch.quantile(train_entropies, 1 - 0.05, interpolation="higher")


# nice plot
if True:
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    plot_boundary(axs[0], model, x_train, y_train, device=device)
    plot_loss_landscape(axs[1], model, x=x_train, y=y_train, device=device)
    plot_entropy_landscape(axs[2], model, x=x_train, y=y_train, device=device)
    plot_information_content_landscape(axs[3], model, x=x_train, y=y_train)
    fig.suptitle(f"Epochs:{n_epochs}")
    plt.tight_layout()
    plt.show()

# get entropies for the calibration set
with torch.no_grad():
    model.eval()
    calib_entropies = []
    for features, labels in calib_loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        entropies = calculate_entropy(logits)
        calib_entropies.append(entropies)

    calib_entropies = torch.cat(calib_entropies)


# find q_hat
alpha = 0.05
q_hat = torch.quantile(calib_entropies, 1 - alpha, interpolation="higher")


# experiment on x_test
with torch.no_grad():
    """
    certain_predictions here means predictions where the entropy is < q_hat
    uncertain_predictions means predictions where entropy >= q_hat
    """
    model.eval()

    certain_accuracy = []
    uncertain_accuracy = []

    uncertain_points = []
    uncertain_labels = []
    certain_points = []
    certain_labels = []

    all_entropies = []

    for features, labels in test_loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)

        # figure out which test points are in uncertain regions
        entropies = calculate_entropy(logits)
        entropies_over_q_hat = entropies >= q_hat

        # get predictions
        preds = logits.argmax(dim=-1)
        is_correct_pred = preds == labels

        # accuracies
        uncertain_preds = is_correct_pred[entropies_over_q_hat]
        uncertain_accuracy.append(uncertain_preds)
        certain_preds = is_correct_pred[~entropies_over_q_hat]
        certain_accuracy.append(certain_preds)

        # points
        uncertain_points.append(features[entropies_over_q_hat].cpu())
        uncertain_labels.append(labels[entropies_over_q_hat].cpu())
        certain_points.append(features[~entropies_over_q_hat].cpu())
        certain_labels.append(labels[~entropies_over_q_hat].cpu())

        all_entropies.append(entropies)

    # combine
    certain_accuracy = torch.cat(certain_accuracy)
    uncertain_accuracy = torch.cat(uncertain_accuracy)

    uncertain_points = torch.cat(uncertain_points)
    certain_points = torch.cat(certain_points)
    uncertain_labels = torch.cat(uncertain_labels)
    certain_labels = torch.cat(certain_labels)

    points = torch.cat((certain_points, uncertain_points))
    labels = torch.cat(
        (
            torch.zeros(len(certain_points)),
            torch.ones(len(uncertain_points)),
        )
    )

    all_entropies = torch.cat(all_entropies)


# accuracy of points with entropy < q_hat
certain_accuracy.float().mean()
# accuracy of points with entropy >= q_hat
uncertain_accuracy.float().mean()


if True:
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    plot_boundary(axs[0], model, x_train, y_train, device=device, plot_x=points, plot_y=labels)
    plot_loss_landscape(axs[1], model, x=x_train, y=y_train, device=device, plot_x=points, plot_y=labels)
    plot_entropy_landscape(axs[2], model, x=x_train, y=y_train, device=device, plot_x=points, plot_y=labels)
    plot_information_content_landscape(axs[3], model, x=x_train, y=y_train, plot_x=points, plot_y=labels)
    fig.suptitle(f"Points coloured by entropy > q_hat")
    plt.tight_layout()
    plt.show()

# the percentage of points that are "uncertain" matches q_hat
(all_entropies > q_hat).float().mean()

# plot the whole entropy distribution
if True:
    plt.axvline(q_hat, color="red")
    plt.hist(all_entropies, bins=50)
    plt.title("Red line is q_hat")
    plt.show()


################ The relationship between q_hat and adversarial examples

attacks = {
    # "fgsm": fb.attacks.FGSM(),
    "fgsm": fb.attacks.fast_gradient_method.L1FastGradientAttack(),
    "bim": fb.attacks.basic_iterative_method.L2AdamBasicIterativeAttack(),
    "deepfool": fb.attacks.deepfool.L2DeepFoolAttack(),
}
results = {}
examples = {}
labels = {}
for name, attack in attacks.items():
    adversarial_examples, original_labels = attack_model(
        model, attack=attack, loader=train_loader, verbose=False, epsilons=0.01
    )
    examples[name] = adversarial_examples.cpu()

    with torch.no_grad():
        model.eval()
        logits = model(adversarial_examples)
        adversarial_entropies = calculate_entropy(logits)
        results[name] = adversarial_entropies.cpu()
        labels[name] = (adversarial_entropies >= q_hat).cpu()


for name, adversarial_entropies in results.items():
    percent_over_q_hat = (adversarial_entropies > q_hat).float().mean()
    print(f"{name}: {percent_over_q_hat.item():.4f}")

if True:
    fig, axs = plt.subplots(3, 4, figsize=(12, 12))
    colour_map = {1: "red", 0: "blue"}
    for i, name in enumerate(results):
        plot_boundary(
            axs[i, 0],
            model,
            x_train,
            y_train,
            device=device,
            plot_x=examples[name],
            plot_y=labels[name],
            colour_map=colour_map,
        )
        plot_loss_landscape(
            axs[i, 1],
            model,
            x=x_train,
            y=y_train,
            device=device,
            plot_x=examples[name],
            plot_y=labels[name],
            colour_map=colour_map,
        )
        plot_entropy_landscape(
            axs[i, 2],
            model,
            x=x_train,
            y=y_train,
            device=device,
            plot_x=examples[name],
            plot_y=labels[name],
            colour_map=colour_map,
        )
        plot_information_content_landscape(
            axs[i, 3],
            model,
            x=x_train,
            y=y_train,
            device=device,
            plot_x=examples[name],
            plot_y=labels[name],
            colour_map=colour_map,
        )
    fig.suptitle(f"Points coloured by entropy > q_hat")
    plt.tight_layout()
    plt.show()
