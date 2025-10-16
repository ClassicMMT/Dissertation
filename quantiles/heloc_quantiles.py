"""
This experiment calculates whether the test data has a distribution shift on the Heloc dataset.
"""

from sklearn.model_selection import train_test_split
import torch
from src.datasets import create_loaders, load_heloc, scale_datasets
from src.models import HelocNet
from src.method import identify_uncertain_points_all
from src.utils import evaluate_model, set_all_seeds, train_model


random_state = 123
g = set_all_seeds(random_state)
device = torch.device("mps")
batch_size = 128

# load data
x_train, x_test, y_train, y_test = load_heloc(batch_size=128, return_raw=True, random_state=random_state)
x_train, x_calib, y_train, y_calib = train_test_split(
    x_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train
)
x_train, x_calib, x_test = scale_datasets(x_train.to_numpy(), x_calib.to_numpy(), x_test.to_numpy())
train_loader, train_datasest = create_loaders(x_train, y_train, batch_size, generator=g)
calib_loader, calib_datasest = create_loaders(x_calib, y_calib, batch_size, generator=g)
test_loader, test_datasest = create_loaders(x_test, y_test, batch_size, generator=g)


# model
model = HelocNet().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model = train_model(model, train_loader, criterion, optimizer, n_epochs=30, device=device, verbose=False)
evaluate_model(model, train_loader, device=device)
evaluate_model(model, calib_loader, device=device)

rejection_proportion_results = {
    "entropy": [],
    "information_content": [],
    "probability_gap": [],
    "any": [],
    "all": [],
}

low_uncertainty_accuracy_results = {key: [] for key in rejection_proportion_results}
high_uncertainty_accuracy_results = {key: [] for key in rejection_proportion_results}
false_positive_results = {key: [] for key in rejection_proportion_results}
false_negative_results = {key: [] for key in rejection_proportion_results}


# get uncertain points
alpha = 0.05
uncertain_results = identify_uncertain_points_all(model, calib_loader, test_loader, alpha=alpha, device=device)

# unpack results
entropy_reject = uncertain_results["highly_uncertain_entropy"]
info_content_reject = uncertain_results["highly_uncertain_information_content"]
probability_gap_reject = uncertain_results["highly_uncertain_probability_gap"]
any_reject = uncertain_results["highly_uncertain_any"]
all_reject = uncertain_results["highly_uncertain_all"]
is_correct = uncertain_results["is_correct_predictions"].cpu()

# accuracy
is_correct[~entropy_reject].float().mean().item()
is_correct[entropy_reject].float().mean().item()

is_correct[~info_content_reject].float().mean().item()
is_correct[info_content_reject].float().mean().item()

is_correct[~probability_gap_reject].float().mean().item()
is_correct[probability_gap_reject].float().mean().item()

is_correct[~any_reject].float().mean().item()
is_correct[any_reject].float().mean().item()

is_correct[~all_reject].float().mean().item()
is_correct[all_reject].float().mean().item()
