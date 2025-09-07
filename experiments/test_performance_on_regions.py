import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from src.utils import load_model, set_all_seeds, sample
from src.datasets import load_heloc, load_spambase
from src.models import HelocNet, SpamBaseNet
from src.attacks import load_all_adversarial_examples

device = torch.device("mps")
random_state = 123
set_all_seeds(random_state)

# load model
model_name = "spambase"
model = load_model(empty_instance=SpamBaseNet(), load_name=model_name)
(train_loader, test_loader), (train_dataset, test_dataset) = load_spambase()

# model_name = "heloc"
# model = load_model(HelocNet(), load_name=model_name)
# (train_loader, test_loader), (train_dataset, test_dataset) = load_heloc()


# Attacks to load
# attacks = ["bim_linf", "cw_l2", "df_l2", "df_linf", "fgsm"]
attacks = ["bim_linf", "df_l2", "df_linf", "fgsm"]
# attacks = ["df_linf", "df_l2"]
# epsilons = [0.01, 0.05, 0.1, 0.15, 0.2]

epsilons = np.linspace(0.001, 0.2, 50).round(3)

# Get all pre-generated adversarial examples
all_adv_examples, _ = load_all_adversarial_examples(
    model_name, include_attacks=attacks, include_epsilons=epsilons
)

# get original examples
original_examples = train_dataset.tensors[0]

# sample adversarial examples to be the same size as original dataset
# all_adv_examples = sample(
#     all_adv_examples, size=len(original_examples), random_state=random_state
# )

# create labels
normal_labels = torch.zeros(len(original_examples))
adv_labels = torch.ones(len(all_adv_examples))

# combine all data
examples = torch.cat((original_examples, all_adv_examples))
labels = torch.cat((normal_labels, adv_labels))

# fit classifier
knn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
knn.fit(examples, labels)

# get test data
test_examples, test_labels = test_dataset.tensors

with torch.no_grad():
    model.eval()
    incorrectly_classified = (model(test_examples).argmax(dim=-1) != test_labels).int()

test_adv_predictions = knn.predict(test_examples)
confusion_matrix(incorrectly_classified, test_adv_predictions)
