import numpy as np
import torch
import umap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from src.attacks import load_all_adversarial_examples
from src.datasets import load_heloc
from src.models import HelocNet
from src.utils import sample, set_all_seeds, load_model

random_state = 123
set_all_seeds(random_state)

(train_loader, test_loader), (train_dataset, test_dataset) = load_heloc()
model = load_model(HelocNet(), "heloc")
model.eval()

# Extract data
x_train = train_dataset.tensors[0]
x_test = test_dataset.tensors[0]
all_data = torch.cat((train_dataset.tensors[0], test_dataset.tensors[0]))
all_labels = torch.cat((train_dataset.tensors[1], test_dataset.tensors[1]))
test_indicator = torch.cat((torch.zeros(len(train_dataset)), torch.ones(len(test_dataset))))

# get predicted labels
with torch.no_grad():
    preds = model(all_data).argmax(dim=-1)
    correct_prediction = preds == all_labels

# get projection
reducer = umap.UMAP(n_components=2, random_state=random_state)
projection = reducer.fit_transform(all_data, n_jobs=1)

# ADD Adversarial attacks and see where they live on this space

adversarial_examples, _, attack_names, epsilons = load_all_adversarial_examples("heloc", return_attack_and_epsilon=True)
adversarial_sample, attack_names_sample, epsilons_sample = sample(
    adversarial_examples, attack_names, epsilons, size=0.1
)
adversarial_projection = reducer.transform(adversarial_sample)

attacks = np.unique(attack_names_sample)
attack_to_index = {str(attack_name): i for i, attack_name in enumerate(attacks)}
attack_indices_sample = np.array(list(map(lambda attack: attack_to_index[attack], attack_names_sample)))

# Plots
s = 7
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

scatter0 = axs[0, 0].scatter(projection[:, 0], projection[:, 1], c=test_indicator, s=s)
legend0 = axs[0, 0].legend(*scatter0.legend_elements(), title="Point in test")
axs[0, 0].add_artist(legend0)
axs[0, 0].set_title("Point in Train/Test")

scatter1 = axs[0, 1].scatter(projection[:, 0], projection[:, 1], c=correct_prediction, s=s)
legend1 = axs[0, 1].legend(*scatter0.legend_elements(), title="Correctly Classified")
axs[0, 1].add_artist(legend1)
axs[0, 1].set_title("Is Correctly Classified")

scatter2 = axs[1, 0].scatter(projection[:, 0], projection[:, 1], c=all_labels, s=s)
legend2 = axs[1, 0].legend(*scatter0.legend_elements(), title="True Label")
axs[1, 0].add_artist(legend2)
axs[1, 0].set_title("True Labels")

xlim, ylim = axs[0, 0].get_xlim(), axs[0, 0].get_ylim()

# Second row
colour_map = {
    0: "#B79F00",
    1: "#F8766D",
    2: "#00BA38",
    3: "#F564E3",
    4: "#00BFC4",
    5: "#619CFF",
}

scatter3 = axs[1, 1].scatter(
    adversarial_projection[:, 0],
    adversarial_projection[:, 1],
    c=[colour_map[i] for i in attack_indices_sample],
    # alpha=0.01,
    s=s,
)
legend_handles = [mpatches.Patch(color=colour_map[attack_to_index[attack]], label=attack) for attack in attacks]
axs[1, 1].legend(handles=legend_handles, title="Attack")
axs[1, 1].set_xlim(xlim)
axs[1, 1].set_ylim(ylim)
axs[1, 1].set_title("By Attack")

fig.suptitle("Heloc", fontsize=30)

fig.tight_layout()
plt.show()
