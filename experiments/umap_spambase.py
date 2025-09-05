import umap
import matplotlib.pyplot as plt
from src.datasets import induce_covariate_shift, load_spambase, scale_datasets
from src.utils import set_all_seeds

random_state = 123
set_all_seeds(random_state)

(train_loader, test_loader), (train_dataset, test_dataset) = load_spambase(scale=False)

test_with_induced_shift = induce_covariate_shift(
    test_dataset, n_features_to_shift=10, intensity=2, random_state=random_state
)

train, test1, test2 = scale_datasets(
    train_dataset.tensors[0], test_dataset.tensors[0], test_with_induced_shift
)

reducer = umap.UMAP(n_components=2, random_state=random_state, n_jobs=1)
train_projection = reducer.fit_transform(train)
test_projection = reducer.transform(test1)
test2_projection = reducer.transform(test2)

# Original Data
plt.scatter(
    train_projection[:, 0],
    train_projection[:, 1],
    c="blue",
    label="Train",
    # alpha=0.1
)
plt.scatter(
    test_projection[:, 0],
    test_projection[:, 1],
    c="green",
    label="Original Test",
    # alpha=0.1,
)
plt.scatter(
    test2_projection[:, 0],
    test2_projection[:, 1],
    c="red",
    label="Shifted Test",
    # alpha=0.1,
)
plt.legend(loc="lower left")
plt.show()
