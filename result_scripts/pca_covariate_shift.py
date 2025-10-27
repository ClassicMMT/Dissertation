"""
The purpose of this file is to induce a covariate shift in the test distrubition and
make a comparison plot of how this distribution has changed from the original.
"""

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.datasets import induce_covariate_shift, load_spambase, scale_datasets
from src.utils import calculate_entropy, set_all_seeds

random_state = 123
set_all_seeds(random_state)

(train_loader, test_loader), (train_dataset, test_dataset) = load_spambase(scale=False, random_state=random_state)

X_test, y_test = test_dataset.tensors
X_train, y_train = train_dataset.tensors

test_with_induced_shift = induce_covariate_shift(X_test, n_features_to_shift=57, intensity=2, random_state=random_state)

# train, test1, test2 = scale_datasets(X_train, X_test, test_with_induced_shift)
scaler = StandardScaler()
train = scaler.fit_transform(X_train)
test1 = scaler.transform(X_test)
test2 = scaler.transform(test_with_induced_shift)

reducer = PCA(n_components=2, random_state=random_state, svd_solver="full")
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
plt.title("UMAP Projection of Induced Covariate Shift (SpamBase)")
plt.show()
