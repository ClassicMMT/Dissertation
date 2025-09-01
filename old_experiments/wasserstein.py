import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from scipy.stats import wasserstein_distance_nd
from sklearn.svm import SVC

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from art.attacks.evasion import HopSkipJump, wasserstein, SimBA as SimpleBlackAttack
from art.estimators.classification import SklearnClassifier


from utils import (
    ss,
    plot_svm,
    get_limits,
    get_accuracy,
    plot_attacks_svm,  # works with only svm but faster
    plot_attacks,  # should work with most classifiers but slower
)

np.random.seed(123)


##### Same scale distributions

x1 = np.random.normal(loc=0, scale=5, size=(200, 1))
x2 = np.vstack(
    (
        np.random.normal(loc=5, scale=1, size=(100, 1)),
        np.random.normal(loc=0, scale=1, size=(100, 1)),
    )
)
x = np.hstack((x1, x2))
y = np.hstack((np.zeros(100), np.ones(100))).astype(int)

normal = SVC(
    kernel="rbf",
    random_state=101,
)

overfit = SVC(
    kernel="rbf",
    random_state=102,
    gamma=50,
)

normal.fit(x, y)
overfit.fit(x, y)

# classifier's original accuracy
(get_accuracy(normal.predict(x), y), get_accuracy(normal.predict(x), y))


# plot_svm(normal, x, y, title="normal")
# plot_svm(overfit, x, y, title="overfit")


#### Attacks


adversarial_normal = HopSkipJump(SklearnClassifier(model=normal)).generate(x)
adversarial_overfit = HopSkipJump(SklearnClassifier(model=overfit)).generate(x)

adversarial_y_normal = normal.predict(adversarial_normal)
adversarial_y_overfit = overfit.predict(adversarial_overfit)

print(
    "Predictions on adv points",
    "\nNormal Accuracy",
    np.mean(adversarial_y_normal == y).item(),
    "\nOverfit Accuracy:",
    np.mean(adversarial_y_overfit == y).item(),
)

plot_attacks_svm(x, y, adversarial_normal, adversarial_overfit, normal, overfit)
plot_attacks(x, y, adversarial_normal, adversarial_overfit, normal, overfit)


wasserstein_distance_nd(x, adversarial_normal)
wasserstein_distance_nd(x, adversarial_overfit)

ss(x, adversarial_normal).mean()
ss(x, adversarial_overfit).mean()

### By class experiment
print(
    "Blue class",
    "\nNormal distance",
    wasserstein_distance_nd(x[y == 0, :], adversarial_normal[y == 0, :]),
    "\nOverfit distance",
    wasserstein_distance_nd(x[y == 0, :], adversarial_overfit[y == 0, :]),
    "\nOrange class",
    "\nNormal distance",
    wasserstein_distance_nd(x[y == 1, :], adversarial_normal[y == 1, :]),
    "\nOverfit distance",
    wasserstein_distance_nd(x[y == 1, :], adversarial_overfit[y == 1, :]),
)

ss(x[y == 0, :], adversarial_normal[y == 0, :]).mean()
ss(x[y == 1, :], adversarial_normal[y == 1, :]).mean()
ss(x[y == 1, :], adversarial_overfit[y == 0, :]).mean()
ss(x[y == 1, :], adversarial_overfit[y == 1, :]).mean()

##### Correlation (wasserstein/ss) experiment


##### Different scale experiment
# what happens to the wasserstein distance when the variance is changed?

np.random.seed(124)

x1 = np.vstack(
    (
        np.random.normal(loc=0, scale=10, size=(100, 1)),
        np.random.normal(loc=0, scale=1, size=(100, 1)),
    )
)
x2 = np.vstack(
    (
        np.random.normal(loc=5, scale=1, size=(100, 1)),
        np.random.normal(loc=0, scale=1, size=(100, 1)),
    )
)
x = np.hstack((x1, x2))
y = np.hstack((np.zeros(100), np.ones(100))).astype(int)

plt.scatter(x[:, 0], x[:, 1], c=y)
plt.show()

normal = SVC(
    kernel="rbf",
    random_state=103,
)

overfit = SVC(
    kernel="rbf",
    random_state=104,
    gamma=50,
)

normal.fit(x, y)
overfit.fit(x, y)
(get_accuracy(normal.predict(x), y), get_accuracy(normal.predict(x), y))

adversarial_normal = HopSkipJump(SklearnClassifier(model=normal)).generate(x)
adversarial_overfit = HopSkipJump(SklearnClassifier(model=overfit)).generate(x)

adversarial_y_normal = normal.predict(adversarial_normal)
adversarial_y_overfit = overfit.predict(adversarial_overfit)

print(
    "Predictions on adv points",
    "\nNormal Accuracy",
    np.mean(adversarial_y_normal == y).item(),
    "\nOverfit Accuracy:",
    np.mean(adversarial_y_overfit == y).item(),
)

plot_attacks_svm(x, y, adversarial_normal, adversarial_overfit, normal, overfit)
plot_attacks(x, y, adversarial_normal, adversarial_overfit, normal, overfit)

wasserstein_distance_nd(x, adversarial_normal)
wasserstein_distance_nd(x, adversarial_overfit)

ss(x, adversarial_normal).mean()
ss(x, adversarial_overfit).mean()

print(  # by class
    "Blue class",
    "\nNormal Accuracy",
    wasserstein_distance_nd(x[y == 0, :], adversarial_normal[y == 0, :]),
    "\nOverfit Accuracy",
    wasserstein_distance_nd(x[y == 0, :], adversarial_overfit[y == 0, :]),
    "\nOrange class",
    "\nNormal Accuracy",
    wasserstein_distance_nd(x[y == 1, :], adversarial_normal[y == 1, :]),
    "\nOverfit Accuracy",
    wasserstein_distance_nd(x[y == 1, :], adversarial_overfit[y == 1, :]),
)


##### Class imbalance experiment

np.random.seed(412)
x1 = np.random.normal(loc=0, scale=5, size=(200, 1))
x2 = np.vstack(
    (
        np.random.normal(loc=5, scale=1, size=(100, 1)),
        np.random.normal(loc=0, scale=1, size=(100, 1)),
    )
)
x = np.hstack((x1, x2))
y = np.hstack((np.zeros(100), np.ones(100))).astype(int)

# create imbalance by removing 1/2 of class 1

indices = np.random.choice(100, 50)
x = np.vstack((x[indices], x[100:]))
y = np.hstack((y[indices], y[100:]))
np.unique(y, return_counts=True)

normal = SVC(
    kernel="rbf",
    random_state=111,
)

overfit = SVC(
    kernel="rbf",
    random_state=112,
    gamma=50,
)

normal.fit(x, y)
overfit.fit(x, y)

(get_accuracy(normal.predict(x), y), get_accuracy(normal.predict(x), y))


# plot_svm(normal, x, y, title="normal")
# plot_svm(overfit, x, y, title="overfit")

adversarial_normal = HopSkipJump(SklearnClassifier(model=normal)).generate(x)
adversarial_overfit = HopSkipJump(SklearnClassifier(model=overfit)).generate(x)

adversarial_y_normal = normal.predict(adversarial_normal)
adversarial_y_overfit = overfit.predict(adversarial_overfit)

print(
    "Predictions on adv points",
    "\nNormal Accuracy",
    np.mean(adversarial_y_normal == y).item(),
    "\nOverfit Accuracy:",
    np.mean(adversarial_y_overfit == y).item(),
)

plot_attacks_svm(x, y, adversarial_normal, adversarial_overfit, normal, overfit)
plot_attacks(x, y, adversarial_normal, adversarial_overfit, normal, overfit)


wasserstein_distance_nd(x, adversarial_normal)
wasserstein_distance_nd(x, adversarial_overfit)

ss(x, adversarial_normal).mean()
ss(x, adversarial_overfit).mean()

### By class
print(
    "Blue class",
    "\nNormal Accuracy",
    wasserstein_distance_nd(x[y == 0, :], adversarial_normal[y == 0, :]),
    "\nOverfit Accuracy",
    wasserstein_distance_nd(x[y == 0, :], adversarial_overfit[y == 0, :]),
    "\nOrange class",
    "\nNormal Accuracy",
    wasserstein_distance_nd(x[y == 1, :], adversarial_normal[y == 1, :]),
    "\nOverfit Accuracy",
    wasserstein_distance_nd(x[y == 1, :], adversarial_overfit[y == 1, :]),
)

ss(x[y == 0, :], adversarial_normal[y == 0, :]).mean()
ss(x[y == 1, :], adversarial_normal[y == 1, :]).mean()
ss(x[y == 1, :], adversarial_overfit[y == 0, :]).mean()
ss(x[y == 1, :], adversarial_overfit[y == 1, :]).mean()


##### Linear separability experiment

##### Test and wasserstein correlations experiment
# does the wasserstein distance maintain the properties we like when using a test set?

##### Different models experiment


##### Approximate CDF distance metric experiment

##### Wasserstein distance but only for successful adversarial points experiment

np.random.seed(124)  # changed from 125 to 124 to match different-scale experiment

x1 = np.vstack(
    (
        np.random.normal(loc=0, scale=10, size=(100, 1)),
        np.random.normal(loc=0, scale=1, size=(100, 1)),
    )
)
x2 = np.vstack(
    (
        np.random.normal(loc=5, scale=1, size=(100, 1)),
        np.random.normal(loc=0, scale=1, size=(100, 1)),
    )
)
x = np.hstack((x1, x2))
y = np.hstack((np.zeros(100), np.ones(100))).astype(int)

# plt.scatter(x[:, 0], x[:, 1], c=y)
# plt.show()

normal = SVC(
    kernel="rbf",
    random_state=103,
)

overfit = SVC(
    kernel="rbf",
    random_state=104,
    gamma=50,
)

normal.fit(x, y)
overfit.fit(x, y)

preds_normal = normal.predict(x)
preds_overfit = overfit.predict(x)
(get_accuracy(normal.predict(x), y), get_accuracy(normal.predict(x), y))

adversarial_normal = HopSkipJump(SklearnClassifier(model=normal)).generate(x)
adversarial_overfit = HopSkipJump(SklearnClassifier(model=overfit)).generate(x)

adversarial_y_normal = normal.predict(adversarial_normal)
adversarial_y_overfit = overfit.predict(adversarial_overfit)

print(
    "Predictions on adv points - attack failure rate",
    "\nNormal Accuracy",
    np.mean(adversarial_y_normal == y).item(),
    "\nOverfit Accuracy:",
    np.mean(adversarial_y_overfit == y).item(),
)

plot_attacks_svm(x, y, adversarial_normal, adversarial_overfit, normal, overfit)
plot_attacks(x, y, adversarial_normal, adversarial_overfit, normal, overfit)


# Attacks which were successful but only where the model got them correct in the first place
i_normal = (adversarial_y_normal != y) & (y == preds_normal)
i_overfit = (adversarial_y_overfit != y) & (y == preds_overfit)

wasserstein_distance_nd(x[i_normal, :], adversarial_normal[i_normal, :])
wasserstein_distance_nd(x[i_overfit, :], adversarial_overfit[i_overfit, :])

# separated by class
print(
    "Blue class",
    "\nNormal Accuracy",
    wasserstein_distance_nd(
        x[i_normal & y == 0, :],
        adversarial_normal[i_normal & y == 0, :],
    ),
    "\nOverfit Accuracy",
    wasserstein_distance_nd(
        x[i_overfit & y == 0, :],
        adversarial_overfit[i_overfit & y == 0, :],
    ),
    "\nOrange class",
    "\nNormal Accuracy",
    wasserstein_distance_nd(
        x[i_normal & y == 1, :],
        adversarial_normal[i_normal & y == 1, :],
    ),
    "\nOverfit Accuracy",
    wasserstein_distance_nd(
        x[i_overfit & y == 1, :],
        adversarial_overfit[i_overfit & y == 1, :],
    ),
)


##### Impact of sample size on wasserstein distance experiment
# Note: this experiment so far is just a copy of the one from above

np.random.seed(124)
sample_size = 100
scaler = StandardScaler()

x1 = np.vstack(
    (
        np.random.normal(loc=0, scale=1, size=(sample_size, 1)),
        np.random.normal(loc=0, scale=1, size=(sample_size, 1)),
    )
)
x2 = np.vstack(
    (
        np.random.normal(loc=5, scale=1, size=(sample_size, 1)),
        np.random.normal(loc=0, scale=1, size=(sample_size, 1)),
    )
)
x = np.hstack((x1, x2))
y = np.hstack((np.zeros(sample_size), np.ones(sample_size))).astype(int)
x = scaler.fit_transform(x)

plt.scatter(x[:, 0], x[:, 1], c=y)
plt.show()

normal = SVC(
    kernel="rbf",
    random_state=103,
)

overfit = SVC(
    kernel="rbf",
    random_state=104,
    gamma=50,
)

normal.fit(x, y)
overfit.fit(x, y)

preds_normal = normal.predict(x)
preds_overfit = overfit.predict(x)
(get_accuracy(normal.predict(x), y), get_accuracy(normal.predict(x), y))

adversarial_normal = HopSkipJump(SklearnClassifier(model=normal)).generate(x)
adversarial_overfit = HopSkipJump(SklearnClassifier(model=overfit)).generate(x)

adversarial_y_normal = normal.predict(adversarial_normal)
adversarial_y_overfit = overfit.predict(adversarial_overfit)

print(
    "Predictions on adv points - attack failure rate",
    "\nNormal Accuracy",
    np.mean(adversarial_y_normal == y).item(),
    "\nOverfit Accuracy:",
    np.mean(adversarial_y_overfit == y).item(),
)

plot_attacks_svm(x, y, adversarial_normal, adversarial_overfit, normal, overfit)
plot_attacks(x, y, adversarial_normal, adversarial_overfit, normal, overfit)


# Attacks which were successful but only where the model got them correct in the first place
i_normal = (adversarial_y_normal != y) & (y == preds_normal)
i_overfit = (adversarial_y_overfit != y) & (y == preds_overfit)

wasserstein_distance_nd(x[i_normal, :], adversarial_normal[i_normal, :])
wasserstein_distance_nd(x[i_overfit, :], adversarial_overfit[i_overfit, :])

# separated by class
print(
    "Blue class 0",
    "\nNormal Distance",
    wasserstein_distance_nd(
        x[i_normal & y == 0, :],
        adversarial_normal[i_normal & y == 0, :],
    ),
    "\nOverfit Distance",
    wasserstein_distance_nd(
        x[i_overfit & y == 0, :],
        adversarial_overfit[i_overfit & y == 0, :],
    ),
    "\nOrange class 1",
    "\nNormal Distance",
    wasserstein_distance_nd(
        x[i_normal & y == 1, :],
        adversarial_normal[i_normal & y == 1, :],
    ),
    "\nOverfit Distance",
    wasserstein_distance_nd(
        x[i_overfit & y == 1, :],
        adversarial_overfit[i_overfit & y == 1, :],
    ),
)


##### Impact of "level of overfitting" experiment
"""Just change gamma and explore what happens to the wasserstein distance when we have more/less overfitting"""

##### Other attacks


##### Plots

plot_attacks(x, y, adversarial_normal, adversarial_overfit, normal, overfit)

plot_attacks_svm(x, y, adversarial_normal, adversarial_overfit, normal, overfit)
