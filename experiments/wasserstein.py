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

import numpy as np

np.vstack()

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


##### Linear separability experiment

##### Test and wasserstein correlations experiment
# does the wasserstein distance maintain the properties we like when using a test set?

##### Different models experiment


##### Approximate CDF distance metric experiment

##### Wasserstein distance but only for successful adversarial points experiment


##### Impact of sample size on wasserstein distance experiment


##### Other attacks


##### Plots

plot_attacks(x, y, adversarial_normal, adversarial_overfit, normal, overfit)

plot_attacks_svm(x, y, adversarial_normal, adversarial_overfit, normal, overfit)
