import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import wasserstein_distance, wasserstein_distance_nd
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr  # kl-divergence
from scipy.stats import entropy

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import ZooAttack, wasserstein
from art.attacks.evasion import HopSkipJump
from art.attacks.evasion import BoundaryAttack


from utils import (
    hellinger_distance,
    plot_svm,
    ss,
    get_limits,
    get_accuracy,
    plot_attacks_svm,
)

from bhattacharyya import bhatta_dist

x, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    n_classes=2,
    shuffle=True,
    # Some over lap with this seed. Both models perform well, but overfit does better on train
    # wasserstein adv is smaller than normal
    # random_state=423,
    # Not much overlap here, both models perform well equally
    # wasserstein adv is smaller than normal, but by a smaller degree
    # random_state=600,
)


# data modifications - Makes the data linearly-separable
# x[y == 1, 0] = x[y == 1, 0] + 5

plt.scatter(x[:, 0], x[:, 1], c=y)
plt.show()


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

# Attacks
attack_normal = HopSkipJump(SklearnClassifier(model=normal))
attack_overfit = HopSkipJump(SklearnClassifier(model=overfit))

adversarial_normal = attack_normal.generate(x)
adversarial_overfit = attack_overfit.generate(x)

adversarial_y_normal = normal.predict(adversarial_normal)
adversarial_y_overfit = overfit.predict(adversarial_overfit)

# classifier's adversarial accuracy
print(
    "Predictions on adv points",
    "\nNormal Accuracy",
    np.mean(adversarial_y_normal == y).item(),
    "\nOverfit Accuracy:",
    np.mean(adversarial_y_overfit == y).item(),
)


plot_attacks_svm(x, y, adversarial_normal, adversarial_overfit, normal, overfit)


# Wasserstein distance - larger values mean more movement so further apart

wasserstein_distance_nd(x, adversarial_normal)
wasserstein_distance_nd(x, adversarial_overfit)

# Battacharyya distance - this function only takes univariate features
# closer to 0 means more similar
(
    "Feature 1",
    bhatta_dist(x[:, 0], adversarial_normal[:, 0]).round(6).item(),
    bhatta_dist(x[:, 0], adversarial_overfit[:, 0]).round(6).item(),
)

(
    "Feature 2",
    bhatta_dist(x[:, 1], adversarial_normal[:, 1]).round(6).item(),
    bhatta_dist(x[:, 1], adversarial_overfit[:, 1]).round(6).item(),
)

# Can be infinite, which is not very good (may not generally be appropriate)
jensenshannon(x, adversarial_normal)
jensenshannon(x, adversarial_overfit)

# KL-divergence (attempts so far)
rel_entr(x, adversarial_normal).sum(axis=0)
rel_entr(x, adversarial_overfit).sum(axis=0)

entropy(x, adversarial_normal, base=2.0)

# hellinger distance
# scaler = MinMaxScaler((0, 1))
# x = scaler.fit_transform(x)
# adversarial_normal = scaler.transform(adversarial_normal)
# adversarial_overfit = scaler.transform(adversarial_overfit)

hellinger_distance(x[:, 0], adversarial_normal[:, 0])
hellinger_distance(x[:, 0], adversarial_overfit[:, 0])
hellinger_distance(x[:, 1], adversarial_normal[:, 1])
hellinger_distance(x[:, 1], adversarial_overfit[:, 1])
