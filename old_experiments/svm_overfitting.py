import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import data
from sklearn.base import estimator_html_repr
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import ZooAttack
from art.attacks.evasion import HopSkipJump
from art.attacks.evasion import BoundaryAttack

from utils import plot_svm, ss, get_limits


# create fake data
x, y = make_classification(
    n_samples=50, n_features=2, n_informative=2, n_redundant=0, random_state=123
)


# Not overfit

# train model
classifier = SVC(
    kernel="rbf",
    random_state=123,
)
classifier.fit(x, y)

print(accuracy := np.mean(classifier.predict(x) == y))

# Plot data
plot_svm(classifier, x, y)


# Attacks
index = 1
art_classifier = SklearnClassifier(model=classifier)
# attack = ZooAttack(art_classifier, targeted=False, max_iter=100, learning_rate=0.01)
# attack = HopSkipJump(art_classifier)
attack = BoundaryAttack(art_classifier, targeted=False)
data_for_attack = x[index : index + 1]
adv = attack.generate(data_for_attack, y[index : index + 1])
adv_prediction = classifier.predict(adv)

plot_svm(classifier, x, y, index_attack=(index, adv))


# OVERFIT

# train model
classifier = SVC(
    kernel="rbf",
    random_state=123,
    # C = ,
    gamma=50,
)
classifier.fit(x, y)

print(accuracy := np.mean(classifier.predict(x) == y))

# Plot data
plot_svm(classifier, x, y)


# Attacks
index = 1
art_classifier = SklearnClassifier(model=classifier)
# attack = ZooAttack(art_classifier, targeted=False, max_iter=100, learning_rate=0.01)
attack = HopSkipJump(art_classifier)
# attack = BoundaryAttack(art_classifier, targeted=False)
data_for_attack = x[index : index + 3]
adv = attack.generate(data_for_attack, y[index : index + 1])
adv_prediction = classifier.predict(adv)
ss(data_for_attack, adv)


plot_svm(classifier, x, y, index_attack=(index, adv))


## Tests

scaler = StandardScaler()
x = scaler.fit_transform(x)

normal = SVC(
    kernel="rbf",
    random_state=123,
)

overfit = SVC(
    kernel="rbf",
    random_state=123,
    gamma=50,
)

normal.fit(x, y)
overfit.fit(x, y)

art_normal = SklearnClassifier(model=normal)
art_overfit = SklearnClassifier(model=overfit)

attack_normal = HopSkipJump(art_normal)
attack_overfit = HopSkipJump(art_overfit)

adversarial_normal = attack_normal.generate(x)
adversarial_overfit = attack_overfit.generate(x)

limits_normal = get_limits(np.concat((x, adversarial_normal)))
limits_overfit = get_limits(np.concat((x, adversarial_overfit)))

adversarial_y_normal = normal.predict(adversarial_normal)
adversarial_y_overfit = overfit.predict(adversarial_overfit)

accuracy_normal = np.mean(adversarial_y_normal == y)
accuracy_overfit = np.mean(adversarial_y_overfit == y)
(accuracy_normal, accuracy_overfit)

# Plot results
fig, ax = plt.subplots(2, 2)

plot_svm(normal, x, y, limits=limits_normal, support_vectors=False, ax=ax[0, 0])
plot_svm(overfit, x, y, limits=limits_overfit, support_vectors=False, ax=ax[0, 1])
plot_svm(
    normal,
    adversarial_normal,
    y,
    limits=limits_normal,
    support_vectors=False,
    ax=ax[1, 0],
)
plot_svm(
    overfit,
    adversarial_overfit,
    y,
    limits=limits_overfit,
    support_vectors=False,
    ax=ax[1, 1],
)
plt.show()

np.sqrt(((x - adversarial_normal) ** 2).sum(axis=1)).mean()
np.sqrt(((x - adversarial_overfit) ** 2).sum(axis=1)).mean()

# np.std(x, axis=0)
# np.mean(x, axis=0)
