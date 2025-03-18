import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import estimator_html_repr
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification

from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import ZooAttack
from art.attacks.evasion import HopSkipJump
from art.attacks.evasion import BoundaryAttack

from utils import plot_svm


# create fake data
x, y = make_classification(
    n_samples=200, n_features=2, n_informative=2, n_redundant=0, random_state=123
)

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
