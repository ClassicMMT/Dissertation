# Tree

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import make_classification

from art.attacks.evasion import DecisionTreeAttack
from art.estimators.classification import SklearnClassifier

# create fake data
x, y = make_classification(
    n_samples=200, n_features=2, n_informative=2, n_redundant=0, random_state=123
)

x, y = make_classification(
    n_samples=500,
    n_clusters_per_class=1,
    n_features=2,
    n_classes=3,
    n_informative=2,
    n_redundant=0,
    random_state=456,
)


# train model
classifier = DecisionTreeClassifier()
classifier.fit(x, y)

print(accuracy := np.mean(classifier.predict(x) == y))

# plot decision boundaries/regions
lower_x1, lower_x2 = x.min(axis=0)
upper_x1, upper_x2 = x.max(axis=0)
x1_points = np.linspace(start=lower_x1, stop=upper_x1, num=50)
x2_points = np.linspace(start=lower_x2, stop=upper_x2, num=50)
x1, x2 = np.meshgrid(x1_points, x2_points)
x1 = x1.flatten()
x2 = x2.flatten()
region_points = np.vstack((x1, x2)).T
region_labels = classifier.predict(region_points)

# plot data
plt.scatter(
    region_points[:, 0], region_points[:, 1], c=region_labels, alpha=0.1, marker="s"
)
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.show()


# Attacks
index = 1
art_classifier = SklearnClassifier(model=classifier)
attack = DecisionTreeAttack(classifier=art_classifier)
data_for_attack = x[1, :].reshape((-1, 2))
adv = attack.generate(data_for_attack)
adv_prediction = classifier.predict(adv)

# plot attack
x_without_index = np.delete(x, (index), axis=0)
y_without_index = np.delete(y, (index), axis=0)
plt.scatter(
    region_points[:, 0], region_points[:, 1], c=region_labels, alpha=0.1, marker="s"
)
plt.scatter(
    x_without_index[:, 0], x_without_index[:, 1], c=y_without_index
)  # original data
plt.scatter(x[index, 0], x[index, 1], c="blue")  # attacked point
plt.scatter(adv[:, 0], adv[:, 1], c="red")  # attack result
plt.show()
