import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import umap
import torch


from art.attacks.evasion import HopSkipJump
from art.estimators.classification import SklearnClassifier


import random

np.random.seed(123)
random.seed(123)


x, y = make_classification(
    n_samples=100,
    n_features=3,
    n_informative=3,
    n_redundant=0,
    n_classes=2,
    random_state=123,
)

clf = SVC(random_state=123)
clf.fit(x, y)
adversarial = HopSkipJump(SklearnClassifier(clf)).generate(x)
correct_and_adversarial = (clf.predict(adversarial) != y) & (clf.predict(x) == y)

x_sub = x[correct_and_adversarial]
y_sub = y[correct_and_adversarial]
adv_sub = adversarial[correct_and_adversarial]

# Figure 1
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(
    x_sub[y_sub == 0, 0],
    x_sub[y_sub == 0, 1],
    x_sub[y_sub == 0, 2],
    label="class 0",
    c="green",
    alpha=0.2,
)
ax.scatter(
    x_sub[y_sub == 1, 0],
    x_sub[y_sub == 1, 1],
    x_sub[y_sub == 1, 2],
    label="class 1",
    c="blue",
    alpha=0.2,
)
ax.scatter(
    adversarial[:, 0],
    adversarial[:, 1],
    adversarial[:, 2],
    label="adversarial",
    c="red",
)
plt.legend()
plt.legend(["class 0", "class 1", "adversarial"])
plt.show()


# Figure 2
reducer = umap.UMAP(n_components=2, random_state=125, n_jobs=-1)
projection = reducer.fit_transform(adversarial)
plt.scatter(projection[:, 0], projection[:, 1], c="red", label="adversarial")
plt.legend()
plt.show()

# Figure 3
clusterer = KMeans(n_clusters=2)
clusters = clusterer.fit_predict(projection)
plt.scatter(projection[:, 0], projection[:, 1], c=np.where(clusters, "red", "orange"))
plt.show()

# Figure 4
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(
    x_sub[y_sub == 0, 0],
    x_sub[y_sub == 0, 1],
    x_sub[y_sub == 0, 2],
    label="class 0",
    c="green",
    alpha=0.2,
)
ax.scatter(
    x_sub[y_sub == 1, 0],
    x_sub[y_sub == 1, 1],
    x_sub[y_sub == 1, 2],
    label="class 1",
    c="blue",
    alpha=0.2,
)
ax.scatter(
    adversarial[:, 0],
    adversarial[:, 1],
    adversarial[:, 2],
    label="adversarial",
    c=np.where(clusters, "red", "orange"),
)
plt.legend()
plt.legend(["class 0", "class 1", "adversarial"])
plt.show()
