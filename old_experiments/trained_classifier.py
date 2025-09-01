import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
from utils import plot_svm

# NOT FINISHED EXPERIMENT

random_state = 100

x, y = make_classification(
    n_samples=300,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_classes=2,
    random_state=random_state,
)

# plot data
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.show()

# fit model
clf = make_pipeline(
    StandardScaler(),
    # C is the misclassification penalty. low C makes decision surface smooth, high C = big penalty
    # gamma how much influence a single observation has
    SVC(kernel="rbf"),
)

clf = SVC(kernel="rbf")
clf.fit(x, y)
preds = clf.predict(x)
(preds == y).mean()  # accuracy

# plot svm boundary

plot_svm(clf=clf, X=x, y=y)
np.arange(10) + np.arange(10)
