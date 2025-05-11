import numpy as np
import pandas as pd
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


x1 = np.random.normal(loc=0, scale=5, size=(200, 1))
x2 = np.vstack(
    (
        np.random.normal(loc=5, scale=1, size=(100, 1)),
        np.random.normal(loc=0, scale=1, size=(100, 1)),
    )
)
x = np.hstack((x1, x2))
y = np.hstack((np.zeros(100), np.ones(100))).astype(int)

points_removed = [5 * i for i in range(20)]
accuracies = []
wassersteins = []
attack_failure_rates = []
by_class = []


for points_to_remove in points_removed:

    # for i in range(2):
    #     points_to_remove = points_removed[i]
    y_sub = y[points_to_remove:]
    x_sub = x[points_to_remove:, :]

    normal = SVC(
        kernel="rbf",
        random_state=101,
    )

    overfit = SVC(
        kernel="rbf",
        random_state=102,
        gamma=50,
    )

    normal.fit(x_sub, y_sub)
    overfit.fit(x_sub, y_sub)

    preds_normal = normal.predict(x_sub)
    preds_overfit = overfit.predict(x_sub)

    # classifier's original accuracy
    original_accuracies = (
        get_accuracy(preds_normal, y_sub),
        get_accuracy(preds_overfit, y_sub),
    )
    accuracies.append(original_accuracies)

    #### Attacks

    adversarial_normal = HopSkipJump(SklearnClassifier(model=normal)).generate(x_sub)
    adversarial_overfit = HopSkipJump(SklearnClassifier(model=overfit)).generate(x_sub)

    adversarial_y_normal = normal.predict(adversarial_normal)
    adversarial_y_overfit = overfit.predict(adversarial_overfit)

    # Predictions on adv points - attack failure rate
    attack_failure = (
        np.mean(adversarial_y_normal == y_sub).item(),
        np.mean(adversarial_y_overfit == y_sub).item(),
    )
    attack_failure_rates.append(attack_failure)

    wasserstein_distances = (
        wasserstein_distance_nd(x_sub, adversarial_normal),
        wasserstein_distance_nd(x_sub, adversarial_overfit),
    )
    wassersteins.append(wasserstein_distances)

    i_normal = (adversarial_y_normal != y_sub) & (y_sub == preds_normal)
    i_overfit = (adversarial_y_overfit != y_sub) & (y_sub == preds_overfit)

    ### By class experiment
    blue_normal, blue_overfit, orange_normal, orange_overfit = (
        wasserstein_distance_nd(
            x_sub[i_normal & y_sub == 0, :],
            adversarial_normal[i_normal & y_sub == 0, :],
        ),
        wasserstein_distance_nd(
            x_sub[i_overfit & y_sub == 0, :],
            adversarial_overfit[i_overfit & y_sub == 0, :],
        ),
        wasserstein_distance_nd(
            x_sub[i_normal & y_sub == 1, :],
            adversarial_normal[i_normal & y_sub == 1, :],
        ),
        wasserstein_distance_nd(
            x_sub[i_overfit & y_sub == 1, :],
            adversarial_overfit[i_overfit & y_sub == 1, :],
        ),
    )
    by_class.append((blue_normal, blue_overfit, orange_normal, orange_overfit))


attack_failure_rates = pd.DataFrame(
    attack_failure_rates,
    columns=("normal_attack_failure_rate", "overfit_attack_failure_rate"),
)
accuracies = pd.DataFrame(accuracies, columns=("normal_accuracy", "overfit_accuracy"))

wassersteins = pd.DataFrame(
    wassersteins, columns=("normal_wasserstein", "overfit_wasserstein")
)
by_class = pd.DataFrame(
    by_class, columns=("0_normal", "0_overfit", "1_normal", "1_overfit")
)
points_removed = pd.DataFrame(points_removed, columns=("points_removed",))

full_data = pd.concat(
    (points_removed, accuracies, attack_failure_rates, wassersteins, by_class), axis=1
)
# full_data.to_csv("data/imbalance_experiment_results.csv", index=False)

# Plots
full_data.columns

plot_attacks_svm(x_sub, y_sub, adversarial_normal, adversarial_overfit, normal, overfit)

if True:
    plt.plot(full_data.points_removed, full_data.normal_wasserstein)
    plt.plot(full_data.points_removed, full_data.overfit_wasserstein)
    plt.xlabel("Points Removed")
    plt.ylabel("Wasserstein")
    plt.title("Imbalance Experiment")
    plt.legend(["Normal", "Overfit"])
    plt.show()

if True:
    plt.plot(full_data.points_removed, full_data["0_normal"])
    plt.plot(full_data.points_removed, full_data["1_normal"])
    plt.plot(full_data.points_removed, full_data["0_overfit"])
    plt.plot(full_data.points_removed, full_data["1_overfit"])
    plt.legend(
        [
            "Class 0 - Normal",
            "Class 1 - Normal",
            "Class 0 - Overfit",
            "Class 1 - Overfit",
        ]
    )
    plt.xlabel("Points Removed")
    plt.ylabel("Wasserstein")
    plt.title("Imbalance Experiment - By Class (Class 0 is removed)")
    plt.show()

if True:
    # NOTE THAT THE CLASS CHOSEN IS BASED OFF THE PLOT ABOVE
    plt.plot(full_data.points_removed, full_data["1_normal"] / full_data["0_normal"])
    plt.plot(full_data.points_removed, full_data["1_overfit"] / full_data["0_overfit"])
    plt.legend(["Normal - (Class 1 / Class 0)", "Overfit - (Class 1 / Class 0)"])
    plt.xlabel("Points Removed")
    plt.ylabel("Wasserstein Ratio")
    plt.title("Imbalance Experiment - Ratio by Class (Class 0 is removed)")
    plt.ylim((0, 100))
    plt.show()

if True:
    plt.plot(full_data.points_removed, full_data.normal_attack_failure_rate)
    plt.plot(full_data.points_removed, full_data.overfit_attack_failure_rate)
    plt.legend(("Normal", "Overfit"))
    plt.xlabel("Points Removed")
    plt.title("Attack Failure Rate")
    plt.ylabel("Failure Rate")
    plt.show()

if True:
    # Not a very interesting plot
    plt.plot(full_data.points_removed, full_data.normal_accuracy)
    plt.plot(full_data.points_removed, full_data.overfit_accuracy)
    plt.legend(("Normal Accuracy", "Overfit Accuracy"))
    plt.xlabel("Points Removed")
    plt.title("Classifier Accuracy")
    plt.ylabel("Accuracy")
    plt.show()
