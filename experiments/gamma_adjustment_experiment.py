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

import multiprocessing, warnings, os

multiprocessing.set_start_method("fork")
warnings.filterwarnings("ignore")

verbose = False

from utils import (
    ss,
    plot_svm,
    get_limits,
    get_accuracy,
    plot_attacks_svm,  # works with only svm but faster
    plot_attacks,  # should work with most classifiers but slower
)

np.random.seed(130)


x1 = np.random.normal(loc=0, scale=5, size=(200, 1))
x2 = np.vstack(
    (
        np.random.normal(loc=5, scale=1, size=(100, 1)),
        np.random.normal(loc=0, scale=1, size=(100, 1)),
    )
)
x = np.hstack((x1, x2))
y = np.hstack((np.zeros(100), np.ones(100))).astype(int)


# gammas = (
#     [gamma for gamma in range(1, 11)]
#     + [gamma for gamma in range(15, 31, 5)]
#     + [gamma for gamma in range(40, 101, 10)]
# )

gammas = [gamma / 10 for gamma in range(1, 21)]


def run(gamma):
    normal = SVC(
        kernel="rbf",
        random_state=int(gamma * 10) + 1,
    )

    overfit = SVC(
        kernel="rbf",
        random_state=int(gamma * 10),
        gamma=gamma,
    )

    normal.fit(x, y)
    overfit.fit(x, y)

    preds_normal = normal.predict(x)
    preds_overfit = overfit.predict(x)

    # classifier's original accuracy
    original_accuracies = (
        get_accuracy(preds_normal, y),
        get_accuracy(preds_overfit, y),
    )

    #### Attacks

    adversarial_normal = HopSkipJump(
        SklearnClassifier(model=normal), verbose=verbose
    ).generate(x)
    adversarial_overfit = HopSkipJump(
        SklearnClassifier(model=overfit), verbose=verbose
    ).generate(x)

    adversarial_y_normal = normal.predict(adversarial_normal)
    adversarial_y_overfit = overfit.predict(adversarial_overfit)

    # Predictions on adv points - attack failure rate
    attack_failure = (
        np.mean(adversarial_y_normal == y).item(),
        np.mean(adversarial_y_overfit == y).item(),
    )

    # make sure the model was correct and the attack was successful
    i_normal = (adversarial_y_normal != y) & (y == preds_normal)
    i_overfit = (adversarial_y_overfit != y) & (y == preds_overfit)

    wasserstein_distances = (
        wasserstein_distance_nd(
            x[i_normal, :],
            adversarial_normal[i_normal, :],
        ),
        wasserstein_distance_nd(
            x[i_overfit, :],
            adversarial_overfit[i_overfit, :],
        ),
    )

    ### By class distances
    blue_normal, blue_overfit, orange_normal, orange_overfit = (
        wasserstein_distance_nd(
            x[i_normal & y == 0, :],
            adversarial_normal[i_normal & y == 0, :],
        ),
        wasserstein_distance_nd(
            x[i_overfit & y == 0, :],
            adversarial_overfit[i_overfit & y == 0, :],
        ),
        wasserstein_distance_nd(
            x[i_normal & y == 1, :],
            adversarial_normal[i_normal & y == 1, :],
        ),
        wasserstein_distance_nd(
            x[i_overfit & y == 1, :],
            adversarial_overfit[i_overfit & y == 1, :],
        ),
    )

    return {
        "gamma": gamma,
        "normal_accuracy": original_accuracies[0],
        "overfit_accuracy": original_accuracies[1],
        "normal_attack_failure_rate": attack_failure[0],
        "overfit_attack_failure_rate": attack_failure[1],
        "normal_wasserstein": wasserstein_distances[0],
        "overfit_wasserstein": wasserstein_distances[1],
        "0_normal": blue_normal,
        "1_normal": orange_normal,
        "0_overfit": blue_overfit,
        "1_overfit": orange_overfit,
        "normal_model": normal,
        "overfit_model": overfit,
        "adversarial_normal": adversarial_normal,
        "adversarial_overfit": adversarial_overfit,
    }


with multiprocessing.Pool(processes=10) as pool:
    results = pool.map(run, gammas)

# save the models and adversarial points for plotting
normal_models = [result["normal_model"] for result in results]
overfit_models = [result["overfit_model"] for result in results]
adversarial_normals = [result["adversarial_normal"] for result in results]
adversarial_overfits = [result["adversarial_overfit"] for result in results]

# remove models from results so we can convert the results to a dataframe
for result in results:
    result.pop("normal_model")
    result.pop("overfit_model")
    result.pop("adversarial_normal")
    result.pop("adversarial_overfit")

# final results
full_data = pd.DataFrame(results)
# full_data.to_csv("data/gamma_experiment.csv", index=False)

old_data = pd.read_csv("data/gamma_experiment.csv")
all_data = pd.concat((old_data, full_data), axis=0)
# all_data.to_csv("data/gamma_experiment.csv", index=False)

# Attack plots
if True:
    i = 3
    gamma = gammas[i]
    adversarial_normal = adversarial_normals[i]
    adversarial_overfit = adversarial_overfits[i]
    normal = normal_models[i]
    overfit = overfit_models[i]
    plot_attacks_svm(
        x,
        y,
        adversarial_normal,
        adversarial_overfit,
        normal,
        overfit,
        title=f"Gamma: {gamma}",
    )


# other plots
if True:
    plt.plot(full_data.gamma, full_data.normal_wasserstein)
    plt.plot(full_data.gamma, full_data.overfit_wasserstein)
    plt.xlabel("Gamma")
    plt.ylabel("Wasserstein")
    plt.title("Overfitting Severity Experiment")
    plt.legend(["Normal", "Overfit"])
    plt.grid(axis="x", color="grey", alpha=0.1)
    plt.xticks(full_data.gamma)
    plt.show()

if True:
    # plt.plot(full_data.gamma, full_data["0_normal"])
    # plt.plot(full_data.gamma, full_data["1_normal"])
    plt.plot(full_data.gamma, full_data["0_overfit"])
    plt.plot(full_data.gamma, full_data["1_overfit"])
    plt.legend(
        [
            # "Class 0 - Normal",
            # "Class 1 - Normal",
            "Class 0 - Overfit",
            "Class 1 - Overfit",
        ]
    )
    plt.xlabel("Gamma")
    plt.ylabel("Wasserstein")
    plt.title("Overfitting Severity Experiment - By Class")
    plt.xticks(full_data.gamma)
    plt.show()

if True:
    # NOTE THAT THE CLASS CHOSEN IS BASED OFF THE PLOT ABOVE
    plt.plot(full_data.gamma, full_data["0_overfit"] / full_data["1_overfit"])
    plt.legend(["Overfit - (Class 0 / Class 1)"])
    plt.xlabel("Gamma")
    plt.ylabel("Wasserstein Ratio")
    plt.title("Overfitting Severity Experiment - Ratio By Class")
    # plt.ylim((0, 20))
    plt.axhline(1, color="grey", alpha=0.5)
    plt.xticks(full_data.gamma)
    plt.show()

if True:
    # plt.plot(full_data.gamma, full_data.normal_attack_failure_rate)
    plt.plot(full_data.gamma, full_data.overfit_attack_failure_rate)
    plt.legend(("Normal", "Overfit"))
    plt.xlabel("Gamma")
    plt.title("Attack Failure Rate")
    plt.ylabel("Failure Rate")
    plt.show()

if True:
    # Not a very interesting plot
    plt.plot(full_data.gamma, full_data.normal_accuracy)
    plt.plot(full_data.gamma, full_data.overfit_accuracy)
    plt.legend(("Normal Accuracy", "Overfit Accuracy"))
    plt.xlabel("Gamma")
    plt.title("Classifier Accuracy")
    plt.ylabel("Accuracy")
    plt.show()
