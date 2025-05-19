from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.plotting import plot_decision_regions  # pip install mlxtend


def get_accuracy(pred, y):
    return (pred == y).mean().item()


def ss(original, adversarial):
    return ((original - adversarial) ** 2).sum(axis=1)


def hellinger_distance(original, adversarial):
    return (
        1
        / np.sqrt(2)
        * np.sqrt(((np.sqrt(original) - np.sqrt(adversarial)) ** 2).sum())
    )


def get_limits(x, eps=0.5):
    """
    For passing in to the plot svm functions "limits" parameter.
    eps determines the blank boundary around points.
    Returns (x_min, x_max, y_min, y_max).
    """
    return (
        x.min(axis=0)[0] - eps,
        x.max(axis=0)[0] + eps,
        x.min(axis=0)[1] - eps,
        x.max(axis=0)[1] + eps,
    )


def plot_attacks(
    x,
    y,
    adv_normal,
    adv_overfit,
    clf_normal,
    clf_overfit,
    extend_limits_with_fake_points=True,
):
    """
    General function for plotting the decision boundary for any classifier.
    Only works in 2D.
    For SVM, use plot_attacks_svm() instead since it's faster.
    """

    limits_normal = get_limits(np.concat((x, adv_normal)))
    limits_overfit = get_limits(np.concat((x, adv_overfit)))

    xlim = (
        min(limits_normal[0], limits_overfit[0]),
        max(limits_normal[1], limits_overfit[1]),
    )
    ylim = (
        min(limits_normal[2], limits_overfit[2]),
        max(limits_normal[3], limits_overfit[3]),
    )

    if extend_limits_with_fake_points:
        fake_points = np.vstack(
            (
                np.array((xlim[0], ylim[0])).reshape(-1, 2) - 5,
                np.array((xlim[1], ylim[1])).reshape(-1, 2) + 5,
            )
        )
        x = np.vstack(
            (
                x,
                fake_points,
            )
        )
        adv_normal = np.vstack(
            (
                adv_normal,
                fake_points,
            )
        )
        adv_overfit = np.vstack(
            (
                adv_overfit,
                fake_points,
            )
        )
        y = np.concatenate((y, np.zeros(2))).astype(int)

    fig, ax = plt.subplots(2, 2)

    plot_decision_regions(x, y, clf=clf_normal, legend=1, ax=ax[0, 0])
    ax[0, 0].set_title("Normal Model")
    ax[0, 0].set_xlim(limits_normal[0], limits_normal[1])
    ax[0, 0].set_ylim(limits_normal[2], limits_normal[3])
    plot_decision_regions(x, y, clf=clf_overfit, legend=1, ax=ax[0, 1])
    ax[0, 1].set_title("Overfit Model")
    ax[0, 1].set_xlim(limits_overfit[0], limits_overfit[1])
    ax[0, 1].set_ylim(limits_overfit[2], limits_overfit[3])
    plot_decision_regions(adv_normal, y, clf=clf_normal, legend=1, ax=ax[1, 0])
    ax[1, 0].set_title("Normal Adv Points")
    ax[1, 0].set_xlim(limits_normal[0], limits_normal[1])
    ax[1, 0].set_ylim(limits_normal[2], limits_normal[3])
    plot_decision_regions(adv_overfit, y, clf=clf_overfit, legend=1, ax=ax[1, 1])
    ax[1, 1].set_title("Overfit Adv Points")
    ax[1, 1].set_xlim(limits_overfit[0], limits_overfit[1])
    ax[1, 1].set_ylim(limits_overfit[2], limits_overfit[3])
    plt.tight_layout()
    plt.show()


# SVM SPECIFIC FUNCTIONS


def plot_attacks_svm(
    x,
    y,
    adversarial_normal,
    adversarial_overfit,
    normal_model,
    overfit_model,
    title=None,
):
    """
    Function producing four plots.
    1. Points with decision boundary for the Normal model
    2. Points with decision boundary for the Overfit model
    3. Adversarial points with boundary for the Normal model
    4. Adversarial points with boundary for the overfit model
    """

    limits_normal = get_limits(np.concat((x, adversarial_normal)))
    limits_overfit = get_limits(np.concat((x, adversarial_overfit)))
    fig, ax = plt.subplots(2, 2)

    plot_svm(
        normal_model,
        x,
        y,
        limits=limits_normal,
        support_vectors=False,
        ax=ax[0, 0],
        title="Normal Model",
    )
    plot_svm(
        overfit_model,
        x,
        y,
        limits=limits_overfit,
        support_vectors=False,
        ax=ax[0, 1],
        title="Overfit Model",
    )
    plot_svm(
        normal_model,
        adversarial_normal,
        y,
        limits=limits_normal,
        support_vectors=False,
        ax=ax[1, 0],
        title="Normal Adv Examples",
    )
    plot_svm(
        overfit_model,
        adversarial_overfit,
        y,
        limits=limits_overfit,
        support_vectors=False,
        ax=ax[1, 1],
        title="Overfit Adv Examples",
    )
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# Taken from: https://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html#sphx-glr-auto-examples-svm-plot-svm-kernels-py
def plot_svm(
    clf,
    X,
    y,
    title="Classifier",
    ax=None,
    support_vectors=False,
    index_attack=(
        None,
        None,
    ),  # tuple containing row index of original point and attack point vector/array
    limits=None,
    extend_limits_with_fake_points=True,
):
    """
    Plot SVC with decision boundary
    limits should be a tuple: (x_min, x_max, y_min, y_max)
    extend_limits_with_fake_points will make fake points (which won't be plotted) but that will extend the decision boundary.
    """

    # will need later
    axes_is_none = ax is None

    attacked_row_index, attack = index_attack
    if limits is None:
        x_min = X.min(axis=0)[0]
        x_max = X.max(axis=0)[0]
        y_min = X.min(axis=0)[1]
        y_max = X.max(axis=0)[1]
    else:
        x_min, x_max, y_min, y_max = limits

    if ax is None:
        fig, ax = plt.subplots()
    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

    if extend_limits_with_fake_points:
        X = np.vstack(
            (
                X,
                np.array((x_min, y_min)).reshape(-1, 2) - 5,
                np.array((x_max, y_max)).reshape(-1, 2) + 5,
            )
        )
        y = np.concatenate((y, np.zeros(2)))

    # Plot samples by color and add legend
    scatter = ax.scatter(X[:, 0], X[:, 1], s=150, c=y, label=y, edgecolors="k")
    ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    ax.set_title("Title")
    # _ = plt.show()

    # Plot the attack
    if attacked_row_index:
        ax.scatter(
            X[attacked_row_index, 0],
            X[attacked_row_index, 1],
            s=150,
            c="blue",
            label=y[attacked_row_index],
            edgecolors="k",
            zorder=10,
        )

    if attack is not None and attacked_row_index:
        ax.scatter(
            attack[:, 0],
            attack[:, 1],
            s=150,
            c="red",
            label=y[attacked_row_index],
            edgecolors="k",
            zorder=10,
        )

    # Settings for plotting
    if ax is None:
        _, ax = plt.subplots()

    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

    # Plot decision boundary and margins
    common_params = {"estimator": clf, "X": X, "ax": ax}
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="predict",
        plot_method="pcolormesh",
        alpha=0.3,
    )
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="decision_function",
        plot_method="contour",
        levels=[-1, 0, 1],
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],
    )

    if support_vectors:
        # Plot bigger circles around samples that serve as support vectors
        ax.scatter(
            clf.support_vectors_[:, 0],
            clf.support_vectors_[:, 1],
            s=150,
            facecolors="none",
            edgecolors="k",
        )

    # Plot samples by color and add legend
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolors="k")
    ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    ax.set_title(title)

    if axes_is_none:
        _ = plt.show()
