from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt


def ss(original, adversarial):
    return ((original - adversarial) ** 2).sum(axis=1)


def get_limits(x, eps=0.5):
    """
    For passing in to the plot svm functions "limits" parameter.

    eps determines the blank boundary around points
    """
    return (
        x.min(axis=0)[0] - eps,
        x.max(axis=0)[0] + eps,
        x.min(axis=0)[1] - eps,
        x.max(axis=0)[1] + eps,
    )


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
):
    """
    Plot SVC with decision boundary
    """

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

    if ax is None:
        _ = plt.show()
