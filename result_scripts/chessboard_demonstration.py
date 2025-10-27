"""
File to make the fig:chessboard_demonstration in overleaf.
"""

from src.utils import set_all_seeds
from src.datasets import make_chessboard
import matplotlib.pyplot as plt


random_state = 123
set_all_seeds(random_state)


x, y = make_chessboard(n_blocks=2, random_state=random_state)
x2, y2 = make_chessboard(n_blocks=4, random_state=random_state)
x3, y3 = make_chessboard(n_blocks=2, random_state=random_state, all_different_classes=True)
x4, y4 = make_chessboard(n_blocks=4, n_points_in_block=200, random_state=random_state, all_different_classes=True)

if True:
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].scatter(x[:, 0], x[:, 1], c=y, edgecolor="k")
    axs[1].scatter(x2[:, 0], x2[:, 1], c=y2, edgecolor="k")
    axs[2].scatter(x3[:, 0], x3[:, 1], c=y3, edgecolor="k")
    axs[3].scatter(x4[:, 0], x4[:, 1], c=y4, edgecolor="k")
    axs[0].set_title("Two Blocks - 2 Classes")
    axs[1].set_title("Four Blocks - 2 Classes")
    axs[2].set_title("Two Blocks - 4 Classes")
    axs[3].set_title("Four Blocks - 16 Classes")

    plt.tight_layout()
    plt.show()
