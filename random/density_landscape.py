"""
This file plots the density landscape.
"""

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from src.datasets import make_chessboard
from src.utils import plot_density_landscape, KNNDensity
import matplotlib.pyplot as plt


random_state = 123

# simulate data
x, y = make_chessboard(n_blocks=2, n_points_in_block=50, random_state=random_state)

# split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=random_state)

# the scaling works like this:
knndensity = KNNDensity(n_neighbours=5)
knndensity = knndensity.fit(x_train)
densities = knndensity.predict(x_test)

fig, axs = plt.subplots()
plot_density_landscape(axs, x, y, "Density Landscape", n_neighbours=5, point_size=10)
plt.show()
