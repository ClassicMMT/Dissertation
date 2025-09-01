import numpy as np
from scipy import sparse
from scipy.spatial import distance_matrix
from scipy.stats import wasserstein_distance, wasserstein_distance_nd

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# The Earth Mover distance is just the mean absolute difference between the distributions
x = np.random.normal(loc=0, scale=1, size=100)
y = np.random.normal(loc=5, scale=1, size=100)
x.sort()
y.sort()
np.abs(x - y).mean()

# Check against scipy
wasserstein_distance(x, y)

# Computing the distance for p > 1
p = 2
(np.abs((x - y) ** p)).mean() ** (1 / p)

# multidimensional
x = np.random.normal(size=(100, 150))
y = np.random.normal(size=(100, 150))


wasserstein_distance_nd(x, y)
m, n = len(x), len(y)
