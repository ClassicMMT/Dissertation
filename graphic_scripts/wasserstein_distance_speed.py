import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance_nd
from src.utils import set_all_seeds
import time
import matplotlib.pyplot as plt

random_state = 123
set_all_seeds(random_state)

sizes = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
speeds = []

for size in sizes:
    # simulate distribution
    d1 = np.random.randn(size, 10)
    d2 = np.random.randn(size, 10)

    start = time.time()

    result = wasserstein_distance_nd(d1, d2)

    end = time.time()
    time_taken = end - start

    speeds.append(round(time_taken, 3))

plt.bar(sizes, height=speeds)
plt.show()


len(speeds)
len(sizes)
