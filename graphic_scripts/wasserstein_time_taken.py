"""
This script measures how much time is taken for
"""

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance_nd
from src.utils import set_all_seeds
import time
import matplotlib.pyplot as plt

random_state = 123
set_all_seeds(random_state)

results = {"sizes": [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], "time_taken_seconds": []}

for size in results["sizes"]:
    # simulate distribution
    d1 = np.random.randn(size, 10)
    d2 = np.random.randn(size, 10)

    start = time.time()

    result = wasserstein_distance_nd(d1, d2)

    end = time.time()
    time_taken = end - start

    results["time_taken_seconds"].append(round(time_taken, 3))


data = pd.DataFrame(results)
data.to_csv("saved_results/wasserstein_time_taken.csv", index=False)
