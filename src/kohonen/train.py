# kohonen.py
import matplotlib.pyplot as plt
import numpy as np
import time
import logging
from functools import wraps

from numpy.typing import NDArray

log = logging.getLogger(__name__)

def timer(func):
    """Decorator that times the execution of a function."""
    @wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        log.debug(f"Function {func.__name__!r} took: {elapsed_time:.4f} seconds")
        return result
    return wrapper_timer

@timer
def train(input_data: NDArray, n_max_iterations : int, width : int, height: int) -> NDArray:
    """Train a Kohonen network.

    Args:
        input_data (string): NDArray of shape (N x 1 x 3)
        n_max_iterations (int): Number of iterations (T).
        width (int): Width of map.
        height (int): Height of map.

    Returns:
        NDArray: Weights after training. 
    """
    sigma_0 = max(width, height) / 2
    alpha_0 = 0.1
    time_constant = n_max_iterations / np.log(sigma_0)
    t_values = np.arange(n_max_iterations)

    weights = np.random.random((width, height, 3))
    
    sigma_schedule = sigma_0 * np.exp(-t_values / time_constant)
    alpha_schedule = alpha_0 * np.exp(-t_values / time_constant)

    xx, yy = np.meshgrid(np.arange(width), np.arange(height), indexing='ij')

    # Epochs
    for t in range(n_max_iterations):
        sigma_t = sigma_schedule[t]
        alpha_t = alpha_schedule[t]

        weights_flat = weights.reshape(-1, 3)

        for vt in input_data:
            #batch
            distances_sq = np.sum((weights_flat - vt) ** 2, axis=1)
            bmu_idx = np.argmin(distances_sq)
            bmu_x, bmu_y = np.unravel_index(bmu_idx, (width, height))

            distances_sq_grid = (xx - bmu_x) ** 2 + (yy - bmu_y) ** 2
            influence = np.exp(-distances_sq_grid / (2 * sigma_t ** 2))
            weights += alpha_t * influence[:, :, np.newaxis] * (vt - weights)
    return weights 