# kohonen.py
import argparse
from datetime import datetime
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.som.train import train, train_stochastic
from src.som.plotting import plot_input, save_plot

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

RUNS_DIR = Path("runs", str(datetime.now().strftime("%Y%m%d_%H%M%S")))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a Kohonen Self-Organizing Map")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic mini-batch training instead of batch training",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size for stochastic training (default: 32)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2042,
        help="Random seed for reproducibility (default: 2042)",
    )
    parser.add_argument(
        "--plot-input",
        action="store_true",
        help="Visualise input data."
    )
    return parser.parse_args()

def main(stochastic: bool = False, batch_size: int = 32, seed: int = 2042) -> None:
    """Train a SOM Kohonen network using batch or stochastic training.

    Args:
        stochastic: If True, use stochastic mini-batch training.
        batch_size: Mini-batch size for stochastic training.
        seed: Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    input_data = rng.random((10, 3))

    if stochastic:
        log.info(f"Using stochastic training with batch size {batch_size}")
        train_fn = lambda *args, **kwargs: train_stochastic(
            *args, batch_size=batch_size, seed=seed, **kwargs
        )
        suffix = "stochastic"
    else:
        log.info("Using batch training")
        train_fn = train
        suffix = "batch"

    for n_iterations, width, height in [(100, 10, 10), (1000, 100, 100)]:
        image_data = train_fn(input_data, n_iterations, width, height)
        save_plot(RUNS_DIR, image_data, f"{n_iterations}_{suffix}.png")

    log.info("Training complete. Plots saved.")

if __name__ == "__main__":
    args = parse_args()
    main(stochastic=args.stochastic, batch_size=args.batch_size, seed=args.seed)