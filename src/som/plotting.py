from logging import log
import logging
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

log = logging.getLogger(__name__)

def save_plot(output_dir: Path, data: np.ndarray, filename: str) -> None:
    """Save an array as an image to the runs directory.

    Args:
        data: Image array to save.
        filename: Output filename.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_dir / filename, data)
    log.info(f"Saved {filename} to {output_dir}")


def plot_input(input_data: np.ndarray) -> None:
    """Plot input data as a 2D scatter coloured by RGB values.

    Args:
        input_data: Input array of shape (n_samples, 3).
    """
    fig, ax = plt.subplots()
    ax.scatter(input_data[:, 0], input_data[:, 1], c=input_data)
    ax.set_xlabel("R")
    ax.set_ylabel("G")
    plt.tight_layout()
    plt.show()
