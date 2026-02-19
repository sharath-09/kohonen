# kohonen.py
import matplotlib.pyplot as plt
import numpy as np
import logging

from src.kohonen.train import train

logging.basicConfig(level=logging.DEBUG)

log = logging.getLogger(__name__)


def main() -> None:
    """
    Train a SOM Kohonen network.
    """
    np.random.seed(42)

    # Generate data
    input_data = np.random.random((10,3))
    image_data = train(input_data, 100, 10, 10)
    plt.imsave('100.png', image_data)

    # Generate data
    image_data = train(input_data, 1000, 100, 100)
    plt.imsave('1000.png', image_data)

    log.info("Training complete. Plots saved.")

    

if __name__ == '__main__':
    main()

