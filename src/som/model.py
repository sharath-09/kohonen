import numpy as np

from numpy.typing import NDArray
from typing import Tuple, overload

class Kohonen:
    """Self-Organizing Map implementation."""
    
    def __init__(
        self,
        width: int,
        height: int,
        input_dim: int,
        num_iterations: int,
        learning_rate: float = 0.1,
        sigma: float = None,
        seed: int = None
    ):
        """SOM Kohonen network.

        Args:
            width (int): Width of map.
            height (int): Height of map.
            input_dim (int): Input dimensions.
            learning_rate (float, optional): Initial Learning Rate. Defaults to 0.1.
            sigma (float, optional): Initial sigma (neighbourhood radius). Defaults to None.
            seed (int, optional): Optional seed, for reproducibility. Defaults to None.
        """        
        self.width = width
        self.height = height
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.sigma = sigma or max(width, height) / 2.0
        self.num_iterations = num_iterations
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize weights
        self.weights = np.random.random((width, height, input_dim))
        
        # Pre-compute coordinate grids (avoid recomputing each iteration)
        self.x_grid, self.y_grid = np.meshgrid(
            np.arange(width), 
            np.arange(height), 
            indexing='ij')
    
    def _find_bmu(self, input_vector: np.ndarray) -> Tuple[int, int]:
        """Find best matching unit for input vector.

        Args:
            input_vector (np.ndarray): Input vector.

        Returns:
            Tuple[int, int]: x,y index of BMU node.
        """     
        distances = np.sum((self.weights - input_vector) ** 2, axis=2)
        bmu_idx = np.argmin(distances)
        return np.unravel_index(bmu_idx, (self.width, self.height))
    
    def _compute_influence(
        self, 
        bmu_x: int, 
        bmu_y: int, 
        sigma: float
    ) -> np.ndarray:
        """Compute influence for all nodes in map.

        Args:
            bmu_x (int): x index of BMU
            bmu_y (int): y index of BMU
            sigma (float): Current influence radius, sigma

        Returns:
            np.ndarray: Influence map
        """        
        distances = np.sqrt(
            (self.x_grid - bmu_x) ** 2 + (self.y_grid - bmu_y) ** 2
        )
        return np.exp(-(distances ** 2) / (2 * sigma ** 2))
    
    def _update_weights(
        self,
        input_vector: np.ndarray,
        bmu_x: int,
        bmu_y: int,
        learning_rate: float,
        sigma: float
    ) -> None:
        """Update weights in map.

        Args:
            input_vector (np.ndarray): Input vector.
            bmu_x (int): X index of BMU
            bmu_y (int): Y index of BMU
            learning_rate (float): Current Learning rate
            sigma (float): Current influence radius
        """
        influence = self._compute_influence(bmu_x, bmu_y, sigma)
        # Broadcast influence across input dimensions
        self.weights += (
            learning_rate 
            * influence[:, :, np.newaxis] 
            * (input_vector - self.weights)
        )
        
    def _step(
        self,
        input_data: np.ndarray,
        iteration: int
    ) -> float:
        """Execute one training step.

        Args:
            input_data (np.ndarray): Training data of shape (n_samples, input_dim)
            iteration (int): Current iteration number

        Returns:
            float: Quantization error for this epoch
        """
        # Decay schedules
        decay_factor = np.exp(-iteration / (self.num_iterations / np.log(self.sigma)))
        current_lr = self.learning_rate * decay_factor
        current_sigma = self.sigma * decay_factor
        
        total_error = 0.0
        
        for input_vector in input_data:
            bmu_x, bmu_y = self._find_bmu(input_vector)
            self._update_weights(input_vector, bmu_x, bmu_y, current_lr, current_sigma)
            
            # Compute quantization error
            bmu_weights = self.weights[bmu_x, bmu_y]
            total_error += np.linalg.norm(input_vector - bmu_weights)
        
        return total_error / len(input_data)
    
    def train(self, input_data: NDArray, verbose: bool) -> None:
        """Train kohonen network.

        Args:
            input_data (NDArray): Input array of shape (n_samples, input_dim)
            verbose (bool): Set to true for verbose logging.
        """
        history = {'quantization_error': []}
        
        for iteration in range(self.num_iterations):
            error = self._step(input_data, iteration)
            history['quantization_error'].append(error)
            
            if verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}/{self.num_iterations}, Error: {error:.4f}")
        
        return history


class KohonenStochastic(Kohonen):
    """Self-Organizing Kohonen Map w/ Stochastic Random sampling."""

    def __init__(
        self,
        width: int,
        height: int,
        input_dim: int,
        num_iterations: int,
        learning_rate: float = 0.1,
        sigma: float = None,
        batch_size: int = 32,
        replacement: bool = False,
        seed: int = None,
    ):
        super().__init__(
            width=width,
            height=height,
            input_dim=input_dim,
            num_iterations=num_iterations,
            learning_rate=learning_rate,
            sigma=sigma,
            seed=seed,
        )
        self.batch_size = batch_size
        self.replacement = replacement
        self.rng = np.random.default_rng(seed)

    def _sample_batch(self, input_data: NDArray) -> NDArray:
        """Sample a random mini-batch from input data.

        Args:
            input_data (NDArray): Full input array of shape (n_samples, input_dim).

        Returns:
            NDArray: Sampled batch of shape (batch_size, input_dim).
        """
        n_samples = len(input_data)
        size = self.batch_size if self.replacement else min(self.batch_size, n_samples)
        indices = self.rng.choice(n_samples, size=size, replace=self.replacement)
        return input_data[indices]

    def _step(self, input_data: NDArray, iteration: int) -> float:
        """Execute one training step on a stochastically sampled mini-batch.

        Args:
            input_data (NDArray): Input array of shape (n_samples, input_dim).
            iteration (int): Current iteration number.

        Returns:
            float: Mean quantization error over the sampled batch.
        """
        decay_factor = np.exp(-iteration / (self.num_iterations / np.log(self.sigma)))
        current_lr = self.learning_rate * decay_factor
        current_sigma = self.sigma * decay_factor

        batch = self._sample_batch(input_data)
        total_error = 0.0

        for input_vector in batch:
            bmu_x, bmu_y = self._find_bmu(input_vector)
            self._update_weights(input_vector, bmu_x, bmu_y, current_lr, current_sigma)
            total_error += np.linalg.norm(input_vector - self.weights[bmu_x, bmu_y])

        return total_error / len(batch)