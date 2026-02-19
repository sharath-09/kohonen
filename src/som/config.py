# som/config.py
from dataclasses import dataclass, field
from typing import Optional
import yaml

@dataclass
class SOMConfig:
    """Configuration for SOM training."""
    
    # Model parameters
    width: int = 10
    height: int = 10
    input_dim: int = 3
    learning_rate: float = 0.1
    sigma: Optional[float] = None
    seed: Optional[int] = 42
    
    # Training parameters
    n_iterations: int = 1000
    batch_size: Optional[int] = None  # For future batching support
    
    # I/O parameters
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    save_frequency: int = 100
    
    @classmethod
    def from_yaml(cls, path: str) -> 'SOMConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f)
    
    def validate(self):
        """Validate configuration parameters."""
        assert self.width > 0, "Width must be positive"
        assert self.height > 0, "Height must be positive"
        assert self.input_dim > 0, "Input dimension must be positive"
        assert 0 < self.learning_rate <= 1, "Learning rate must be in (0, 1]"
        assert self.n_iterations > 0, "Number of iterations must be positive"