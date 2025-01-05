import random
import numpy as np
import torch

def set_random_seeds(seed):
    """
    Sets random seeds for reproducibility across random, numpy, and torch.

    Parameters:
    - seed: Integer seed value for reproducibility.
    """
    random.seed(seed)       # Set seed for Python's random module
    np.random.seed(seed)    # Set seed for NumPy's random functions
    torch.manual_seed(seed) # Set seed for PyTorch operations
