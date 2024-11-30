import torch
import numpy as np
import random


# Function to set random seeds
def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Uncomment these lines if deterministic behavior is required
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False