# This script provides a utility function to set a random seed for all relevant libraries and components.
# It is useful for making the training process deterministic and the results reproducible.

import random
import torch
import numpy as np

def seed_everything(seed):
    """
    Set a random seed for all relevant libraries and components.

    Args:
        seed (int): The random seed value.
    """
    random_seed = seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)