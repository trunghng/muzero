import random

import numpy as np
import torch


def ftensor(x: np.ndarray, device=None):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
