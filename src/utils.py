import os
import random
import numpy as np

def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)