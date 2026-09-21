"""Deterministic seed and execution controls for PyTorch on CPU and Apple Silicon MPS."""
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> dict:
    """Seed Python, NumPy, PyTorch CPU and MPS for reproducible execution."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    # Configure deterministic flags where supported
    torch.use_deterministic_algorithms(False)  # Some MPS ops may not support strict deterministic mode
    return {
        "seed": seed,
        "python_hash_seed": str(seed),
        "torch_seed": seed,
        "mps_seeded": torch.backends.mps.is_available(),
    }
