import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42):
    # 1. Python built-in random module
    random.seed(seed)

    # 2. Python hash seed
    # Note: This only affects subprocesses created after setting the variable.
    # To apply it to the current process, set the environment variable
    # in the terminal before running the script.
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 3. NumPy
    np.random.seed(seed)

    # 4. PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU training

    # 5. cuDNN & deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Enable deterministic algorithms across PyTorch
    # Prevent potential CUDA >= 10.2 issues with deterministic execution
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)

    print(
        f"[*] Global random seed set to {seed} for Python, NumPy, PyTorch, and cuDNN."
    )


def seed_worker(worker_id):
    """Initialize a deterministic random seed for each DataLoader worker."""
    # Get the current PyTorch seed assigned by the DataLoader and Generator
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
