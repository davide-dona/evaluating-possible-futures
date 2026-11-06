import os

import numpy as np
import torch


def deterministic(seed: int) -> np.random.Generator:
    """Seed torch and make it use deterministic kernels only.

    Must run before the first CUDA call, since cuBLAS reads its workspace setting when it starts.

    Returns:
        A NumPy generator seeded the same way.
    """
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    return np.random.default_rng(seed)
