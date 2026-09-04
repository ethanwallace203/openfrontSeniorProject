"""Centralized reproducibility. Call ``seed_everything`` once, at the very
start of a training script, before any data loading or model
construction.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seeds Python's ``random``, ``numpy``, and ``torch`` (CPU and, if
    available, all CUDA devices) from a single seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
