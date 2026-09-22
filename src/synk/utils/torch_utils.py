"""Reproducibility helpers: seeding, device selection, deterministic settings."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and torch; set CUDA determinism when available."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Return the best available torch device (CPU fallback is the default path)."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_summary() -> str:
    return f"torch={torch.__version__} cuda={torch.cuda.is_available()} device={get_device()}"


def worker_init_fn(worker_id: int) -> None:  # pragma: no cover - used by DataLoader
    """Deterministic worker seeding for DataLoaders."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
