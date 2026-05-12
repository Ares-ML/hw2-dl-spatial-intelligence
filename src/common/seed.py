"""Reproducibility helpers for Python, NumPy, and PyTorch."""

from __future__ import annotations

import os
import random
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - NumPy is installed in the HW2 env.
    np = None  # type: ignore[assignment]

try:
    import torch
except ImportError:  # pragma: no cover - lets non-training tools import this module.
    torch = None  # type: ignore[assignment]


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for this operation.")
    return torch


def set_seed(seed: int = 42, deterministic: bool = False) -> int:
    """Seed Python, NumPy, PyTorch, and CUDA RNGs.

    Args:
        seed: The seed value to apply.
        deterministic: If true, request deterministic PyTorch algorithms where
            practical. This can reduce training speed but improves repeatability.

    Returns:
        The seed value, which is convenient to store in configs and logs.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    if np is not None:
        np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:  # pragma: no cover - older torch fallback.
                torch.use_deterministic_algorithms(True)
        else:
            torch.backends.cudnn.benchmark = True

    return seed


def seed_worker(worker_id: int) -> None:
    """Seed a PyTorch DataLoader worker from PyTorch's per-worker seed."""

    del worker_id  # PyTorch already folded worker_id into initial_seed().
    torch_module = _require_torch()
    worker_seed = torch_module.initial_seed() % 2**32
    random.seed(worker_seed)
    if np is not None:
        np.random.seed(worker_seed)


def make_generator(seed: int = 42):
    """Create a CPU PyTorch generator seeded for DataLoader shuffling."""

    torch_module = _require_torch()
    generator = torch_module.Generator()
    generator.manual_seed(seed)
    return generator
