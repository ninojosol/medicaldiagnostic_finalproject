"""Reproducibility helpers.

Read docs/REPRODUCIBILITY.md for the honest limits of what these guarantee.
Seeding removes *most* run-to-run variation, but bit-exact reproducibility across
different GPUs, driver versions, or cuDNN versions is not achievable.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    """Seed Python, NumPy and PyTorch RNGs.

    Parameters
    ----------
    seed:
        The seed applied to every RNG.
    deterministic:
        When True, ask cuDNN for deterministic algorithms. This costs speed and
        some ops still have no deterministic implementation, so we use
        ``warn_only=True`` rather than crashing mid-training.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Needed for deterministic behaviour of some CUDA matmul kernels.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (RuntimeError, TypeError):  # older torch builds
            pass
    else:
        torch.backends.cudnn.benchmark = True

    return seed


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` so augmentations are reproducible per worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = 42) -> torch.Generator:
    """A torch Generator to hand to DataLoader(generator=...) for stable shuffling."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
