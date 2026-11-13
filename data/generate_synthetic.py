"""Simulated Brownian paths for the numerical experiments.

Every experiment in this replication operates on simulated stochastic
processes rather than observed market data (Sec. 4.2-4.4 of the paper), so
the only "data" generated here is Monte Carlo Brownian motion. No external
market-data service is used.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def brownian_increments(n_paths: int, n_steps: int, T: float, dim: int, seed: int) -> np.ndarray:
    """IID N(0, h) increments, shape (n_paths, n_steps, dim), h = T / n_steps."""
    if n_paths <= 0 or n_steps <= 0 or dim <= 0:
        raise ValueError("n_paths, n_steps and dim must be positive")
    h = T / n_steps
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=np.sqrt(h), size=(n_paths, n_steps, dim))


def aggregate_increments(dW_ref: np.ndarray, n_ref: int, n_coarse: int) -> np.ndarray:
    """Sum reference increments into n_coarse blocks of equal size.

    Requires n_ref to be an integer multiple of n_coarse, so that the
    coarse-grid Brownian values are an exact subsample of the reference path
    (same underlying realisation, coupled discretisations).
    """
    if n_ref % n_coarse != 0:
        raise ValueError("n_ref must be a multiple of n_coarse")
    block = n_ref // n_coarse
    n_paths, _, dim = dW_ref.shape
    return dW_ref.reshape(n_paths, n_coarse, block, dim).sum(axis=2)


def time_extended_deltas(dW: np.ndarray, T: float) -> np.ndarray:
    """Prepend the deterministic time increment h=T/N to each row of dW.

    Returns an array of shape (n_paths, N, 1 + dim) with letter 0 = time.
    """
    n_paths, N, dim = dW.shape
    h = T / N
    time_col = np.full((n_paths, N, 1), h)
    return np.concatenate([time_col, dW], axis=2)


def checkpoint_fractions(test_times: Tuple[float, ...], N: int, T: float):
    """Map test times to (segment_index, fraction) pairs on an N-step grid."""
    h = T / N
    out = []
    for t in test_times:
        pos = t / h
        k = int(np.floor(pos - 1e-9))
        k = min(max(k, 0), N - 1)
        f = pos - k
        f = min(max(f, 0.0), 1.0)
        if f == 0.0:
            k = max(k - 1, 0)
            f = 1.0
        out.append((k, f))
    return out
