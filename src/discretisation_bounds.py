"""Word-by-word discretisation error bounds (Sec. 5.2/5.3, p.22-26):
Proposition 5.14 (strong, off-grid rate N^{-1/2}), Proposition 5.15 (strong,
grid points in d=1, rate N^{-1}) and Proposition 5.17 (weak, rate N^{-1}).

``Ŵ_t^v`` denotes the (Stratonovich) signature of Brownian motion, realised
numerically as the signature of its piecewise-linear interpolation on the
fine reference grid (large N_ref); ``W̃_t^v(N)`` is the signature of the
coarser N-step interpolation of the *same* path, following Sec. 5's
discretisation scheme and the Wong-Zakai convergence of signatures of
piecewise-linear approximations to Brownian motion.
"""

from __future__ import annotations

from math import floor
from typing import Tuple

import numpy as np
from scipy.special import gammaln

from src.tensor_algebra import Word


def weighted_length(v: Word) -> int:
    """||v|| = |v| + n_0(v): time letters (0) are counted twice (Sec. 2/5)."""
    return len(v) + sum(1 for letter in v if letter == 0)


def c_d(d: int) -> float:
    return 1.0 + d / 2.0


def strong_bound_rhs(v: Word, T: float, N: int, C: float = 1.0) -> float:
    """RHS of Proposition 5.14: C e^{C||v||} T^{||v||/2} N^{-1/2} / floor(||v||/2)!."""
    w = weighted_length(v)
    log_val = (
        np.log(C) + C * w + 0.5 * w * np.log(T) - 0.5 * np.log(N) - gammaln(floor(w / 2) + 1)
    )
    return float(np.exp(log_val))


def strong_bound_rhs_grid(v: Word, T: float, N: int, C: float = 1.0) -> float:
    """Proposition 5.15 analogue at grid points, d=1: the same prefactor with
    the off-grid N^{-1/2} replaced by the improved N^{-1} rate."""
    w = weighted_length(v)
    log_val = (
        np.log(C) + C * w + 0.5 * w * np.log(T) - np.log(N) - gammaln(floor(w / 2) + 1)
    )
    return float(np.exp(log_val))


def weak_bound_rhs(v: Word, T: float, N: int, d: int) -> float:
    """RHS of Proposition 5.17: zero for odd ||v||, else
    2(|v|+1)(|v|+2) (3 c_d T)^{||v||/2} / floor(||v||/2)! * N^{-1}."""
    w = weighted_length(v)
    if w % 2 != 0:
        return 0.0
    m = len(v)
    log_val = (
        np.log(2.0 * (m + 1) * (m + 2))
        + 0.5 * w * np.log(3 * c_d(d) * T)
        - gammaln(floor(w / 2) + 1)
        - np.log(N)
    )
    return float(np.exp(log_val))


def rosenthal_levy_area_moment(q: int, h: float, C_p: float = 1.0) -> float:
    """Rosenthal-type bound E[A_ij^{2q}] <~ C_p h^{2q} for the Levy area A_ij
    over one step of length h (Lemma 5.13); used only as an order-of-magnitude
    sanity check, not as a tight constant."""
    return C_p * h ** (2 * q)
