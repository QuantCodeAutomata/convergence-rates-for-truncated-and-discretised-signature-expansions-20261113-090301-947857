"""Coefficient family ell^DE and reference solver for the linear delay equation
of Sec. 4.2 of the paper.

    dX_t = sum_i (a_i + b_i X_t + int_0^t K_i(t-s) X_s ds) dW-hat^i_t,
    K_i(t) = sum_k alpha_i^k exp(beta_i^k t), i in {0, 1}, X_0 = x.

ell^DE is built from the tensor-algebra operations of ``src.tensor_algebra``
following the construction given in the paper's implementation notes:

    c_i = b_i*empty + sum_k alpha_i^k * [0 . exp_(x)(beta_i^k * 0)]
    p   = x*empty + sum_i a_i*(i) - (1/2)(sum_i a_i b_i)*(0)
    q   = sum_i c_i.(i) - (1/2)(sum_i b_i c_i).(0)
    ell^DE = p . (empty - q)^{-1}
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src import tensor_algebra as ta
from src.tensor_algebra import Series


def build_ell_DE(
    x0: float,
    a: Sequence[float],
    b: Sequence[float],
    alpha: Sequence[float],
    beta: Sequence[float],
    max_len: int,
) -> Series:
    """Build ell^DE for i in {0, 1} with a single exponential term per kernel."""
    c = []
    for i in (0, 1):
        ci = ta.empty_series(b[i])
        ci = ta.series_add(ci, ta.exp_kernel_series(alpha[i], beta[i], max_len))
        c.append(ci)

    p = ta.empty_series(x0)
    for i in (0, 1):
        p = ta.series_add(p, ta.single_letter(i, a[i]))
    p = ta.series_add(p, ta.single_letter(0, -0.5 * (a[0] * b[0] + a[1] * b[1])))

    q: Series = {}
    for i in (0, 1):
        q = ta.series_add(q, ta.append_letter(c[i], i, max_len))
    b_dot_c: Series = {}
    for i in (0, 1):
        b_dot_c = ta.series_add(b_dot_c, ta.series_scale(c[i], b[i]))
    q = ta.series_add(q, ta.series_scale(ta.append_letter(b_dot_c, 0, max_len), -0.5))

    neumann = ta.neumann_series(q, max_len)
    ell = ta.concat(p, neumann, max_len)
    return ell


def simulate_reference(
    dW_ref: np.ndarray,
    x0: float,
    a: Sequence[float],
    b: Sequence[float],
    alpha: Sequence[float],
    beta: Sequence[float],
    T: float,
) -> np.ndarray:
    """Stratonovich-Heun (predictor-corrector) reference solution on the fine grid.

    Returns X at every grid point, shape (n_paths, N_ref + 1). The exponential
    kernel memory terms are carried as the Markovian lift
    Y_i(t) = int_0^t exp(beta_i (t-s)) X_s ds, which satisfies the ODE
    dY_i/dt = beta_i Y_i + X_t (standard affine-Volterra lift), so that the
    whole state (X, Y_0, Y_1) is simulated without recomputing the memory
    integral from scratch at every step.
    """
    n_paths, N_ref, _ = dW_ref.shape
    h = T / N_ref
    X = np.full(n_paths, x0)
    Y = np.zeros((2, n_paths))
    X_path = np.empty((n_paths, N_ref + 1))
    X_path[:, 0] = X

    def drift_diff(Xv, Yv):
        f0 = a[0] + b[0] * Xv + alpha[0] * Yv[0]
        f1 = a[1] + b[1] * Xv + alpha[1] * Yv[1]
        return f0, f1

    for k in range(N_ref):
        dW = dW_ref[:, k, 0]
        f0, f1 = drift_diff(X, Y)
        X_pred = X + f0 * h + f1 * dW
        Y_pred = np.stack([Y[i] + (beta[i] * Y[i] + X) * h for i in (0, 1)])
        f0p, f1p = drift_diff(X_pred, Y_pred)
        X_new = X + 0.5 * (f0 + f0p) * h + 0.5 * (f1 + f1p) * dW
        Y_new = np.stack([Y[i] + 0.5 * ((beta[i] * Y[i] + X) + (beta[i] * Y_pred[i] + X_pred)) * h
                           for i in (0, 1)])
        X, Y = X_new, Y_new
        X_path[:, k + 1] = X
    return X_path
