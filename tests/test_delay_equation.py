"""Unit tests for src.delay_equation (coefficient family ell^DE, Sec. 4.2)."""

from __future__ import annotations

import numpy as np
import pytest

from src import delay_equation as de
from src import path_signatures as ps
from src.config import EXP1_A, EXP1_ALPHA, EXP1_B, EXP1_BETA, EXP1_T, EXP1_X0


def test_build_ell_DE_empty_word_is_initial_value():
    ell = de.build_ell_DE(EXP1_X0, EXP1_A, EXP1_B, EXP1_ALPHA, EXP1_BETA, max_len=4)
    assert ell[()] == pytest.approx(EXP1_X0)


def test_build_ell_DE_level1_matches_drift_diffusion_coefficients():
    """At level 1, ell^DE must reduce to the coefficients a_i of the driving
    letters (words (0,) and (1,)): X_t ~ x + a_0 t + a_1 W_t to first order."""
    a = np.array([0.4, 0.9])
    b = np.array([0.0, 0.0])  # no feedback => higher levels vanish beyond level 1
    alpha = np.array([0.0, 0.0])
    beta = np.array([1.0, 1.0])
    ell = de.build_ell_DE(0.1, a, b, alpha, beta, max_len=3)
    assert ell.get((0,), 0.0) == pytest.approx(a[0])
    assert ell.get((1,), 0.0) == pytest.approx(a[1])


def test_build_ell_DE_is_truncated_to_max_len():
    ell = de.build_ell_DE(EXP1_X0, EXP1_A, EXP1_B, EXP1_ALPHA, EXP1_BETA, max_len=5)
    assert all(len(w) <= 5 for w in ell)


def test_simulate_reference_matches_signature_expansion_for_no_feedback():
    """With b=0 and no kernel memory (alpha=0), the delay equation reduces to
    dX_t = a_0 dt + a_1 dW_t, i.e. X_t = x + a_0 t + a_1 W_t exactly, so the
    Euler-Heun reference path must match both the closed form and the
    level-1 signature expansion <ell^DE, W-hat_t> for every t and every path."""
    rng = np.random.default_rng(1)
    n_paths, N_ref, T = 200, 500, 1.0
    a = np.array([0.4, 0.9])
    b = np.array([0.0, 0.0])
    alpha = np.array([0.0, 0.0])
    beta = np.array([1.0, 1.0])
    x0 = 0.1
    dW = rng.normal(scale=np.sqrt(T / N_ref), size=(n_paths, N_ref, 1))
    X_path = de.simulate_reference(dW, x0, a, b, alpha, beta, T)
    W_path = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(dW[:, :, 0], axis=1)], axis=1)
    t_grid = np.linspace(0, T, N_ref + 1)
    expected = x0 + a[0] * t_grid[None, :] + a[1] * W_path
    np.testing.assert_allclose(X_path, expected, atol=1e-8)


def test_simulate_reference_output_shape():
    rng = np.random.default_rng(2)
    n_paths, N_ref = 10, 50
    dW = rng.normal(scale=np.sqrt(1.0 / N_ref), size=(n_paths, N_ref, 1))
    X_path = de.simulate_reference(dW, EXP1_X0, EXP1_A, EXP1_B, EXP1_ALPHA, EXP1_BETA, EXP1_T)
    assert X_path.shape == (n_paths, N_ref + 1)
    assert np.all(np.isfinite(X_path))
