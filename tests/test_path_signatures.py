"""Unit tests for src.path_signatures (batched Chen's-identity signature engine)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src import path_signatures as ps


def test_segment_signature_matches_tensor_exponential():
    """Level n of a single affine segment with increment delta is
    delta^{(x)n} / n! (standard fact, Sec. 2 of the paper)."""
    delta = np.array([[1.0, 2.0]])  # one path, 2 letters
    M = 3
    levels = ps.segment_signature(delta, M)
    assert levels[0].shape == (1,)
    assert levels[0][0] == pytest.approx(1.0)
    assert levels[1][0] == pytest.approx(delta[0])
    expected_2 = np.outer(delta[0], delta[0]) / 2.0
    np.testing.assert_allclose(levels[2][0], expected_2)
    expected_3 = np.einsum("i,j,k->ijk", delta[0], delta[0], delta[0]) / 6.0
    np.testing.assert_allclose(levels[3][0], expected_3)


def test_chen_combine_reproduces_two_step_signature():
    """Chen's identity: signature of concatenated segments equals the tensor
    product sum; checked here against the direct level-2 formula for a
    two-step piecewise-linear path with explicit increments."""
    d1 = np.array([[1.0, 0.0]])
    d2 = np.array([[0.0, 1.0]])
    M, L = 2, 2
    A = ps.segment_signature(d1, M)
    B = ps.segment_signature(d2, M)
    C = ps.chen_combine(A, B, M, L)
    # Level 1 must be d1 + d2
    np.testing.assert_allclose(C[1][0], d1[0] + d2[0])
    # Level 2 word (0,1) must get a contribution 1 (from A_1 x B_1) since the
    # straight segments have no own level-2 contribution along distinct axes.
    assert C[2][0, 0, 1] == pytest.approx(1.0)
    assert C[2][0, 1, 0] == pytest.approx(0.0)


def test_identity_levels_are_neutral_for_chen_combine():
    delta = np.array([[0.3, -0.2]])
    M, L = 3, 2
    seg = ps.segment_signature(delta, M)
    identity = ps.identity_levels(1, M, L)
    combined = ps.chen_combine(identity, seg, M, L)
    for a, b in zip(seg, combined):
        np.testing.assert_allclose(a, b)


def test_inner_product_restricted_matches_full_when_within_level():
    coeffs = {(): 1.0, (0,): 2.0, (0, 1): 3.0}
    delta = np.array([[0.5, 0.1]])
    levels = ps.segment_signature(delta, 2)
    full = ps.inner_product(coeffs, levels)
    restricted = ps.truncated_inner_product(coeffs, levels, M=1)
    assert full != pytest.approx(restricted.item())
    only_level1 = ps.inner_product({(): 1.0, (0,): 2.0}, levels)
    np.testing.assert_allclose(restricted, only_level1)


def test_evaluate_along_path_matches_brownian_motion_identity():
    """For coeffs = single letter 1 (word (1,)), <ell, W-hat_t> = W_t exactly,
    at every checkpoint along the piecewise-linear path."""
    rng = np.random.default_rng(0)
    n_paths, N = 5, 20
    dW = rng.normal(scale=np.sqrt(1.0 / N), size=(n_paths, N, 1))
    time_col = np.full((n_paths, N, 1), 1.0 / N)
    deltas = np.concatenate([time_col, dW], axis=2)
    coeffs = {(1,): 1.0}
    fractions = [(N - 1, 1.0), (N // 2, 1.0)]
    out = ps.evaluate_along_path(deltas, coeffs, M=1, L=2, fractions=fractions)
    w_full = dW[:, :, 0].sum(axis=1)
    np.testing.assert_allclose(out[(N - 1, 1.0)], w_full, atol=1e-10)
    w_half = dW[:, : N // 2 + 1, 0].sum(axis=1)
    np.testing.assert_allclose(out[(N // 2, 1.0)], w_half, atol=1e-10)


def test_evaluate_along_path_time_word_matches_elapsed_time():
    """<ell, W-hat_t> for ell = word (0,) must equal the elapsed time t."""
    n_paths, N, T = 3, 10, 1.0
    dW = np.zeros((n_paths, N, 1))
    time_col = np.full((n_paths, N, 1), T / N)
    deltas = np.concatenate([time_col, dW], axis=2)
    coeffs = {(0,): 1.0}
    fractions = [(4, 1.0)]
    out = ps.evaluate_along_path(deltas, coeffs, M=1, L=2, fractions=fractions)
    np.testing.assert_allclose(out[(4, 1.0)], np.full(n_paths, 5 * T / N))
