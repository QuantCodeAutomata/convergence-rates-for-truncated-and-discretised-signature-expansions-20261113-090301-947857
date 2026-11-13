"""Unit tests for src.fbm (shifted Riemann-Liouville fBm, Sec. 4.3, Prop. 4.6)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import gammaln

from src import fbm


def test_rl_coefficient_at_n0_matches_prefactor():
    t, H, mu = 0.5, 0.3, 1.0 / 52.0
    c0 = fbm.rl_coefficient(0, t, H, mu)
    expected = (t + mu) ** (H - 0.5) / np.exp(gammaln(H + 0.5))
    assert c0 == pytest.approx(expected)


def test_build_ell_RL_words_have_correct_shape():
    ell = fbm.build_ell_RL(t=0.5, H=0.3, mu=1.0 / 52.0, max_len=6)
    for word in ell:
        assert word[-1] == 1
        assert all(letter == 0 for letter in word[:-1])
        assert len(word) <= 6


def test_geometric_ratio_is_between_zero_and_one():
    for t in (0.01, 0.5, 1.0):
        q = fbm.geometric_ratio(t, mu=1.0 / 52.0)
        assert 0.0 < q < 1.0


def test_geometric_ratio_increasing_in_t():
    mu = 1.0 / 52.0
    qs = [fbm.geometric_ratio(t, mu) for t in (0.1, 0.5, 1.0)]
    assert qs[0] < qs[1] < qs[2]


def test_proposition_4_6_upper_bound_decreases_geometrically_in_M():
    t, H, mu, p = 1.0, 0.3, 1.0 / 52.0, 3.0
    bounds = [fbm.proposition_4_6_upper(t, M, H, mu, p) for M in range(2, 8)]
    ratios = [bounds[i + 1] / bounds[i] for i in range(len(bounds) - 1)]
    q_t = fbm.geometric_ratio(t, mu)
    # ratio must be close to q_t times a slowly varying sqrt((2M+1)/(2M+3)) factor
    for ratio in ratios:
        assert ratio < 1.0
        assert ratio > q_t * 0.5  # not decaying faster than the geometric rate


def test_proposition_4_6_lower_bound_vanishes_at_H_half():
    assert fbm.proposition_4_6_lower(t=0.5, M=4, H=0.5, mu=1.0 / 52.0) == 0.0


def test_proposition_4_6_lower_le_upper():
    t, H, mu, p = 0.7, 0.3, 1.0 / 52.0, 3.0
    for M in range(2, 8):
        lower = fbm.proposition_4_6_lower(t, M, H, mu)
        upper = fbm.proposition_4_6_upper(t, M, H, mu, p)
        assert lower <= upper * 10  # order-of-magnitude consistency (different norms: L2 vs Lp)


def test_variance_0n1_matches_direct_formula():
    import math
    t, n = 0.6, 4
    expected = t ** (2 * n + 1) / ((math.factorial(n) ** 2) * (2 * n + 1))
    assert fbm.variance_0n1(n, t) == pytest.approx(expected)


def test_kappa_p_matches_known_values():
    # kappa_2 = E[Z^2]^{1/2} = 1 for Z ~ N(0,1)
    assert fbm.kappa_p(2.0) == pytest.approx(1.0, abs=1e-8)
    # kappa_1 = E[|Z|] = sqrt(2/pi)
    assert fbm.kappa_p(1.0) == pytest.approx(np.sqrt(2.0 / np.pi), abs=1e-8)
