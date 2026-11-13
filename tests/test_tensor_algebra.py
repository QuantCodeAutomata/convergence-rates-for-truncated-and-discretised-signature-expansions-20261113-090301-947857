"""Unit tests for src.tensor_algebra (free tensor algebra on scalar word series)."""

from __future__ import annotations

import math

import pytest

from src import tensor_algebra as ta


def test_series_add_cancels_zero_coefficients():
    a = {(0,): 1.0, (1,): 2.0}
    b = {(0,): -1.0, (1,): 1.0}
    out = ta.series_add(a, b)
    assert out == {(1,): 3.0}


def test_concat_is_associative():
    a = ta.single_letter(0, 2.0)
    b = ta.single_letter(1, 3.0)
    c = ta.single_letter(0, 0.5)
    left = ta.concat(ta.concat(a, b, 10), c, 10)
    right = ta.concat(a, ta.concat(b, c, 10), 10)
    assert left == right
    assert left == {(0, 1, 0): 3.0}


def test_concat_truncates_to_max_len():
    a = ta.single_letter(0, 1.0)
    out = ta.tensor_power(a, 5, max_len=3)
    assert all(len(w) <= 3 for w in out)
    assert (0, 0, 0) not in out or len(out) <= 1


def test_neumann_series_is_two_sided_inverse():
    """(empty - q)^{-1} (empty - q) = empty, truncated to max_len words."""
    max_len = 6
    q = {(0,): 0.3, (1,): -0.2, (0, 1): 0.1}
    inv = ta.neumann_series(q, max_len)
    empty_minus_q = ta.series_add(ta.empty_series(1.0), ta.series_scale(q, -1.0))
    product = ta.concat(inv, empty_minus_q, max_len)
    assert product.get((), 0.0) == pytest.approx(1.0)
    for word, coeff in product.items():
        if word != ():
            assert coeff == pytest.approx(0.0, abs=1e-10)


def test_neumann_series_rejects_constant_term():
    with pytest.raises(ValueError):
        ta.neumann_series({(): 1.0}, max_len=3)


def test_shuffle_product_is_commutative_and_counts_interleavings():
    a = ta.single_letter(0, 1.0)
    b = ta.single_letter(1, 1.0)
    sh_ab = ta.shuffle_product(a, b, max_len=5)
    sh_ba = ta.shuffle_product(b, a, max_len=5)
    assert sh_ab == sh_ba
    # shuffle of two single letters gives the two interleavings with coeff 1 each
    assert sh_ab == {(0, 1): 1.0, (1, 0): 1.0}


def test_shuffle_product_word_count():
    """|wa (shuffle) wb| has binom(|wa|+|wb|, |wa|) words, each counted with
    multiplicity equal to the number of interleavings producing it."""
    wa = ta.single_letter(1, 1.0)
    wa = ta.tensor_power(wa, 2, max_len=10)  # word (1,1)
    wb = ta.single_letter(1, 1.0)
    wb = ta.tensor_power(wb, 1, max_len=10)  # word (1,)
    out = ta.shuffle_product(wa, wb, max_len=10)
    # shuffle of (1,1) and (1,) over a single-letter alphabet collapses to one
    # word (1,1,1) with multiplicity C(3,1) = 3
    assert out == {(1, 1, 1): 3.0}


def test_restrict_last_inverts_append_letter():
    ell = {(0,): 2.0, (1, 0): 3.0, (1,): 5.0}
    appended = ta.append_letter(ell, 0, max_len=5)
    restricted = ta.restrict_last(appended, 0)
    assert restricted == ell


def test_exp_kernel_series_matches_exponential_expansion():
    alpha, beta, max_len = 1.5, 0.7, 8
    series = ta.exp_kernel_series(alpha, beta, max_len)
    for n in range(max_len):
        word = (0,) * (n + 1)
        expected = alpha * beta ** n / math.factorial(n)
        assert series.get(word, 0.0) == pytest.approx(expected)


def test_words_up_to_counts_correctly():
    words = list(ta.words_up_to(alphabet_size=2, max_len=3))
    # 1 (empty) + 2 + 4 + 8 = 15
    assert len(words) == 15
    assert () in words
    assert (0, 1, 0) in words
