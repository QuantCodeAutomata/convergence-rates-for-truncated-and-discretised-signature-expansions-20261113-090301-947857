"""Truncated tensor-algebra operations on scalar word series.

Custom implementation (paper Sec. 3.1 / Sec. 7): the paper builds coefficient
families such as ell^DE (Sec. 4.2) and the map Phi (Lemma 7.1) from scalar
tensor-algebra operations -- concatenation product, tensor powers, the
Neumann series (empty - q)^{-1} = sum_n q^{(x)n}, the shuffle product, and the
restriction operator ell|_i -- on the free tensor algebra T((A)) over a finite
alphabet A = {0, ..., d}. No standard quantitative library (iisignature,
signatory, sympy's free-algebra tools) implements this specific dual
coefficient-space algebra, so it is implemented here directly as dictionaries
mapping words (tuples of letters) to scalar coefficients, truncated to a
maximal word length.

A "series" is represented as dict[tuple[int, ...], float]; the empty word is
the key () and all operations truncate results to words of length <= max_len.
"""

from __future__ import annotations

import math
from itertools import product as iter_product
from typing import Dict, Tuple

Word = Tuple[int, ...]
Series = Dict[Word, float]


def series_add(a: Series, b: Series) -> Series:
    """Sum of two truncated series."""
    out = dict(a)
    for w, c in b.items():
        out[w] = out.get(w, 0.0) + c
    return {w: c for w, c in out.items() if c != 0.0}


def series_scale(a: Series, factor: float) -> Series:
    """Scalar multiple of a series."""
    return {w: factor * c for w, c in a.items() if factor * c != 0.0}


def single_letter(letter: int, coeff: float = 1.0) -> Series:
    """The series coeff * (letter) (one word of length 1)."""
    return {(letter,): coeff} if coeff != 0.0 else {}


def empty_series(coeff: float = 1.0) -> Series:
    """The series coeff * (empty word)."""
    return {(): coeff} if coeff != 0.0 else {}


def concat(a: Series, b: Series, max_len: int) -> Series:
    """Concatenation product (a . b)^v = sum_{v=v1 v2} a^{v1} b^{v2}, truncated.

    This is the natural (non-commutative) product of the free tensor algebra
    T((A)), dual to deconcatenation; it is the product used to build ell^DE
    via p . (empty - q)^{-1} (paper Sec. 4.2) and the Neumann series below.
    """
    out: Series = {}
    for wa, ca in a.items():
        if len(wa) > max_len:
            continue
        for wb, cb in b.items():
            w = wa + wb
            if len(w) > max_len:
                continue
            out[w] = out.get(w, 0.0) + ca * cb
    return {w: c for w, c in out.items() if c != 0.0}


def tensor_power(a: Series, n: int, max_len: int) -> Series:
    """a^{(x)n} under concatenation, truncated to max_len."""
    if n == 0:
        return empty_series(1.0)
    out = dict(a)
    for _ in range(n - 1):
        out = concat(out, a, max_len)
        if not out:
            break
    return out


def neumann_series(q: Series, max_len: int) -> Series:
    """(empty - q)^{-1} = sum_{n>=0} q^{(x)n}, truncated to max_len words.

    Finite because q has no constant term (q^{()} = 0 is required and
    checked), so q^{(x)n} vanishes in word-length once n > max_len.
    """
    if q.get((), 0.0) != 0.0:
        raise ValueError("Neumann series requires q to have no constant term")
    out = empty_series(1.0)
    term = empty_series(1.0)
    for n in range(1, max_len + 1):
        term = concat(term, q, max_len)
        if not term:
            break
        out = series_add(out, term)
    return out


def shuffle_product(a: Series, b: Series, max_len: int) -> Series:
    """Shuffle product a (shuffle) b, truncated to max_len.

    Used to build the Ito correction term in Phi (ell (shuffle) 2, Sec. 7):
    the shuffle of two words is the sum over all interleavings that preserve
    the relative order of each word's letters.
    """
    out: Series = {}
    for wa, ca in a.items():
        for wb, cb in b.items():
            if len(wa) + len(wb) > max_len:
                continue
            coeff = ca * cb
            for word in _shuffles(wa, wb):
                out[word] = out.get(word, 0.0) + coeff
    return {w: c for w, c in out.items() if c != 0.0}


def _shuffles(wa: Word, wb: Word):
    """All distinct interleavings of wa and wb (standard shuffle recursion)."""
    if not wa:
        yield wb
        return
    if not wb:
        yield wa
        return
    for tail in _shuffles(wa[1:], wb):
        yield (wa[0],) + tail
    for tail in _shuffles(wa, wb[1:]):
        yield (wb[0],) + tail


def restrict_last(ell: Series, letter: int) -> Series:
    """ell|_i : (ell|_i)(u) = ell(u + (i,)) for every word u (Lemma 7.1).

    Drops the trailing occurrence of `letter` from each word of ell and keeps
    only words that actually end with `letter`.
    """
    out: Series = {}
    for w, c in ell.items():
        if w and w[-1] == letter:
            out[w[:-1]] = out.get(w[:-1], 0.0) + c
    return out


def append_letter(ell: Series, letter: int, max_len: int) -> Series:
    """ell . (letter): append `letter` to every word of ell, truncated."""
    return concat(ell, single_letter(letter), max_len)


def exp_kernel_series(alpha: float, beta: float, max_len: int) -> Series:
    """0 . exp_{\\otimes}(beta * 0) with coefficient alpha (paper Sec. 4.2).

    exp_{\\otimes}(x) = sum_n x^{\\otimes n}/n! is the CONCATENATION
    (tensor) exponential. For the single-letter series beta*(0), the n-th
    concatenation power is beta^n * (0)^{\\otimes n} (the word 0 repeated n
    times), so exp_{\\otimes}(beta*(0)) = sum_n beta^n/n! * (0)^{\\otimes n}.
    Concatenating the letter 0 on the left then gives the word 0^{n+1} with
    coefficient alpha * beta^n / n!.
    """
    out: Series = {}
    for n in range(0, max_len):
        word = (0,) * (n + 1)
        out[word] = alpha * (beta ** n) / math.factorial(n)
    return {w: c for w, c in out.items() if c != 0.0}


def truncate(ell: Series, max_len: int) -> Series:
    """Restrict a series to words of length <= max_len."""
    return {w: c for w, c in ell.items() if len(w) <= max_len}


def words_up_to(alphabet_size: int, max_len: int):
    """Enumerate every word of length 0..max_len over {0,...,alphabet_size-1}."""
    yield ()
    for n in range(1, max_len + 1):
        for w in iter_product(range(alphabet_size), repeat=n):
            yield w
