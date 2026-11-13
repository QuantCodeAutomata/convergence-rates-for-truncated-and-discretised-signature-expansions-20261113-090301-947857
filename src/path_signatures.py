"""Batched, Monte-Carlo signature engine for piecewise-linear paths.

Custom implementation (paper Sec. 2 / Eq. (2.2) Chen's identity): computes,
for a batch of Monte Carlo paths simultaneously, the truncated signature of a
piecewise-linear interpolation of a time-extended path (time plus one or more
Brownian directions) and its inner product with a scalar coefficient family
(``src.tensor_algebra.Series``). Signature levels are stored as dense numpy
arrays indexed by letter rather than by iisignature's internal word order, so
that they can be paired directly with the word dictionaries built in
``src.tensor_algebra`` without reverse-engineering a third-party index
convention.

Level n of an alphabet of size L is an array of shape (n_paths,) + (L,) * n;
level 0 is the array of ones (empty word). Word (i1, ..., in) corresponds to
index [: , i1, ..., in].
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.tensor_algebra import Series, Word

Levels = List[np.ndarray]


def segment_signature(delta: np.ndarray, M: int) -> Levels:
    """Exact signature (levels 0..M) of one affine segment with increment delta.

    For a straight-line path the level-n component of word (i1,...,in) is
    delta_{i1} * ... * delta_{in} / n!, i.e. the truncated tensor exponential
    of delta (standard fact used throughout Sec. 2 of the paper).
    """
    n_paths = delta.shape[0]
    levels: Levels = [np.ones(n_paths)]
    if M >= 1:
        levels.append(delta.copy())
    cur = delta
    for n in range(2, M + 1):
        cur = np.einsum("p...,pj->p...j", cur, delta) / n
        levels.append(cur)
    return levels


def _outer_batched(a: np.ndarray, b: np.ndarray, L: int) -> np.ndarray:
    """Batched outer product of a (levels k) and b (level n-k) over letters."""
    n_paths = a.shape[0]
    ka = a.reshape(n_paths, -1, 1)
    kb = b.reshape(n_paths, 1, -1)
    out = (ka * kb).reshape((n_paths,) + (L,) * (a.ndim - 1 + b.ndim - 1))
    return out


def chen_combine(A: Levels, B: Levels, M: int, L: int) -> Levels:
    """Chen's identity: signature of the concatenation of two path segments.

    C_n = sum_{k=0}^{n} A_k (x) B_{n-k}  (Eq. (2.2) of the paper).
    """
    C: Levels = []
    for n in range(M + 1):
        acc = None
        for k in range(n + 1):
            term = _outer_batched(A[k], B[n - k], L)
            acc = term if acc is None else acc + term
        C.append(acc)
    return C


def identity_levels(n_paths: int, M: int, L: int) -> Levels:
    """Signature of the empty path: level 0 = 1, all higher levels = 0."""
    levels: Levels = [np.ones(n_paths)]
    for n in range(1, M + 1):
        levels.append(np.zeros((n_paths,) + (L,) * n))
    return levels


def inner_product(coeffs: Series, levels: Levels) -> np.ndarray:
    """<ell, signature> = sum_v ell^v * signature^v, batched over paths."""
    n_paths = levels[0].shape[0]
    out = np.zeros(n_paths)
    for word, c in coeffs.items():
        n = len(word)
        if n >= len(levels):
            continue
        out = out + c * levels[n][(slice(None),) + word]
    return out


def truncated_inner_product(coeffs: Series, levels: Levels, M: int) -> np.ndarray:
    """<ell^{<=M}, signature>, restricting ``coeffs`` to words of length <= M
    and reusing signature levels already computed at a higher max level."""
    restricted = {w: c for w, c in coeffs.items() if len(w) <= M}
    return inner_product(restricted, levels)


def levels_along_path(
    deltas: np.ndarray,
    M: int,
    L: int,
    fractions: Sequence[Tuple[int, float]],
) -> Dict[Tuple[int, float], Levels]:
    """Raw signature levels (0..M) of the path at each requested checkpoint.

    Computing the levels once at the highest required M and reusing them for
    every truncation M' <= M (via ``inner_product`` with the coefficients
    restricted to words of length <= M') avoids recomputing Chen's identity
    from scratch for every truncation level tested in an experiment.
    """
    n_paths, N, _ = deltas.shape
    by_segment: Dict[int, List[float]] = {}
    for k, f in fractions:
        by_segment.setdefault(k, []).append(f)

    running = identity_levels(n_paths, M, L)
    out: Dict[Tuple[int, float], Levels] = {}
    for k in range(N):
        delta_k = deltas[:, k, :]
        if k in by_segment:
            for f in by_segment[k]:
                if f >= 1.0:
                    continue
                partial = segment_signature(f * delta_k, M)
                out[(k, f)] = chen_combine(running, partial, M, L)
        full = segment_signature(delta_k, M)
        running = chen_combine(running, full, M, L)
        if k in by_segment and 1.0 in by_segment[k]:
            out[(k, 1.0)] = [level.copy() for level in running]
    return out


def evaluate_along_path(
    deltas: np.ndarray,
    coeffs: Series,
    M: int,
    L: int,
    fractions: Sequence[Tuple[int, float]],
) -> Dict[Tuple[int, float], np.ndarray]:
    """<ell^{<=M}, path signature> at requested (segment_index, fraction) points.

    ``deltas`` has shape (n_paths, N, L): the N affine-segment increments of
    the piecewise-linear path (segment k runs from grid point k to k+1).
    ``fractions`` lists points as (k, f) with f in (0, 1]; f=1 is the grid
    point at the end of segment k, f<1 is strictly inside segment k (the
    piecewise-linear interpolation restricted to [0, local fraction f] of an
    affine segment is itself affine with increment f * delta_k).
    """
    n_paths, N, _ = deltas.shape
    by_segment: Dict[int, List[float]] = {}
    for k, f in fractions:
        by_segment.setdefault(k, []).append(f)

    running = identity_levels(n_paths, M, L)
    out: Dict[Tuple[int, float], np.ndarray] = {}
    for k in range(N):
        delta_k = deltas[:, k, :]
        if k in by_segment:
            for f in by_segment[k]:
                if f >= 1.0:
                    continue
                partial = segment_signature(f * delta_k, M)
                combined = chen_combine(running, partial, M, L)
                out[(k, f)] = inner_product(coeffs, combined)
        full = segment_signature(delta_k, M)
        running = chen_combine(running, full, M, L)
        if k in by_segment and 1.0 in by_segment[k]:
            out[(k, 1.0)] = inner_product(coeffs, running)
    return out
