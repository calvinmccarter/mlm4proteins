"""Multi-objective scalarization: NDS (Pareto) and STS (smooth Tchebycheff)."""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def _to_minimize(scores: np.ndarray, directions: list[str]) -> np.ndarray:
    """Flip signs so every column is minimization-direction.

    ``scores`` is shape ``[N, M]`` with M objectives; ``directions[j]`` is
    ``"minimize"`` or ``"maximize"`` for column j.
    """
    out = scores.astype(np.float64).copy()
    for j, d in enumerate(directions):
        if d == "maximize":
            out[:, j] = -out[:, j]
        elif d != "minimize":
            raise ValueError(f"unknown direction {d!r} at column {j}")
    return out


def zscore(values: np.ndarray, eps: float = 0.01) -> np.ndarray:
    """Standardize each column of ``values``. ``eps`` is added to the std."""
    mu = values.mean(axis=0, keepdims=True)
    sigma = values.std(axis=0, keepdims=True) + eps
    return (values - mu) / sigma


def smooth_tchebycheff(
    scores: np.ndarray,
    directions: list[str],
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Smooth Tchebycheff scalarization (Lin et al. 2024), minimize-direction.

    All inputs are first converted to minimization direction by sign-flipping
    columns marked ``"maximize"``. They are then z-scored across the batch.
    The aggregate is ``log Σ_j exp(c_j * z_j)`` with all ``c_j > 0`` (the
    paper's convention after sign-flipping) — lower output is better.

    Args:
        scores: ``[N, M]`` raw objective values across the candidate set.
        directions: list of length M, each ``"minimize"`` or ``"maximize"``.
        weights: optional ``[M]`` non-negative per-objective weights. Larger
            weight = greater emphasis. Defaults to all ones.

    Returns:
        ``[N]`` aggregated scores. Lower is preferred.
    """
    n, m = scores.shape
    if weights is None:
        weights = np.ones(m)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (m,) or (weights < 0).any():
        raise ValueError("weights must be a non-negative vector of length M")

    minimized = _to_minimize(scores, directions)
    z = zscore(minimized)
    return logsumexp(z * weights[None, :], axis=1)


def non_dominated_rank(
    scores: np.ndarray,
    directions: list[str],
) -> np.ndarray:
    """Pareto non-dominated sort. Rank 0 = first front (best), 1 = second, etc.

    All objectives are converted to minimization direction first. Returns
    ``int[N]`` ranks. Lower rank is preferred.
    """
    minimized = _to_minimize(scores, directions)
    n = minimized.shape[0]
    ranks = np.full(n, -1, dtype=np.int64)
    remaining = np.arange(n)
    current = 0
    while remaining.size > 0:
        sub = minimized[remaining]
        # i is non-dominated if no j in sub strictly dominates it
        dominated = np.zeros(sub.shape[0], dtype=bool)
        for k in range(sub.shape[0]):
            le = np.all(sub <= sub[k], axis=1)
            lt = np.any(sub < sub[k], axis=1)
            if (le & lt & (np.arange(sub.shape[0]) != k)).any():
                dominated[k] = True
        front_idx = remaining[~dominated]
        ranks[front_idx] = current
        remaining = remaining[dominated]
        current += 1
    return ranks


def nds_score(
    scores: np.ndarray,
    directions: list[str],
    tiebreaker: np.ndarray | None = None,
) -> np.ndarray:
    """Combine non-dominated rank with a tiebreaker into a single sortable score.

    Lower output is preferred. ``tiebreaker`` (e.g. pseudo-perplexity) is
    z-scored and added as a small fractional adjustment so it only matters
    within the same Pareto front.
    """
    ranks = non_dominated_rank(scores, directions)
    if tiebreaker is None:
        return ranks.astype(np.float64)
    tb = zscore(tiebreaker.reshape(-1, 1)).ravel()
    # Scale tiebreaker into (-0.5, 0.5) so it can't cross front boundaries.
    tb = 0.5 * np.tanh(tb / 4.0)
    return ranks.astype(np.float64) + tb
