"""Unit tests for MOO scalarization."""
from __future__ import annotations

import numpy as np

from mmmlm.moo import non_dominated_rank, smooth_tchebycheff


def test_nds_two_objectives_minimize():
    """Classic 5-point Pareto example."""
    # Objectives: minimize both
    points = np.array([
        [1.0, 5.0],   # rank 0
        [2.0, 3.0],   # rank 0
        [3.0, 2.0],   # rank 0
        [4.0, 1.0],   # rank 0
        [3.0, 5.0],   # dominated by [2, 3] and [3, 2]: rank 1
    ])
    ranks = non_dominated_rank(points, ["minimize", "minimize"])
    assert ranks.tolist() == [0, 0, 0, 0, 1]


def test_nds_with_maximize_axis():
    """Flip-direction handling: maximizing the second objective."""
    points = np.array([
        [1.0, 1.0],   # we want low first, high second
        [2.0, 2.0],
        [3.0, 3.0],
    ])
    ranks = non_dominated_rank(points, ["minimize", "maximize"])
    # All three points are mutually non-dominated under the first/second tradeoff.
    assert ranks.tolist() == [0, 0, 0]


def test_sts_minimization_orders_consistently():
    """Smaller (better) inputs should yield smaller (better) STS scores."""
    points = np.array([
        [1.0, 1.0],   # best
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0],   # worst
    ])
    out = smooth_tchebycheff(points, ["minimize", "minimize"])
    # Lower is better; the all-1 row should score lowest.
    assert np.argmin(out) == 0
    assert np.argmax(out) == 3


def test_sts_weight_emphasis():
    """A larger weight on objective j makes it dominate the aggregation."""
    points = np.array([
        [1.0, 5.0],   # good on first, bad on second
        [5.0, 1.0],   # bad on first, good on second
    ])
    # With heavy weight on the FIRST objective, the first row should win.
    s_w_first = smooth_tchebycheff(points, ["minimize", "minimize"],
                                   weights=np.array([5.0, 1.0]))
    assert s_w_first[0] < s_w_first[1]
    # With heavy weight on the SECOND objective, the second row should win.
    s_w_second = smooth_tchebycheff(points, ["minimize", "minimize"],
                                    weights=np.array([1.0, 5.0]))
    assert s_w_second[1] < s_w_second[0]
