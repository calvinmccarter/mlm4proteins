"""Scoring functions for guidance during beam search."""
from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ScoringFunction(Protocol):
    """A function that scores a batch of clean (non-masked) protein sequences.

    Used for multi-objective guidance in :func:`mmmlm.sampling.beam.beam_search`.
    """

    name: str
    direction: Literal["minimize", "maximize"]

    def __call__(self, sequences: list[str]) -> np.ndarray:
        """Score sequences. Returns float[N] scoring each sequence.

        The semantics of "better" depend on ``direction``: if ``minimize``, lower
        scores are better; if ``maximize``, higher scores are better. The
        scalarization machinery in :mod:`mmmlm.moo` flips signs internally so
        downstream all objectives are minimization.
        """
        ...
