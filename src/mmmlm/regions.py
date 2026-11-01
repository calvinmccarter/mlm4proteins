"""Mutable-region masks: which positions in a seed sequence are eligible for substitution."""
from __future__ import annotations

import numpy as np


def mask_from_indices(seq_length: int, indices: list[int]) -> np.ndarray:
    """Return a bool[L] array, True at the given positions, False elsewhere."""
    mask = np.zeros(seq_length, dtype=bool)
    for i in indices:
        if not (0 <= i < seq_length):
            raise ValueError(f"index {i} out of range for length {seq_length}")
        mask[i] = True
    return mask


def mask_from_slice(seq_length: int, start: int, end: int) -> np.ndarray:
    """Return a bool[L] array, True for positions in [start, end), False elsewhere."""
    mask = np.zeros(seq_length, dtype=bool)
    mask[start:end] = True
    return mask


def all_positions_mask(seq_length: int) -> np.ndarray:
    return np.ones(seq_length, dtype=bool)
