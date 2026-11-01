"""Protocol for protein masked language models used by samplers in this package."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class ProteinMLM(Protocol):
    """A masked language model that scores protein sequences position-wise.

    The single load-bearing method is :meth:`template_logits`, which returns the
    matrix ``Y`` whose row ``i`` is the model's logits at position ``i`` when
    that position is replaced with the mask token. Logits are restricted and
    re-ordered to the 20-letter alphabet ``constants.AMINO_ACIDS``.

    Sampling code in :mod:`mmmlm.sampling` derives the wild-type marginal log
    probability matrix ``A`` from these logits at a chosen temperature, then
    uses ``A`` to score all single-substitution neighbors of the template.
    """

    name: str

    def template_logits(
        self,
        sequence: str,
        positions: np.ndarray | None = None,
    ) -> torch.Tensor:
        """Compute per-position masked logits for ``sequence``.

        Args:
            sequence: Length-L protein sequence over the 20-letter alphabet.
            positions: Optional bool[L] array. If provided, only positions where
                the mask is True are evaluated; the returned tensor's other
                rows are filled with NaN. This lets callers skip forward passes
                for positions that are not eligible for mutation.

        Returns:
            ``Tensor[L, 20]`` of logits in the canonical amino-acid order.
            Rows for positions excluded by ``positions`` contain NaN.
        """
        ...

    def multi_mask_logits(
        self,
        sequence: str,
        mask_positions: list[int],
    ) -> torch.Tensor:
        """Forward-pass with all ``mask_positions`` masked simultaneously.

        Used by mutation-centric samplers such as denoising sampling
        (Hoogeboom et al. 2021 / Hayes et al. 2025), where the model conditions
        on the unmasked context and predicts the masked residues jointly. A
        single forward pass returns logits at each of the masked positions.

        Returns:
            ``Tensor[L, 20]`` of logits in the canonical amino-acid order.
            Rows for positions outside ``mask_positions`` contain NaN.
        """
        ...
