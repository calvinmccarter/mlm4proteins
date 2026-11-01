"""Denoising sampling (Hoogeboom et al. 2021; adopted for protein optimization
by Hayes et al. 2025 for ESM-3).

Algorithm, given a seed sequence, a mutable region, and an edit budget E:

  1. Choose E positions within the mutable region (uniform without replacement).
  2. Mask all E positions simultaneously.
  3. For E iterations: forward-pass the partially-masked sequence, pick the
     masked position with the lowest entropy (or highest argmax probability),
     sample (or argmax) a residue at that position from the conditional, and
     unmask. The next iteration starts with one fewer masked position.

This is mutation-centric: each residue is irreversibly committed based on a
local conditional, in contrast to beam search which scores whole sequences.
Each call to :func:`denoising_sample_one` produces ONE output sequence; the
realized edits to the seed can be ≤ E because a sampled residue may equal the
seed residue at that position.

That shortfall is intrinsic to the method as commonly defined, and is left in
place deliberately — do not "fix" it by mirroring what
:mod:`mmmlm.sampling.gibbs` does. Gibbs is free to keep taking updates until it
has accumulated E edits, because its updates are exchangeable resamples of a
single position. Denoising commits each position once, in a schedule fixed when
the E positions are chosen, so there is no analogous knob: forcing E edits would
mean either re-masking committed positions (no longer denoising) or excluding the
seed residue from the draw (no longer the model's conditional).

The magnitude scales with how much mass the model puts on the wild-type residue,
which varies ~5x across the models benchmarked here. Measured on Trastuzumab H3
at E=7: ESM2-650M (p_wt=0.13) realizes ~6.1 of 7 requested edits, AbMPNN
(p_wt=0.63) only ~3.0, so the high-edit cells stay thin for structure-conditioned
models even though the Gibbs ones now fill. This is a known drawback of denoising
for edit-budgeted design — and part of why Gibbs is the primary mutation-centric
baseline here — so it belongs in the results rather than being engineered away.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from ..constants import AMINO_ACIDS
from ..models.base import ProteinMLM


PositionSelection = Literal["lowest_entropy", "max_probability"]


@dataclass
class DenoisingConfig:
    edit_budget: int                                         # E
    temperature: float = 1.0                                 # τ for sampling
    argmax: bool = False                                     # decode the argmax instead of sampling
    position_selection: PositionSelection = "lowest_entropy"
    seed: int | None = 0


def _entropy_per_row(log_probs: np.ndarray) -> np.ndarray:
    """Shannon entropy of each row of a [L, 20] log-probability matrix."""
    # NaN rows propagate; we'll filter those out before using the result.
    probs = np.exp(log_probs)
    return -np.nansum(probs * log_probs, axis=-1)


def denoising_sample_one(
    seed_sequence: str,
    model: ProteinMLM,
    mutable_mask: np.ndarray,
    config: DenoisingConfig,
    rng: np.random.Generator | None = None,
) -> tuple[str, list[int]]:
    """Run E denoising iterations starting from ``seed_sequence``."""
    if rng is None:
        rng = np.random.default_rng(config.seed)
    if len(seed_sequence) != mutable_mask.shape[0]:
        raise ValueError("seed length does not match mask length")

    cand_positions = np.flatnonzero(mutable_mask)
    if cand_positions.size == 0:
        raise ValueError("mutable_mask has no True positions")
    if config.edit_budget > cand_positions.size:
        raise ValueError(
            f"edit_budget {config.edit_budget} exceeds mutable region size "
            f"{cand_positions.size}"
        )

    chosen = rng.choice(cand_positions, size=config.edit_budget, replace=False)
    masked_remaining = set(int(p) for p in chosen)
    chars = list(seed_sequence)

    for _ in range(config.edit_budget):
        logits = model.multi_mask_logits("".join(chars), list(masked_remaining))
        log_probs_t = torch.log_softmax(logits / config.temperature, dim=-1)
        log_probs = log_probs_t.cpu().numpy()                # [L, 20]
        entropies = _entropy_per_row(log_probs)              # [L]

        masked_list = sorted(masked_remaining)
        if config.position_selection == "lowest_entropy":
            ents = [(entropies[p], p) for p in masked_list]
            chosen_pos = min(ents, key=lambda x: x[0])[1]
        elif config.position_selection == "max_probability":
            max_log_probs = [(np.nanmax(log_probs[p]), p) for p in masked_list]
            chosen_pos = max(max_log_probs, key=lambda x: x[0])[1]
        else:
            raise ValueError(f"unknown position_selection {config.position_selection!r}")

        row_log_probs = log_probs[chosen_pos]
        if config.argmax:
            r = int(np.argmax(row_log_probs))
        else:
            shifted = row_log_probs - np.nanmax(row_log_probs)
            probs = np.exp(shifted)
            probs = probs / probs.sum()
            r = int(rng.choice(len(probs), p=probs))
        chars[chosen_pos] = AMINO_ACIDS[r]
        masked_remaining.remove(chosen_pos)

    seq = "".join(chars)
    mutated = [i for i in range(len(seq)) if seq[i] != seed_sequence[i]]
    return seq, mutated


def denoising_sample_n(
    seed_sequence: str,
    model: ProteinMLM,
    mutable_mask: np.ndarray,
    config: DenoisingConfig,
    n_sequences: int,
) -> list[tuple[str, list[int]]]:
    """Run independent denoising chains; return up to ``n_sequences`` unique outputs."""
    rng = np.random.default_rng(config.seed)
    seen: dict[str, list[int]] = {}
    for _ in range(n_sequences):
        seq, muts = denoising_sample_one(
            seed_sequence, model, mutable_mask, config, rng=rng
        )
        if seq not in seen:
            seen[seq] = muts
    return [(s, m) for s, m in seen.items()]
