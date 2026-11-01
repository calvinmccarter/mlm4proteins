"""Gibbs-style mutation-centric sampling from an MLM.

Each iteration: pick one mutable position uniformly at random, mask it, sample a
new residue from the MLM's conditional (random-scan Gibbs). The chain runs until
the sequence differs from the seed at exactly ``edit_budget`` positions, so E
counts *realized edits* — the axis every plot is grouped by — rather than a
number of updates.

Positions are drawn **with replacement**, which is what makes that stopping rule
reachable: a position whose resample came back wild-type has to get another
chance. Every transition is an unmodified Gibbs update, so the chain is a genuine
Gibbs chain for the model's distribution; what the stopping rule adds is
conditioning on the first hitting time of ``{Hamming == E}``, so outputs follow
that hitting-time distribution rather than the model's own.

Two alternatives were measured and rejected:

  * **E updates, positions without replacement.** Cheap, but E draws from ``m``
    positions realize only ``E · (1 − p_wt)`` edits on average, and the shortfall
    is severe exactly for models confident in the wild type: on Trastuzumab H3 at
    E=7, ESM2-650M (p_wt=0.13) reached 6.1 edits while AbMPNN (p_wt=0.63) reached
    2.4, leaving the high-edit cells empty for the structure-conditioned models.
    Conditioning on realized edits, it is statistically indistinguishable from
    the current rule (every 95% bootstrap CI on the per-cell difference in oracle
    p(binder) contains zero, for both models), so nothing is lost by dropping it
    — while coverage improves from 176/132/54/9/1/0 designs at 2..7 edits to
    113/135/154/151/152/135.
  * **Restricting the draw to ``x_k ≠ current``.** Guarantees E edits in E steps,
    but it is a Metropolized-Gibbs proposal without its acceptance correction, so
    the chain no longer leaves the model's distribution invariant.

The cost is real: reaching E edits takes more updates than E, by a factor that
grows with the model's wild-type mass (ESM2 needs ~13 updates for 7 edits,
AbMPNN ~207). ``max_steps`` caps it, and chains that fall short are reported.

The variant ``argmax=True`` deterministically picks the highest-probability
residue, which tends to produce fewer unique outputs but follows the MLM's
preferences more strictly (used for antibody humanization in Gordon et al. 2024).

Each call to :func:`gibbs_sample_one` produces ONE output sequence. To obtain
``N`` sequences, run it ``N`` times — independent runs are not guaranteed to
produce unique sequences, so the caller deduplicates.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..constants import AA_TO_IDX, AMINO_ACIDS
from ..models.base import ProteinMLM


@dataclass
class GibbsConfig:
    edit_budget: int          # E
    temperature: float = 1.0  # τ for sampling from the conditional
    argmax: bool = False      # if True, pick argmax instead of sampling
    seed: int | None = 0
    # Update cap; None means 60 · E. A model that strongly prefers the wild type
    # may need many updates to accumulate E edits, and occasionally will not get
    # there at all.
    max_steps: int | None = None
    # Emit the chain's state at the first time it reaches *each* edit count
    # 1..E, not just E. Because the stopping rule is a first hitting time and one
    # update moves the Hamming distance by at most one, the chain must pass
    # through every intermediate count, and its state there is by construction a
    # sample from this same rule with ``edit_budget`` set to that count. So one
    # E-edit chain yields a valid design for every budget below it, the way a
    # single beam search already returns every realized-edit bucket up to E.
    # Off by default: it changes how many designs a chain produces.
    capture_intermediates: bool = False


def gibbs_sample_one(
    seed_sequence: str,
    model: ProteinMLM,
    mutable_mask: np.ndarray,
    config: GibbsConfig,
    rng: np.random.Generator | None = None,
    snapshots: dict[int, str] | None = None,
) -> tuple[str, list[int]]:
    """Run one Gibbs chain from ``seed_sequence`` until it has E realized edits.

    Returns the final sequence and the sorted indices that ended up different
    from the seed. That list has exactly ``edit_budget`` entries unless the
    update cap was hit first.

    If ``snapshots`` is given it is filled with ``{n_edits: sequence}`` recording
    the state the *first* time the chain reaches each edit count in 1..E -- see
    ``GibbsConfig.capture_intermediates``. Keyed on the measured Hamming distance
    rather than on the step index, because the distance is not monotone:
    resampling an already-mutated position back to wild type moves it *down*, so
    "the state after the first accepted mutation" and "the first state at Hamming
    1" are not always the same sequence.
    """
    if rng is None:
        rng = np.random.default_rng(config.seed)
    if len(seed_sequence) != mutable_mask.shape[0]:
        raise ValueError("seed length does not match mask length")

    seq = seed_sequence
    candidate_positions = np.flatnonzero(mutable_mask)
    if candidate_positions.size == 0:
        raise ValueError("mutable_mask has no True positions")
    if config.edit_budget > candidate_positions.size:
        raise ValueError(
            f"edit_budget {config.edit_budget} exceeds the mutable region size "
            f"{candidate_positions.size}"
        )

    def _update(sequence: str, k: int) -> str:
        """One Gibbs update: resample position ``k`` from its conditional."""
        single_pos = np.zeros_like(mutable_mask)
        single_pos[k] = True
        logits = model.template_logits(sequence, positions=single_pos)
        log_probs = torch.log_softmax(logits[k] / config.temperature, dim=-1).cpu().numpy()
        if config.argmax:
            r = int(np.argmax(log_probs))
        else:
            probs = np.exp(log_probs - log_probs.max())
            probs = probs / probs.sum()
            r = int(rng.choice(len(probs), p=probs))
        return sequence[:k] + AMINO_ACIDS[r] + sequence[k + 1:]

    def _n_edits(sequence: str) -> int:
        return sum(1 for a, b in zip(sequence, seed_sequence) if a != b)

    # Positions are drawn with replacement: a position whose resample came back
    # wild-type has to get another chance, which is what makes the edit-count
    # stopping rule reachable at all.
    max_steps = config.max_steps if config.max_steps is not None else 60 * config.edit_budget
    for _ in range(max_steps):
        n_now = _n_edits(seq)
        if snapshots is not None and 0 < n_now <= config.edit_budget:
            snapshots.setdefault(n_now, seq)
        if n_now == config.edit_budget:
            break
        seq = _update(seq, int(rng.choice(candidate_positions)))

    mutated = [
        i for i in range(len(seq))
        if seq[i] != seed_sequence[i]
    ]
    return seq, mutated


def gibbs_sample_n(
    seed_sequence: str,
    model: ProteinMLM,
    mutable_mask: np.ndarray,
    config: GibbsConfig,
    n_sequences: int,
) -> list[tuple[str, list[int]]]:
    """Run independent Gibbs chains and return ``n_sequences`` (deduplicated) outputs.

    With ``argmax=True``, this often returns fewer than ``n_sequences`` unique
    outputs; the returned list is whatever was produced.

    The fraction of chains that hit the update cap without reaching E edits is
    printed rather than swallowed: it measures how hard the model resists moving
    that far from the seed, which is a result about the model and not a bug to
    be hidden.

    With ``config.capture_intermediates`` the return can *exceed* ``n_sequences``:
    each chain contributes a design at every edit count it passed through, so
    ``n_sequences`` becomes a chain budget rather than an output count.
    """
    rng = np.random.default_rng(config.seed)
    seen: dict[str, list[int]] = {}
    short = 0
    for _ in range(n_sequences):
        snaps: dict[int, str] | None = {} if config.capture_intermediates else None
        seq, muts = gibbs_sample_one(seed_sequence, model, mutable_mask, config,
                                     rng=rng, snapshots=snaps)
        if len(muts) != config.edit_budget:
            short += 1
        if seq not in seen:
            seen[seq] = muts
        if snaps:
            # Every lower-count snapshot as well. The endpoint is already snaps[E]
            # whenever E was reached; adding it above first is what keeps a
            # cap-truncated chain's final state from being dropped.
            for snap in snaps.values():
                if snap not in seen:
                    seen[snap] = [i for i in range(len(snap))
                                  if snap[i] != seed_sequence[i]]
    if short:
        print(f"    [gibbs E={config.edit_budget}] {short}/{n_sequences} chains hit "
              f"the update cap without reaching {config.edit_budget} edits", flush=True)
    return [(s, m) for s, m in seen.items()]
