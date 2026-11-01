"""Sequence-centric stochastic beam search over MLM pseudo-log-likelihoods.

Implements the central method of the paper. For a template sequence ``x`` and
its log-softmax marginal matrix ``A`` (where ``A[i, r] = log softmax_τ(Y_i)[r]``
and ``Y_i`` is the model's logits at position ``i`` when that position is
masked), the wild-type marginal PLL of any single-substitution neighbor ``x'``
that differs at position ``k`` is

    PLL_wt(x') ≈ Σ_i A[i, x'_i]
              = (template's self-PLL) − A[k, x_k] + A[k, x'_k]

so all neighbors of a given template can be scored in O(L · 20) once ``A`` has
been computed. The beam expands the current top-B templates each step and keeps
the top-B scored neighbors as the next beam, for E steps. Stochastic ranking
adds Gumbel(0, 1) noise to balance likelihood and diversity (Kool et al. 2019).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from ..constants import AA_TO_IDX, AMINO_ACIDS, NUM_AMINO_ACIDS
from ..models.base import ProteinMLM
from ..moo import nds_score, smooth_tchebycheff
from ..scoring.base import ScoringFunction


@dataclass
class BeamCandidate:
    sequence: str
    parent_pll: float          # template self-PLL at the moment of expansion
    pll: float                 # wild-type-marginal PLL of this child
    edits: int                 # realized Hamming distance to the seed
    depth: int = 0             # number of beam search steps taken to reach this child
    extra_scores: dict[str, float] = field(default_factory=dict)
    sort_key: float = 0.0      # value used for ranking at the most recent step


def _log_softmax_at_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Return log_softmax(logits / τ) along the last axis."""
    return torch.log_softmax(logits / temperature, dim=-1)


def _self_pll_from_log_marginals(
    seq: str,
    log_marginals: torch.Tensor,
    mutable_mask: np.ndarray,
) -> float:
    """Sum log A[i, x_i] over rows where the marginals are defined.

    Rows outside the mutable mask hold NaN by the ProteinMLM contract; we sum
    only the in-mask rows. The result is comparable across templates as long as
    the mutable mask is held fixed throughout a search.
    """
    idx = np.array([AA_TO_IDX[c] for c in seq], dtype=np.int64)
    rows = np.flatnonzero(mutable_mask)
    return float(log_marginals[rows, idx[rows]].sum().item())


def _enumerate_children(
    template: BeamCandidate,
    seed_sequence: str,
    log_marginals: torch.Tensor,
    mutable_mask: np.ndarray,
) -> list[BeamCandidate]:
    """All 1-substitution children of ``template`` within the mutable region.

    Each child's PLL is computed via the closed-form delta from the template.
    The child's ``edits`` is its Hamming distance to ``seed_sequence`` (so a
    substitution that reverts a previously-changed position back to the seed
    residue *decreases* the realized edit count). Search depth — how many beam
    steps the child has been through — is the loop counter, tracked separately
    via the outer search loop and ``depth``.
    """
    seq = template.sequence
    L = len(seq)
    seq_idx = np.array([AA_TO_IDX[c] for c in seq], dtype=np.int64)
    parent_pll = template.pll  # the freshly-computed self-PLL of this template
    log_m = log_marginals.cpu().numpy()
    children: list[BeamCandidate] = []
    for k in np.flatnonzero(mutable_mask):
        a_k_xk = log_m[k, seq_idx[k]]
        seed_at_k = seed_sequence[k]
        template_at_k = seq[k]
        for r in range(NUM_AMINO_ACIDS):
            if r == seq_idx[k]:
                continue  # skip the identity substitution
            new_aa = AMINO_ACIDS[r]
            new_seq = seq[:k] + new_aa + seq[k + 1:]
            child_pll = parent_pll - a_k_xk + log_m[k, r]
            # Hamming-distance update: only position k changed between
            # template and child, so we adjust template.edits by the delta at k.
            if template_at_k == seed_at_k and new_aa != seed_at_k:
                edits_delta = +1
            elif template_at_k != seed_at_k and new_aa == seed_at_k:
                edits_delta = -1
            else:
                edits_delta = 0
            children.append(
                BeamCandidate(
                    sequence=new_seq,
                    parent_pll=parent_pll,
                    pll=float(child_pll),
                    edits=template.edits + edits_delta,
                    depth=template.depth + 1,
                )
            )
    return children


GuidanceMode = str  # "none" | "sts" | "nds"


@dataclass
class BeamSearchConfig:
    edit_budget: int               # E (search depth = number of beam steps)
    beam_keep: int                 # candidates kept per step for final pool
    beam_forward: int              # candidates forwarded to the next step
    n_final: int                   # candidates returned **per realized-edit bucket**
    temperature: float = 1.5       # τ for log_softmax(logits / τ)
    gumbel: bool = True            # add Gumbel(0, 1) noise to PLL before ranking
    guidance: GuidanceMode = "none"
    sts_weights: dict[str, float] | None = None  # name -> c_i (positive)
    seed: int | None = 0


def beam_search(
    seed_sequence: str,
    model: ProteinMLM,
    mutable_mask: np.ndarray,
    config: BeamSearchConfig,
    scoring_functions: list[ScoringFunction] | None = None,
) -> list[BeamCandidate]:
    """Run sequence-centric stochastic beam search.

    Args:
        seed_sequence: Starting sequence. Search proceeds outward from here.
        model: A :class:`ProteinMLM`. We call ``template_logits`` once per
            template per beam step.
        mutable_mask: ``bool[L]`` where True positions are eligible for
            substitution.
        config: hyperparameters.
        scoring_functions: optional list of per-sequence scorers used for
            multi-objective guidance. The MLM pseudo-perplexity is always
            included alongside these as an additional objective.

    Returns:
        Up to ``config.n_final`` :class:`BeamCandidate`s, ranked by the most
        recent step's ``sort_key``.
    """
    if len(seed_sequence) != mutable_mask.shape[0]:
        raise ValueError(
            f"seed length {len(seed_sequence)} != mask length {mutable_mask.shape[0]}"
        )
    if config.guidance != "none" and not scoring_functions:
        raise ValueError(f"guidance={config.guidance!r} requires scoring_functions")

    rng = np.random.default_rng(config.seed)

    # Step 0: evaluate seed, seed the beam.
    seed_logits = model.template_logits(seed_sequence, positions=mutable_mask)
    seed_log_marg = _log_softmax_at_temperature(seed_logits, config.temperature)
    seed_pll = _self_pll_from_log_marginals(seed_sequence, seed_log_marg, mutable_mask)
    seed_cand = BeamCandidate(
        sequence=seed_sequence,
        parent_pll=seed_pll,
        pll=seed_pll,
        edits=0,
        depth=0,
    )

    forward_pool: list[BeamCandidate] = [seed_cand]
    keep_pool: list[BeamCandidate] = []  # accumulate across steps; do not include seed

    for step in range(1, config.edit_budget + 1):
        # Expand each forwarded template to its single-substitution neighbors.
        all_children: list[BeamCandidate] = []
        for template in forward_pool:
            t_logits = model.template_logits(template.sequence, positions=mutable_mask)
            t_log_marg = _log_softmax_at_temperature(t_logits, config.temperature)
            # Recompute the template's self-PLL with these freshly-masked marginals.
            template = BeamCandidate(
                sequence=template.sequence,
                parent_pll=template.parent_pll,
                pll=_self_pll_from_log_marginals(template.sequence, t_log_marg, mutable_mask),
                edits=template.edits,
                depth=template.depth,
                extra_scores=dict(template.extra_scores),
            )
            all_children.extend(
                _enumerate_children(template, seed_sequence, t_log_marg, mutable_mask)
            )
        if not all_children:
            break

        # Deduplicate sequences reached via multiple parents at this step.
        # Keep the highest-PLL copy (it'll dominate any ranking anyway).
        seen: dict[str, BeamCandidate] = {}
        for c in all_children:
            if c.sequence not in seen or c.pll > seen[c.sequence].pll:
                seen[c.sequence] = c
        all_children = list(seen.values())

        # Score with auxiliary functions if guidance is enabled.
        if scoring_functions:
            seqs = [c.sequence for c in all_children]
            for fn in scoring_functions:
                vals = np.asarray(fn(seqs), dtype=np.float64)
                for c, v in zip(all_children, vals):
                    c.extra_scores[fn.name] = float(v)

        # Apply Gumbel perturbation to the PLL before any ranking.
        plls = np.array([c.pll for c in all_children])
        if config.gumbel:
            u = rng.uniform(size=plls.shape).clip(1e-12, 1 - 1e-12)
            gumbel = -np.log(-np.log(u))
            perturbed_pll = plls + gumbel
        else:
            perturbed_pll = plls

        # Build the sort key (lower is better in MOO conventions; for unguided
        # we negate PLL so "lower = worse PLL" matches "lower = worse").
        if config.guidance == "none":
            # sort by perturbed PLL DESCENDING (highest PLL = best); we use
            # -perturbed_pll so ascending sort gives the best first.
            sort_key = -perturbed_pll
        else:
            # Pseudo-perplexity: lower is better. Use -perturbed_pll directly
            # as the "perplexity-like" objective (it's already minimization).
            obj_names = ["pseudo_perplexity"] + [fn.name for fn in scoring_functions]
            obj_dirs = ["minimize"] + [fn.direction for fn in scoring_functions]
            obj_matrix = np.column_stack(
                [-perturbed_pll]
                + [np.array([c.extra_scores[fn.name] for c in all_children]) for fn in scoring_functions]
            )
            if config.guidance == "sts":
                weights = np.ones(obj_matrix.shape[1])
                if config.sts_weights:
                    for j, name in enumerate(obj_names):
                        if name in config.sts_weights:
                            weights[j] = float(config.sts_weights[name])
                sort_key = smooth_tchebycheff(obj_matrix, obj_dirs, weights=weights)
            elif config.guidance == "nds":
                sort_key = nds_score(obj_matrix, obj_dirs, tiebreaker=-perturbed_pll)
            else:
                raise ValueError(f"unknown guidance mode {config.guidance!r}")

        for c, k in zip(all_children, sort_key):
            c.sort_key = float(k)

        # Sort ascending: smaller sort_key is better.
        all_children.sort(key=lambda c: c.sort_key)
        # keep_pool admits everything (refines and single reverts included)
        # so the final output retains some local diversity around each
        # frontier point.
        keep_pool.extend(all_children[: config.beam_keep])
        # Frontier maintains the realized-edits == depth invariant: the
        # forward pool only accepts children that strictly increase the
        # realized edit count vs their parent (which by invariant has
        # edits == step - 1 at this point in the loop).
        forward_eligible = [c for c in all_children if c.edits == step]
        forward_pool = forward_eligible[: config.beam_forward]
        # Ensure the frontier itself is represented in the output even when
        # guidance ranks reverts/refines higher globally. Without this,
        # under heavy guidance pull toward the seed, +1-progress children
        # never enter keep_pool because the top-beam_keep slice fills up
        # with low-edit refines/reverts before reaching them.
        keep_pool.extend(forward_pool)
        if not forward_pool:
            break

    # Final ranking: dedupe across steps, then return ``n_final`` candidates
    # per realized-edit bucket so output coverage spans every edit level the
    # frontier reached (and every -1 / 0 cluster around it). Sequences with
    # ``edits == 0`` (full revert back to the seed) are excluded.
    final_seen: dict[str, BeamCandidate] = {}
    for c in keep_pool:
        if c.sequence not in final_seen or c.sort_key < final_seen[c.sequence].sort_key:
            final_seen[c.sequence] = c
    by_edits: dict[int, list[BeamCandidate]] = {}
    for c in final_seen.values():
        if c.edits == 0:
            continue
        by_edits.setdefault(c.edits, []).append(c)
    final: list[BeamCandidate] = []
    for e in sorted(by_edits):
        bucket = sorted(by_edits[e], key=lambda c: c.sort_key)
        final.extend(bucket[: config.n_final])
    return final
