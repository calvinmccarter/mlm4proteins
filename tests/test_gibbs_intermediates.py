"""Tests for ``capture_intermediates`` on the Gibbs samplers.

One E-edit chain passes through every lower edit count, so its state at the first
hit of each count is a valid sample under the same stopping rule with that budget.
Capturing them turns one chain into a design for every bucket up to E -- the prefix
property beam search already has.

The distributional claim rests on the stopping rule being a *first hitting time*
and on one update moving the Hamming distance by at most one; both are asserted
here, since either breaking would silently invalidate the intermediates.
"""
import numpy as np
import pytest
import torch

from mmmlm.constants import AMINO_ACIDS, NUM_AMINO_ACIDS
from mmmlm.sampling.gibbs import GibbsConfig, gibbs_sample_n, gibbs_sample_one

SEED_SEQ = "AAAAAAAAAAAA"


class _CyclingMLM:
    """Deterministic stand-in: every position prefers a fixed non-wild-type residue.

    Returns logits over the 20 canonical acids for the requested positions, peaked
    on a residue chosen by position index so chains actually accumulate edits.
    """

    name = "cycling"

    def template_logits(self, sequence, positions=None):
        L = len(sequence)
        out = torch.full((L, NUM_AMINO_ACIDS), -20.0)
        rows = np.flatnonzero(positions) if positions is not None else range(L)
        for k in rows:
            favoured = AMINO_ACIDS[(int(k) + 3) % NUM_AMINO_ACIDS]
            out[k, AMINO_ACIDS.index(favoured)] = 10.0
            # leave some mass on wild type so the chain sometimes stalls, which is
            # what makes the stopping rule non-trivial
            out[k, AMINO_ACIDS.index(sequence[k])] = 8.0
        return out


def _mask(n, upto):
    m = np.zeros(n, dtype=bool)
    m[:upto] = True
    return m


def test_default_is_off_so_existing_results_reproduce():
    assert GibbsConfig(edit_budget=3).capture_intermediates is False


def test_snapshots_cover_every_edit_count_up_to_e():
    cfg = GibbsConfig(edit_budget=4, temperature=1.0, argmax=False, seed=0)
    snaps: dict[int, str] = {}
    seq, muts = gibbs_sample_one(SEED_SEQ, _CyclingMLM(), _mask(12, 8), cfg,
                                 rng=np.random.default_rng(0), snapshots=snaps)
    assert sorted(snaps) == [1, 2, 3, 4]
    # each snapshot really is at its own Hamming distance
    for j, s in snaps.items():
        assert sum(a != b for a, b in zip(s, SEED_SEQ)) == j
    # and the endpoint is the top snapshot
    assert snaps[4] == seq and len(muts) == 4


def test_hamming_moves_by_at_most_one_per_update():
    """The claim that a chain must pass through every count depends on this."""
    cfg = GibbsConfig(edit_budget=6, temperature=1.0, argmax=False, seed=1)
    mask, mlm = _mask(12, 10), _CyclingMLM()
    rng = np.random.default_rng(1)
    snaps: dict[int, str] = {}
    gibbs_sample_one(SEED_SEQ, mlm, mask, cfg, rng=rng, snapshots=snaps)
    counts = sorted(snaps)
    assert counts == list(range(1, 7)), "a count was skipped -- prefix claim broken"


def test_snapshot_is_first_hit_not_last():
    """setdefault, not assignment: a later return to the same count must not overwrite."""
    snaps = {2: "first"}
    cfg = GibbsConfig(edit_budget=3, seed=0)
    # calling again must leave the recorded state alone for counts already seen
    gibbs_sample_one(SEED_SEQ, _CyclingMLM(), _mask(12, 8), cfg,
                     rng=np.random.default_rng(0), snapshots=snaps)
    assert snaps[2] == "first"


def test_sample_n_yields_more_designs_than_chains():
    """n_sequences becomes a chain budget, so the output can exceed it."""
    mask, mlm = _mask(12, 10), _CyclingMLM()
    n = 6
    plain = gibbs_sample_n(SEED_SEQ, mlm, mask,
                           GibbsConfig(edit_budget=3, seed=0), n_sequences=n)
    with_inter = gibbs_sample_n(SEED_SEQ, mlm, mask,
                                GibbsConfig(edit_budget=3, seed=0,
                                            capture_intermediates=True),
                                n_sequences=n)
    assert len(with_inter) > len(plain)
    # every returned design carries the edit list matching its own sequence
    for s, muts in with_inter:
        assert muts == [i for i in range(len(s)) if s[i] != SEED_SEQ[i]]
        assert 1 <= len(muts) <= 3


def test_intermediates_fill_the_lower_buckets():
    mask, mlm = _mask(12, 10), _CyclingMLM()
    out = gibbs_sample_n(SEED_SEQ, mlm, mask,
                         GibbsConfig(edit_budget=3, seed=0, capture_intermediates=True),
                         n_sequences=8)
    by_bucket = {}
    for s, muts in out:
        by_bucket.setdefault(len(muts), []).append(s)
    assert set(by_bucket) == {1, 2, 3}, f"buckets present: {sorted(by_bucket)}"


def test_endpoint_kept_even_when_it_is_not_a_snapshot():
    """A cap-truncated chain's final state must still be returned.

    The endpoint is added before the snapshots, so a chain that ran out of updates
    without reaching E -- whose final state need not equal any first-hit state,
    since the distance can wander back down -- is not dropped.
    """
    cfg = GibbsConfig(edit_budget=9, max_steps=2, seed=0, capture_intermediates=True)
    out = gibbs_sample_n(SEED_SEQ, _CyclingMLM(), _mask(12, 10), cfg, n_sequences=3)
    assert out, "cap-truncated chains produced nothing"
    assert all(len(m) < 9 for _, m in out)


def test_unguided_and_guided_configs_agree_on_the_flag_name():
    """Both samplers expose it identically so a runner can pass one argument."""
    from mmmlm.sampling.gibbs_guided import GuidedGibbsConfig

    assert hasattr(GibbsConfig(edit_budget=1), "capture_intermediates")
    assert hasattr(GuidedGibbsConfig(edit_budget=1), "capture_intermediates")
    assert GuidedGibbsConfig(edit_budget=1).capture_intermediates is False
