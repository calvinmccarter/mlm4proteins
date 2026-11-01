"""Gibbs runs until E *realized* edits, not E updates.

A stub MLM with a tunable wild-type mass lets us assert the mechanism directly.
The rejected alternative — take exactly E updates — realizes only
``E · (1 − p_wt)`` edits, which is what left the high-edit cells empty for
structure-conditioned models (AbMPNN, p_wt ≈ 0.63, reached 2.4 of 7).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mmmlm.constants import AA_TO_IDX, NUM_AMINO_ACIDS
from mmmlm.sampling.gibbs import GibbsConfig, gibbs_sample_n, gibbs_sample_one
from mmmlm.sampling.gibbs_guided import GuidedGibbsConfig, gibbs_guided_sample_one

SEED_SEQ = "ACDEFGHIKL" * 3          # length 30
MUTABLE = slice(10, 20)              # a 10-residue "CDR"


class StubMLM:
    """Puts ``p_wt`` on the residue already present, spreading the rest evenly."""

    name = "stub"

    def __init__(self, p_wt: float):
        self.p_wt = p_wt
        self.calls = 0

    def template_logits(self, sequence: str, positions=None) -> torch.Tensor:
        L = len(sequence)
        out = torch.full((L, NUM_AMINO_ACIDS), float("nan"))
        rows = np.flatnonzero(np.ones(L, bool) if positions is None else positions)
        rest = (1.0 - self.p_wt) / (NUM_AMINO_ACIDS - 1)
        for i in rows:
            self.calls += 1
            probs = torch.full((NUM_AMINO_ACIDS,), rest)
            probs[AA_TO_IDX[sequence[i]]] = self.p_wt
            out[i] = probs.log()
        return out

    def multi_mask_logits(self, sequence: str, mask_positions):  # pragma: no cover
        raise NotImplementedError


class StubScorer:
    """Flat scorer: leaves the MLM term untouched so guidance is a no-op."""

    name = "flat"
    direction = "maximize"

    def __call__(self, sequences):
        return np.zeros(len(sequences))


def _mask() -> np.ndarray:
    m = np.zeros(len(SEED_SEQ), dtype=bool)
    m[MUTABLE] = True
    return m


def _n_edits(seq: str) -> int:
    return sum(1 for a, b in zip(seq, SEED_SEQ) if a != b)


@pytest.mark.parametrize("p_wt", [0.0, 0.5, 0.9])
@pytest.mark.parametrize("E", [1, 4, 7])
def test_reaches_the_edit_target_exactly(p_wt, E):
    """Regardless of how much mass sits on the wild type."""
    for seed in range(3):
        cfg = GibbsConfig(edit_budget=E, seed=seed)
        seq, muts = gibbs_sample_one(SEED_SEQ, StubMLM(p_wt), _mask(), cfg)
        assert len(muts) == E
        assert _n_edits(seq) == E


def test_edits_land_inside_the_mutable_region():
    cfg = GibbsConfig(edit_budget=7, seed=0)
    seq, muts = gibbs_sample_one(SEED_SEQ, StubMLM(0.7), _mask(), cfg)
    assert all(MUTABLE.start <= i < MUTABLE.stop for i in muts)
    assert seq[:MUTABLE.start] == SEED_SEQ[:MUTABLE.start]
    assert seq[MUTABLE.stop:] == SEED_SEQ[MUTABLE.stop:]


def test_update_count_grows_with_wildtype_mass():
    """The cost of the stopping rule: resistant models need more updates.

    Two separate costs stack. Even at ``p_wt = 0``, where every update changes
    its position, drawing with replacement means revisits only swap one
    non-wild-type residue for another — so reaching E edits needs E *distinct*
    positions, which costs ``m·(H_m − H_{m−E})`` ≈ 11 draws for 7 of 10
    (coupon collector). On top of that, wild-type mass wastes updates outright.
    """
    cheap, dear = StubMLM(0.0), StubMLM(0.9)
    gibbs_sample_one(SEED_SEQ, cheap, _mask(), GibbsConfig(7, seed=0))
    gibbs_sample_one(SEED_SEQ, dear, _mask(), GibbsConfig(7, seed=0))
    assert cheap.calls >= 7            # never fewer than one update per edit
    assert cheap.calls < 30            # but collisions alone stay modest
    assert dear.calls > 2 * cheap.calls  # wild-type mass dominates the cost


def test_max_steps_cap_is_respected_and_reported(capsys):
    """A cap that cannot be met returns short rather than looping forever."""
    cfg = GibbsConfig(edit_budget=7, seed=0, max_steps=3)
    mlm = StubMLM(0.95)
    seq, muts = gibbs_sample_one(SEED_SEQ, mlm, _mask(), cfg)
    assert mlm.calls <= 3
    assert len(muts) < 7
    gibbs_sample_n(SEED_SEQ, StubMLM(0.95), _mask(), cfg, n_sequences=4)
    assert "hit the update cap" in capsys.readouterr().out


def test_default_cap_scales_with_budget():
    """max_steps=None means 60*E, enough for a p_wt=0.9 model to reach E=7."""
    cfg = GibbsConfig(edit_budget=7, seed=0)
    assert cfg.max_steps is None
    _, muts = gibbs_sample_one(SEED_SEQ, StubMLM(0.9), _mask(), cfg)
    assert len(muts) == 7


def test_budget_beyond_mutable_region_is_rejected():
    with pytest.raises(ValueError, match="exceeds the mutable region size"):
        gibbs_sample_one(SEED_SEQ, StubMLM(0.5), _mask(), GibbsConfig(11, seed=0))


def test_guided_sampler_uses_the_same_stopping_rule():
    """Otherwise guided and unguided curves would sit at different edit distances."""
    for E in (3, 7):
        cfg = GuidedGibbsConfig(edit_budget=E, seed=0)
        seq, muts = gibbs_guided_sample_one(
            SEED_SEQ, StubMLM(0.9), _mask(), [StubScorer()], cfg
        )
        assert len(muts) == E
        assert _n_edits(seq) == E


def test_guided_budget_beyond_mutable_region_is_rejected():
    cfg = GuidedGibbsConfig(edit_budget=11, seed=0)
    with pytest.raises(ValueError, match="exceeds the mutable region size"):
        gibbs_guided_sample_one(SEED_SEQ, StubMLM(0.5), _mask(), [StubScorer()], cfg)
