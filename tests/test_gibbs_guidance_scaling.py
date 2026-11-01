"""Tests for ``GuidedGibbsConfig.guidance_scaling``.

The PoE adds the oracle term straight onto the MLM's log-softmax, so what decides
how much it can move the sample is the term's *spread* across the 20 candidate
residues, not its magnitude. A regression oracle in arbitrary units (the AlphaSeq
surrogate is log10 nM K_D) is therefore silently mis-weighted under ``"raw"``:
measured on real CDR positions the MLM spans ~7.7 nats against the affinity term's
~1.2, so gamma=1.0 is roughly a 6:1 loss for the oracle.

``"mlm_matched"`` rescales the oracle term to the MLM term's own std so gamma is a
relative weight, mirroring how ``moo.smooth_tchebycheff`` z-scores each objective.
"""
import numpy as np
import pytest

from mmmlm.constants import AMINO_ACIDS, NUM_AMINO_ACIDS
from mmmlm.sampling.gibbs_guided import (
    GuidedGibbsConfig,
    _match_to_mlm,
    _oracle_logits,
)


class _AffinityStub:
    """Regression scorer in log10-nM-like units; prefers residue index ``best``."""

    name = "affinity"
    direction = "minimize"

    def __init__(self, best: int = 3, spread: float = 1.2, offset: float = 3.0):
        self.best, self.spread, self.offset = best, spread, offset

    def __call__(self, sequences):
        # A small spread on a large offset -- the shape that breaks "raw".
        out = np.full(len(sequences), self.offset, dtype=np.float64)
        out[self.best] -= self.spread
        return out


def test_config_rejects_unknown_scaling():
    with pytest.raises(ValueError, match="guidance_scaling"):
        GuidedGibbsConfig(edit_budget=1, guidance_scaling="zscore")


def test_default_is_raw_so_existing_results_reproduce():
    assert GuidedGibbsConfig(edit_budget=1).guidance_scaling == "raw"


def test_matched_term_takes_the_mlm_terms_spread():
    """The rescaled term's std equals the MLM term's std -- that is what parity means."""
    rng = np.random.default_rng(0)
    log_p_mlm = np.log(rng.dirichlet(np.ones(NUM_AMINO_ACIDS)))
    oracle = _oracle_logits([""] * NUM_AMINO_ACIDS, _AffinityStub(), True)

    matched = _match_to_mlm(oracle, log_p_mlm, 1e-6)
    assert matched.std() == pytest.approx(log_p_mlm.std(), rel=1e-9)
    # Rescaling must not reorder preferences, only requantify them.
    assert np.argmax(matched) == np.argmax(oracle)
    assert np.argsort(matched).tolist() == np.argsort(oracle).tolist()


def test_raw_leaves_a_large_offset_in_place_and_matched_removes_it():
    """A constant offset cannot change a softmax, but it does hide the real spread."""
    oracle = _oracle_logits([""] * NUM_AMINO_ACIDS, _AffinityStub(offset=3.0), True)
    # direction="minimize" negates, so the offset is the big shared component
    assert abs(oracle.mean()) > 2.0            # raw term dominated by the offset
    matched = _match_to_mlm(oracle, np.log(np.full(NUM_AMINO_ACIDS, 1 / 20)), 1e-6)
    assert matched.mean() == pytest.approx(0.0, abs=1e-9)


def _log_softmax(x):
    x = np.asarray(x, dtype=np.float64)
    return x - (np.log(np.exp(x - x.max()).sum()) + x.max())


def test_matched_flips_the_argmax_at_parity_where_raw_needs_double():
    """The behavioural point, on a realistically spread MLM conditional (~9 nats).

    gamma=1.0 with matched scaling lets the oracle carry the position; raw needs
    gamma=2 to do the same, which is the mis-weighting the flag exists to remove.
    """
    mlm_favours, oracle_favours = 0, 5
    log_p_mlm = _log_softmax(np.linspace(0.0, -9.0, NUM_AMINO_ACIDS))
    assert np.argmax(log_p_mlm) == mlm_favours
    oracle = _oracle_logits([""] * NUM_AMINO_ACIDS,
                            _AffinityStub(best=oracle_favours, spread=1.2), True)
    matched = _match_to_mlm(oracle, log_p_mlm, 1e-6)

    assert np.argmax(log_p_mlm + 1.0 * matched) == oracle_favours
    assert np.argmax(log_p_mlm + 1.0 * oracle) == mlm_favours
    assert np.argmax(log_p_mlm + 2.0 * oracle) == oracle_favours


def test_matched_stays_conservative_against_an_overwhelmingly_peaked_mlm():
    """Parity is in std, not in worst-case range, so a spike is not overridden at 1.0.

    This is deliberate: when the MLM puts almost all mass on one residue, a weak
    oracle preference should not flip it at equal weight. gamma remains the knob.
    """
    mlm_favours, oracle_favours = 0, 5
    log_p_mlm = np.full(NUM_AMINO_ACIDS, -8.0)
    log_p_mlm[mlm_favours] = -0.1
    oracle = _oracle_logits([""] * NUM_AMINO_ACIDS,
                            _AffinityStub(best=oracle_favours, spread=1.2), True)
    matched = _match_to_mlm(oracle, log_p_mlm, 1e-6)

    assert np.argmax(log_p_mlm + 1.0 * matched) == mlm_favours
    assert np.argmax(log_p_mlm + 1.5 * matched) == oracle_favours
    # raw cannot get there at any strength this side of absurd
    assert np.argmax(log_p_mlm + 3.0 * oracle) == mlm_favours


def test_degenerate_oracle_contributes_nothing():
    """All-equal scores express no preference; don't divide by ~0 and invent one."""
    log_p_mlm = np.log(np.full(NUM_AMINO_ACIDS, 1 / 20))
    flat = np.full(NUM_AMINO_ACIDS, 2.5)
    assert np.allclose(_match_to_mlm(flat, log_p_mlm, 1e-6), 0.0)


def test_gamma_zero_is_plain_gibbs_under_both_modes():
    """gamma=0 must reduce to the MLM conditional, so the ablation pair stays clean."""
    log_p_mlm = np.log(np.random.default_rng(1).dirichlet(np.ones(NUM_AMINO_ACIDS)))
    oracle = _oracle_logits([""] * NUM_AMINO_ACIDS, _AffinityStub(), True)
    for term in (oracle, _match_to_mlm(oracle, log_p_mlm, 1e-6)):
        assert np.allclose(log_p_mlm + 0.0 * term, log_p_mlm)


def test_probability_oracle_becomes_log_p_under_raw():
    """Why raw stays the default: a p-in-(0,1) oracle is already on a log-prob scale.

    Trastuzumab's CNN is maximize-direction with p in (0,1), so ``_oracle_logits``
    takes the log branch and the PoE really is p_MLM * p_oracle**gamma. Rescaling
    that would change published results for no benefit.
    """
    probs = np.linspace(0.05, 0.95, NUM_AMINO_ACIDS)

    class _ProbStub:
        name, direction = "binder", "maximize"

        def __call__(self, sequences):
            return probs

    term = _oracle_logits([""] * NUM_AMINO_ACIDS, _ProbStub(), True)
    assert np.allclose(term, np.log(probs))
    # commensurate with an MLM log-softmax by construction, unlike log10-nM affinity
    assert -3.1 < term.min() < -2.9 and term.max() < 0.0


def test_amino_acid_alphabet_length_matches_candidate_construction():
    """Guards the assumption that the PoE scores exactly one candidate per AA."""
    assert len(AMINO_ACIDS) == NUM_AMINO_ACIDS == 20
