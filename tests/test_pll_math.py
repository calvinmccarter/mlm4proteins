"""Verify the wild-type marginal PLL math: closed-form delta == direct Σ_i A[i, x'_i]."""
from __future__ import annotations

import numpy as np
import torch

from mmmlm.constants import AA_TO_IDX, AMINO_ACIDS, NUM_AMINO_ACIDS
from mmmlm.models.base import ProteinMLM
from mmmlm.sampling.beam import (
    BeamSearchConfig,
    _enumerate_children,
    _log_softmax_at_temperature,
    _self_pll_from_log_marginals,
    beam_search,
)


class FixedLogitsMLM:
    """A mock MLM whose template_logits returns a deterministic Tensor[L, 20].

    The matrix depends on the input sequence so different templates yield
    different marginals (necessary for a meaningful test).
    """

    name = "fixed-logits"

    def __init__(self, length: int, seed: int = 0):
        self.length = length
        self._rng = np.random.default_rng(seed)
        # Per-template logits = base + perturbation depending on the input
        self._base = torch.tensor(self._rng.normal(size=(length, 20)), dtype=torch.float32)

    def _seq_perturbation(self, sequence: str) -> torch.Tensor:
        # Make the perturbation depend on each residue in a reproducible way.
        idx = torch.tensor([AA_TO_IDX[c] for c in sequence], dtype=torch.long)
        # outer product: per-position bias from each other position's residue
        bias = torch.zeros(self.length, 20)
        for i in range(self.length):
            for j in range(self.length):
                if i != j:
                    bias[i, idx[j].item()] += 0.05 * (i - j) / self.length
        return bias

    def template_logits(self, sequence, positions=None):
        assert len(sequence) == self.length
        out = self._base + self._seq_perturbation(sequence)
        if positions is not None:
            mask = torch.from_numpy(positions)
            out = out.clone()
            out[~mask] = float("nan")
        return out

    def multi_mask_logits(self, sequence, mask_positions):
        out = torch.full((self.length, 20), float("nan"))
        if not mask_positions:
            return out
        # Independent of the masking pattern for this mock; what matters is
        # the protocol shape, not realism.
        full = self._base + self._seq_perturbation(sequence)
        for p in mask_positions:
            out[p] = full[p]
        return out


def _direct_wt_pll(seq: str, log_marg: torch.Tensor, mutable_mask: np.ndarray) -> float:
    """Reference: directly evaluate Σ_i A[i, x_i] over masked rows."""
    s = 0.0
    for i, c in enumerate(seq):
        if mutable_mask[i]:
            s += float(log_marg[i, AA_TO_IDX[c]].item())
    return s


def test_protocol_compliance():
    mlm = FixedLogitsMLM(length=10)
    assert isinstance(mlm, ProteinMLM)


def test_closed_form_delta_matches_direct_pll():
    """For each child of a template, the closed-form delta computation in
    _enumerate_children should equal the direct Σ_i A[i, x'_i]."""
    L = 12
    mlm = FixedLogitsMLM(length=L, seed=42)
    rng = np.random.default_rng(0)
    seq = "".join(rng.choice(list(AMINO_ACIDS), size=L))
    mask = np.zeros(L, dtype=bool)
    mask[3:9] = True  # arbitrary mutable region

    logits = mlm.template_logits(seq, positions=mask)
    log_marg = _log_softmax_at_temperature(logits, temperature=1.5)
    self_pll = _self_pll_from_log_marginals(seq, log_marg, mask)

    from mmmlm.sampling.beam import BeamCandidate
    template = BeamCandidate(sequence=seq, parent_pll=self_pll, pll=self_pll, edits=0)
    children = _enumerate_children(template, seq, log_marg, mask)

    assert len(children) == mask.sum() * (NUM_AMINO_ACIDS - 1)
    for child in children:
        direct = _direct_wt_pll(child.sequence, log_marg, mask)
        assert abs(child.pll - direct) < 1e-5, (
            f"mismatch for {child.sequence}: closed-form={child.pll}, direct={direct}"
        )


def test_beam_search_runs_unguided():
    L = 15
    mlm = FixedLogitsMLM(length=L, seed=7)
    rng = np.random.default_rng(0)
    seq = "".join(rng.choice(list(AMINO_ACIDS), size=L))
    mask = np.zeros(L, dtype=bool)
    mask[5:10] = True

    cfg = BeamSearchConfig(
        edit_budget=3,
        beam_keep=20,
        beam_forward=5,
        n_final=10,
        temperature=1.5,
        gumbel=True,
        seed=0,
    )
    out = beam_search(seq, mlm, mask, cfg)
    # n_final is now per realized-edit bucket; with budget=3 we expect
    # buckets at edits in {1, 2, 3} (edits=0 is filtered out).
    assert 1 <= len(out) <= cfg.edit_budget * cfg.n_final
    # all returned sequences should respect the mutable region
    for c in out:
        for i, (a, b) in enumerate(zip(seq, c.sequence)):
            if not mask[i]:
                assert a == b, f"non-mutable position {i} changed: {a} -> {b}"
        assert 1 <= c.edits <= cfg.edit_budget
    # The strict +1 frontier guarantees at least one candidate at each
    # realized-edit level from 1 up to edit_budget under a mutable region
    # large enough to support that depth.
    realized = {c.edits for c in out}
    assert cfg.edit_budget in realized, (
        f"frontier failed to reach realized edits = {cfg.edit_budget}; "
        f"observed: {sorted(realized)}"
    )
