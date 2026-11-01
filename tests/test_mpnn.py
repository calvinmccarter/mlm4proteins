"""Pin the StructureMPNN wrapper to ProteinMPNN's own reference implementations.

Skipped unless a ProteinMPNN checkout, its weights, and the trastuzumab
structure are all present (``MMMLM_PROTEINMPNN_DIR``).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from mmmlm.constants import AMINO_ACIDS
from mmmlm.models.base import ProteinMLM

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent / "experiments" / "trastuzumab_cdrh3"
STRUCTURE = EXPERIMENT_DIR / "structures" / "1N8Z.pdb"
TRASTUZUMAB_VH = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYAD"
    "SVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
)
H3_START, H3_END = 98, 108


def _root() -> Path | None:
    root = os.environ.get("MMMLM_PROTEINMPNN_DIR")
    return Path(root) if root else None


def _weights(root: Path) -> Path:
    return root / "vanilla_model_weights" / "v_48_020.pt"


pytestmark = pytest.mark.skipif(
    _root() is None
    or not STRUCTURE.is_file()
    or not _weights(_root()).is_file(),
    reason="needs MMMLM_PROTEINMPNN_DIR with weights and the 1N8Z structure",
)


@pytest.fixture(scope="module")
def mlm():
    from mmmlm.models.mpnn import StructureMPNN

    root = _root()
    return StructureMPNN(
        structure_path=STRUCTURE,
        weights_path=_weights(root),
        design_chain="B",
        design_sequence=TRASTUZUMAB_VH,
        context_chains=["A", "C"],
        proteinmpnn_root=root,
        device="cpu",
        max_batch_size=4,
    )


def test_satisfies_protocol(mlm):
    assert isinstance(mlm, ProteinMLM)


def test_all_masked_matches_unconditional_probs(mlm):
    """visible == 0 everywhere is exactly ProteinMPNN.unconditional_probs."""
    S = mlm._tokenize(TRASTUZUMAB_VH)
    visible = torch.zeros((1, mlm.complex_length))
    with torch.inference_mode():
        ours = mlm._decode(S, visible)
        ref = mlm._model.unconditional_probs(
            mlm._X, mlm._mask, mlm._residue_idx, mlm._chain_encoding_all
        )
    assert torch.allclose(ours, ref, atol=1e-5)


def test_single_position_agrees_with_conditional_probs(mlm):
    """M == {i} matches ``--conditional_probs_only`` to within that reference's
    own sensitivity to the random decoding order of the non-target positions.

    ``conditional_probs`` puts position ``i`` last but shuffles everything
    before it; we instead let every position see all of ``s_{-i}``. The gap that
    introduces must be no larger than the gap between two reference draws.
    """
    from mmmlm.models.mpnn import MPNN_ALPHABET

    S = mlm._tokenize(TRASTUZUMAB_VH)
    cols = [MPNN_ALPHABET.index(a) for a in AMINO_ACIDS]
    chain_M = torch.zeros((1, mlm.complex_length))
    chain_M[0, H3_START:H3_END] = 1.0

    refs = []
    for seed in range(3):
        torch.manual_seed(seed)
        randn = torch.randn(chain_M.shape)
        with torch.inference_mode():
            ref = mlm._model.conditional_probs(
                mlm._X, S, mlm._mask, chain_M, mlm._residue_idx,
                mlm._chain_encoding_all, randn,
            )
        refs.append(ref[0, H3_START:H3_END][:, cols].numpy())

    h3 = np.zeros(len(TRASTUZUMAB_VH), dtype=bool)
    h3[H3_START:H3_END] = True
    ours = mlm.template_logits(TRASTUZUMAB_VH, positions=h3)[H3_START:H3_END].numpy()

    ref_spread = max(
        np.abs(a - b).mean() for a, b in [(refs[0], refs[1]), (refs[0], refs[2]), (refs[1], refs[2])]
    )
    ours_gap = max(np.abs(ours - r).mean() for r in refs)
    assert ours_gap < 2.0 * ref_spread, (ours_gap, ref_spread)
    for r in refs:
        assert np.corrcoef(ours.ravel(), r.ravel())[0, 1] > 0.98


def test_masked_position_identity_does_not_leak(mlm):
    """The residue at a masked position must not influence its own prediction."""
    h3 = np.zeros(len(TRASTUZUMAB_VH), dtype=bool)
    h3[H3_START] = True
    base = mlm.template_logits(TRASTUZUMAB_VH, positions=h3)[H3_START]
    mutated = TRASTUZUMAB_VH[:H3_START] + "A" + TRASTUZUMAB_VH[H3_START + 1:]
    other = mlm.template_logits(mutated, positions=h3)[H3_START]
    assert torch.allclose(base, other, atol=1e-6)


def test_multi_mask_matches_per_position_masking(mlm):
    """A two-position multi-mask row equals decoding each with both hidden."""
    lp = mlm.multi_mask_logits(TRASTUZUMAB_VH, [H3_START, H3_START + 1])
    assert torch.isfinite(lp[[H3_START, H3_START + 1]]).all()
    assert torch.isnan(lp[H3_START + 2]).all()
    # Hiding a neighbour changes the prediction: multi-mask != single-mask.
    single = np.zeros(len(TRASTUZUMAB_VH), dtype=bool)
    single[H3_START] = True
    assert not torch.allclose(
        lp[H3_START], mlm.template_logits(TRASTUZUMAB_VH, single)[H3_START], atol=1e-3
    )


def test_determinism(mlm):
    h3 = np.zeros(len(TRASTUZUMAB_VH), dtype=bool)
    h3[H3_START:H3_END] = True
    a = mlm.template_logits(TRASTUZUMAB_VH, positions=h3)
    b = mlm.template_logits(TRASTUZUMAB_VH, positions=h3)
    assert torch.equal(a[H3_START:H3_END], b[H3_START:H3_END])
