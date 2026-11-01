"""Pin AbLang2's batched ``template_logits`` to the ``stepwise_masking`` result.

``template_logits`` used to make one ``stepwise_masking=True`` call, which asks
AbLang2 for masked marginals at *every* position in the chain. Samplers only ever
request the positions inside the mutable region, so it now masks each requested
position with ``'*'`` and scores those variants in a single batched call instead.

Both condition each position on the same unmasked context, so they must agree;
these tests assert that against the reference call, and that batching is
transparent (chunk boundaries do not change the answer).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mmmlm.constants import AMINO_ACIDS

ablang2 = pytest.importorskip("ablang2", reason="needs the ablang2 package")

TRASTUZUMAB_VH = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYAD"
    "SVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
)
TRASTUZUMAB_VL = (
    "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRF"
    "SGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"
)
H3_START, H3_END = 98, 108


@pytest.fixture(scope="module")
def mlm():
    from mmmlm.models.ablang2 import AbLang2

    return AbLang2(vl_sequence=TRASTUZUMAB_VL, device="cpu")


def _h3_mask() -> np.ndarray:
    m = np.zeros(len(TRASTUZUMAB_VH), dtype=bool)
    m[H3_START:H3_END] = True
    return m


def _stepwise_reference(mlm, sequence: str) -> torch.Tensor:
    """The pre-optimization implementation, kept here as the oracle."""
    offset = mlm._row_offset()
    probs = np.asarray(
        mlm._model(mlm._pair(sequence), mode="probability", stepwise_masking=True)[0]
    )
    chain = probs[offset: offset + len(sequence)]
    with np.errstate(divide="ignore"):
        logp = np.log(chain.clip(min=1e-30))
    return torch.from_numpy(logp[:, mlm._aa_columns]).to(torch.float32)


def test_matches_stepwise_masking(mlm):
    ours = mlm.template_logits(TRASTUZUMAB_VH, _h3_mask())[H3_START:H3_END]
    ref = _stepwise_reference(mlm, TRASTUZUMAB_VH)[H3_START:H3_END]
    assert torch.allclose(ours, ref, atol=1e-4), (ours - ref).abs().max()
    assert ours.argmax(-1).tolist() == ref.argmax(-1).tolist()


def test_batching_is_transparent(mlm):
    """Chunking must not change the result: batch of 16 vs forced batch of 3."""
    full = mlm.template_logits(TRASTUZUMAB_VH, _h3_mask())
    original = mlm.max_batch_size
    try:
        mlm.max_batch_size = 3
        chunked = mlm.template_logits(TRASTUZUMAB_VH, _h3_mask())
    finally:
        mlm.max_batch_size = original
    assert torch.equal(full[H3_START:H3_END], chunked[H3_START:H3_END])


def test_single_position_unchanged(mlm):
    """The Gibbs-style one-position query is a batch of one and still exact."""
    one = np.zeros(len(TRASTUZUMAB_VH), dtype=bool)
    one[H3_START] = True
    row = mlm.template_logits(TRASTUZUMAB_VH, one)
    assert torch.isnan(row[H3_START + 1]).all()
    assert torch.isfinite(row[H3_START]).all()
    # Same position requested alone or as part of the window -> same value.
    window = mlm.template_logits(TRASTUZUMAB_VH, _h3_mask())
    assert torch.allclose(row[H3_START], window[H3_START], atol=1e-6)


def test_out_of_mask_rows_are_nan(mlm):
    logits = mlm.template_logits(TRASTUZUMAB_VH, _h3_mask())
    assert torch.isnan(logits[:H3_START]).all()
    assert torch.isnan(logits[H3_END:]).all()
    assert torch.isfinite(logits[H3_START:H3_END]).all()
    assert logits.shape == (len(TRASTUZUMAB_VH), len(AMINO_ACIDS))
