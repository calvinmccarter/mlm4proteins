"""AbLang-2 wrapper exposing the :class:`ProteinMLM` interface for one chain.

AbLang-2 (Olsen et al. 2024) is a paired heavy/light chain MLM. The wrapper is
constructed with a fixed partner chain so the standard ``template_logits`` API
can be called with just the *variable* chain's sequence:

  - ``variable_chain="H"`` (default): the ``sequence`` argument is the VH and a
    fixed VL is supplied via ``vl_sequence``. This is the original behavior.
  - ``variable_chain="L"``: the ``sequence`` argument is the VL and a fixed VH is
    supplied via ``vh_sequence``.

Internally we use ``ablang2``'s ``mode="probability"``. AbLang2 tokenizes a pair
as ``< VH > | < VL >`` with 5 special tokens, so VH residue rows begin at index
``1`` and VL residue rows begin at index ``len(VH) + 4``. We extract the rows for
the variable chain, log them (a log-softmax_τ=1 matrix up to a per-row constant),
and re-order columns to the canonical 20 AA order. A temperature τ ≠ 1 is applied
downstream by the sampler since ``softmax_τ(log_probs)`` is invariant to the
unknown per-row offset.

``template_logits`` masks each requested position with ``'*'`` and scores all the
variants in one batched call, rather than using AbLang2's ``stepwise_masking``
(which computes marginals for every position in the chain, when samplers only ask
for the few inside the mutable region). Equivalent to float32 noise and ~21x
faster on a 10-residue CDR — see ``tests/test_ablang2_batched_marginals.py``.
"""
from __future__ import annotations

from functools import cached_property
from typing import Optional

import numpy as np
import torch

from ..constants import AMINO_ACIDS

# AbLang-2's tokenizer uses "<VH>|<VL>" with 5 specials (<, >, |, <, >).
# VH residues occupy rows [1, 1+VH_len); VL residues occupy rows
# [VH_len + 4, VH_len + 4 + VL_len).
_VH_OFFSET = 1
_NUM_SPECIALS = 5


class AbLang2(torch.nn.Module):
    """AbLang-2 :class:`ProteinMLM` wrapper varying one chain with the other fixed."""

    def __init__(
        self,
        vl_sequence: Optional[str] = None,
        device: str | torch.device | None = None,
        variable_chain: str = "H",
        vh_sequence: Optional[str] = None,
        max_batch_size: int = 16,
    ):
        super().__init__()
        self.name = "ablang2"
        self.max_batch_size = max_batch_size
        if variable_chain not in ("H", "L"):
            raise ValueError(f"variable_chain must be 'H' or 'L', got {variable_chain!r}")
        self.variable_chain = variable_chain
        if variable_chain == "H":
            if vl_sequence is None:
                raise ValueError("variable_chain='H' requires vl_sequence (the fixed light chain)")
            self.vl_sequence = vl_sequence
            self.vh_sequence = vh_sequence  # unused in heavy mode
        else:  # "L"
            if vh_sequence is None:
                raise ValueError("variable_chain='L' requires vh_sequence (the fixed heavy chain)")
            self.vh_sequence = vh_sequence
            self.vl_sequence = vl_sequence  # unused in light mode
        self.device = (
            torch.device(device) if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        import ablang2  # local import so the rest of mmmlm doesn't hard-depend on it
        self._model = ablang2.pretrained(device=str(self.device))

    @cached_property
    def _aa_columns(self) -> np.ndarray:
        """Indices into AbLang2's alphabet that correspond to canonical AAs."""
        aa_to_token = self._model.tokenizer.aa_to_token
        out = np.zeros(len(AMINO_ACIDS), dtype=np.int64)
        for i, aa in enumerate(AMINO_ACIDS):
            if aa not in aa_to_token:
                raise RuntimeError(f"AbLang2 tokenizer missing canonical AA {aa!r}")
            out[i] = int(aa_to_token[aa])
        return out

    def _pair(self, variable_seq: str) -> list[str]:
        """Order the [heavy, light] pair AbLang2 expects, inserting the variable chain."""
        if self.variable_chain == "H":
            return [variable_seq, self.vl_sequence]
        return [self.vh_sequence, variable_seq]

    def _row_offset(self) -> int:
        """First probability row of the variable chain."""
        if self.variable_chain == "H":
            return _VH_OFFSET
        # Light chain rows start after "< VH > | <" == VH_len + 4 tokens.
        return len(self.vh_sequence) + (_NUM_SPECIALS - 1)

    @torch.inference_mode()
    def template_logits(
        self,
        sequence: str,
        positions: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        L = len(sequence)
        if positions is None:
            positions = np.ones(L, dtype=bool)
        if positions.shape[0] != L:
            raise ValueError(f"positions length {positions.shape[0]} != sequence length {L}")

        keep_rows = np.flatnonzero(positions)
        out = torch.full((L, len(AMINO_ACIDS)), float("nan"), dtype=torch.float32)
        if keep_rows.size == 0:
            return out
        offset = self._row_offset()

        # Mask each requested position with '*' and score all the variants in
        # one batched forward pass. AbLang2 accepts a list of [VH, VL] pairs, so
        # a whole CDR window costs a single call.
        #
        # The obvious alternative -- one call with ``stepwise_masking=True`` --
        # is what this used to do, but it makes AbLang2 compute marginals for
        # *every* position in the chain when a sampler only ever asks for the
        # handful inside the mutable region. Measured on the 10-residue
        # Trastuzumab H3: 749 ms stepwise vs 36 ms batched, a ~21x saving on the
        # dominant beam-search operation. The two agree to float32 noise
        # (max |diff| 5.7e-6) since both condition on the same context; see
        # tests/test_ablang2_batched_marginals.py.
        for start in range(0, keep_rows.size, self.max_batch_size):
            chunk = keep_rows[start: start + self.max_batch_size]
            pairs = [
                self._pair(sequence[:k] + "*" + sequence[k + 1:])
                for k in chunk.tolist()
            ]
            batch = np.asarray(self._model(pairs, mode="probability"))
            # batch[j] is the full pair; take the masked position's own row.
            rows = np.stack([batch[j, offset + int(k)] for j, k in enumerate(chunk)])
            with np.errstate(divide="ignore"):
                log_probs = np.log(np.clip(rows, 1e-30, None))
            out[chunk] = torch.from_numpy(log_probs[:, self._aa_columns]).to(out.dtype)
        return out

    @torch.inference_mode()
    def multi_mask_logits(
        self,
        sequence: str,
        mask_positions: list[int],
    ) -> torch.Tensor:
        L = len(sequence)
        out = torch.full((L, len(AMINO_ACIDS)), float("nan"), dtype=torch.float32)
        if not mask_positions:
            return out
        offset = self._row_offset()
        chars = list(sequence)
        for pos in mask_positions:
            if not (0 <= pos < L):
                raise ValueError(f"mask position {pos} out of range for length {L}")
            chars[pos] = "*"
        masked_seq = "".join(chars)
        # Single forward pass without stepwise_masking — AbLang2 fills the
        # masked positions conditioned on the unmasked context.
        seq_probs = np.asarray(self._model(self._pair(masked_seq), mode="probability")[0])
        for pos in mask_positions:
            row = seq_probs[offset + pos]
            with np.errstate(divide="ignore"):
                log_probs = np.log(np.clip(row, 1e-30, None))
            out[pos] = torch.from_numpy(log_probs[self._aa_columns]).to(out.dtype)
        return out
