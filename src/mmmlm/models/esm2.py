"""ESM-2 wrapper exposing the :class:`ProteinMLM` interface.

Loads ``facebook/esm2_t33_650M_UR50D`` (650M params) by default via HuggingFace
``transformers``. To compute the marginal at position ``i``, we run a forward
pass with that position replaced by ``<mask>``; multiple positions are batched.
The returned ``Tensor[L, 20]`` follows the canonical 20-letter alphabet order.
"""
from __future__ import annotations

from functools import cached_property
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from ..constants import AMINO_ACIDS


class ESM2(torch.nn.Module):
    """Thin :class:`ProteinMLM` wrapper around an ESM-2 checkpoint."""

    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        max_batch_size: int = 32,
    ):
        super().__init__()
        self.name = model_name
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype
        self.max_batch_size = max_batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._hf = AutoModelForMaskedLM.from_pretrained(model_name, dtype=dtype)
        self._hf.to(self.device).eval()

    @cached_property
    def _aa_token_ids(self) -> torch.Tensor:
        """Map canonical AA index -> ESM tokenizer id, shape [20]."""
        ids = [self.tokenizer.convert_tokens_to_ids(a) for a in AMINO_ACIDS]
        if any(i is None or i == self.tokenizer.unk_token_id for i in ids):
            missing = [a for a, i in zip(AMINO_ACIDS, ids)
                       if i is None or i == self.tokenizer.unk_token_id]
            raise RuntimeError(f"ESM tokenizer missing canonical AAs: {missing}")
        return torch.tensor(ids, dtype=torch.long, device=self.device)

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

        mask_indices = np.flatnonzero(positions)
        if mask_indices.size == 0:
            return torch.full((L, len(AMINO_ACIDS)), float("nan"))

        # Tokenize once: [<cls>, ...seq..., <eos>] of shape [1, L+2].
        enc = self.tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
        base_ids = enc["input_ids"].to(self.device)  # [1, L+2]
        attention = enc["attention_mask"].to(self.device)  # [1, L+2]

        seq_len = base_ids.shape[1]
        # Sanity: ESM2 prepends <cls> (1 token) and appends <eos> (1 token).
        assert seq_len == L + 2, f"unexpected ESM tokenization length {seq_len} for seq of length {L}"

        mask_id = self.tokenizer.mask_token_id

        out_logits = torch.full(
            (L, len(AMINO_ACIDS)), float("nan"),
            device=self.device, dtype=torch.float32,
        )

        for chunk_start in range(0, mask_indices.size, self.max_batch_size):
            chunk = mask_indices[chunk_start: chunk_start + self.max_batch_size]
            B = chunk.size
            batch_ids = base_ids.expand(B, -1).clone()
            for row, pos in enumerate(chunk):
                batch_ids[row, pos + 1] = mask_id  # +1 for <cls>
            batch_attention = attention.expand(B, -1)
            out = self._hf(input_ids=batch_ids, attention_mask=batch_attention)
            # Extract logits at the masked position of each row.
            row_idx = torch.arange(B, device=self.device)
            mask_pos = torch.tensor(chunk + 1, device=self.device)  # +1 for <cls>
            row_logits = out.logits[row_idx, mask_pos]  # [B, vocab]
            # Restrict to canonical 20 AA columns.
            aa_logits = row_logits.index_select(dim=-1, index=self._aa_token_ids)
            out_logits[chunk] = aa_logits.float()

        return out_logits.cpu()

    @torch.inference_mode()
    def multi_mask_logits(
        self,
        sequence: str,
        mask_positions: list[int],
    ) -> torch.Tensor:
        L = len(sequence)
        if not mask_positions:
            return torch.full((L, len(AMINO_ACIDS)), float("nan"))
        enc = self.tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
        ids = enc["input_ids"].to(self.device).clone()        # [1, L+2]
        attention = enc["attention_mask"].to(self.device)
        mask_id = self.tokenizer.mask_token_id
        for pos in mask_positions:
            if not (0 <= pos < L):
                raise ValueError(f"mask position {pos} out of range for length {L}")
            ids[0, pos + 1] = mask_id  # +1 for <cls>
        out = self._hf(input_ids=ids, attention_mask=attention)
        out_logits = torch.full(
            (L, len(AMINO_ACIDS)), float("nan"),
            device=self.device, dtype=torch.float32,
        )
        positions_t = torch.tensor([p + 1 for p in mask_positions], device=self.device)
        rows = out.logits[0, positions_t].index_select(dim=-1, index=self._aa_token_ids)
        for i, pos in enumerate(mask_positions):
            out_logits[pos] = rows[i].float()
        return out_logits.cpu()
