"""ProteinMPNN / AbMPNN wrappers exposing the :class:`ProteinMLM` interface.

ProteinMPNN (Dauparas et al. 2022) and its antibody-finetuned variant AbMPNN
(Dreyer et al. 2023) are *inverse folding* models: they factorize
``p(sequence | backbone)`` autoregressively over a random decoding order. They
are not masked language models, but they expose exactly the quantity the
samplers in :mod:`mmmlm.sampling` need — the conditional distribution at a
position given the backbone and the identity of every *other* residue:

    A[i, r] = log p(s_i = r | s_{-M}, X),   M = the set of "masked" positions

Both the ProteinMPNN encoder (which depends only on the fixed backbone ``X``)
and the decoder support this. The decoder's ``mask_bw`` / ``mask_fw`` gates
decide, per position, which neighbours contribute *sequence* information
(already-decoded) versus *backbone-only* information (not yet decoded). Setting

    order_mask_backward[q, p] = 1  iff  p not in M

makes every position see the identities of all non-masked residues and none of
the masked ones, in a single decoder pass. This is the "decode M simultaneously"
variant of ProteinMPNN's own ``conditional_probs`` (which instead samples a
random decoding order for the non-target positions), and it is what makes the
model behave like the ESM-2 / AbLang-2 wrappers: mask a position, let everything
else be visible. It is also deterministic — no ``randn`` draw is involved.

Two limiting cases pin the implementation to the reference code:

  - ``M`` = every position reproduces ``ProteinMPNN.unconditional_probs``
    bit-for-bit (both amount to ``order_mask_backward == 0``).
  - ``M`` = ``{i}`` is the single-position conditional, i.e. ProteinMPNN's
    ``--conditional_probs_only`` mode up to the decoding order of the
    non-target positions.

See ``tests/test_mpnn.py`` for both checks.

The backbone is fixed for the whole search, so the encoder is run **once** at
construction and cached; each ``template_logits`` call is a batch of decoder
passes only. Backbone noise (``augment_eps``) is 0 at inference, matching
``protein_mpnn_run.py``'s default, so the cached encoding is exact.

Only one chain segment is designable. The wrapper is constructed with the
structure, the chain to design, and the seed sequence of the designed segment;
all other chains (partner chain, antigen, constant domains) stay as
crystallized and are always visible context.
"""
from __future__ import annotations

import os
import sys
from functools import cached_property
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from ..constants import AMINO_ACIDS

# ProteinMPNN's fixed 21-letter alphabet (20 canonical + X).
MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"

# Architecture constants hard-coded in protein_mpnn_run.py.
_HIDDEN_DIM = 128
_NUM_LAYERS = 3


def _import_proteinmpnn(root: str | os.PathLike | None):
    """Import the ProteinMPNN package from a checkout on disk.

    ProteinMPNN is not distributed on PyPI, so the caller points us at a clone
    of https://github.com/dauparas/ProteinMPNN (either explicitly or via the
    ``MMMLM_PROTEINMPNN_DIR`` environment variable).
    """
    if root is None:
        root = os.environ.get("MMMLM_PROTEINMPNN_DIR")
    if root is None:
        raise RuntimeError(
            "ProteinMPNN checkout not found. Pass proteinmpnn_root=... or set "
            "MMMLM_PROTEINMPNN_DIR to a clone of "
            "https://github.com/dauparas/ProteinMPNN"
        )
    root = Path(root).expanduser().resolve()
    if not (root / "protein_mpnn_utils.py").is_file():
        raise RuntimeError(f"{root} does not look like a ProteinMPNN checkout")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import protein_mpnn_utils  # noqa: F401  (side effect: importable module)

    return protein_mpnn_utils


class StructureMPNN(torch.nn.Module):
    """ProteinMPNN-family :class:`ProteinMLM` over a fixed backbone.

    Args:
        structure_path: PDB file of the complex.
        weights_path: ``.pt`` checkpoint (``vanilla_model_weights/v_48_020.pt``
            for ProteinMPNN, ``abmpnn.pt`` for AbMPNN).
        design_chain: chain ID whose sequence is varied.
        design_sequence: seed sequence of the designable segment. It must occur
            in ``design_chain``'s structure sequence; its location fixes the
            mapping from ``template_logits`` sequence positions to structure
            positions. Defaults to the whole chain.
        context_chains: other chains to include as fixed visible context.
            ``None`` includes every chain in the file except ``design_chain``.
        name: value of the ``ProteinMLM.name`` attribute.
        max_batch_size: decoder passes batched per forward.
    """

    def __init__(
        self,
        structure_path: str | os.PathLike,
        weights_path: str | os.PathLike,
        design_chain: str,
        design_sequence: Optional[str] = None,
        context_chains: Optional[Sequence[str]] = None,
        name: str = "proteinmpnn",
        device: str | torch.device | None = None,
        max_batch_size: int = 4,
        backbone_noise: float = 0.0,
        proteinmpnn_root: str | os.PathLike | None = None,
    ):
        super().__init__()
        self.name = name
        self.max_batch_size = max_batch_size
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        mpnn = _import_proteinmpnn(proteinmpnn_root)
        self._gather_nodes = mpnn.gather_nodes
        self._cat_neighbors_nodes = mpnn.cat_neighbors_nodes

        available = sorted(
            k[len("seq_chain_"):]
            for k in mpnn.parse_PDB(str(structure_path))[0]
            if k.startswith("seq_chain_")
        )
        if design_chain not in available:
            raise ValueError(
                f"design_chain {design_chain!r} not in {structure_path} "
                f"(chains: {available})"
            )
        if context_chains is None:
            context_chains = [c for c in available if c != design_chain]
        else:
            context_chains = [c for c in context_chains if c != design_chain]
            missing = [c for c in context_chains if c not in available]
            if missing:
                raise ValueError(f"context chains {missing} not in {structure_path}")

        # Re-parse with only the chains we keep: tied_featurize derives the
        # padded length from pdb['seq'], so unused chains must not be present.
        pdb = mpnn.parse_PDB(
            str(structure_path), input_chain_list=[design_chain] + list(context_chains)
        )[0]

        # tied_featurize concatenates sorted(masked_chains) + sorted(visible_chains),
        # so putting the design chain alone in `masked_chains` places it first and
        # gives it global offset 0.
        chain_seq = pdb[f"seq_chain_{design_chain}"]
        if design_sequence is None:
            design_sequence = chain_seq.replace("-", "X")
            design_offset = 0
        else:
            design_offset = chain_seq.find(design_sequence)
            if design_offset < 0:
                raise ValueError(
                    f"design_sequence (len {len(design_sequence)}) not found in chain "
                    f"{design_chain} of {structure_path} (len {len(chain_seq)}). "
                    "The seed sequence must match the deposited sequence exactly."
                )
        self.design_length = len(design_sequence)
        self.design_sequence = design_sequence
        # Global (complex-wide) index of designable-segment position 0.
        self._global_offset = design_offset
        self.design_chain = design_chain
        self.context_chains = list(context_chains)

        feats = mpnn.tied_featurize(
            [pdb], self.device, {pdb["name"]: ([design_chain], list(context_chains))}
        )
        X, S_wt, mask = feats[0], feats[1], feats[2]
        chain_encoding_all, residue_idx = feats[5], feats[12]

        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        model = mpnn.ProteinMPNN(
            num_letters=21,
            node_features=_HIDDEN_DIM,
            edge_features=_HIDDEN_DIM,
            hidden_dim=_HIDDEN_DIM,
            num_encoder_layers=_NUM_LAYERS,
            num_decoder_layers=_NUM_LAYERS,
            augment_eps=backbone_noise,
            k_neighbors=checkpoint["num_edges"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device).eval()
        self._model = model
        self.train_noise_level = float(checkpoint["noise_level"])
        self.k_neighbors = int(checkpoint["num_edges"])

        self._S_wt = S_wt
        self._mask = mask
        # Kept so tests can call the reference conditional_probs /
        # unconditional_probs implementations on the same featurized input.
        self._X = X
        self._residue_idx = residue_idx
        self._chain_encoding_all = chain_encoding_all
        self.complex_length = int(S_wt.shape[1])

        # The backbone never changes, so encode once. augment_eps must be 0 for
        # this to be exact (ProteinFeatures adds Gaussian noise to X otherwise).
        if backbone_noise != 0.0:
            raise ValueError(
                "backbone_noise must be 0.0: the encoder is cached once, so "
                "coordinate noise would silently be frozen across all calls."
            )
        with torch.inference_mode():
            E, E_idx = model.features(X, mask, residue_idx, chain_encoding_all)
            h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
            h_E = model.W_e(E)
            attend = mpnn.gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
            attend = mask.unsqueeze(-1) * attend
            for layer in model.encoder_layers:
                h_V, h_E = layer(h_V, h_E, E_idx, mask, attend)
        self._E_idx = E_idx
        self._h_V_enc = h_V
        self._h_E = h_E

    @cached_property
    def _aa_columns(self) -> torch.Tensor:
        """Canonical-AA index -> ProteinMPNN alphabet column, shape [20]."""
        return torch.tensor(
            [MPNN_ALPHABET.index(a) for a in AMINO_ACIDS],
            dtype=torch.long, device=self.device,
        )

    def _tokenize(self, sequence: str) -> torch.Tensor:
        """WT complex tokens with the designable segment replaced by ``sequence``."""
        if len(sequence) != self.design_length:
            raise ValueError(
                f"sequence length {len(sequence)} != designable segment length "
                f"{self.design_length}"
            )
        S = self._S_wt.clone()
        try:
            tokens = [MPNN_ALPHABET.index(c) for c in sequence]
        except ValueError as exc:
            raise ValueError(f"sequence contains a non-ProteinMPNN letter: {exc}") from exc
        S[0, self._global_offset: self._global_offset + self.design_length] = torch.tensor(
            tokens, dtype=S.dtype, device=S.device
        )
        return S

    @torch.inference_mode()
    def _decode(self, S: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
        """Run the decoder with an explicit per-row visibility mask.

        Args:
            S: long[B, L] complex tokens.
            visible: float[B, L]; ``visible[b, p] == 1`` means row ``b`` may use
                residue ``p``'s identity. Positions with ``visible == 0`` are the
                "masked" set ``M`` — every position, including ``p`` itself via
                the self-edge, is blocked from seeing them.

        Returns:
            float[B, L, 21] log-probabilities.
        """
        model = self._model
        B, L = S.shape
        E_idx = self._E_idx.expand(B, -1, -1)
        h_E = self._h_E.expand(B, -1, -1, -1)
        h_V_enc = self._h_V_enc.expand(B, -1, -1)
        mask = self._mask.expand(B, -1)

        h_S = model.W_s(S)
        h_ES = self._cat_neighbors_nodes(h_S, h_E, E_idx)
        h_EX_encoder = self._cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = self._cat_neighbors_nodes(h_V_enc, h_EX_encoder, E_idx)

        # order_mask_backward[b, q, p] = visible[b, p] is independent of q, so
        # gathering it along the neighbour axis is just a node gather.
        mask_attend = self._gather_nodes(visible.unsqueeze(-1), E_idx)  # [B, L, K, 1]
        mask_1D = mask.view(B, L, 1, 1)
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1.0 - mask_attend)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder

        h_V = h_V_enc
        for layer in model.decoder_layers:
            h_ESV = self._cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask)
        return torch.log_softmax(model.W_out(h_V), dim=-1)

    def _global(self, positions: np.ndarray) -> np.ndarray:
        return positions + self._global_offset

    @torch.inference_mode()
    def template_logits(
        self,
        sequence: str,
        positions: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        L = self.design_length
        if len(sequence) != L:
            raise ValueError(f"sequence length {len(sequence)} != {L}")
        if positions is None:
            positions = np.ones(L, dtype=bool)
        if positions.shape[0] != L:
            raise ValueError(
                f"positions length {positions.shape[0]} != sequence length {L}"
            )
        out = torch.full((L, len(AMINO_ACIDS)), float("nan"), dtype=torch.float32)
        idx = np.flatnonzero(positions)
        if idx.size == 0:
            return out

        S = self._tokenize(sequence)
        for start in range(0, idx.size, self.max_batch_size):
            chunk = idx[start: start + self.max_batch_size]
            B = chunk.size
            g = torch.as_tensor(self._global(chunk), device=self.device)
            visible = torch.ones((B, self.complex_length), device=self.device)
            rows = torch.arange(B, device=self.device)
            visible[rows, g] = 0.0
            log_p = self._decode(S.expand(B, -1), visible)
            out[chunk] = (
                log_p[rows, g].index_select(dim=-1, index=self._aa_columns).float().cpu()
            )
        return out

    @torch.inference_mode()
    def multi_mask_logits(
        self,
        sequence: str,
        mask_positions: list[int],
    ) -> torch.Tensor:
        L = self.design_length
        out = torch.full((L, len(AMINO_ACIDS)), float("nan"), dtype=torch.float32)
        if not mask_positions:
            return out
        for pos in mask_positions:
            if not (0 <= pos < L):
                raise ValueError(f"mask position {pos} out of range for length {L}")
        g = torch.as_tensor(
            self._global(np.asarray(mask_positions, dtype=np.int64)), device=self.device
        )
        visible = torch.ones((1, self.complex_length), device=self.device)
        visible[0, g] = 0.0
        # One pass: every masked position is predicted from the same
        # non-masked context, none of them sees any other.
        log_p = self._decode(self._tokenize(sequence), visible)
        rows = log_p[0, g].index_select(dim=-1, index=self._aa_columns).float().cpu()
        out[np.asarray(mask_positions, dtype=np.int64)] = rows
        return out


def ProteinMPNNMLM(**kwargs) -> StructureMPNN:
    """:class:`StructureMPNN` with vanilla ProteinMPNN defaults."""
    kwargs.setdefault("name", "proteinmpnn")
    return StructureMPNN(**kwargs)


def AbMPNN(**kwargs) -> StructureMPNN:
    """:class:`StructureMPNN` with AbMPNN defaults (same architecture, ab weights)."""
    kwargs.setdefault("name", "abmpnn")
    return StructureMPNN(**kwargs)
