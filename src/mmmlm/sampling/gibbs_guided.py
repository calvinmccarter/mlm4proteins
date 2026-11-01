"""Oracle-guided Gibbs sampling (Raghu et al. 2025-style product of experts).

For each chain we pre-pick ``E`` mutation positions in the mutable region (with
replacement, matching ``GuidedSamplingGenerator`` in the bh-ai codebase). At
each step ``t`` we draw a residue ``r`` for position ``k_t`` from a product of
experts distribution combining the MLM's masked-position conditional and one
or more oracle scores:

    p_PoE(x_{k_t} = r | x_{¬k_t}) ∝ p_MLM(x_{k_t} = r | x_{¬k_t})
                                     · exp(Σ_o γ_o · f_o(x with x_{k_t}=r))

(equation 4 of Raghu, Ober, Kazman, Elliott, "Guided Sequence-Structure
Generative Modeling for Iterative Antibody Optimization", GEM @ ICLR 2025).

The MLM contribution is computed by masking ``k_t`` and reading off the
log-softmax(τ_mlm) at that row. The oracle contribution is computed by
constructing all 20 candidate sequences (one per amino acid at ``k_t``) and
scoring each with the oracle, then optionally rescaling each oracle's
contribution by a per-oracle guidance strength γ. After summation, an optional
top-p (nucleus) filter removes the tail.

For an oracle that returns probabilities in (0, 1) (such as the trastuzumab
CNN), we use ``log p`` as the additive log-likelihood term — that is, the PoE
multiplies ``p_MLM`` by ``p_oracle ** γ``. Direction is respected: a
``"minimize"`` scorer is negated before being added in.

Scaling (``guidance_scaling``)
-----------------------------
The oracle term is *added to log-probabilities*, so what decides how much it can
actually move the sample is the term's spread across the 20 candidate residues —
not its absolute magnitude.

``"raw"`` (default, the original behaviour) adds ``γ · f_o`` in the oracle's own
units. That is well-posed only when ``f_o`` is already on a log-probability scale,
which holds for a probability-valued oracle under ``log_on_probabilities``.

For a *regression* oracle in arbitrary units it silently under- or over-weights.
Measured on the AlphaSeq affinity surrogate (log10 nM K_D): across the 20
substitutions at one CDR position the affinity term spans ~1.2 while the MLM
log-softmax spans ~7.7 nats, so at γ=1.0 the MLM out-votes the oracle ~6:1 and
guidance is nearly a no-op.

``"mlm_matched"`` fixes that by standardizing the oracle term over the 20
candidates and rescaling it to the MLM term's own standard deviation:

    contribution = γ · std(log p_MLM) · zscore(f_o)

γ then means a *relative* weight — 1.0 is parity with the MLM, 2.0 is twice its
pull — mirroring how :func:`mmmlm.moo.smooth_tchebycheff` z-scores each objective
before applying weights, which is why STS guidance works at weight 1.0 while raw
PoE at γ=1.0 does not.

Two properties are deliberate. The MLM term itself is never rescaled, so γ=0
still reproduces plain Gibbs exactly and ``mlm_temperature`` keeps its meaning.
And the match is done per position, because the MLM's spread varies widely by
position (~4-12 nats in the sequence above); a single global constant would still
be swamped at exactly the confident positions where guidance must push hardest.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..constants import AA_TO_IDX, AMINO_ACIDS, NUM_AMINO_ACIDS
from ..models.base import ProteinMLM
from ..scoring.base import ScoringFunction


@dataclass
class GuidedGibbsConfig:
    edit_budget: int                          # E realized edits per chain
    # Update cap; None means 60 · E. Reaching E edits takes more than E updates
    # whenever the model puts mass on the wild-type residue.
    max_steps: int | None = None
    mlm_temperature: float = 1.0              # τ for log_softmax(MLM logits / τ)
    guidance_strength: float = 1.0            # γ — single shared strength
    guidance_strengths: dict[str, float] | None = None  # name -> γ override
    top_p: float = 1.0                        # nucleus filter; 1.0 disables
    log_on_probabilities: bool = True         # treat p∈(0,1) oracle outputs as log p
    # "raw": add γ·f_o in the oracle's own units (original behaviour, and correct for
    # log-probability-valued oracles). "mlm_matched": z-score f_o over the 20
    # candidates and rescale to the MLM term's std, making γ a relative weight.
    # See the module docstring. Default stays "raw" so existing results reproduce.
    guidance_scaling: str = "raw"
    scaling_eps: float = 1e-6                 # floor on the oracle-term std
    # Emit the chain's state at the first time it reaches each edit count 1..E, not
    # just E. Valid for the same reason as in the unguided sampler -- see
    # ``mmmlm.sampling.gibbs.GibbsConfig.capture_intermediates``. Off by default.
    capture_intermediates: bool = False
    seed: int | None = 0

    def __post_init__(self):
        if self.guidance_scaling not in ("raw", "mlm_matched"):
            raise ValueError("guidance_scaling must be 'raw' or 'mlm_matched', "
                             f"got {self.guidance_scaling!r}")


def _strength_for(fn_name: str, cfg: GuidedGibbsConfig) -> float:
    if cfg.guidance_strengths and fn_name in cfg.guidance_strengths:
        return float(cfg.guidance_strengths[fn_name])
    return cfg.guidance_strength


def _match_to_mlm(oracle_term: np.ndarray, log_p_mlm: np.ndarray, eps: float) -> np.ndarray:
    """Standardize ``oracle_term`` and rescale it to ``log_p_mlm``'s spread.

    Returns a vector whose std equals the MLM term's std, so γ=1.0 gives the oracle
    the same pull as the MLM. A degenerate oracle term (every candidate scored
    alike) expresses no preference, so this returns zeros rather than dividing by
    ~0 and manufacturing one.
    """
    sd = float(oracle_term.std())
    if sd <= eps:
        return np.zeros_like(oracle_term)
    return (oracle_term - oracle_term.mean()) / sd * float(log_p_mlm.std())


def _oracle_logits(
    candidates: list[str],
    scorer: ScoringFunction,
    log_on_probabilities: bool,
) -> np.ndarray:
    """Score 20 candidate sequences and convert to a 20-vector of log-likelihood terms."""
    raw = np.asarray(scorer(candidates), dtype=np.float64)
    if scorer.direction == "minimize":
        raw = -raw
    if log_on_probabilities and raw.min() >= 0.0 and raw.max() <= 1.0:
        return np.log(np.clip(raw, 1e-30, None))
    return raw


def _top_p_filter(logits: np.ndarray, top_p: float) -> np.ndarray:
    """Standard nucleus filter: zero out the tail beyond cumulative mass top_p."""
    if top_p >= 1.0:
        return logits
    order = np.argsort(-logits)
    sorted_logits = logits[order]
    sorted_probs = np.exp(sorted_logits - sorted_logits.max())
    sorted_probs /= sorted_probs.sum()
    cum = np.cumsum(sorted_probs)
    keep_count = int(np.searchsorted(cum, top_p) + 1)
    out = np.full_like(logits, -np.inf)
    out[order[:keep_count]] = logits[order[:keep_count]]
    return out


def gibbs_guided_sample_one(
    seed_sequence: str,
    model: ProteinMLM,
    mutable_mask: np.ndarray,
    scoring_functions: list[ScoringFunction],
    config: GuidedGibbsConfig,
    rng: np.random.Generator | None = None,
    snapshots: dict[int, str] | None = None,
) -> tuple[str, list[int]]:
    """Run PoE-guided Gibbs from ``seed_sequence`` until it has E realized edits.

    ``snapshots``, if given, is filled with ``{n_edits: sequence}`` at the first time
    the chain reaches each edit count in 1..E -- see ``capture_intermediates``.

    Stopping rule matches :func:`mmmlm.sampling.gibbs.gibbs_sample_one` — E counts
    realized edits, not updates — so the guided and unguided variants land at the
    same edit distances and stay comparable in plots grouped by realized edits.
    Taking exactly E updates instead would make the guided methods stop short by
    the model's wild-type mass, which is severe for structure-conditioned models
    (AbMPNN realizes ~2.4 of 7 requested edits that way).
    """
    if rng is None:
        rng = np.random.default_rng(config.seed)
    if not scoring_functions:
        raise ValueError("at least one scoring function is required for guided Gibbs")

    cand_positions = np.flatnonzero(mutable_mask)
    if cand_positions.size == 0:
        raise ValueError("mutable_mask has no True positions")
    if config.edit_budget > cand_positions.size:
        raise ValueError(
            f"edit_budget {config.edit_budget} exceeds the mutable region size "
            f"{cand_positions.size}"
        )

    def _n_edits(sequence: str) -> int:
        return sum(1 for a, b in zip(sequence, seed_sequence) if a != b)

    seq = seed_sequence
    max_steps = config.max_steps if config.max_steps is not None else 60 * config.edit_budget
    for _ in range(max_steps):
        n_now = _n_edits(seq)
        if snapshots is not None and 0 < n_now <= config.edit_budget:
            snapshots.setdefault(n_now, seq)
        if n_now == config.edit_budget:
            break
        k = int(rng.choice(cand_positions))

        # MLM contribution: mask position k, get log-softmax over the 20 AAs.
        single_pos = np.zeros_like(mutable_mask)
        single_pos[k] = True
        mlm_logits = model.template_logits(seq, positions=single_pos)
        log_p_mlm = (
            torch.log_softmax(mlm_logits[k] / config.mlm_temperature, dim=-1)
            .cpu()
            .numpy()
        )

        # Oracle contributions: 20 candidates, one per AA substitution at k.
        candidates = [seq[:k] + AMINO_ACIDS[r] + seq[k + 1:] for r in range(NUM_AMINO_ACIDS)]
        combined = log_p_mlm.copy()
        for fn in scoring_functions:
            gamma = _strength_for(fn.name, config)
            term = _oracle_logits(
                candidates, fn, log_on_probabilities=config.log_on_probabilities
            )
            if config.guidance_scaling == "mlm_matched":
                term = _match_to_mlm(term, log_p_mlm, config.scaling_eps)
            combined += gamma * term

        combined = _top_p_filter(combined, config.top_p)
        # Sample from the renormalized distribution.
        m = combined.max()
        probs = np.exp(combined - m)
        probs = probs / probs.sum()
        r = int(rng.choice(NUM_AMINO_ACIDS, p=probs))
        seq = seq[:k] + AMINO_ACIDS[r] + seq[k + 1:]

    mutated = [i for i in range(len(seq)) if seq[i] != seed_sequence[i]]
    return seq, mutated


def gibbs_guided_sample_n(
    seed_sequence: str,
    model: ProteinMLM,
    mutable_mask: np.ndarray,
    scoring_functions: list[ScoringFunction],
    config: GuidedGibbsConfig,
    n_sequences: int,
) -> list[tuple[str, list[int]]]:
    """Run independent guided-Gibbs chains; return up to ``n_sequences`` unique outputs.

    Chains that hit the update cap without reaching E edits are reported, as in
    the unguided sampler — it measures the model's resistance, not a failure.
    """
    rng = np.random.default_rng(config.seed)
    seen: dict[str, list[int]] = {}
    short = 0
    for _ in range(n_sequences):
        snaps: dict[int, str] | None = {} if config.capture_intermediates else None
        seq, muts = gibbs_guided_sample_one(
            seed_sequence, model, mutable_mask, scoring_functions, config, rng=rng,
            snapshots=snaps,
        )
        if len(muts) != config.edit_budget:
            short += 1
        if seq not in seen:
            seen[seq] = muts
        if snaps:
            for snap in snaps.values():
                if snap not in seen:
                    seen[snap] = [i for i in range(len(snap))
                                  if snap[i] != seed_sequence[i]]
    if short:
        print(f"    [gibbs_guided E={config.edit_budget}] {short}/{n_sequences} chains "
              f"hit the update cap without reaching {config.edit_budget} edits", flush=True)
    return [(s, m) for s, m in seen.items()]
