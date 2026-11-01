# mmmlm

Reference implementation accompanying *How to make the most of your masked
language model for protein engineering* (NeurIPS 2026) —
[arXiv](https://arxiv.org/abs/2603.10302) ·
[OpenReview](https://openreview.net/forum?id=b0V2pP79ls).

This repo provides a small, dependency-light implementation of:

- Sequence-centric **stochastic beam search** over MLM pseudo-log-likelihoods,
  using the wild-type marginal approximation so that all single-substitution
  neighbors of a template are scored at the cost of computing the template's
  PLL itself.
- **Gibbs sampling** for MLMs (mutation-centric baseline).
- **Multi-objective guidance** via non-dominated sorting (NDS) and smooth
  Tchebycheff scalarization (STS).
- A **`ProteinMLM`** protocol with reference wrappers for ESM2-650M, AbLang-2,
  and the structure-conditioned inverse-folding models **ProteinMPNN** and
  **AbMPNN** (which expose `log p(s_i | s_{-M}, backbone)` rather than a
  mask-token marginal — see `src/mmmlm/models/mpnn.py`).

A reproducible in-silico experiment using **Trastuzumab CDR-H3** as the seed
and a binder-classification CNN (Mason et al. 2021 / Tz_her2_affinity_and_beyond)
as the oracle is provided under `experiments/trastuzumab_cdrh3/`.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[esm2,ablang2,trastuzumab,dev]"
```

## Quickstart

```python
from mmmlm.models.esm2 import ESM2
from mmmlm.regions import mask_from_slice
from mmmlm.sampling.beam import BeamSearchConfig, beam_search

mlm = ESM2()  # facebook/esm2_t33_650M_UR50D, on cuda if available
seed_vh = "EVQLVESGGGLVQ..."             # your VH
mask = mask_from_slice(len(seed_vh), 98, 108)  # mutate H3 only

cfg = BeamSearchConfig(
    edit_budget=4, beam_keep=200, beam_forward=20, n_final=200,
    temperature=1.5, gumbel=True,
)
candidates = beam_search(seed_vh, mlm, mask, cfg)
for c in candidates[:10]:
    print(c.edits, c.pll, c.sequence)
```

## Trastuzumab experiment

```bash
# 1) Train the CNN oracle (~minutes on a single GPU)
.venv/bin/python experiments/trastuzumab_cdrh3/oracle.py \
  --data-dir <path-to>/Tz_her2_affinity_and_beyond/data/affinity_data/her2/her2_aff_large/processed/remove_overlap/random_split/0.7_0.15_0.15 \
  --out experiments/trastuzumab_cdrh3/checkpoints/cnn.pt

# 2) Run all 9 sampling methods (5 beam + 3 gibbs + denoising) under both MLMs.
#    Single command — sequential across models:
.venv/bin/python experiments/trastuzumab_cdrh3/run.py \
  --oracle-checkpoint experiments/trastuzumab_cdrh3/checkpoints/cnn.pt

#    OR split across GPUs (one model per process), then merge:
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/trastuzumab_cdrh3/run.py \
  --oracle-checkpoint experiments/trastuzumab_cdrh3/checkpoints/cnn.pt \
  --models esm2 --out-dir experiments/trastuzumab_cdrh3/results/rerun_esm2 &
CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/trastuzumab_cdrh3/run.py \
  --oracle-checkpoint experiments/trastuzumab_cdrh3/checkpoints/cnn.pt \
  --models ablang2 --out-dir experiments/trastuzumab_cdrh3/results/rerun_ablang2 &
wait
.venv/bin/python experiments/trastuzumab_cdrh3/merge_results.py \
  --inputs experiments/trastuzumab_cdrh3/results/rerun_{esm2,ablang2}/all_sequences.csv \
  --out    experiments/trastuzumab_cdrh3/results/all_sequences.csv

# 3) Post-hoc OASis humanness scoring (only OASis-guided methods populate
#    score_oasis_percentile during the run; this fills it in for the rest):
.venv/bin/python experiments/trastuzumab_cdrh3/score_oasis.py \
  --csv experiments/trastuzumab_cdrh3/results/all_sequences.csv

# 4) Plots:
.venv/bin/python experiments/trastuzumab_cdrh3/analyze.py \
  --results-dir experiments/trastuzumab_cdrh3/results
```

Outputs land in `experiments/trastuzumab_cdrh3/results/`.

### Structure-conditioned models (ProteinMPNN / AbMPNN)

`--models proteinmpnn abmpnn` needs three extra inputs: a ProteinMPNN checkout
(the package is not on PyPI, and it ships the vanilla weights), the AbMPNN
weights, and a backbone. The backbone is **1N8Z** — the trastuzumab
Fab / HER2-domain-IV crystal structure, whose chain B begins with exactly the
seed VH, chain A with the seed VL, and chain C is the antigen:

```bash
git clone https://github.com/dauparas/ProteinMPNN.git ../ProteinMPNN
export MMMLM_PROTEINMPNN_DIR=$PWD/../ProteinMPNN

mkdir -p experiments/trastuzumab_cdrh3/structures
curl -o experiments/trastuzumab_cdrh3/structures/1N8Z.pdb \
  https://files.rcsb.org/download/1N8Z.pdb
curl -Lo experiments/trastuzumab_cdrh3/structures/abmpnn.pt \
  https://zenodo.org/api/records/8164693/files/abmpnn.pt/content

.venv/bin/python experiments/trastuzumab_cdrh3/run.py \
  --oracle-checkpoint experiments/trastuzumab_cdrh3/checkpoints/cnn.pt \
  --models proteinmpnn abmpnn \
  --proteinmpnn-dir $MMMLM_PROTEINMPNN_DIR \
  --out-dir experiments/trastuzumab_cdrh3/results/rerun_mpnn
```

The H3 loop conformation comes from wild-type trastuzumab and is held fixed for
every design, so these two models score H3 substitutions *in the wild-type
backbone* — they do not re-fold the loop. `--structure-context-chains A` drops
the antigen (chain C) if you want the apo-Fv comparison; the default `A C`
keeps it, since the epitope is what makes inverse folding informative about
binding. The default weights are `vanilla_model_weights/v_48_020.pt` for
ProteinMPNN and `abmpnn.pt` for AbMPNN; both use k=48 neighbours and run with
`augment_eps=0` (no backbone noise at inference).

`tests/test_mpnn.py` pins the wrapper to ProteinMPNN's own
`unconditional_probs` / `conditional_probs` implementations, and is skipped
unless `MMMLM_PROTEINMPNN_DIR` and the two downloads above are present.

#### Apo (no-antigen) appendix figures

`results/summary_{proteinmpnn,abmpnn}-apo.png` come from a separate sweep with
the HER2 chain dropped from the fixed context, so the models score H3 without
the epitope. They are held apart from the main figure set on purpose: the apo
rows live in `results/apo/all_sequences.csv` (gitignored) and are labelled
`proteinmpnn_apo` / `abmpnn_apo` so they can never be confused with the
with-antigen rows of the same model.

```bash
# 1) sweep, antigen dropped (--structure-context-chains omits chain C)
.venv/bin/python experiments/trastuzumab_cdrh3/run.py \
  --oracle-checkpoint experiments/trastuzumab_cdrh3/checkpoints/cnn.pt \
  --guidance-checkpoint experiments/trastuzumab_cdrh3/checkpoints/cnn_mason.pt \
  --models proteinmpnn abmpnn --edit-budgets 2 3 4 5 6 7 \
  --beam-keep 2000 --gibbs-n 200 --structure-context-chains A \
  --oasis-db <path-to>/OASis_9mers_v1.db --oasis-workers 24 \
  --proteinmpnn-dir $MMMLM_PROTEINMPNN_DIR \
  --out-dir experiments/trastuzumab_cdrh3/results/apo
# 2) suffix the model column, then score OASis for the sampler rows
python -c "import pandas as pd; p='experiments/trastuzumab_cdrh3/results/apo/all_sequences.csv'; \
d=pd.read_csv(p); d['model']+='_apo'; d.to_csv(p,index=False)"
.venv/bin/python experiments/trastuzumab_cdrh3/score_oasis.py \
  --csv experiments/trastuzumab_cdrh3/results/apo/all_sequences.csv --db-path <db> --n-workers 24
# 3) render into a scratch dir and lift out just the two summary panels, so the
#    main figures are untouched (analyze.py has no per-model output filter)
mkdir -p experiments/trastuzumab_cdrh3/results/apo/figs
cp experiments/trastuzumab_cdrh3/results/apo/all_sequences.csv \
   experiments/trastuzumab_cdrh3/results/apo/figs/
.venv/bin/python experiments/trastuzumab_cdrh3/analyze.py \
  --results-dir experiments/trastuzumab_cdrh3/results/apo/figs
cd experiments/trastuzumab_cdrh3/results
cp apo/figs/summary_proteinmpnnapo.png summary_proteinmpnn-apo.png
cp apo/figs/summary_abmpnnapo.png      summary_abmpnn-apo.png
```

Dropping the antigen costs binding across the board — AbMPNN's unguided beam at
4 realized edits falls from 0.935 to 0.575 mean oracle p(binder) — and lowers
p(wild-type) on H3 (AbMPNN 0.632 → 0.577, ProteinMPNN 0.351 → 0.272), which in
turn makes the Gibbs samplers 2.5–3.1x cheaper because fewer updates are wasted
returning the wild-type residue.
