# ProteinMPNN / AbMPNN scaffolding — prepared but deliberately not run

This directory contains a complete, smoke-tested pipeline for the
structure-conditioned one-edit PLL evaluation (ProteinMPNN + AbMPNN), but it was
**not run at scale** as part of the analysis. This note records why, and how to
run it if a reviewer asks.

## Files
- `mpnn_structures.py` — the 4 antibody→(PDB, VH/VL chain) config + a trimmer
  (keeps the Fab + its contacting antigen, drops redundant copies).
- `prep_mpnn_structures.py` — downloads the 4 PDBs + AbMPNN weights and writes
  the trimmed complexes into `structures/` (gitignored).
- `pll_one_edit_mpnn.py` — the eval, mirroring the sequence-model scripts
  (per-chain design; VH scored in VL+antigen context and vice-versa; approx /
  true / Gumbel PLL; caching; `--device cuda`).

## Why it wasn't run (the principled reason)
The most interesting regime — where the WT-marginal approximation diverges from
the true PLL — is the **near-germline UCAs**. UCAs are *inferred* germline
ancestors with **no experimental structure by construction**, so a
structure-conditioned model **cannot probe the phenomenon of interest at all**.
Concretely, of the 7 antibodies:
- `medi8852_uca`, `mab114_uca` — inferred UCAs, no structure exists.
- `c143` — no exact-sequence-match deposit (only ~96–100% homologs).
- `medi8852`, `mab114`, `s309`, `regn10987` — have structures, but all are
  **somatically matured**, i.e. the regime where the approximation is already
  tight (Approx-vs-True Spearman ≈ 0.93). So MPNN could only re-confirm the easy
  case.

Additionally, the method targets the **structure-free** setting (Hie et al.),
which is the whole point of sequence-based antibody optimization; ProteinMPNN /
AbMPNN require a backbone and answer a different question.

(Note: the disordered residues in the 4 available structures are **not** a good
argument — they are just flexible chain termini / a couple FR1 residues, all
CDRs are resolved, and `pll_one_edit_mpnn.py` already handles them by scoring
only the modeled residues.)

## How to run it (4 matured antibodies, GPU) if needed
```bash
export MMMLM_PROTEINMPNN_DIR=/path/to/ProteinMPNN     # github.com/dauparas/ProteinMPNN
python experiments/hie_antibodies/prep_mpnn_structures.py      # download + trim + AbMPNN weights
python experiments/hie_antibodies/pll_one_edit_mpnn.py --device cuda
```
Cost is ~tens of hours on CPU per antibody; run on a CUDA box (MPS is
unsupported — ProteinMPNN's decoder gather op). Smoke-tested on CPU with
`--antibodies regn10987 --max-variants 6`.
