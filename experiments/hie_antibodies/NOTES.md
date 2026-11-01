# Hie et al. 2024 one-edit PLL evaluation — decisions to review

Repeats the Trastuzumab one-edit PLL evaluation (true vs. WT-marginal-approx vs.
Gumbel-perturbed PLL, correlations at tau=1) for the seven antibodies from Hie
et al. 2024 (Nat. Biotech., "Efficient evolution of human antibodies from
general protein language models").

Script: `pll_one_edit_vhvl.py`. Output: `results/pll_one_edit_vhvl.parquet`
(+ `results/per_antibody/<name>.parquet` per antibody). Run log: `/tmp/hie_pll.log`.

## Judgment calls made while you were away (please sanity-check)

1. **Antibodies (7).** Taken verbatim from Brian Hie's `efficient-evolution`
   repo (`bin/{medi8852,medi_uca,mab114,mab114_uca,s309,regn10987,c143}.py`).
   MEDI8852 and mAb114 each contribute the matured antibody AND its UCA — that's
   how the count reaches seven. Named here: `medi8852`, `medi8852_uca`, `mab114`,
   `mab114_uca`, `s309`, `regn10987`, `c143`. VH/VL lengths 118–129 / 103–110.

2. **Region = full paired VH+VL** (your call: "full sequence… full VH and VL").
   One-edit variants enumerated over every position of both chains
   ((len VH + len VL) × 19 ≈ 4.4k/antibody). PLL summed over all positions.

3. **Model = AbLang-2 only.** ESM2 would be ~37 h here for this; the
   structure-conditioned models (ProteinMPNN/AbMPNN) need per-antibody structures
   we don't have. AbLang-2 is natively paired, which is why full VH+VL is natural.

4. **Scored directly on the paired model, not via the mmmlm wrapper.** The
   `AbLang2` wrapper fixes the context chain at construction, so it can't vary VH
   while masking VL (and vice-versa). This script masks each position across the
   concatenated `<VH>|<VL>` input and reads that position's own masked marginal —
   a single paired pass conditioning on both chains, exactly as you asked. PLL
   math reuses the method's own `_log_softmax_at_temperature` and
   `_self_pll_from_log_marginals`, so the definition matches beam_search.

5. **tau = 1, seed = 0.** Matches the reviewer-facing Trastuzumab tau=1 table.
   Gumbel = one uniform draw per variant, `-log(-log(u))`, added to approx_pll.

6. **Exact true PLL (no approximation).** Each variant's true PLL is a fresh
   full-pair masked-marginal pass in that variant's context. ~3.7 s/variant on
   MPS → ~4.5 h/antibody → ~32 h total. Running in background, resumable.

7. **Caching.** Each paired context's raw (pre-temperature) log-prob marginal
   matrix is cached to `cache/marginals/<sha1(VH|VL)>.npy`. Lets a killed run
   resume instantly and lets us recompute at another temperature / re-draw Gumbel
   with no model calls. Per-antibody parquets persist as each finishes.
   (Both `cache/` and `results/per_antibody/` are already gitignored; the
   top-level combined parquet is trackable.)

## Not yet done
- Nothing committed yet — left for your review of the above.
- A per-antibody figure (like the Trastuzumab grid) could be added once data lands.
- The Gumbel columns are stochastic (seed 0), same caveat as before.
