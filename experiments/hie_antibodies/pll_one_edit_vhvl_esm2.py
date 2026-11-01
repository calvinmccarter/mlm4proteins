"""ESM2-650M counterpart to ``pll_one_edit_vhvl.py`` (AbLang-2).

Runs the *same* exact-vs-approximate one-edit PLL analysis, but with a general
protein LM (ESM2) instead of the antibody-specific ALM (AbLang-2), to test
whether the wild-type-marginal approximation's breakdown on high-probability
(germline / UCA) seeds is specific to ALMs. If ESM2's residue-to-residue
coupling on antibody sequences is weaker, its approx PLL should track the exact
PLL well even where AbLang-2's collapses.

ESM2 is not a paired model, so ``--pairing`` chooses how VH+VL is scored:

  * ``concat``   -- score the *concatenated* VH+VL as one chain (each position
                    conditioned on the full VH+VL, mirroring AbLang-2's paired
                    forward). Fairest model-only comparison; the VH/VL junction
                    is out-of-distribution for ESM2 but only touches a couple of
                    boundary positions out of ~230.
  * ``perchain`` -- score VH and VL independently (VH positions see only VH, VL
                    only VL). This is how ESM2 is "meant" to be used, but it
                    zeroes cross-chain coupling by construction, changing the
                    task relative to AbLang-2.

``--sample-variants N`` scores a reproducible random subset of the
(len(VH)+len(VL)) x 19 one-edit variants (drawn uniformly across all positions
and mutations) instead of all of them -- a few hundred points already pin the
per-antibody Pearson/Spearman tightly, turning a ~day-long full sweep into a few
GPU-hours. Set 0 for the full enumeration.

Cache, output columns, and filenames are namespaced by model+pairing so nothing
collides with the AbLang-2 run's ``cache/marginals`` or ``results/``.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR))                       # sibling script
sys.path.insert(0, str(EXPERIMENT_DIR.parent.parent / "src"))  # mmmlm package

from mmmlm.constants import AA_TO_IDX, AMINO_ACIDS
# Reuse the antibody panel and the shared PLL / reporting helpers verbatim.
from pll_one_edit_vhvl import (
    ANTIBODIES,
    _corr_table,
    _pll,
    _print_padded,
    _variants,
)


def _pick_device(requested: str | None) -> str:
    if requested is not None:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


class ESM2Scorer:
    """ESM2 masked-marginal scorer over VH+VL, with an on-disk marginal cache.

    ``raw_logmarg`` returns ``float32[len(VH)+len(VL), 20]`` of per-position
    masked logits over the canonical AAs (row i<len(VH) is VH position i, row
    len(VH)+j is VL position j). Logits, not log-probs -- the downstream
    ``_pll`` applies ``log_softmax`` which is invariant to the per-row offset,
    so this matches the AbLang-2 pipeline exactly.
    """

    def __init__(self, device: str, batch_size: int, cache_dir: Path,
                 pairing: str, model_name: str, dtype: str):
        from mmmlm.models.esm2 import ESM2

        self.pairing = pairing
        self.model = ESM2(model_name=model_name, device=device,
                          dtype=_DTYPES[dtype], max_batch_size=batch_size)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, vh: str, vl: str) -> Path:
        h = hashlib.sha1(f"{self.pairing}|{vh}|{vl}".encode()).hexdigest()
        return self.cache_dir / f"{h}.npy"

    def raw_logmarg(self, vh: str, vl: str) -> np.ndarray:
        path = self._cache_path(vh, vl)
        if path.exists():
            return np.load(path)

        if self.pairing == "concat":
            mat = self.model.template_logits(vh + vl).cpu().numpy()
        else:  # perchain
            h = self.model.template_logits(vh).cpu().numpy()
            l = self.model.template_logits(vl).cpu().numpy()
            mat = np.concatenate([h, l], axis=0)
        mat = mat.astype(np.float32)

        tmp = path.parent / (path.name + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, mat)
        tmp.rename(path)
        return mat


def _select_variants(vh: str, vl: str, sample_n: int, sample_seed: int):
    """All one-edit variants, or a reproducible uniform random subset of them."""
    all_variants = _variants(vh, vl)
    if not sample_n or sample_n >= len(all_variants):
        return all_variants, False
    rng = np.random.default_rng(sample_seed)
    idx = np.sort(rng.choice(len(all_variants), size=sample_n, replace=False))
    return [all_variants[i] for i in idx], True


def _score_antibody(name: str, vh: str, vl: str, scorer: ESM2Scorer, tau: float,
                    seed: int, sample_n: int, sample_seed: int) -> pd.DataFrame:
    Lh = len(vh)
    combined = vh + vl
    seed_raw = scorer.raw_logmarg(vh, vl)
    seed_pll, A = _pll(seed_raw, combined, tau)
    model_tag = f"esm2_{scorer.pairing}"
    print(f"  [{name}] VH={Lh} VL={len(vl)} seed full-pair PLL = {seed_pll:.4f}",
          flush=True)

    variants, sampled = _select_variants(vh, vl, sample_n, sample_seed)
    rng = np.random.default_rng(seed)
    uniforms = rng.uniform(size=len(variants)).clip(1e-12, 1 - 1e-12)
    gumbel_noise = -np.log(-np.log(uniforms))

    rows = []
    for i, (pos, wt, mut) in enumerate(tqdm(variants, desc=f"  {name}", leave=False)):
        approx = float(seed_pll - A[pos, AA_TO_IDX[wt]] + A[pos, AA_TO_IDX[mut]])
        if pos < Lh:
            vvh, vvl, chain = vh[:pos] + mut + vh[pos + 1:], vl, "H"
        else:
            q = pos - Lh
            vvh, vvl, chain = vh, vl[:q] + mut + vl[q + 1:], "L"
        true_pll, _ = _pll(scorer.raw_logmarg(vvh, vvl), vvh + vvl, tau)
        rows.append({
            "antibody": name, "model": model_tag, "temperature": tau,
            "chain": chain, "pos": pos,
            "pos_in_chain": pos if chain == "H" else pos - Lh,
            "wt_aa": wt, "mut_aa": mut,
            "true_pll": true_pll, "approx_pll": approx,
            "gumbel_pll": approx + float(gumbel_noise[i]),
            "gumbel_noise": float(gumbel_noise[i]),
            "uniform_draw": float(uniforms[i]),
            "approx_minus_true": approx - true_pll, "seed_pll": seed_pll,
            "sampled": sampled, "n_all_variants": len(_variants(vh, vl)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--antibodies", nargs="+", default=list(ANTIBODIES),
                        choices=list(ANTIBODIES))
    parser.add_argument("--pairing", choices=["concat", "perchain"], default="concat")
    parser.add_argument("--sample-variants", type=int, default=0,
                        help="Random subset size per antibody (0 = all variants).")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=128,
                        help="ESM2 masked-position batch (per-variant forward).")
    parser.add_argument("--model-name", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--dtype", choices=list(_DTYPES), default="float32")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path,
                        default=EXPERIMENT_DIR / "results_esm2")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = _pick_device(args.device)
    cache_dir = args.cache_dir or (EXPERIMENT_DIR / "cache" / f"marginals_esm2_{args.pairing}")
    out = args.out or (args.out_dir / f"pll_one_edit_vhvl_esm2_{args.pairing}.parquet")
    per_ab_dir = args.out_dir / "per_antibody"
    per_ab_dir.mkdir(parents=True, exist_ok=True)
    tag = f"esm2_{args.pairing}"
    print(f"[hie pll esm2] device={device} dtype={args.dtype} pairing={args.pairing} "
          f"tau={args.temperature} seed={args.seed} batch={args.batch_size} "
          f"sample={args.sample_variants or 'all'} model={args.model_name}", flush=True)

    scorer = ESM2Scorer(device, args.batch_size, cache_dir, args.pairing,
                        args.model_name, args.dtype)

    frames = []
    for name in args.antibodies:
        vh, vl = ANTIBODIES[name]
        out_pq = per_ab_dir / f"{name}__{tag}.parquet"
        if out_pq.exists() and not args.force and args.sample_variants == 0:
            print(f"=== {name} (cached parquet) ===", flush=True)
            frames.append(pd.read_parquet(out_pq))
            continue
        print(f"=== {name} ===", flush=True)
        t0 = time.time()
        df = _score_antibody(name, vh, vl, scorer, args.temperature, args.seed,
                             args.sample_variants, args.sample_seed)
        # Sampled runs still checkpoint, but under a sample-suffixed name so they
        # never shadow a future full run's parquet.
        ckpt = out_pq if args.sample_variants == 0 else \
            per_ab_dir / f"{name}__{tag}__s{args.sample_variants}.parquet"
        df.to_parquet(ckpt, index=False)
        frames.append(df)
        print(f"  [{name}] {len(df)} variants in {(time.time()-t0)/60:.1f} min "
              f"-> {ckpt.name}", flush=True)

    full = pd.concat(frames, ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(out, index=False)
    print(f"\n[hie pll esm2] wrote {len(full)} rows -> {out}", flush=True)

    print(f"\nApprox / Gumbel vs true PLL correlation ({tag}, tau={args.temperature:g}):")
    _print_padded(_corr_table(full))


if __name__ == "__main__":
    main()
