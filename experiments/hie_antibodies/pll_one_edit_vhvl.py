"""Exact vs. approximate PLL for every one-edit variant of the seven Hie et al.
antibodies, over the full paired VH+VL sequence, using AbLang-2.

Repeats the Trastuzumab one-edit PLL evaluation for the seven antibodies matured
in Hie et al. 2024 (Nat. Biotech., "Efficient evolution of human antibodies from
general protein language models"). Differences from the Trastuzumab script:

  * Region is the FULL paired sequence (all VH + all VL positions), not a CDR
    window. AbLang-2 is a paired heavy/light model, so a single forward on the
    "<VH>|<VL>" input conditions on both chains; we mask each position (in VH or
    VL) and read its own masked marginal. The PLL is summed over every position
    of both chains.
  * Every one-edit variant is enumerated across both chains:
    (len(VH) + len(VL)) x 19 per antibody.
  * Only AbLang-2 is run (ESM2 is ~37 h here for this; the structure-conditioned
    models need per-antibody structures we don't have).

For each variant we record, at a shared temperature ``tau`` (default 1.0):

  * ``true_pll``   -- exact masked-marginal PLL: each position is masked IN THE
                      VARIANT's paired context and the variant residue's log-prob
                      is read from a fresh forward. (len(VH)+len(VL) masked
                      forwards per variant.)
  * ``approx_pll`` -- wild-type-marginal approximation: the seed's paired
                      marginal matrix is computed once and every variant scored
                      as ``sum_i A[i, x'_i]`` = ``seed_pll - A[k,wt] + A[k,mut]``.
  * ``gumbel_pll`` -- ``approx_pll`` + one Gumbel(0,1) draw per variant.

Caching: the raw (pre-temperature) masked-marginal matrix for each paired
context is cached to disk keyed by ``sha1(VH|VL)``. This makes a killed run
resume instantly, and lets the PLLs be recomputed at a different temperature (or
the Gumbel re-drawn) with no model calls. Per-antibody result parquets are
written as each antibody finishes, so completed work always persists.
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
sys.path.insert(0, str(EXPERIMENT_DIR))                         # sibling: antibodies
sys.path.insert(0, str(EXPERIMENT_DIR.parent.parent / "src"))

from mmmlm.constants import AA_TO_IDX, AMINO_ACIDS
from mmmlm.sampling.beam import (
    _log_softmax_at_temperature,
    _self_pll_from_log_marginals,
)
from antibodies import ANTIBODIES  # single source of truth; re-exported here

# AbLang-2 concatenates "<VH>|<VL>" with 5 special tokens (<, >, |, <, >).
# VH residues occupy rows [1, 1+len(VH)); VL residues occupy rows
# [len(VH)+4, len(VH)+4+len(VL)). Mirrors mmmlm/models/ablang2.py.
_VH_ROW0 = 1
_VL_ROW_GAP = 4


def _pick_device(requested: str | None) -> str:
    if requested is not None:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class PairedScorer:
    """AbLang-2 paired masked-marginal scorer with an on-disk marginal cache."""

    def __init__(self, device: str, batch_size: int, cache_dir: Path):
        import ablang2

        self.model = ablang2.pretrained(device=device)
        aa_to_token = self.model.tokenizer.aa_to_token
        self.aa_cols = np.array([int(aa_to_token[a]) for a in AMINO_ACIDS])
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, vh: str, vl: str) -> Path:
        h = hashlib.sha1(f"{vh}|{vl}".encode()).hexdigest()
        return self.cache_dir / f"{h}.npy"

    def raw_logmarg(self, vh: str, vl: str) -> np.ndarray:
        """Return ``float32[len(VH)+len(VL), 20]`` of natural-log masked-marginal
        probabilities over the canonical AAs, conditioned on the full pair.

        Row ``i < len(VH)`` is VH position ``i``; row ``len(VH)+j`` is VL
        position ``j``. Cached to disk by ``sha1(VH|VL)``.
        """
        path = self._cache_path(vh, vl)
        if path.exists():
            return np.load(path)

        Lh, Ll = len(vh), len(vl)
        vl_row0 = Lh + _VL_ROW_GAP
        pairs: list[list[str]] = []
        out_rows: list[int] = []  # model-output row for each masked position
        for p in range(Lh):
            pairs.append([vh[:p] + "*" + vh[p + 1:], vl])
            out_rows.append(_VH_ROW0 + p)
        for q in range(Ll):
            pairs.append([vh, vl[:q] + "*" + vl[q + 1:]])
            out_rows.append(vl_row0 + q)

        out = np.full((Lh + Ll, len(AMINO_ACIDS)), np.nan, dtype=np.float32)
        for s in range(0, len(pairs), self.batch_size):
            chunk = pairs[s: s + self.batch_size]
            probs = np.asarray(self.model(chunk, mode="probability"))  # (B, T, alpha)
            for j in range(len(chunk)):
                gi = s + j
                pr = probs[j, out_rows[gi]][self.aa_cols]
                out[gi] = np.log(np.clip(pr, 1e-30, None))

        # Atomic-ish write: save to a temp path then rename. Use a file handle so
        # np.save does not re-append ".npy" to the temp name.
        tmp = path.parent / (path.name + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, out)
        tmp.rename(path)
        return out


def _pll(raw: np.ndarray, combined_seq: str, tau: float) -> tuple[float, np.ndarray]:
    """(self-PLL over the full sequence, temperature-scaled log-marginals A)."""
    A = _log_softmax_at_temperature(torch.from_numpy(raw), tau)
    mask = np.ones(len(combined_seq), dtype=bool)
    pll = _self_pll_from_log_marginals(combined_seq, A, mask)
    return pll, A.cpu().numpy()


def _variants(vh: str, vl: str) -> list[tuple[int, str, str]]:
    """All one-edit variants over the concatenated VH+VL, as (pos, wt, mut)."""
    combined = vh + vl
    out = []
    for pos, wt in enumerate(combined):
        for mut in AMINO_ACIDS:
            if mut != wt:
                out.append((pos, wt, mut))
    return out


def _score_antibody(name: str, vh: str, vl: str, scorer: PairedScorer,
                    tau: float, seed: int, max_variants: int = 0) -> pd.DataFrame:
    Lh = len(vh)
    combined = vh + vl
    seed_raw = scorer.raw_logmarg(vh, vl)
    seed_pll, A = _pll(seed_raw, combined, tau)
    print(f"  [{name}] VH={Lh} VL={len(vl)} seed full-pair PLL = {seed_pll:.4f}",
          flush=True)

    variants = _variants(vh, vl)
    rng = np.random.default_rng(seed)
    uniforms = rng.uniform(size=len(variants)).clip(1e-12, 1 - 1e-12)
    gumbel_noise = -np.log(-np.log(uniforms))
    if max_variants:
        variants = variants[:max_variants]

    rows = []
    for i, (pos, wt, mut) in enumerate(tqdm(variants, desc=f"  {name}", leave=False)):
        approx = float(seed_pll - A[pos, AA_TO_IDX[wt]] + A[pos, AA_TO_IDX[mut]])
        if pos < Lh:
            vvh, vvl = vh[:pos] + mut + vh[pos + 1:], vl
            chain = "H"
        else:
            q = pos - Lh
            vvh, vvl = vh, vl[:q] + mut + vl[q + 1:]
            chain = "L"
        true_pll, _ = _pll(scorer.raw_logmarg(vvh, vvl), vvh + vvl, tau)
        rows.append({
            "antibody": name,
            "model": "ablang2",
            "temperature": tau,
            "chain": chain,
            "pos": pos,
            "pos_in_chain": pos if chain == "H" else pos - Lh,
            "wt_aa": wt,
            "mut_aa": mut,
            "true_pll": true_pll,
            "approx_pll": approx,
            "gumbel_pll": approx + float(gumbel_noise[i]),
            "gumbel_noise": float(gumbel_noise[i]),
            "uniform_draw": float(uniforms[i]),
            "approx_minus_true": approx - true_pll,
            "seed_pll": seed_pll,
        })
    return pd.DataFrame(rows)


def _corr_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in df["antibody"].unique():
        s = df[df["antibody"] == name]
        rows.append({
            "Antibody": name,
            "Approx_Pearson": round(float(pearsonr(s.true_pll, s.approx_pll).statistic), 3),
            "Approx_Spearman": round(float(spearmanr(s.true_pll, s.approx_pll).statistic), 3),
            "Gumbel_Pearson": round(float(pearsonr(s.true_pll, s.gumbel_pll).statistic), 3),
            "Gumbel_Spearman": round(float(spearmanr(s.true_pll, s.gumbel_pll).statistic), 3),
        })
    return pd.DataFrame(rows)


def _print_padded(df: pd.DataFrame) -> None:
    cols = list(df.columns)
    cells = [cols] + df.astype(str).values.tolist()
    w = [max(len(r[i]) for r in cells) for i in range(len(cols))]
    for r, row in enumerate(cells):
        print("  ".join(
            c.ljust(w[i]) if i == 0 else c.rjust(w[i]) for i, c in enumerate(row)
        ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--antibodies", nargs="+", default=list(ANTIBODIES),
                        choices=list(ANTIBODIES))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-variants", type=int, default=0,
                        help="Cap variants per antibody (0 = all); for smoke tests.")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if an antibody's parquet exists.")
    parser.add_argument("--cache-dir", type=Path,
                        default=EXPERIMENT_DIR / "cache" / "marginals")
    parser.add_argument("--out-dir", type=Path,
                        default=EXPERIMENT_DIR / "results")
    parser.add_argument("--out", type=Path,
                        default=EXPERIMENT_DIR / "results" / "pll_one_edit_ablang2.parquet")
    args = parser.parse_args()

    device = _pick_device(args.device)
    per_ab_dir = args.out_dir / "per_antibody"
    per_ab_dir.mkdir(parents=True, exist_ok=True)
    print(f"[hie pll] device={device} tau={args.temperature} seed={args.seed} "
          f"batch={args.batch_size} models=ablang2", flush=True)

    scorer = PairedScorer(device, args.batch_size, args.cache_dir)

    frames = []
    for name in args.antibodies:
        vh, vl = ANTIBODIES[name]
        out_pq = per_ab_dir / f"{name}.parquet"
        if out_pq.exists() and not args.force and args.max_variants == 0:
            print(f"=== {name} (cached parquet) ===", flush=True)
            frames.append(pd.read_parquet(out_pq))
            continue
        print(f"=== {name} ===", flush=True)
        t0 = time.time()
        df = _score_antibody(name, vh, vl, scorer, args.temperature, args.seed,
                             max_variants=args.max_variants)
        if args.max_variants == 0:
            df.to_parquet(out_pq, index=False)
        frames.append(df)
        print(f"  [{name}] {len(df)} variants in {(time.time()-t0)/60:.1f} min",
              flush=True)

    full = pd.concat(frames, ignore_index=True)
    if args.max_variants == 0:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        full.to_parquet(args.out, index=False)
        print(f"\n[hie pll] wrote {len(full)} rows -> {args.out}", flush=True)

    print(f"\nApprox / Gumbel vs true PLL correlation (tau={args.temperature:g}):")
    _print_padded(_corr_table(full))


if __name__ == "__main__":
    main()
