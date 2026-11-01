"""One-edit PLL evaluation for the seven Hie et al. antibodies using AbLang-2 run
*as if it were unpaired* -- the same model as ``pll_one_edit_vhvl.py`` but each
chain scored in isolation, to isolate the effect of paired conditioning.

AbLang-2 is a paired model, but its ``ablang2`` API accepts a single chain by
passing an empty partner: ``[VH, '']`` scores the heavy chain alone (residue
rows at offset 1) and ``['', VL]`` the light chain alone (residue rows at offset
2). Both were verified to give sensible marginals (seed argmax==residue ~0.9).
So VH positions are conditioned only on VH and VL only on VL -- no cross-chain
coupling -- exactly the unpaired setup of the Sapiens run, but holding the model
fixed at AbLang-2.

Consequences, identical to the Sapiens script:

    PLL(VH, VL) = PLL_H(VH) + PLL_L(VL),

so a one-edit variant only re-scores the mutated chain (the other chain's PLL is
a constant), which also makes this far cheaper than the paired AbLang-2 run.

Same protocol otherwise: every one-edit variant over both chains, exact
masked-marginal true PLL (each position masked with AbLang-2's '*' token), the
wild-type-marginal approximation, and a Gumbel-perturbed score, at tau=1. Raw
per-(chain, sequence) log-prob marginals are cached to disk. Output schema
matches the AbLang-2 script with ``model='ablang2_unpaired'``.
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
sys.path.insert(0, str(EXPERIMENT_DIR))
sys.path.insert(0, str(EXPERIMENT_DIR.parent.parent / "src"))

from mmmlm.constants import AA_TO_IDX, AMINO_ACIDS
from mmmlm.sampling.beam import (
    _log_softmax_at_temperature,
    _self_pll_from_log_marginals,
)
from pll_one_edit_vhvl import ANTIBODIES  # single source of truth for sequences

# Residue-row offsets for AbLang-2 fed a single chain with an empty partner,
# determined empirically (seed argmax vs residue): [VH, ''] -> VH at row 1;
# ['', VL] -> VL at row 2.
_OFFSET = {"H": 1, "L": 2}


def _pick_device(requested: str | None) -> str:
    if requested is not None:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class AbLang2UnpairedScorer:
    """AbLang-2 masked-marginal scorer for a single chain (empty partner)."""

    def __init__(self, device: str, batch_size: int, cache_dir: Path):
        import ablang2

        self.model = ablang2.pretrained(device=device)
        aa_to_token = self.model.tokenizer.aa_to_token
        self.aa_cols = np.array([int(aa_to_token[a]) for a in AMINO_ACIDS])
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, chain: str, seq: str) -> Path:
        h = hashlib.sha1(f"ablang2_unpaired|{chain}|{seq}".encode()).hexdigest()
        return self.cache_dir / f"{h}.npy"

    def _pair(self, masked_seq: str, chain: str) -> list[str]:
        return [masked_seq, ""] if chain == "H" else ["", masked_seq]

    def raw_logmarg(self, seq: str, chain: str) -> np.ndarray:
        """``float32[len(seq), 20]`` natural-log masked-marginal probs over the
        canonical AAs for ``seq`` scored alone under AbLang-2's ``chain`` slot."""
        path = self._cache_path(chain, seq)
        if path.exists():
            return np.load(path)

        L = len(seq)
        offset = _OFFSET[chain]
        pairs = [self._pair(seq[:i] + "*" + seq[i + 1:], chain) for i in range(L)]
        out = np.full((L, len(AMINO_ACIDS)), np.nan, dtype=np.float32)
        for s in range(0, L, self.batch_size):
            chunk = pairs[s: s + self.batch_size]
            probs = np.asarray(self.model(chunk, mode="probability"))  # (B, T, alpha)
            for j in range(len(chunk)):
                i = s + j
                pr = probs[j, offset + i][self.aa_cols]
                out[i] = np.log(np.clip(pr, 1e-30, None))

        tmp = path.parent / (path.name + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, out)
        tmp.rename(path)
        return out


def _chain_pll(scorer: AbLang2UnpairedScorer, seq: str, chain: str,
               tau: float) -> tuple[float, np.ndarray]:
    A = _log_softmax_at_temperature(torch.from_numpy(scorer.raw_logmarg(seq, chain)), tau)
    mask = np.ones(len(seq), dtype=bool)
    return _self_pll_from_log_marginals(seq, A, mask), A.cpu().numpy()


def _score_antibody(name: str, vh: str, vl: str, scorer: AbLang2UnpairedScorer,
                    tau: float, seed: int, max_variants: int = 0) -> pd.DataFrame:
    Lh, Ll = len(vh), len(vl)
    pll_h_seed, A_h = _chain_pll(scorer, vh, "H", tau)
    pll_l_seed, A_l = _chain_pll(scorer, vl, "L", tau)
    seed_pll = pll_h_seed + pll_l_seed
    print(f"  [{name}] VH={Lh} VL={Ll} seed PLL = {seed_pll:.4f} "
          f"(H {pll_h_seed:.2f} + L {pll_l_seed:.2f})", flush=True)

    variants = []
    for pos, wt in enumerate(vh + vl):
        for mut in AMINO_ACIDS:
            if mut != wt:
                variants.append((pos, wt, mut))
    rng = np.random.default_rng(seed)
    uniforms = rng.uniform(size=len(variants)).clip(1e-12, 1 - 1e-12)
    gumbel_noise = -np.log(-np.log(uniforms))
    if max_variants:
        variants = variants[:max_variants]

    rows = []
    for i, (pos, wt, mut) in enumerate(tqdm(variants, desc=f"  {name}", leave=False)):
        if pos < Lh:  # heavy-chain mutant: only VH re-scored, VL PLL constant
            chain = "H"
            vseq = vh[:pos] + mut + vh[pos + 1:]
            approx = float(seed_pll - A_h[pos, AA_TO_IDX[wt]] + A_h[pos, AA_TO_IDX[mut]])
            true_pll = _chain_pll(scorer, vseq, "H", tau)[0] + pll_l_seed
            pos_in_chain = pos
        else:  # light-chain mutant
            chain = "L"
            q = pos - Lh
            vseq = vl[:q] + mut + vl[q + 1:]
            approx = float(seed_pll - A_l[q, AA_TO_IDX[wt]] + A_l[q, AA_TO_IDX[mut]])
            true_pll = pll_h_seed + _chain_pll(scorer, vseq, "L", tau)[0]
            pos_in_chain = q
        rows.append({
            "antibody": name,
            "model": "ablang2_unpaired",
            "temperature": tau,
            "chain": chain,
            "pos": pos,
            "pos_in_chain": pos_in_chain,
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
    for row in cells:
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
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cache-dir", type=Path,
                        default=EXPERIMENT_DIR / "cache" / "ablang2_unpaired_marginals")
    parser.add_argument("--out-dir", type=Path, default=EXPERIMENT_DIR / "results")
    parser.add_argument("--out", type=Path,
                        default=EXPERIMENT_DIR / "results" / "pll_one_edit_ablang2_unpaired.parquet")
    args = parser.parse_args()

    device = _pick_device(args.device)
    per_ab_dir = args.out_dir / "per_antibody_ablang2_unpaired"
    per_ab_dir.mkdir(parents=True, exist_ok=True)
    print(f"[hie ablang2-unpaired] device={device} tau={args.temperature} "
          f"seed={args.seed} batch={args.batch_size}", flush=True)

    scorer = AbLang2UnpairedScorer(device, args.batch_size, args.cache_dir)

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
        print(f"\n[hie ablang2-unpaired] wrote {len(full)} rows -> {args.out}", flush=True)

    print(f"\nApprox / Gumbel vs true PLL correlation (ablang2_unpaired, tau={args.temperature:g}):")
    _print_padded(_corr_table(full))


if __name__ == "__main__":
    main()
