"""One-edit PLL eval (true vs approx vs Gumbel) for the Hie antibodies with the
structure-conditioned ProteinMPNN and AbMPNN, over the full paired VH+VL.

ProteinMPNN/AbMPNN are inverse-folding models: they factorize p(sequence |
backbone) and their ``StructureMPNN.template_logits`` exposes the same
masked-marginal-style conditional the MLM samplers use -- mask a position,
condition on the backbone and every other (fixed) residue. They design one chain
at a time, so as with Sapiens the full PLL is scored per chain: the VH is
designed with the VL + antigen as fixed context and vice-versa, and a one-edit
variant only re-scores its mutated chain (the other chain's PLL is constant).

Only the four Hie antibodies with an exact-match deposited structure are scored
(see mpnn_structures.py); the UCAs and c143 are omitted. Requires the trimmed
structures + weights from prep_mpnn_structures.py, and a ProteinMPNN checkout
($MMMLM_PROTEINMPNN_DIR). ProteinMPNN's gather op is unsupported on MPS, so this
runs on CUDA (fast) or CPU (slow); do the full run on a GPU box.

Same protocol/columns/caching as pll_one_edit_sapiens.py (model in
{proteinmpnn, abmpnn}); each variant's true PLL is exact, at tau=1 by default.

    export MMMLM_PROTEINMPNN_DIR=../ProteinMPNN
    python experiments/hie_antibodies/prep_mpnn_structures.py
    python experiments/hie_antibodies/pll_one_edit_mpnn.py --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import os
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
from mmmlm.models.mpnn import StructureMPNN
from mmmlm.sampling.beam import (
    _log_softmax_at_temperature,
    _self_pll_from_log_marginals,
)
from antibodies import ANTIBODIES
from mpnn_structures import STRUCT_DIR, STRUCTURES

MODELS = ("proteinmpnn", "abmpnn")


def _pick_device(requested: str | None) -> str:
    if requested is not None:
        return requested
    # No MPS: ProteinMPNN's decoder gather op is unimplemented on the Metal backend.
    return "cuda" if torch.cuda.is_available() else "cpu"


class MPNNScorer:
    """ProteinMPNN/AbMPNN scorer over the trimmed Hie structures, with a disk cache.

    One StructureMPNN instance per (model, antibody, chain) -- the backbone is
    encoded once per instance and reused for every variant of that chain.
    """

    def __init__(self, device: str, proteinmpnn_root: Path, abmpnn_weights: Path,
                 cache_dir: Path, max_batch_size: int):
        self.device = device
        self.root = proteinmpnn_root
        self.abmpnn_weights = abmpnn_weights
        self.max_batch_size = max_batch_size
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._inst: dict[tuple[str, str, str], StructureMPNN] = {}
        self._design: dict[tuple[str, str], str] = {}
        if str(proteinmpnn_root) not in sys.path:
            sys.path.insert(0, str(proteinmpnn_root))
        import protein_mpnn_utils  # noqa: F401
        self._parse_pdb = protein_mpnn_utils.parse_PDB

    def _weights(self, model: str) -> Path:
        if model == "proteinmpnn":
            return self.root / "vanilla_model_weights" / "v_48_020.pt"
        return self.abmpnn_weights

    def design_seq(self, ab: str, chain: str) -> str:
        """Resolved VH/VL sequence = the Hie residues actually modeled in the
        structure (a prefix of the modeled chain up to CH1/CL). Inverse-folding
        can only score residues with backbone coordinates, so disordered residues
        are dropped."""
        key = (ab, chain)
        if key not in self._design:
            cfg = STRUCTURES[ab]
            vh, vl = ANTIBODIES[ab]
            dchain = cfg["vh_chain"] if chain == "H" else cfg["vl_chain"]
            hie = vh if chain == "H" else vl
            pdb = self._parse_pdb(str(STRUCT_DIR / f"{ab}_trim.pdb"),
                                  input_chain_list=[dchain])[0]
            modeled = pdb[f"seq_chain_{dchain}"]
            # Longest prefix of the modeled chain that is a subsequence of the Hie
            # V-domain (stops where CH1/CL begins): the resolved V-domain residues.
            i = k = 0
            for c in modeled:
                j = i
                while j < len(hie) and hie[j] != c:
                    j += 1
                if j < len(hie):
                    i = j + 1
                    k += 1
                else:
                    break
            resolved = modeled[:k]
            dropped = len(hie) - k
            if dropped:
                print(f"  [{ab} {chain}] {dropped} disordered residue(s) dropped "
                      f"(scoring {k}/{len(hie)})", flush=True)
            self._design[key] = resolved
        return self._design[key]

    def instance(self, model: str, ab: str, chain: str) -> StructureMPNN:
        key = (model, ab, chain)
        if key not in self._inst:
            cfg = STRUCTURES[ab]
            design_chain = cfg["vh_chain"] if chain == "H" else cfg["vl_chain"]
            design_seq = self.design_seq(ab, chain)
            self._inst[key] = StructureMPNN(
                structure_path=str(STRUCT_DIR / f"{ab}_trim.pdb"),
                weights_path=str(self._weights(model)),
                design_chain=design_chain,
                design_sequence=design_seq,
                context_chains=None,  # all other chains (partner chain + antigen)
                name=model,
                device=self.device,
                max_batch_size=self.max_batch_size,
                proteinmpnn_root=str(self.root),
            )
        return self._inst[key]

    def raw_logits(self, model: str, ab: str, chain: str, seq: str) -> np.ndarray:
        """[len(seq), 20] log-prob marginals for ``seq`` on the (model, ab, chain)
        backbone. Cached by sha1(model|ab|chain|seq)."""
        h = hashlib.sha1(f"{model}|{ab}|{chain}|{seq}".encode()).hexdigest()
        path = self.cache_dir / f"{h}.npy"
        if path.exists():
            return np.load(path)
        full = np.ones(len(seq), dtype=bool)
        out = self.instance(model, ab, chain).template_logits(seq, positions=full)
        arr = out.cpu().numpy().astype(np.float32)
        tmp = path.parent / (path.name + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, arr)
        tmp.rename(path)
        return arr


def _chain_pll(scorer, model, ab, chain, seq, tau):
    A = _log_softmax_at_temperature(torch.from_numpy(scorer.raw_logits(model, ab, chain, seq)), tau)
    mask = np.ones(len(seq), dtype=bool)
    return _self_pll_from_log_marginals(seq, A, mask), A.cpu().numpy()


def _score(model, ab, scorer, tau, seed, max_variants=0):
    # Resolved (modeled) VH/VL -- inverse folding only scores residues with coords.
    vh = scorer.design_seq(ab, "H")
    vl = scorer.design_seq(ab, "L")
    Lh = len(vh)
    pll_h, A_h = _chain_pll(scorer, model, ab, "H", vh, tau)
    pll_l, A_l = _chain_pll(scorer, model, ab, "L", vl, tau)
    seed_pll = pll_h + pll_l
    print(f"  [{model}/{ab}] VH={Lh} VL={len(vl)} seed PLL={seed_pll:.4f} "
          f"(H {pll_h:.2f} + L {pll_l:.2f})", flush=True)

    variants = [(pos, wt, mut) for pos, wt in enumerate(vh + vl)
                for mut in AMINO_ACIDS if mut != wt]
    rng = np.random.default_rng(seed)
    uniforms = rng.uniform(size=len(variants)).clip(1e-12, 1 - 1e-12)
    gumbel = -np.log(-np.log(uniforms))
    if max_variants:
        variants = variants[:max_variants]

    rows = []
    for i, (pos, wt, mut) in enumerate(tqdm(variants, desc=f"  {model}/{ab}", leave=False)):
        if pos < Lh:
            chain, seq0, pic = "H", vh, pos
            vseq = vh[:pos] + mut + vh[pos + 1:]
            approx = float(seed_pll - A_h[pos, AA_TO_IDX[wt]] + A_h[pos, AA_TO_IDX[mut]])
            true_pll = _chain_pll(scorer, model, ab, "H", vseq, tau)[0] + pll_l
        else:
            q = pos - Lh
            chain, pic = "L", q
            vseq = vl[:q] + mut + vl[q + 1:]
            approx = float(seed_pll - A_l[q, AA_TO_IDX[wt]] + A_l[q, AA_TO_IDX[mut]])
            true_pll = pll_h + _chain_pll(scorer, model, ab, "L", vseq, tau)[0]
        rows.append({
            "antibody": ab, "model": model, "temperature": tau, "chain": chain,
            "pos": pos, "pos_in_chain": pic, "wt_aa": wt, "mut_aa": mut,
            "true_pll": true_pll, "approx_pll": approx,
            "gumbel_pll": approx + float(gumbel[i]),
            "gumbel_noise": float(gumbel[i]), "uniform_draw": float(uniforms[i]),
            "approx_minus_true": approx - true_pll, "seed_pll": seed_pll,
        })
    return pd.DataFrame(rows)


def _corr_table(df):
    rows = []
    for (ab, model), s in df.groupby(["antibody", "model"]):
        rows.append({
            "antibody": ab, "model": model,
            "Approx_Pearson": round(float(pearsonr(s.true_pll, s.approx_pll).statistic), 3),
            "Approx_Spearman": round(float(spearmanr(s.true_pll, s.approx_pll).statistic), 3),
            "Gumbel_Pearson": round(float(pearsonr(s.true_pll, s.gumbel_pll).statistic), 3),
            "Gumbel_Spearman": round(float(spearmanr(s.true_pll, s.gumbel_pll).statistic), 3),
        })
    return pd.DataFrame(rows).sort_values(["model", "antibody"]).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--antibodies", nargs="+", default=list(STRUCTURES),
                   choices=list(STRUCTURES))
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--max-batch-size", type=int, default=32)
    p.add_argument("--max-variants", type=int, default=0,
                   help="Cap variants per antibody (0 = all); for smoke tests.")
    p.add_argument("--force", action="store_true")
    p.add_argument("--proteinmpnn-dir", type=Path,
                   default=Path(os.environ.get("MMMLM_PROTEINMPNN_DIR", "")))
    p.add_argument("--abmpnn-weights", type=Path, default=STRUCT_DIR / "abmpnn.pt")
    p.add_argument("--cache-dir", type=Path, default=EXPERIMENT_DIR / "cache" / "mpnn_marginals")
    p.add_argument("--out-dir", type=Path, default=EXPERIMENT_DIR / "results")
    p.add_argument("--out", type=Path,
                   default=EXPERIMENT_DIR / "results" / "pll_one_edit_mpnn.parquet")
    args = p.parse_args()

    device = _pick_device(args.device)
    per_dir = args.out_dir / "per_antibody_mpnn"
    per_dir.mkdir(parents=True, exist_ok=True)
    print(f"[hie mpnn] device={device} tau={args.temperature} seed={args.seed} "
          f"models={args.models}", flush=True)
    scorer = MPNNScorer(device, args.proteinmpnn_dir, args.abmpnn_weights,
                        args.cache_dir, args.max_batch_size)

    frames = []
    for model in args.models:
        for ab in args.antibodies:
            out_pq = per_dir / f"{model}_{ab}.parquet"
            if out_pq.exists() and not args.force and args.max_variants == 0:
                print(f"=== {model}/{ab} (cached) ===", flush=True)
                frames.append(pd.read_parquet(out_pq))
                continue
            print(f"=== {model}/{ab} ===", flush=True)
            t0 = time.time()
            df = _score(model, ab, scorer, args.temperature, args.seed,
                        max_variants=args.max_variants)
            if args.max_variants == 0:
                df.to_parquet(out_pq, index=False)
            frames.append(df)
            print(f"  [{model}/{ab}] {len(df)} variants in {(time.time()-t0)/60:.1f} min",
                  flush=True)

    full = pd.concat(frames, ignore_index=True)
    if args.max_variants == 0:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        full.to_parquet(args.out, index=False)
        print(f"\n[hie mpnn] wrote {len(full)} rows -> {args.out}", flush=True)
    print(f"\nApprox / Gumbel vs true PLL (tau={args.temperature:g}):")
    print(_corr_table(full).to_string(index=False))


if __name__ == "__main__":
    main()
