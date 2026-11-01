"""Exact vs. approximate PLL for every one-edit CDR-H3 variant of Trastuzumab.

For the Trastuzumab VH seed, restricted to the 10-residue CDR-H3, this script
enumerates all single-substitution variants (10 positions x 19 alternative amino
acids = 190 variants) and, for each, records three scores:

  * ``true_pll``    -- the exact masked-marginal pseudo-log-likelihood of the
                       variant, summed over the H3 positions. Each H3 position is
                       masked *in the variant* and the log-prob of the variant's
                       residue is read off from a fresh forward pass, so the
                       context at every position reflects the mutation. This is
                       ``L_mask`` forward passes per variant.

  * ``approx_pll``  -- the wild-type-marginal approximation this method actually
                       uses in ``mmmlm.sampling.beam``. The seed's marginal
                       matrix ``A`` (log softmax_tau of the per-position masked
                       logits) is computed *once* from the seed, and every
                       variant is scored as ``sum_i A[i, x'_i]``. For a one-edit
                       variant differing at position ``k`` this equals
                       ``seed_pll - A[k, wt_k] + A[k, mut_k]``, i.e. all neighbors
                       are free once the seed's ``A`` is in hand.

  * ``gumbel_pll``  -- the stochastic-ranking score: ``approx_pll`` plus one
                       Gumbel(0, 1) draw per variant, produced from a single
                       uniform draw ``u`` via ``-log(-log(u))`` exactly as
                       ``beam_search`` does before ranking.

Both PLLs use the same temperature ``tau`` (default 1.5, the method's default)
so the only difference between ``true_pll`` and ``approx_pll`` is the wild-type-
marginal approximation itself -- ``approx_pll - true_pll`` is the approximation
error.

The computation is repeated for each requested model (ESM2-650M, AbLang-2, and
the structure-conditioned ProteinMPNN), all results are concatenated into one
parquet file (one ``model`` column), and a small table of the approximate-vs-
true PLL Pearson and Spearman correlations per model is printed at the end.

Example
-------
    export MMMLM_PROTEINMPNN_DIR=../ProteinMPNN            # for proteinmpnn
    .venv/bin/python experiments/trastuzumab_cdrh3/pll_one_edit_h3.py \
        --models esm2 ablang2 proteinmpnn \
        --out experiments/trastuzumab_cdrh3/results/pll_one_edit_h3.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR.parent.parent / "src"))

from mmmlm.constants import AMINO_ACIDS
from mmmlm.regions import mask_from_slice
from mmmlm.sampling.beam import (
    _log_softmax_at_temperature,
    _self_pll_from_log_marginals,
)

# Trastuzumab VH / VL and its CDR-H3, matching experiments/trastuzumab_cdrh3/run.py.
TRASTUZUMAB_VH = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYAD"
    "SVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
)
TRASTUZUMAB_VL = (
    "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRF"
    "SGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"
)
TRASTUZUMAB_H3 = "WGGDGFYAMD"

# Pretty labels for the summary table, keyed by the --models token.
MODEL_PRETTY = {
    "esm2": "ESM2-650M",
    "ablang2": "AbLang2",
    "proteinmpnn": "ProteinMPNN",
    "abmpnn": "AbMPNN",
}


def _pick_device(requested: str | None) -> str:
    if requested is not None:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_model(model_key: str, args, device: str):
    if model_key == "esm2":
        from mmmlm.models.esm2 import ESM2

        return ESM2(device=device)
    if model_key == "ablang2":
        from mmmlm.models.ablang2 import AbLang2

        return AbLang2(vl_sequence=TRASTUZUMAB_VL, device=device)
    if model_key in ("proteinmpnn", "abmpnn"):
        from mmmlm.models.mpnn import StructureMPNN

        # ProteinMPNN/AbMPNN share the StructureMPNN wrapper (and both need the
        # ProteinMPNN checkout for the code); only the weights differ. Their
        # decoder uses a gather op (GatherND) the Metal/MPS backend does not
        # implement, so they run on CPU by default. Both are small, so CPU is fast.
        mpnn_device = args.mpnn_device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model_key == "proteinmpnn":
            weights = args.proteinmpnn_weights
            if weights is None:
                if args.proteinmpnn_dir is None:
                    raise RuntimeError(
                        "proteinmpnn needs --proteinmpnn-weights or --proteinmpnn-dir "
                        "(or $MMMLM_PROTEINMPNN_DIR)."
                    )
                weights = args.proteinmpnn_dir / "vanilla_model_weights" / "v_48_020.pt"
        else:  # abmpnn
            weights = args.abmpnn_weights
            if not Path(weights).is_file():
                raise RuntimeError(
                    f"abmpnn weights not found at {weights}; download abmpnn.pt from "
                    "zenodo.org/records/8164693 or pass --abmpnn-weights."
                )
        return StructureMPNN(
            structure_path=args.structure,
            weights_path=weights,
            design_chain=args.structure_vh_chain,
            design_sequence=TRASTUZUMAB_VH,
            context_chains=args.structure_context_chains,
            name=model_key,
            device=mpnn_device,
            max_batch_size=args.mpnn_batch_size,
            proteinmpnn_root=args.proteinmpnn_dir,
        )
    raise ValueError(f"unknown model {model_key!r}")


def _score_model(model, model_key, vh, h3_mask, h3_start, h3_end, tau, seed):
    """Return the per-variant rows (list[dict]) for one model."""
    h3_positions = list(range(h3_start, h3_end))

    # Seed marginals A (computed once) -> approximate PLL for every variant.
    seed_logits = model.template_logits(vh, positions=h3_mask)
    seed_log_marg = _log_softmax_at_temperature(seed_logits, tau)
    seed_log_marg_np = seed_log_marg.cpu().numpy()
    wt_pll = _self_pll_from_log_marginals(vh, seed_log_marg, h3_mask)
    print(f"  seed (WT) self-PLL over H3 = {wt_pll:.4f}", flush=True)

    variants = []
    for pos in h3_positions:
        wt_aa = vh[pos]
        for mut_aa in AMINO_ACIDS:
            if mut_aa == wt_aa:
                continue
            variants.append((pos, wt_aa, mut_aa))

    # One uniform draw per variant for the Gumbel perturbation (matching
    # beam_search: gumbel = -log(-log(u))). A fresh RNG per model keeps the
    # results independent of model order and reproducible.
    rng = np.random.default_rng(seed)
    uniforms = rng.uniform(size=len(variants)).clip(1e-12, 1 - 1e-12)
    gumbel_noise = -np.log(-np.log(uniforms))

    model_name = getattr(model, "name", model_key)
    rows = []
    for i, (pos, wt_aa, mut_aa) in enumerate(
        tqdm(variants, desc=f"  {model_key}", leave=False)
    ):
        var_vh = vh[:pos] + mut_aa + vh[pos + 1:]

        wt_idx = AMINO_ACIDS.index(wt_aa)
        mut_idx = AMINO_ACIDS.index(mut_aa)
        approx_pll = float(
            wt_pll - seed_log_marg_np[pos, wt_idx] + seed_log_marg_np[pos, mut_idx]
        )

        var_logits = model.template_logits(var_vh, positions=h3_mask)
        var_log_marg = _log_softmax_at_temperature(var_logits, tau)
        true_pll = _self_pll_from_log_marginals(var_vh, var_log_marg, h3_mask)

        rows.append({
            "model": model_name,
            "model_key": model_key,
            "temperature": tau,
            "vh": var_vh,
            "h3": var_vh[h3_start:h3_end],
            "pos_vh": pos,
            "pos_h3": pos - h3_start,
            "wt_aa": wt_aa,
            "mut_aa": mut_aa,
            "edits": 1,
            "true_pll": true_pll,
            "approx_pll": approx_pll,
            "gumbel_pll": approx_pll + float(gumbel_noise[i]),
            "gumbel_noise": float(gumbel_noise[i]),
            "uniform_draw": float(uniforms[i]),
            "approx_minus_true": approx_pll - true_pll,
            "wt_pll": wt_pll,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+",
                        default=["esm2", "ablang2", "proteinmpnn", "abmpnn"],
                        choices=["esm2", "ablang2", "proteinmpnn", "abmpnn"])
    parser.add_argument("--temperature", type=float, default=1.5,
                        help="tau for log_softmax(logits / tau); applied to "
                             "both the true and approximate PLL.")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for the per-variant Gumbel uniform draws.")
    parser.add_argument("--device", default=None,
                        help="torch device (default: cuda>mps>cpu autodetect).")
    parser.add_argument("--out", type=Path,
                        default=EXPERIMENT_DIR / "results" / "pll_one_edit_h3.parquet")
    # Structure-conditioned model (ProteinMPNN) inputs, mirroring run.py.
    parser.add_argument("--structure", type=Path,
                        default=EXPERIMENT_DIR / "structures" / "1N8Z.pdb",
                        help="Backbone PDB (1N8Z: chain B=VH, A=VL, C=HER2).")
    parser.add_argument("--structure-vh-chain", default="B")
    parser.add_argument("--structure-context-chains", nargs="*", default=["A", "C"])
    parser.add_argument("--proteinmpnn-dir", type=Path, default=None,
                        help="Clone of github.com/dauparas/ProteinMPNN "
                             "(also read from $MMMLM_PROTEINMPNN_DIR).")
    parser.add_argument("--proteinmpnn-weights", type=Path, default=None,
                        help="Defaults to <proteinmpnn-dir>/vanilla_model_weights/v_48_020.pt")
    parser.add_argument("--abmpnn-weights", type=Path,
                        default=EXPERIMENT_DIR / "structures" / "abmpnn.pt",
                        help="abmpnn.pt from zenodo.org/records/8164693 (needs the "
                             "ProteinMPNN checkout too, for the model code).")
    parser.add_argument("--mpnn-batch-size", type=int, default=4)
    parser.add_argument("--mpnn-device", default="cpu",
                        help="Device for ProteinMPNN. Defaults to cpu because the "
                             "MPS backend lacks the decoder's GatherND op.")
    args = parser.parse_args()

    device = _pick_device(args.device)
    tau = args.temperature
    print(f"[pll_one_edit_h3] models={args.models} device={device} tau={tau} "
          f"seed={args.seed}", flush=True)

    vh = TRASTUZUMAB_VH
    L = len(vh)
    h3_start = vh.index(TRASTUZUMAB_H3)
    h3_end = h3_start + len(TRASTUZUMAB_H3)
    h3_mask = mask_from_slice(L, h3_start, h3_end)
    print(f"[pll_one_edit_h3] VH length={L}, H3=[{h3_start},{h3_end}) "
          f"='{vh[h3_start:h3_end]}'", flush=True)

    all_rows: list[dict] = []
    scored_keys: list[str] = []
    for model_key in args.models:
        print(f"\n=== {model_key} ===", flush=True)
        try:
            model = _load_model(model_key, args, device)
            rows = _score_model(model, model_key, vh, h3_mask, h3_start, h3_end,
                                tau, args.seed)
        except Exception as exc:  # noqa: BLE001 - report and skip unusable models
            print(f"  [skip {model_key}] {type(exc).__name__}: {exc}", flush=True)
            continue
        all_rows.extend(rows)
        scored_keys.append(model_key)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_rows:
        raise SystemExit("no models produced results; nothing to write.")

    df = pd.DataFrame(all_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"\n[pll_one_edit_h3] wrote {len(df)} rows "
          f"({len(scored_keys)} models x 190 variants) -> {args.out}", flush=True)

    # Summary table: approximate-vs-true PLL correlation per model.
    summary = []
    for model_key in scored_keys:
        sub = df[df["model_key"] == model_key]
        pear = pearsonr(sub["true_pll"], sub["approx_pll"]).statistic
        spear = spearmanr(sub["true_pll"], sub["approx_pll"]).statistic
        summary.append({
            "Model": MODEL_PRETTY.get(model_key, model_key),
            "Pearson": round(float(pear), 3),
            "Spearman": round(float(spear), 3),
        })
    table = pd.DataFrame(summary)
    print(f"\nApproximate-vs-true PLL correlation (tau={tau:g}):")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
