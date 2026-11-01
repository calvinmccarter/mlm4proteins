"""Trastuzumab CDR-H3 reproduction experiment.

Compares three samplers across two MLMs on a single seed (Trastuzumab VH),
restricted to the 10-residue CDR-H3:

  - unguided stochastic beam search
  - unguided Gibbs sampling
  - guided stochastic beam search (CNN oracle as second objective)

Models: ESM2-650M, AbLang2.
The CNN oracle is the PyTorch port of the Mason/Tz_her2 binder classifier.

Outputs:
  - results/<MODEL>_<METHOD>_<E>edits.csv with per-sequence scores
  - results/summary.csv with method-level aggregates
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Allow `python run.py` from this directory.
EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR))
sys.path.insert(0, str(EXPERIMENT_DIR.parent.parent / "src"))

from mmmlm.constants import AMINO_ACIDS
from mmmlm.regions import mask_from_slice
from mmmlm.sampling.beam import BeamSearchConfig, beam_search
from mmmlm.sampling.denoising import DenoisingConfig, denoising_sample_n
from mmmlm.sampling.gibbs import GibbsConfig, gibbs_sample_n
from mmmlm.sampling.gibbs_guided import GuidedGibbsConfig, gibbs_guided_sample_n

from oracle import TrastuzumabCNN, load_oracle, predict


def _gpu_context(device: str) -> tuple[str, int | None]:
    """Describe the accelerator, and count processes already using it.

    The process count is the load-bearing part: two ``run.py`` processes sharing
    one GPU inflate every wall time roughly 2x, uniformly. Without it recorded,
    a ``timings.csv`` cannot be compared across runs, only within one. Returns
    ``None`` for the count when it cannot be determined.
    """
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return str(device), None
    name = torch.cuda.get_device_name(torch.cuda.current_device())
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
        # This process has not allocated yet at startup, so every PID listed is
        # somebody else. Add one for ourselves.
        n_other = len([ln for ln in out.splitlines() if ln.strip()])
        return name, n_other + 1
    except Exception:
        return name, None


TRASTUZUMAB_VH = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYAD"
    "SVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
)
TRASTUZUMAB_VL = (
    "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRF"
    "SGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"
)
TRASTUZUMAB_H3 = "WGGDGFYAMD"


def _h3_slice(vh: str = TRASTUZUMAB_VH, h3: str = TRASTUZUMAB_H3) -> tuple[int, int]:
    """Return (start, end) of the H3 in the VH (end is exclusive)."""
    start = vh.index(h3)
    return start, start + len(h3)


class H3OracleScorer:
    """ScoringFunction adapter: slices H3 out of a full VH and predicts via CNN.

    Returns binder probability in [0, 1] (we want to maximize it).
    """

    name = "tz_cnn_p_binder"
    direction = "maximize"

    def __init__(self, model: TrastuzumabCNN, h3_start: int, h3_len: int = 10):
        self._model = model
        self._h3_start = h3_start
        self._h3_len = h3_len

    def __call__(self, sequences: list[str]) -> np.ndarray:
        h3s = [s[self._h3_start: self._h3_start + self._h3_len] for s in sequences]
        return predict(self._model, h3s)


def _records_from_beam(
    cands,
    method: str,
    model_name: str,
    edit_budget: int,
    seed: int,
    extra_oracle: H3OracleScorer | None,
) -> list[dict]:
    rows = []
    for c in cands:
        h3_start = TRASTUZUMAB_VH.index(TRASTUZUMAB_H3)
        h3 = c.sequence[h3_start: h3_start + 10]
        rows.append({
            "method": method,
            "model": model_name,
            "edit_budget": edit_budget,
            "seed": seed,
            "vh": c.sequence,
            "h3": h3,
            "edits": c.edits,
            "pll": c.pll,
            "sort_key": c.sort_key,
            **{f"score_{k}": v for k, v in c.extra_scores.items()},
        })
    if extra_oracle is not None and rows:
        seqs = [r["vh"] for r in rows]
        p = extra_oracle(seqs)
        for row, val in zip(rows, p):
            row["score_tz_cnn_p_binder"] = float(val)
    return rows


def _records_from_gibbs(
    outputs,
    method: str,
    model_name: str,
    edit_budget: int,
    seed: int,
    oracle: H3OracleScorer | None,
) -> list[dict]:
    h3_start = TRASTUZUMAB_VH.index(TRASTUZUMAB_H3)
    seqs = [s for s, _ in outputs]
    rows = []
    for (s, muts) in outputs:
        rows.append({
            "method": method,
            "model": model_name,
            "edit_budget": edit_budget,
            "seed": seed,
            "vh": s,
            "h3": s[h3_start: h3_start + 10],
            "edits": len(muts),
            "pll": float("nan"),
            "sort_key": float("nan"),
        })
    if oracle is not None and rows:
        p = oracle(seqs)
        for row, val in zip(rows, p):
            row["score_tz_cnn_p_binder"] = float(val)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-checkpoint", type=Path, required=True,
                        help=".pt checkpoint of the strong evaluation CNN. "
                             "All generated sequences are scored with this model.")
    parser.add_argument("--guidance-checkpoint", type=Path, default=None,
                        help="Optional .pt checkpoint of a separate guidance CNN. "
                             "If given, the `beam_weakguided` method uses it for "
                             "STS guidance during beam search; the strong oracle "
                             "is still used to evaluate the resulting sequences. "
                             "If not given, `beam_weakguided` is skipped.")
    parser.add_argument("--out-dir", type=Path, default=EXPERIMENT_DIR / "results")
    parser.add_argument("--edit-budgets", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--n-final", type=int, default=200,
                        help="Sequences kept per (method, model, E)")
    parser.add_argument("--beam-keep", type=int, default=200)
    parser.add_argument("--beam-forward", type=int, default=20)
    parser.add_argument("--beam-keep-guided", type=int, default=200)
    parser.add_argument("--beam-forward-guided", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--gibbs-temperature", type=float, default=1.0)
    parser.add_argument("--gibbs-n", type=int, default=400,
                        help="Independent Gibbs chains per (model, E)")
    parser.add_argument("--gibbs-max-steps", type=int, default=None,
                        help="Update cap for the Gibbs samplers, which run until "
                             "E *realized* edits (default 60·E). Chains that hit "
                             "it without reaching E edits are reported, not dropped.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models", nargs="+", default=["esm2", "ablang2"])
    parser.add_argument("--proteinmpnn-dir", type=Path, default=None,
                        help="Clone of github.com/dauparas/ProteinMPNN. Required "
                             "for the `proteinmpnn` / `abmpnn` models (also "
                             "readable from $MMMLM_PROTEINMPNN_DIR).")
    parser.add_argument("--structure", type=Path,
                        default=EXPERIMENT_DIR / "structures" / "1N8Z.pdb",
                        help="Backbone for the structure-conditioned models. "
                             "Default is the trastuzumab Fab / HER2 crystal "
                             "structure (chain B = VH, A = VL, C = HER2).")
    parser.add_argument("--structure-vh-chain", default="B")
    parser.add_argument("--structure-context-chains", nargs="*", default=["A", "C"],
                        help="Chains held fixed as visible context. Pass just "
                             "the light chain to drop the antigen.")
    parser.add_argument("--proteinmpnn-weights", type=Path, default=None,
                        help="Defaults to <proteinmpnn-dir>/vanilla_model_weights/v_48_020.pt")
    parser.add_argument("--abmpnn-weights", type=Path,
                        default=EXPERIMENT_DIR / "structures" / "abmpnn.pt",
                        help="abmpnn.pt from zenodo.org/records/8164693")
    parser.add_argument("--mpnn-batch-size", type=int, default=4,
                        help="Decoder passes batched per template_logits call.")
    parser.add_argument("--methods", nargs="+",
                        default=["beam", "gibbs", "denoising", "beam_guided",
                                 "beam_weakguided", "gibbs_guided", "gibbs_weakguided"])
    parser.add_argument("--denoising-temperature", type=float, default=1.0)
    parser.add_argument("--guidance-strength", type=float, default=1.0,
                        help="γ used for product-of-experts guided Gibbs.")
    parser.add_argument("--oasis-db", type=Path, default=None,
                        help="Path to OASis_9mers_v1.db. Required for the "
                             "`beam_oasis_guided` / `beam_oasis_weakguided` methods.")
    parser.add_argument("--oasis-workers", type=int, default=32)
    parser.add_argument("--oasis-cache-path", type=Path,
                        default=EXPERIMENT_DIR / "cache" / "oasis_cache.pkl",
                        help="Persistent disk cache for OASis percentiles.")
    parser.add_argument("--oasis-warm-from", type=Path, default=None,
                        help="Optional CSV (with vh + score_oasis_percentile cols) "
                             "to seed the OASis cache.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    accelerator, gpu_procs = _gpu_context(device)
    print(f"[trastuzumab_cdrh3] device={device} ({accelerator}), "
          f"gpu_procs_at_start={gpu_procs}", flush=True)

    h3_start, h3_end = _h3_slice()
    L = len(TRASTUZUMAB_VH)
    h3_mask = mask_from_slice(L, h3_start, h3_end)
    print(f"[trastuzumab_cdrh3] VH length={L}, H3=[{h3_start},{h3_end}) "
          f"='{TRASTUZUMAB_VH[h3_start:h3_end]}'", flush=True)

    cnn = load_oracle(args.oracle_checkpoint, device=device)
    oracle = H3OracleScorer(cnn, h3_start=h3_start, h3_len=h3_end - h3_start)

    if args.guidance_checkpoint is not None:
        weak_cnn = load_oracle(args.guidance_checkpoint, device=device)
        weak_guide = H3OracleScorer(
            weak_cnn, h3_start=h3_start, h3_len=h3_end - h3_start
        )
        weak_guide.name = "tz_cnn_weak_p_binder"
    else:
        weak_guide = None

    if args.oasis_db is not None:
        from oasis import OASisHumannessScorer
        oasis_scorer = OASisHumannessScorer(
            db_path=str(args.oasis_db),
            n_workers=args.oasis_workers,
            cache_path=args.oasis_cache_path,
        )
        if args.oasis_warm_from is not None and args.oasis_warm_from.exists():
            oasis_scorer.warm_from_csv(args.oasis_warm_from)
    else:
        oasis_scorer = None

    all_rows: list[dict] = []
    # Per-cell wall times. Written alongside summary.csv so method cost is
    # recoverable after the fact -- it used to exist only in stdout.
    timing_rows: list[dict] = []

    for model_key in args.models:
        if model_key == "esm2":
            from mmmlm.models.esm2 import ESM2
            mlm = ESM2(device=device)
        elif model_key == "ablang2":
            from mmmlm.models.ablang2 import AbLang2
            mlm = AbLang2(vl_sequence=TRASTUZUMAB_VL, device=device)
        elif model_key in ("proteinmpnn", "abmpnn"):
            from mmmlm.models.mpnn import StructureMPNN
            mpnn_dir = args.proteinmpnn_dir
            if model_key == "proteinmpnn":
                weights = args.proteinmpnn_weights
                if weights is None:
                    if mpnn_dir is None:
                        raise ValueError(
                            "--proteinmpnn-weights or --proteinmpnn-dir is required "
                            "for the proteinmpnn model"
                        )
                    weights = mpnn_dir / "vanilla_model_weights" / "v_48_020.pt"
            else:
                weights = args.abmpnn_weights
            mlm = StructureMPNN(
                structure_path=args.structure,
                weights_path=weights,
                design_chain=args.structure_vh_chain,
                design_sequence=TRASTUZUMAB_VH,
                context_chains=args.structure_context_chains,
                name=model_key,
                device=device,
                max_batch_size=args.mpnn_batch_size,
                proteinmpnn_root=mpnn_dir,
            )
            print(f"[{model_key}] structure={args.structure.name} "
                  f"design=chain {args.structure_vh_chain} "
                  f"context={args.structure_context_chains} "
                  f"complex_length={mlm.complex_length} "
                  f"k={mlm.k_neighbors} train_noise={mlm.train_noise_level}A",
                  flush=True)
        else:
            raise ValueError(f"unknown model {model_key}")
        print(f"\n=== {model_key} ===", flush=True)

        for E in args.edit_budgets:
            for method in args.methods:
                t0 = time.time()
                if method == "beam":
                    cfg = BeamSearchConfig(
                        edit_budget=E,
                        beam_keep=args.beam_keep,
                        beam_forward=args.beam_forward,
                        n_final=args.n_final,
                        temperature=args.temperature,
                        gumbel=True,
                        guidance="none",
                        seed=args.seed,
                    )
                    out = beam_search(TRASTUZUMAB_VH, mlm, h3_mask, cfg)
                    rows = _records_from_beam(
                        out, method, mlm.name, E, args.seed, oracle
                    )
                elif method == "beam_guided":
                    cfg = BeamSearchConfig(
                        edit_budget=E,
                        beam_keep=args.beam_keep_guided,
                        beam_forward=args.beam_forward_guided,
                        n_final=args.n_final,
                        temperature=args.temperature,
                        gumbel=True,
                        guidance="sts",
                        sts_weights={"pseudo_perplexity": 1.0,
                                     "tz_cnn_p_binder": 2.0},
                        seed=args.seed,
                    )
                    out = beam_search(
                        TRASTUZUMAB_VH, mlm, h3_mask, cfg,
                        scoring_functions=[oracle],
                    )
                    rows = _records_from_beam(
                        out, method, mlm.name, E, args.seed, None
                    )
                elif method == "beam_weakguided":
                    if weak_guide is None:
                        print(f"  [skip beam_weakguided E={E}] "
                              f"--guidance-checkpoint not provided", flush=True)
                        continue
                    cfg = BeamSearchConfig(
                        edit_budget=E,
                        beam_keep=args.beam_keep_guided,
                        beam_forward=args.beam_forward_guided,
                        n_final=args.n_final,
                        temperature=args.temperature,
                        gumbel=True,
                        guidance="sts",
                        sts_weights={"pseudo_perplexity": 1.0,
                                     "tz_cnn_weak_p_binder": 2.0},
                        seed=args.seed,
                    )
                    out = beam_search(
                        TRASTUZUMAB_VH, mlm, h3_mask, cfg,
                        scoring_functions=[weak_guide],
                    )
                    # Evaluate with the STRONG oracle for fair comparison.
                    rows = _records_from_beam(
                        out, method, mlm.name, E, args.seed, oracle,
                    )
                elif method == "denoising":
                    cfg = DenoisingConfig(
                        edit_budget=E,
                        temperature=args.denoising_temperature,
                        argmax=False,
                        position_selection="lowest_entropy",
                        seed=args.seed,
                    )
                    out = denoising_sample_n(
                        TRASTUZUMAB_VH, mlm, h3_mask, cfg, n_sequences=args.gibbs_n,
                    )
                    rows = _records_from_gibbs(
                        out, method, mlm.name, E, args.seed, oracle
                    )
                elif method == "gibbs":
                    cfg = GibbsConfig(
                        edit_budget=E,
                        temperature=args.gibbs_temperature,
                        argmax=False,
                        seed=args.seed,
                        max_steps=args.gibbs_max_steps,
                    )
                    out = gibbs_sample_n(
                        TRASTUZUMAB_VH, mlm, h3_mask, cfg, n_sequences=args.gibbs_n,
                    )
                    rows = _records_from_gibbs(
                        out, method, mlm.name, E, args.seed, oracle
                    )
                elif method in ("beam_oasis_guided", "beam_oasis_weakguided"):
                    if oasis_scorer is None:
                        print(f"  [skip {method} E={E}] --oasis-db not provided",
                              flush=True)
                        continue
                    binder = oracle if method == "beam_oasis_guided" else weak_guide
                    if binder is None:
                        print(f"  [skip {method} E={E}] no guidance CNN", flush=True)
                        continue
                    cfg = BeamSearchConfig(
                        edit_budget=E,
                        beam_keep=args.beam_keep_guided,
                        beam_forward=args.beam_forward_guided,
                        n_final=args.n_final,
                        temperature=args.temperature,
                        gumbel=True,
                        guidance="sts",
                        sts_weights={
                            "pseudo_perplexity": 1.0,
                            binder.name: 2.0,
                            "oasis_percentile": 2.0,
                        },
                        seed=args.seed,
                    )
                    out = beam_search(
                        TRASTUZUMAB_VH, mlm, h3_mask, cfg,
                        scoring_functions=[binder, oasis_scorer],
                    )
                    rows = _records_from_beam(
                        out, method, mlm.name, E, args.seed, oracle,
                    )
                elif method == "gibbs_guided":
                    # Product-of-experts guided Gibbs using the strong oracle.
                    cfg = GuidedGibbsConfig(
                        edit_budget=E,
                        mlm_temperature=args.gibbs_temperature,
                        guidance_strength=args.guidance_strength,
                        seed=args.seed,
                    )
                    out = gibbs_guided_sample_n(
                        TRASTUZUMAB_VH, mlm, h3_mask, [oracle], cfg,
                        n_sequences=args.gibbs_n,
                    )
                    rows = _records_from_gibbs(
                        out, method, mlm.name, E, args.seed, oracle
                    )
                elif method == "gibbs_weakguided":
                    if weak_guide is None:
                        print(f"  [skip gibbs_weakguided E={E}] "
                              f"--guidance-checkpoint not provided", flush=True)
                        continue
                    cfg = GuidedGibbsConfig(
                        edit_budget=E,
                        mlm_temperature=args.gibbs_temperature,
                        guidance_strength=args.guidance_strength,
                        seed=args.seed,
                    )
                    out = gibbs_guided_sample_n(
                        TRASTUZUMAB_VH, mlm, h3_mask, [weak_guide], cfg,
                        n_sequences=args.gibbs_n,
                    )
                    # Evaluate with the strong oracle for fair comparison.
                    rows = _records_from_gibbs(
                        out, method, mlm.name, E, args.seed, oracle
                    )
                else:
                    raise ValueError(f"unknown method {method}")
                dt = time.time() - t0
                if rows:
                    p_vals = [r.get("score_tz_cnn_p_binder", float("nan")) for r in rows]
                    p_mean = float(np.nanmean(p_vals))
                    p_succ = float(np.nanmean([1.0 if (p > 0.5) else 0.0 for p in p_vals]))
                else:
                    p_mean = p_succ = float("nan")
                print(f"  [{method:12s} E={E}] n={len(rows):4d}  "
                      f"mean_p_binder={p_mean:.3f}  frac_p>0.5={p_succ:.3f}  "
                      f"({dt:.1f}s)", flush=True)
                timing_rows.append({
                    "model": mlm.name,
                    "method": method,
                    "edit_budget": E,
                    "seconds": dt,
                    "n": len(rows),
                    "unique_h3": len({r["h3"] for r in rows}),
                    "device": accelerator,
                    "gpu_procs_at_start": gpu_procs,
                })
                all_rows.extend(rows)
                # Persist incremental progress so a crash doesn't lose work.
                pd.DataFrame(all_rows).to_csv(
                    args.out_dir / "all_sequences.csv", index=False
                )
                pd.DataFrame(timing_rows).to_csv(
                    args.out_dir / "timings.csv", index=False
                )

        # Free model memory before loading next.
        del mlm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(all_rows)
    df.to_csv(args.out_dir / "all_sequences.csv", index=False)

    # Summary aggregates per (model, method, edit_budget)
    summary = (
        df.groupby(["model", "method", "edit_budget"])
        .agg(
            n=("vh", "count"),
            mean_p_binder=("score_tz_cnn_p_binder", "mean"),
            frac_p_gt_0_5=("score_tz_cnn_p_binder", lambda s: (s > 0.5).mean()),
            mean_pll=("pll", "mean"),
            mean_edits=("edits", "mean"),
            unique_h3=("h3", "nunique"),
        )
        .reset_index()
    )
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    print("\nsummary:")
    print(summary.to_string(index=False))

    timings = pd.DataFrame(timing_rows)
    if not timings.empty:
        timings.to_csv(args.out_dir / "timings.csv", index=False)
        print("\ntiming (seconds summed over edit budgets):")
        print(
            timings.pivot_table(index="method", columns="model",
                                values="seconds", aggfunc="sum")
            .round(1).to_string()
        )
        if gpu_procs and gpu_procs > 1:
            print(f"NOTE: {gpu_procs} processes shared this GPU, so these wall "
                  f"times are inflated and comparable only within this run.",
                  flush=True)

    cfg = vars(args) | {"accelerator": accelerator, "gpu_procs_at_start": gpu_procs}
    with open(args.out_dir / "config.json", "w") as f:
        json.dump(cfg, f, default=str, indent=2)


if __name__ == "__main__":
    main()
