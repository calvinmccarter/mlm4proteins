"""Beam-size sweep for the Trastuzumab CDR-H3 experiment.

Runs three beam-search methods (Beam, Beam + STS Mason CNN, Beam + STS
Mason CNN + OASis) at several values of ``beam_forward`` (the frontier
width) on each MLM, holds everything else (``beam_keep`` = ``n_final`` =
200, edit budget sweep E ∈ 2..7, temperature 1.5, seed 0) fixed, and
writes one combined CSV per model with an extra ``beam_forward`` column.

Outputs go under ``experiments/trastuzumab_cdrh3/results/beam_size_sweep/``
to keep them disjoint from the main ``all_sequences.csv``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
SWEEP_DIR = RESULTS_DIR / "beam_size_sweep"
sys.path.insert(0, str(EXPERIMENT_DIR))

from oasis import (  # noqa: E402
    DEFAULT_ABNUMBER_DIR,
    DEFAULT_BIOPHI_DIR,
    DEFAULT_OASIS_DB,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-checkpoint", type=Path,
                        default=EXPERIMENT_DIR / "checkpoints" / "cnn.pt")
    parser.add_argument("--guidance-checkpoint", type=Path,
                        default=EXPERIMENT_DIR / "checkpoints" / "cnn_mason.pt")
    parser.add_argument("--oasis-db", type=Path, default=DEFAULT_OASIS_DB)
    parser.add_argument("--oasis-cache-path", type=Path,
                        default=EXPERIMENT_DIR / "cache" / "oasis_cache.pkl")
    parser.add_argument("--beam-forwards", type=int, nargs="+",
                        default=[1, 2, 3, 5, 10, 20, 30, 50, 100])
    parser.add_argument("--edit-budgets", type=int, nargs="+",
                        default=[2, 3, 4, 5, 6, 7])
    parser.add_argument("--n-final", type=int, default=200)
    parser.add_argument("--beam-keep", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models", nargs="+", default=["esm2", "ablang2"])
    parser.add_argument("--methods", nargs="+",
                        default=["beam", "beam_weakguided", "beam_oasis_weakguided"])
    parser.add_argument("--gpu-by-model", default="esm2:0,ablang2:1",
                        help="Mapping ``model:cuda_index`` for parallel runs.")
    parser.add_argument("--sweep-dir", type=Path, default=SWEEP_DIR,
                        help="Output root for per-cell subdirs and combined "
                             "CSV; override to run a disjoint sweep.")
    args = parser.parse_args()

    sweep_dir = args.sweep_dir
    sweep_dir.mkdir(parents=True, exist_ok=True)

    gpu_map: dict[str, str] = {}
    for chunk in args.gpu_by_model.split(","):
        if not chunk.strip():
            continue
        m, idx = chunk.split(":")
        gpu_map[m.strip()] = idx.strip()

    accumulated: list[pd.DataFrame] = []

    for model in args.models:
        for bf in args.beam_forwards:
            run_out = sweep_dir / f"{model}_bf{bf}"
            run_out.mkdir(parents=True, exist_ok=True)
            csv = run_out / "all_sequences.csv"
            if csv.exists():
                print(f"[skip] {csv} already exists ({len(pd.read_csv(csv))} rows)",
                      flush=True)
            else:
                env = {"PYTHONPATH": f"{DEFAULT_ABNUMBER_DIR}:{DEFAULT_BIOPHI_DIR}",
                       "HF_HUB_DISABLE_XET": "1"}
                if model in gpu_map:
                    env["CUDA_VISIBLE_DEVICES"] = gpu_map[model]
                cmd = [
                    sys.executable, str(EXPERIMENT_DIR / "run.py"),
                    "--oracle-checkpoint", str(args.oracle_checkpoint),
                    "--guidance-checkpoint", str(args.guidance_checkpoint),
                    "--oasis-db", str(args.oasis_db),
                    "--oasis-cache-path", str(args.oasis_cache_path),
                    "--oasis-workers", "16",
                    "--edit-budgets", *(str(e) for e in args.edit_budgets),
                    "--n-final", str(args.n_final),
                    "--beam-keep", str(args.beam_keep),
                    "--beam-forward", str(bf),
                    "--beam-keep-guided", str(args.beam_keep),
                    "--beam-forward-guided", str(bf),
                    "--gibbs-n", "0",
                    "--temperature", str(args.temperature),
                    "--seed", str(args.seed),
                    "--models", model,
                    "--methods", *args.methods,
                    "--out-dir", str(run_out),
                ]
                merged_env = {**__import__("os").environ, **env}
                print(f"[run] model={model} beam_forward={bf} "
                      f"-> {run_out.name}", flush=True)
                subprocess.run(cmd, env=merged_env, check=True)

            df = pd.read_csv(csv)
            df["beam_forward"] = bf
            accumulated.append(df)

    combined = pd.concat(accumulated, ignore_index=True)
    out_csv = sweep_dir / "all_sequences.csv"
    combined.to_csv(out_csv, index=False)
    print(f"\nwrote combined sweep CSV: {out_csv} "
          f"({len(combined)} rows, {combined['beam_forward'].nunique()} beam sizes)",
          flush=True)
    with open(sweep_dir / "config.json", "w") as f:
        json.dump(vars(args), f, default=str, indent=2)


if __name__ == "__main__":
    main()
