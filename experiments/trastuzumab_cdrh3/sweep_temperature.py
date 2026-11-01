"""Temperature sweep for the Trastuzumab CDR-H3 stochastic beam search.

Runs five beam-search methods (Beam, Beam + STS Mason CNN, Beam + STS
Mason CNN + OASis, Beam + STS oracle, Beam + STS oracle + OASis) at
several values of the Gumbel ``--temperature`` on each MLM, holds
everything else (``beam_keep = n_final = 200``, ``beam_forward = 20``,
edit budget sweep ``E ∈ 2..7``, seed 0) fixed, and writes one combined
CSV per model with an extra ``temperature`` column.

Outputs go under ``experiments/trastuzumab_cdrh3/results/temperature_sweep/``
to keep them disjoint from the main ``all_sequences.csv``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
SWEEP_DIR = RESULTS_DIR / "temperature_sweep"
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
    parser.add_argument("--temperatures", type=float, nargs="+",
                        default=[0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.5, 5.0])
    parser.add_argument("--edit-budgets", type=int, nargs="+",
                        default=[2, 3, 4, 5, 6, 7])
    parser.add_argument("--n-final", type=int, default=200)
    parser.add_argument("--beam-keep", type=int, default=200)
    parser.add_argument("--beam-forward", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models", nargs="+", default=["esm2", "ablang2"])
    parser.add_argument("--methods", nargs="+",
                        default=["beam", "beam_weakguided", "beam_oasis_weakguided",
                                 "beam_guided", "beam_oasis_guided"])
    parser.add_argument("--gpu-by-model", default="esm2:0,ablang2:1",
                        help="Mapping ``model:cuda_index`` for sequential runs.")
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
        for T in args.temperatures:
            run_out = sweep_dir / f"{model}_T{T:g}"
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
                    "--beam-forward", str(args.beam_forward),
                    "--beam-keep-guided", str(args.beam_keep),
                    "--beam-forward-guided", str(args.beam_forward),
                    "--gibbs-n", "0",
                    "--temperature", str(T),
                    "--seed", str(args.seed),
                    "--models", model,
                    "--methods", *args.methods,
                    "--out-dir", str(run_out),
                ]
                merged_env = {**__import__("os").environ, **env}
                print(f"[run] model={model} temperature={T} "
                      f"-> {run_out.name}", flush=True)
                subprocess.run(cmd, env=merged_env, check=True)

            df = pd.read_csv(csv)
            df["temperature"] = T
            accumulated.append(df)

    combined = pd.concat(accumulated, ignore_index=True)
    out_csv = sweep_dir / "all_sequences.csv"
    combined.to_csv(out_csv, index=False)
    print(f"\nwrote combined sweep CSV: {out_csv} "
          f"({len(combined)} rows, "
          f"{combined['temperature'].nunique()} temperatures)",
          flush=True)
    with open(sweep_dir / "config.json", "w") as f:
        json.dump(vars(args), f, default=str, indent=2)


if __name__ == "__main__":
    main()
