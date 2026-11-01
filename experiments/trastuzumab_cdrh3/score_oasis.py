"""Score the OASis humanness percentile of every generated VH in ``all_sequences.csv``.

Adds a ``score_oasis_percentile`` column. Idempotent: previously-scored rows
are reused from the existing CSV (cached by ``vh`` sequence).

Run after ``run.py`` has written ``all_sequences.csv``. Uses a Pool of workers
to parallelize across CPUs since OASis lookups are ~1.5 s per sequence single-
threaded.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR))

from oasis import (  # noqa: E402
    DEFAULT_CACHE_PATH,
    DEFAULT_OASIS_DB,
    OASisHumannessScorer,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True,
                        help="Path to all_sequences.csv (will be updated in place).")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_OASIS_DB)
    parser.add_argument("--n-workers", type=int, default=32)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"loaded {len(df)} rows from {args.csv}")

    # Seed scorer from any rows that already have OASis values (idempotent re-runs).
    seed: dict[str, float] = {}
    if "score_oasis_percentile" in df.columns:
        for vh, val in zip(df["vh"], df["score_oasis_percentile"]):
            if isinstance(val, (int, float)) and not np.isnan(val):
                seed[vh] = float(val)
        print(f"seeding scorer with {len(seed)} previously-scored sequences from CSV")

    scorer = OASisHumannessScorer(
        db_path=str(args.db_path), n_workers=args.n_workers,
        cache=seed, cache_path=args.cache_path,
    )
    cache = scorer._cache  # the live cache: includes seed + disk + new computations

    try:
        unique_vhs = list(df["vh"].unique())
        new_to_score = [s for s in unique_vhs if s not in cache]
        print(f"{len(unique_vhs)} unique VHs total, "
              f"{len(new_to_score)} need fresh OASis lookups")

        # Score in chunks so we can save progress along the way.
        chunk = 500
        t0 = time.time()
        for i in range(0, len(new_to_score), chunk):
            sub = new_to_score[i: i + chunk]
            scorer(sub)  # populates cache (which is scorer._cache)
            df["score_oasis_percentile"] = [cache.get(s, float("nan")) for s in df["vh"]]
            df.to_csv(args.csv, index=False)
            done = min(i + chunk, len(new_to_score))
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-9)
            eta = (len(new_to_score) - done) / max(rate, 1e-9)
            print(f"  scored {done}/{len(new_to_score)} unique VHs  "
                  f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)", flush=True)
    finally:
        scorer.close()

    df["score_oasis_percentile"] = [cache.get(s, float("nan")) for s in df["vh"]]
    df.to_csv(args.csv, index=False)
    populated = df["score_oasis_percentile"].notna().sum()
    print(f"\nfinal: {populated}/{len(df)} rows have OASis; "
          f"range [{df['score_oasis_percentile'].min():.3f}, "
          f"{df['score_oasis_percentile'].max():.3f}]")


if __name__ == "__main__":
    main()
