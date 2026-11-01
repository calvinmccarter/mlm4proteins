"""Concatenate per-model ``run.py`` outputs into the canonical results CSV.

``run.py`` accepts ``--models esm2 ablang2`` and writes a single
``all_sequences.csv``. When you split the work across GPUs by launching two
processes (one per model) into separate ``--out-dir`` directories, this
script glues their outputs back together so the analysis pipeline
(``recompute_edits.py`` → ``score_oasis.py`` → ``analyze.py``) sees one CSV.

Idempotent: each input CSV is read independently, the rows are concatenated
in the order given, and the merged frame is written verbatim.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs", nargs="+", type=Path, required=True,
        help="Per-model all_sequences.csv files to merge.",
    )
    parser.add_argument(
        "--out", type=Path, required=True,
        help="Destination CSV (typically results/all_sequences.csv).",
    )
    args = parser.parse_args()

    frames = []
    for p in args.inputs:
        df = pd.read_csv(p)
        print(f"  {p}: {len(df)} rows")
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(merged)} rows")


if __name__ == "__main__":
    main()
