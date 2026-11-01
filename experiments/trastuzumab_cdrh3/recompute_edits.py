"""Recompute the ``edits`` column of an existing ``all_sequences.csv``.

Sets ``edits = Hamming(vh, seed_vh)`` for every row. Useful for CSVs that
were produced before the beam-search bug fix in which ``edits`` counted
positions *touched* during the search rather than the realized Hamming
distance to the seed (so a substitution that reverts a previously-changed
position back to the seed residue was double-counted).

We only do single-substitution mutations and never indels, so plain
character-by-character comparison is sufficient — no alignment needed.

Adds a ``budget`` column (the original edit-budget parameter the run was
invoked with) so plots that group by realized edits can still distinguish
runs that started with different budgets.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR))

from run import TRASTUZUMAB_VH  # noqa: E402


def hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    return sum(1 for x, y in zip(a, b) if x != y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--seed-vh", default=TRASTUZUMAB_VH,
                        help="Seed VH; default is the canonical Trastuzumab VH.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "vh" not in df.columns:
        raise SystemExit("CSV missing required `vh` column")

    # Move existing edit_budget into a clearer name; keep both for backward compat.
    # Rows merged in from a later run have `edit_budget` but no `budget`, so fill
    # rather than only creating the column when it is absent entirely.
    if "edit_budget" in df.columns:
        if "budget" not in df.columns:
            df["budget"] = df["edit_budget"]
        else:
            df["budget"] = df["budget"].fillna(df["edit_budget"])

    old = df.get("edits")
    new = df["vh"].apply(lambda v: hamming(v, args.seed_vh))
    df["edits"] = new

    if old is not None:
        diffs = (old.astype(int) != new.astype(int)).sum()
        print(f"updated {diffs} of {len(df)} rows where realized edits "
              f"differed from the previously-recorded value.")
    else:
        print(f"populated `edits` for {len(df)} rows.")
    print(f"realized edits distribution per method:")
    print(df.groupby("method")["edits"].describe()[["min", "mean", "max"]].to_string())

    df.to_csv(args.csv, index=False)
    print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
