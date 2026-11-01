"""Prepare the Mason et al. (DMS_opt) HER2 H3 dataset into train/val/test CSVs.

Combines the AgPos (binders) and AgNeg (non-binders) tables from
``DMS_opt/data/mHER_H3_AgPos.csv`` and ``mHER_H3_AgNeg.csv``, deduplicates by
sequence, drops anything that isn't a clean 10-residue H3 over the standard
20-letter alphabet, and writes train/val/test CSVs with columns
``seq,label`` matching the format consumed by :mod:`oracle.train_oracle`.

Output columns are intentionally the same as the larger her2_aff_large CSVs so
the same training script works on both datasets.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def _load_split_inputs(dms_opt_dir: Path) -> pd.DataFrame:
    pos = pd.read_csv(dms_opt_dir / "mHER_H3_AgPos.csv")
    neg = pd.read_csv(dms_opt_dir / "mHER_H3_AgNeg.csv")
    df = pd.concat([pos, neg], ignore_index=True)
    df = df[["AASeq", "AgClass"]].rename(columns={"AASeq": "seq", "AgClass": "label"})

    # Filter to clean 10-aa sequences over the canonical alphabet.
    df = df[df["seq"].str.len() == 10]
    df = df[df["seq"].apply(lambda s: all(c in VALID_AA for c in s))]

    # Some sequences may appear in both AgPos and AgNeg with conflicting
    # labels — drop those entirely rather than make a silent choice.
    label_counts = df.groupby("seq")["label"].nunique()
    conflict = label_counts[label_counts > 1].index
    if len(conflict):
        print(f"dropping {len(conflict)} sequences with conflicting labels")
    df = df[~df["seq"].isin(conflict)]

    # Deduplicate (keep one row per unique sequence).
    df = df.drop_duplicates("seq").reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dms-opt-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    args = parser.parse_args()

    df = _load_split_inputs(args.dms_opt_dir)
    print(f"clean dataset size: {len(df)}, "
          f"binders: {(df['label']==1).sum()}, non-binders: {(df['label']==0).sum()}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(df))
    n_test = int(round(args.test_frac * len(df)))
    n_val = int(round(args.val_frac * len(df)))
    test_idx = perm[:n_test]
    val_idx = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.iloc[train_idx].to_csv(args.out_dir / "train.csv", index=False)
    df.iloc[val_idx].to_csv(args.out_dir / "val.csv", index=False)
    df.iloc[test_idx].to_csv(args.out_dir / "test.csv", index=False)
    print(f"wrote train/val/test of sizes "
          f"{len(train_idx)}/{len(val_idx)}/{len(test_idx)} to {args.out_dir}")


if __name__ == "__main__":
    main()
