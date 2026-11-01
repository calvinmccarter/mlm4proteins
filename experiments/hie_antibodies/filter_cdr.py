"""Filter a one-edit PLL parquet down to variants whose mutated position lies in
a CDR (Chothia CDR definition), writing ``<stem>_cdr.parquet``.

The full-sequence PLLs are unchanged -- this only restricts *which* one-edit
variants are kept (those mutating a CDR residue), so it is a pure subset of the
``full`` results and needs no model re-run. CDR positions are obtained per
antibody per chain with abnumber (scheme='chothia', cdr_definition='chothia');
the conserved Cys that abnumber includes in L1/L3 is part of its Chothia CDR
convention.

Works on any parquet written by the one-edit PLL scripts (ablang2 / esm2 /
sapiens): they share the ``antibody``, ``chain``, ``pos_in_chain`` columns.

    python experiments/hie_antibodies/filter_cdr.py \
      --parquet experiments/hie_antibodies/results/pll_one_edit_sapiens.parquet
    python experiments/hie_antibodies/corr_table.py \
      --parquet experiments/hie_antibodies/results/pll_one_edit_sapiens_cdr.parquet \
      --label sapiens_cdr
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR))

from pll_one_edit_vhvl import ANTIBODIES


def cdr_positions(seq: str, chain: str, scheme: str = "chothia",
                  use_anarcii: bool = False) -> set[int]:
    """0-indexed positions of ``seq`` that fall in a CDR, via abnumber.

    ``chain`` is 'H' or 'L'. Positions abnumber cannot align (outside the V
    domain) are framework by construction and excluded. ``use_anarcii`` selects
    abnumber's HMMER-free anarcii backend instead of classic ANARCI; for
    canonical human V-domains the Chothia CDR boundaries are identical.
    """
    from abnumber import Chain

    c = Chain(seq, scheme=scheme, cdr_definition=scheme, use_anarcii=use_anarcii)
    items = [(aa, pos.is_in_cdr()) for pos, aa in c]
    chain_seq = "".join(aa for aa, _ in items)
    offset = seq.index(chain_seq)  # abnumber keeps a contiguous V-domain slice
    return {offset + i for i, (_, in_cdr) in enumerate(items) if in_cdr}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", type=Path, required=True,
                   help="Full one-edit PLL parquet (all positions).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output parquet (default: <stem>_cdr.parquet).")
    p.add_argument("--scheme", default="chothia",
                   help="abnumber numbering + CDR definition (default: chothia).")
    p.add_argument("--use-anarcii", action="store_true",
                   help="Use abnumber's HMMER-free anarcii backend instead of classic "
                        "ANARCI (identical Chothia CDRs for canonical V-domains).")
    args = p.parse_args()

    df = pd.read_parquet(args.parquet)

    # CDR position sets per (antibody, chain), computed once.
    cdr: dict[tuple[str, str], set[int]] = {}
    for name, (vh, vl) in ANTIBODIES.items():
        if name not in set(df["antibody"]):
            continue
        cdr[(name, "H")] = cdr_positions(vh, "H", args.scheme, args.use_anarcii)
        cdr[(name, "L")] = cdr_positions(vl, "L", args.scheme, args.use_anarcii)

    keep = df.apply(
        lambda r: r["pos_in_chain"] in cdr.get((r["antibody"], r["chain"]), set()),
        axis=1,
    )
    sub = df[keep].reset_index(drop=True)

    out = args.out or args.parquet.with_name(args.parquet.stem + "_cdr.parquet")
    sub.to_parquet(out, index=False)

    per_ab = sub.groupby("antibody").size()
    print(f"[filter_cdr] scheme={args.scheme}: kept {len(sub)}/{len(df)} variants "
          f"({len(sub)/len(df)*100:.1f}%) -> {out}")
    print(per_ab.to_string())


if __name__ == "__main__":
    main()
