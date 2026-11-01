"""Render the approx/Gumbel-vs-true PLL correlation table for a one-edit PLL
parquet as a space-padded, border-less plain-text table (nice in a monospace
textbox), one row per antibody in alphabetical order.

Works on any parquet written by ``pll_one_edit_vhvl.py`` (AbLang-2) or its ESM2
counterpart ``pll_one_edit_vhvl_esm2.py`` -- they share the ``true_pll /
approx_pll / gumbel_pll`` columns the correlations are computed from.

Columns: Antibody  Approx_Pearson  Approx_Spearman  Gumbel_Pearson  Gumbel_Spearman

The output filename is ``corr_table_<label>.txt``, where ``--label`` encodes the
model and the scored region, e.g. ``ablang2_full`` / ``esm2-650m_full`` (``full``
= the whole VH+VL; a future CDR-only sweep would use e.g. ``..._cdr``).

    python experiments/hie_antibodies/corr_table.py \
      --parquet experiments/hie_antibodies/results/pll_one_edit_ablang2.parquet \
      --label ablang2_full
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR))                       # sibling scripts
sys.path.insert(0, str(EXPERIMENT_DIR.parent.parent / "src"))  # mmmlm package

# Reuse the exact correlation definition the run scripts print at the end.
from pll_one_edit_vhvl import _corr_table


def format_padded(df: pd.DataFrame, ndigits: int = 3) -> str:
    """First column left-justified, numeric columns right-justified to a fixed
    decimal count, single-space-padded to equal width with a two-space gutter."""
    disp = df.copy()
    for c in disp.columns[1:]:
        disp[c] = disp[c].map(lambda v: f"{float(v):.{ndigits}f}")
    cols = list(disp.columns)
    cells = [cols] + disp.astype(str).values.tolist()
    widths = [max(len(row[i]) for row in cells) for i in range(len(cols))]
    return "\n".join(
        "  ".join(cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i])
                  for i, cell in enumerate(row))
        for row in cells
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", type=Path,
                   default=EXPERIMENT_DIR / "results" / "pll_one_edit_ablang2.parquet")
    p.add_argument("--label", default=None,
                   help="Tag placed in the filename as corr_table_<label>.txt, encoding "
                        "model + region, e.g. 'ablang2_full' or 'esm2-650m_full'. "
                        "Defaults to the parquet stem.")
    p.add_argument("--out-dir", type=Path, default=EXPERIMENT_DIR / "results",
                   help="Directory for the table (default: the experiment's results/, "
                        "which is git-tracked even though bulk run outputs are not).")
    p.add_argument("--out", type=Path, default=None,
                   help="Explicit output path; overrides --out-dir/--label.")
    p.add_argument("--digits", type=int, default=3)
    args = p.parse_args()

    df = pd.read_parquet(args.parquet)
    tbl = _corr_table(df).sort_values("Antibody").reset_index(drop=True)
    text = format_padded(tbl, args.digits)

    label = args.label or args.parquet.stem
    out = args.out or (args.out_dir / f"corr_table_{label}.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(text)
    print(f"\n[corr_table] {len(tbl)} antibodies -> {out}", flush=True)


if __name__ == "__main__":
    main()
