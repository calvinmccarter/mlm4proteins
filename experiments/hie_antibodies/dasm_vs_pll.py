"""Correlate DASM selection scores against a model's approximate & true PLL.

DASM (netam, Deep Amino-acid Selection Model ``DASMHumV1.0-4M``) is not a masked
LM: a single forward pass on an antibody's paired (VH, VL) sequence yields log
selection factors for every single-substitution mutation -- an [L, 20] matrix
per chain, with the wild-type column at each site forced to 0 (log). So the
"sequence score" of a one-edit variant, sum_i logsel[i, x'_i], collapses to
exactly ``logsel[k, mut_k]`` (all other sites are wild-type -> 0): the mutation's
own log selection factor. This is a wild-type-context score, computed from one
pass on the parent -- directly analogous to the wild-type-marginal *approximate*
PLL, but from a dedicated selection model rather than an MLM.

For each of the seven Hie et al. antibodies we score every one-edit VH+VL variant
with DASM and, per antibody, correlate it against the chosen model's
**approximate** (wild-type-marginal) and exact **true** (re-masked) PLL from a
one-edit PLL parquet (``--pll-parquet``: the paired AbLang2 or the Sapiens run).
The DASM score of a variant is independent of the comparison model, so this only
changes which approx_pll / true_pll it is correlated against.

Runs in the isolated netam venv (``.venv-netam``); DASM scoring is 7 forward
passes (one per antibody), so CPU is fine.

    .venv-netam/bin/python experiments/hie_antibodies/dasm_vs_pll.py \
      --pll-parquet experiments/hie_antibodies/results/pll_one_edit_sapiens.parquet \
      --label sapiens
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR))

from antibodies import ANTIBODIES

# DASM's 20 selection-factor columns follow netam.sequences.AA_STR_SORTED.
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}


def dasm_log_selection(device: str) -> dict[tuple[str, str], np.ndarray]:
    """(antibody, chain) -> [L, 20] natural-log DASM selection factors."""
    from netam import pretrained

    crepe = pretrained.load("DASMHumV1.0-4M", device=device)
    out: dict[tuple[str, str], np.ndarray] = {}
    for name, (vh, vl) in ANTIBODIES.items():
        h, l = crepe.model.selection_factors_of_aa_str((vh, vl))  # linear [L,20]
        out[(name, "H")] = np.log(h.detach().cpu().numpy())
        out[(name, "L")] = np.log(l.detach().cpu().numpy())
    return out


def _corr_row(name: str, s: pd.DataFrame) -> dict:
    return {
        "Antibody": name,
        "DASMvApprox_Pearson": round(float(pearsonr(s.dasm_score, s.approx_pll).statistic), 3),
        "DASMvApprox_Spearman": round(float(spearmanr(s.dasm_score, s.approx_pll).statistic), 3),
        "DASMvTrue_Pearson": round(float(pearsonr(s.dasm_score, s.true_pll).statistic), 3),
        "DASMvTrue_Spearman": round(float(spearmanr(s.dasm_score, s.true_pll).statistic), 3),
    }


def _print_padded(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    cells = [cols] + df.astype(str).values.tolist()
    w = [max(len(r[i]) for r in cells) for i in range(len(cols))]
    lines = ["  ".join(c.ljust(w[i]) if i == 0 else c.rjust(w[i])
                       for i, c in enumerate(r)) for r in cells]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pll-parquet", type=Path,
                   default=EXPERIMENT_DIR / "results" / "pll_one_edit_ablang2.parquet",
                   help="One-edit PLL parquet with approx_pll + true_pll "
                        "(paired AbLang2 or Sapiens).")
    p.add_argument("--label", default=None,
                   help="Model tag for output filenames + header, e.g. 'ablang2' or "
                        "'sapiens'. Defaults to the parquet stem after 'pll_one_edit_'.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", type=Path, default=EXPERIMENT_DIR / "results")
    args = p.parse_args()

    label = args.label or args.pll_parquet.stem.replace("pll_one_edit_", "")
    df = pd.read_parquet(args.pll_parquet)
    sel = dasm_log_selection(args.device)

    # DASM score of each variant = log selection factor of its substitution.
    df["dasm_score"] = [
        float(sel[(ab, ch)][pos, AA_IDX[mut]])
        for ab, ch, pos, mut in zip(df.antibody, df.chain, df.pos_in_chain, df.mut_aa)
    ]
    out_pq = args.out_dir / f"dasm_vs_{label}.parquet"
    out_tbl = args.out_dir / f"corr_table_dasm_vs_{label}.txt"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_pq, index=False)

    rows = [_corr_row(name, df[df.antibody == name])
            for name in sorted(df.antibody.unique())]
    table = pd.DataFrame(rows)
    text = _print_padded(table)
    out_tbl.write_text(text + "\n")
    print(f"\nDASM vs {label} PLL correlation (n={len(df)} variants):")
    print(text)
    print(f"\n[dasm_vs_pll] -> {out_pq}\n[dasm_vs_pll] -> {out_tbl}")


if __name__ == "__main__":
    main()
