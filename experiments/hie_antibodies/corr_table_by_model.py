"""Pivot the approx-vs-true PLL correlation table across models, one row per
antibody and one column per model, for a single named metric (default:
``Approx_Spearman``).

Each input parquet must have the ``true_pll / approx_pll / gumbel_pll``
columns shared by ``pll_one_edit_vhvl.py`` (AbLang-2), ``pll_one_edit_sapiens.py``
(Sapiens), and ``pll_one_edit_vhvl_esm2.py`` (ESM2) -- the per-antibody
correlations are computed the same way as ``corr_table.py``, just pivoted
across models instead of printed one model at a time.

Prints a plain-text table (like ``corr_table.py``) and, with ``--latex``,
a copy-pasteable LaTeX ``tabular``.

    python experiments/hie_antibodies/corr_table_by_model.py \
      --parquet results/pll_one_edit_sapiens.parquet Sapiens \
      --parquet results/pll_one_edit_ablang2.parquet AbLang2 \
      --parquet results/pll_one_edit_esm2.parquet ESM2 \
      --metric Approx_Spearman --latex
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

METRICS = ["Approx_Pearson", "Approx_Spearman", "Gumbel_Pearson", "Gumbel_Spearman"]


def format_padded(df: pd.DataFrame, ndigits: int = 3) -> str:
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


def _format_antibody(name: str, texttt: bool) -> str:
    if not texttt:
        return name
    # \texttt{} is plain text mode, so swap the underscore for a hyphen rather
    # than escaping it (matches how these names get typeset in the paper, e.g.
    # \texttt{mab114-uca}).
    return rf"\texttt{{{name.replace('_', '-')}}}"


def to_latex(df: pd.DataFrame, metric: str, ndigits: int = 3, texttt: bool = False,
             caption: str | None = None, label: str = "tab:pll_one_edit_hie_by_model") -> str:
    model_cols = list(df.columns[1:])
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{l" + "c" * len(model_cols) + "}",
        r"\toprule",
        "Antibody & " + " & ".join(model_cols) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        vals = " & ".join(f"{row[c]:.{ndigits}f}" for c in model_cols)
        antibody = _format_antibody(row["Antibody"], texttt)
        lines.append(f"{antibody} & {vals} \\\\")
    if caption is None:
        metric_desc = {
            "Approx_Pearson": "Pearson",
            "Approx_Spearman": "Spearman",
            "Gumbel_Pearson": "Gumbel-perturbed Pearson",
            "Gumbel_Spearman": "Gumbel-perturbed Spearman",
        }[metric]
        caption = (
            f"Approximate-vs-true PLL {metric_desc} correlation, one-edit "
            "full VH+VL variants, over the seven-antibody panel from Hie et al."
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet", nargs=2, action="append", metavar=("PATH", "LABEL"),
        default=None,
        help="A (parquet path, model label) pair; repeat once per model. "
             "Defaults to Sapiens/AbLang2/ESM2 full-VH+VL results.",
    )
    parser.add_argument("--metric", default="Approx_Spearman", choices=METRICS)
    parser.add_argument("--digits", type=int, default=3)
    parser.add_argument("--latex", action="store_true",
                        help="Also print a LaTeX tabular for the chosen metric.")
    parser.add_argument("--tex-out", type=Path, default=None,
                        help="If set (implies --latex), also write the LaTeX to this path.")
    parser.add_argument("--texttt", action="store_true",
                        help="Typeset the Antibody column as \\texttt{...} with "
                             "underscores swapped for hyphens, e.g. mab114_uca -> "
                             "\\texttt{mab114-uca}.")
    parser.add_argument("--caption", default=None,
                        help="LaTeX \\caption text; defaults to a generic description "
                             "of the metric and panel.")
    parser.add_argument("--label", default="tab:pll_one_edit_hie_by_model",
                        help="LaTeX \\label{...} key.")
    args = parser.parse_args()

    pairs = args.parquet or [
        [str(EXPERIMENT_DIR / "results" / "pll_one_edit_sapiens.parquet"), "Sapiens"],
        [str(EXPERIMENT_DIR / "results" / "pll_one_edit_ablang2.parquet"), "AbLang2"],
        [str(EXPERIMENT_DIR / "results" / "pll_one_edit_esm2.parquet"), "ESM2"],
    ]

    per_model = {}
    for path, label in pairs:
        df = pd.read_parquet(path)
        per_model[label] = _corr_table(df).set_index("Antibody")[args.metric]

    combined = pd.DataFrame(per_model)
    combined = combined.reindex(sorted(combined.index)).reset_index()
    combined = combined.rename(columns={"index": "Antibody"})

    print(format_padded(combined, args.digits))

    if args.latex or args.tex_out:
        tex = to_latex(combined, args.metric, args.digits, args.texttt,
                       args.caption, args.label)
        print()
        print(tex)
        if args.tex_out:
            args.tex_out.parent.mkdir(parents=True, exist_ok=True)
            args.tex_out.write_text(tex + "\n")
            print(f"\n[corr_table_by_model] wrote {args.tex_out}", flush=True)


if __name__ == "__main__":
    main()
