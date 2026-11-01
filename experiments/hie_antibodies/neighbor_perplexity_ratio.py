"""Per-mutated-position pseudo-perplexity ratio: neighbor residue vs seed residue.

For each one-edit variant, the changed residue's masked-marginal log-prob equals
``A[k, mut]`` and the seed residue's is ``A[k, wt]`` (they coincide with the true
PLL at the mutated site, since masking k leaves the seed context). The per-site
log-perplexity increase is therefore ``A[k, wt] - A[k, mut] = seed_pll -
approx_pll`` (available directly in the one-edit PLL parquets). We report the
geometric-mean fold-increase per antibody per model,

    ratio = exp( mean_over_neighbors( seed_pll - approx_pll ) ),

i.e. "the typical single substitution makes the changed residue `ratio`x more
perplex than the wild-type residue it replaced". Geometric mean is used because
the arithmetic mean of per-site ratios is dominated by a few catastrophic
substitutions (e.g. Pro in a strand) and runs into the millions for AbLang2.

Defaults to the paired-chain-setting full-VH+VL parquets (Sapiens, paired
AbLang2, concat ESM2). Prints a whitespace-padded table and optionally saves it.

    python experiments/hie_antibodies/neighbor_perplexity_ratio.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS = EXPERIMENT_DIR / "results"

# Column label -> one-edit PLL parquet (paired chain setting, full VH+VL).
DEFAULT_SOURCES = {
    "Sapiens": "pll_one_edit_sapiens.parquet",
    "AbLang2": "pll_one_edit_ablang2.parquet",
    "ESM2": "pll_one_edit_esm2.parquet",
}


def neighbor_ratio(parquet: Path) -> dict[str, float]:
    """antibody -> geometric-mean per-site neighbor/seed perplexity ratio."""
    df = pd.read_parquet(parquet)
    out = {}
    for ab, s in df.groupby("antibody"):
        drop = s["seed_pll"].to_numpy() - s["approx_pll"].to_numpy()  # A[k,wt]-A[k,mut]
        out[ab] = float(np.exp(drop.mean()))
    return out


def padded_table(data: dict[str, dict[str, float]], digits: int) -> str:
    labels = list(data)
    antibodies = sorted(set().union(*(d.keys() for d in data.values())))
    header = ["Antibody"] + labels
    rows = [header] + [
        [ab] + [f"{data[lab][ab]:.{digits}f}" for lab in labels]
        for ab in antibodies
    ]
    w = [max(len(r[i]) for r in rows) for i in range(len(header))]
    return "\n".join(
        "  ".join(c.ljust(w[i]) if i == 0 else c.rjust(w[i]) for i, c in enumerate(r))
        for r in rows
    )


def _format_antibody(name: str, texttt: bool) -> str:
    if not texttt:
        return name
    # Plain text mode: swap the underscore for a hyphen rather than escaping
    # it, e.g. mab114_uca -> \texttt{mab114-uca}.
    return rf"\texttt{{{name.replace('_', '-')}}}"


def to_latex(data: dict[str, dict[str, float]], digits: int, texttt: bool = False,
             caption: str | None = None, label: str = "tab:neighbor_perplexity_ratio") -> str:
    labels = list(data)
    antibodies = sorted(set().union(*(d.keys() for d in data.values())))
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{l" + "c" * len(labels) + "}",
        r"\toprule",
        "Antibody & " + " & ".join(labels) + r" \\",
        r"\midrule",
    ]
    for ab in antibodies:
        vals = " & ".join(f"{data[lab][ab]:.{digits}f}" for lab in labels)
        lines.append(f"{_format_antibody(ab, texttt)} & {vals} \\\\")
    if caption is None:
        caption = (
            "Per-mutated-position pseudo-perplexity ratio (neighbor residue "
            "vs. seed residue, geometric mean), over the seven-antibody panel "
            "from Hie et al."
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", nargs="+", default=None,
                   help="Optional Label=parquet overrides (space-separated). "
                        "Defaults to the paired Sapiens/AbLang2/ESM2 full parquets.")
    p.add_argument("--digits", type=int, default=1)
    p.add_argument("--out", type=Path, default=None,
                   help="Optional path to save the table text.")
    p.add_argument("--latex", action="store_true",
                   help="Also print a LaTeX tabular.")
    p.add_argument("--tex-out", type=Path, default=None,
                   help="If set (implies --latex), also write the LaTeX to this path.")
    p.add_argument("--texttt", action="store_true",
                   help="Typeset the Antibody column as \\texttt{...} with "
                        "underscores swapped for hyphens, e.g. mab114_uca -> "
                        "\\texttt{mab114-uca}.")
    p.add_argument("--caption", default=None,
                   help="LaTeX \\caption text; defaults to a generic description.")
    p.add_argument("--label", default="tab:neighbor_perplexity_ratio",
                   help="LaTeX \\label{...} key.")
    args = p.parse_args()

    if args.sources:
        sources = dict(s.split("=", 1) for s in args.sources)
    else:
        sources = DEFAULT_SOURCES

    data = {label: neighbor_ratio(RESULTS / f if not Path(f).is_absolute() else Path(f))
            for label, f in sources.items()}
    title = ("Per-mutated-position perplexity ratio  "
             "(neighbor residue / seed residue, geometric mean)")
    text = title + "\n\n" + padded_table(data, args.digits)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
        print(f"\n-> {args.out}")

    if args.latex or args.tex_out:
        tex = to_latex(data, args.digits, args.texttt, args.caption, args.label)
        print()
        print(tex)
        if args.tex_out:
            args.tex_out.parent.mkdir(parents=True, exist_ok=True)
            args.tex_out.write_text(tex + "\n")
            print(f"\n[neighbor_perplexity_ratio] wrote {args.tex_out}", flush=True)


if __name__ == "__main__":
    main()
