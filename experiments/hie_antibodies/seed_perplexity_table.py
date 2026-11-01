"""Per-residue pseudo-perplexity of the full seed (wild-type) sequence, per
antibody per model, pivoted into one table.

Each one-edit PLL parquet stores ``seed_pll``, the self-PLL of the unmutated
seed summed over every scored position (one row per one-edit variant, so the
same ``seed_pll`` repeats ``L * 19`` times per antibody, where ``L`` is the
number of scored positions). The per-residue pseudo-perplexity is

    ppl = exp(-seed_pll / L),        L = n_variants(antibody) / 19

i.e. the same masked-marginal PLL the one-edit sweep already computes, just
read off at zero edits and normalized per residue. Low ppl -> the model
assigns unusually high likelihood to that exact sequence.

Defaults to the paired full-VH+VL parquets (Sapiens, AbLang2, ESM2). Prints a
plain-text table and, with ``--latex``, a copy-pasteable LaTeX ``tabular``.

    python experiments/hie_antibodies/seed_perplexity_table.py --latex --texttt
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

N_SUBS_PER_POSITION = 19  # 20 amino acids, minus the wild-type residue itself.


def seed_perplexity(parquet: Path) -> dict[str, float]:
    """antibody -> per-residue pseudo-perplexity of the full seed sequence."""
    df = pd.read_parquet(parquet)
    out = {}
    for ab, s in df.groupby("antibody"):
        n_positions = len(s) / N_SUBS_PER_POSITION
        seed_pll = float(s["seed_pll"].iloc[0])
        out[ab] = float(np.exp(-seed_pll / n_positions))
    return out


def format_padded(data: dict[str, dict[str, float]], digits: int) -> str:
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
             caption: str | None = None, label: str = "tab:seed_perplexity") -> str:
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
            "Per-residue pseudo-perplexity of the full seed (wild-type) "
            "sequence, over the seven-antibody panel from Hie et al."
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
    p.add_argument("--digits", type=int, default=3)
    p.add_argument("--out", type=Path, default=None,
                   help="Optional path to save the plain-text table.")
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
    p.add_argument("--label", default="tab:seed_perplexity",
                   help="LaTeX \\label{...} key.")
    args = p.parse_args()

    sources = (dict(s.split("=", 1) for s in args.sources) if args.sources
               else DEFAULT_SOURCES)
    data = {label: seed_perplexity(RESULTS / f if not Path(f).is_absolute() else Path(f))
            for label, f in sources.items()}

    title = "Per-residue pseudo-perplexity of the full seed sequence"
    text = title + "\n\n" + format_padded(data, args.digits)
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
            print(f"\n[seed_perplexity_table] wrote {args.tex_out}", flush=True)


if __name__ == "__main__":
    main()
