"""LM-vs-DASM Spearman table: how well each LM's approximate and true PLL tracks
the DASM selection score, per antibody, with a grouped two-line header.

Reads the ``dasm_vs_<model>.parquet`` files (produced by dasm_vs_pll.py -- each
adds a ``dasm_score`` column to a one-edit PLL parquet) and reports, per antibody
per model, Spearman(dasm_score, approx_pll) and Spearman(dasm_score, true_pll).

The Approx-vs-True gap within a model is a model-external readout of that LM's
WT-marginal approximation quality: ~0 where approx==true (Sapiens, ESM2), and
large -- with True going negative on the near-germline UCAs -- for paired AbLang2.

    python experiments/hie_antibodies/lm_vs_dasm_table.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS = EXPERIMENT_DIR / "results"

# Column-group label -> dasm_vs_* parquet (paired chain setting).
DEFAULT_MODELS = {
    "Sapiens": "dasm_vs_sapiens.parquet",
    "AbLang2": "dasm_vs_ablang2.parquet",
    "ESM2": "dasm_vs_esm2.parquet",
}
SUBS = ("Approx", "True")


def spearmans(parquet: Path) -> dict[str, tuple[float, float]]:
    """antibody -> (Spearman(dasm, approx_pll), Spearman(dasm, true_pll))."""
    df = pd.read_parquet(parquet)
    out = {}
    for ab, s in df.groupby("antibody"):
        out[ab] = (float(spearmanr(s.dasm_score, s.approx_pll).statistic),
                   float(spearmanr(s.dasm_score, s.true_pll).statistic))
    return out


def render(per: dict[str, dict[str, tuple[float, float]]], digits: int) -> str:
    models = list(per)
    antibodies = sorted(set().union(*(d.keys() for d in per.values())))
    fmt = lambda v: f"{v:.{digits}f}"

    ab_w = max(len("Antibody"), *(len(a) for a in antibodies))
    sub_w = max(len(s) for s in SUBS)
    for m in models:
        for ab in antibodies:
            for v in per[m][ab]:
                sub_w = max(sub_w, len(fmt(v)))
    gut = "  "
    group_w = sub_w * 2 + len(gut)

    line1 = " " * ab_w + gut + gut.join(m.center(group_w) for m in models)
    line2 = ("Antibody".ljust(ab_w) + gut
             + gut.join(gut.join(s.rjust(sub_w) for s in SUBS) for _ in models))
    lines = [line1, line2]
    for ab in antibodies:
        cells = [ab.ljust(ab_w)]
        cells += [gut.join(fmt(v).rjust(sub_w) for v in per[m][ab]) for m in models]
        lines.append(gut.join(cells))
    return "\n".join(lines)


def _format_antibody(name: str, texttt: bool) -> str:
    if not texttt:
        return name
    # Plain text mode: swap the underscore for a hyphen rather than escaping
    # it, e.g. mab114_uca -> \texttt{mab114-uca}.
    return rf"\texttt{{{name.replace('_', '-')}}}"


def to_latex(per: dict[str, dict[str, tuple[float, float]]], digits: int,
             texttt: bool = False, caption: str | None = None,
             label: str = "tab:lm_vs_dasm") -> str:
    models = list(per)
    antibodies = sorted(set().union(*(d.keys() for d in per.values())))
    fmt = lambda v: f"{v:.{digits}f}"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{l" + "cc" * len(models) + "}",
        r"\toprule",
        "& " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{m}}}" for m in models) + r" \\",
        " ".join(
            rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(models))
        ),
        "Antibody & " + " & ".join(" & ".join(SUBS) for _ in models) + r" \\",
        r"\midrule",
    ]
    for ab in antibodies:
        vals = " & ".join(fmt(v) for m in models for v in per[m][ab])
        lines.append(f"{_format_antibody(ab, texttt)} & {vals} \\\\")
    if caption is None:
        caption = (
            "Spearman correlation of each model's approximate and true PLL "
            "against the DASM selection score, per antibody, over the "
            "seven-antibody panel from Hie et al."
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
    p.add_argument("--models", nargs="+", default=None,
                   help="Optional Label=parquet overrides. Defaults to the paired "
                        "Sapiens/AbLang2/ESM2 dasm_vs_* parquets.")
    p.add_argument("--digits", type=int, default=2)
    p.add_argument("--out", type=Path, default=None)
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
    p.add_argument("--label", default="tab:lm_vs_dasm",
                   help="LaTeX \\label{...} key.")
    args = p.parse_args()

    src = (dict(s.split("=", 1) for s in args.models) if args.models
           else DEFAULT_MODELS)
    per = {label: spearmans(RESULTS / f if not Path(f).is_absolute() else Path(f))
           for label, f in src.items()}
    title = "LM-vs-DASM Spearman (paired chain setting)"
    text = title + "\n\n" + render(per, args.digits)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
        print(f"\n-> {args.out}")

    if args.latex or args.tex_out:
        tex = to_latex(per, args.digits, args.texttt, args.caption, args.label)
        print()
        print(tex)
        if args.tex_out:
            args.tex_out.parent.mkdir(parents=True, exist_ok=True)
            args.tex_out.write_text(tex + "\n")
            print(f"\n[lm_vs_dasm_table] wrote {args.tex_out}", flush=True)


if __name__ == "__main__":
    main()
