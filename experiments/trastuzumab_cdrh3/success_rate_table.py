"""Fraction of realized-edit-E sequences with oracle p(binder) > 0.5, one row
per model configuration and one column per method.

Reads ``results/all_sequences.csv`` (the holo/with-antigen sweep) and,
optionally, an apo (no-antigen) sweep CSV in the same schema -- see the
"Apo (no-antigen) appendix figures" section of the top-level README for how
to regenerate ``results/apo/all_sequences.csv`` (it's gitignored, so it isn't
shipped with the repo). ProteinMPNN/AbMPNN get both a "(holo)" row from the
main CSV and a "(apo)" row from ``--apo-csv`` if given; ESM2-650M and AbLang2
have no structure input so they get a single row each.

Column order matches the five "realistic" (Mason-CNN-guided, non-tautological)
methods: Beam, Gibbs, Beam+STS(Mason CNN) ("Beam:Mason"), Gibbs+PoE(Mason CNN)
("Gibbs:Mason"), Beam+STS(Mason CNN + OASis) ("Beam:Mason+OASis").

Prints a plain-text table and, with ``--latex``, a copy-pasteable LaTeX
``tabular`` at ``--digits`` decimal places (default 3).

    python experiments/trastuzumab_cdrh3/success_rate_table.py --latex --digits 3
    python experiments/trastuzumab_cdrh3/success_rate_table.py --latex \
      --apo-csv results/apo/all_sequences.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent

METHODS = ["beam", "gibbs", "beam_weakguided", "gibbs_weakguided", "beam_oasis_weakguided"]
METHOD_PRETTY = {
    "beam": "Beam",
    "gibbs": "Gibbs",
    "beam_weakguided": "Beam:Mason",
    "gibbs_weakguided": "Gibbs:Mason",
    "beam_oasis_weakguided": "Beam:Mason+OASis",
}

# (row label, model key, source) -- source is "holo" (main CSV) or "apo" (--apo-csv).
ROWS = [
    ("ESM2-650M", "facebook/esm2_t33_650M_UR50D", "holo"),
    ("AbLang2", "ablang2", "holo"),
    ("ProteinMPNN (holo)", "proteinmpnn", "holo"),
    ("ProteinMPNN (apo)", "proteinmpnn_apo", "apo"),
    ("AbMPNN (holo)", "abmpnn", "holo"),
    ("AbMPNN (apo)", "abmpnn_apo", "apo"),
]


def success_rates(df: pd.DataFrame, model_key: str, edits: int) -> dict[str, float]:
    """method -> fraction of rows with p(binder) > 0.5, at the given edit count."""
    sub = df[(df["model"] == model_key) & (df["edits"] == edits) & (df["method"].isin(METHODS))]
    return {
        m: float((s["score_tz_cnn_p_binder"] > 0.5).mean())
        for m, s in sub.groupby("method")
    }


def format_padded(table: list[tuple[str, dict[str, float]]], digits: int) -> str:
    header = ["Configuration"] + [METHOD_PRETTY[m] for m in METHODS]
    rows = [header] + [
        [label] + [f"{rates.get(m, float('nan')):.{digits}f}" for m in METHODS]
        for label, rates in table
    ]
    w = [max(len(r[i]) for r in rows) for i in range(len(header))]
    return "\n".join(
        "  ".join(c.ljust(w[i]) if i == 0 else c.rjust(w[i]) for i, c in enumerate(r))
        for r in rows
    )


def to_latex(table: list[tuple[str, dict[str, float]]], digits: int,
             edits: int, caption: str | None = None,
             label: str = "tab:success_rate_e") -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{l" + "c" * len(METHODS) + "}",
        r"\toprule",
        "Configuration & " + " & ".join(METHOD_PRETTY[m] for m in METHODS) + r" \\",
        r"\midrule",
    ]
    for label_, rates in table:
        vals = " & ".join(f"{rates.get(m, float('nan')):.{digits}f}" for m in METHODS)
        lines.append(f"{label_} & {vals} \\\\")
    if caption is None:
        caption = (
            f"Fraction of one-edit-CDR-H3 variants with oracle p(binder) $>$ 0.5 "
            f"at $E={edits}$ realized edits, with and without the HER2 antigen "
            "(subdomain IV) present during structure-conditioned design."
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
    p.add_argument("--csv", type=Path, default=EXPERIMENT_DIR / "results" / "all_sequences.csv",
                    help="Holo (with-antigen) sweep, e.g. results/all_sequences.csv.")
    p.add_argument("--apo-csv", type=Path, default=None,
                    help="Apo (no-antigen) sweep, e.g. results/apo/all_sequences.csv. "
                         "Gitignored / not shipped with the repo -- see README to regenerate. "
                         "If omitted, the two (apo) rows are dropped from the table.")
    p.add_argument("--edits", type=int, default=6, help="Realized edit count E to filter on.")
    p.add_argument("--digits", type=int, default=3)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--latex", action="store_true")
    p.add_argument("--tex-out", type=Path, default=None)
    p.add_argument("--caption", default=None)
    p.add_argument("--label", default="tab:success_rate_e")
    args = p.parse_args()

    holo = pd.read_csv(args.csv)
    apo = pd.read_csv(args.apo_csv) if args.apo_csv and args.apo_csv.is_file() else None
    if args.apo_csv and apo is None:
        print(f"[success_rate_table] --apo-csv {args.apo_csv} not found; "
              "dropping (apo) rows. See README's apo appendix section to regenerate.")

    table = []
    for row_label, model_key, source in ROWS:
        if source == "apo":
            if apo is None:
                continue
            rates = success_rates(apo, model_key, args.edits)
        else:
            rates = success_rates(holo, model_key, args.edits)
        table.append((row_label, rates))

    text = format_padded(table, args.digits)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
        print(f"\n-> {args.out}")

    if args.latex or args.tex_out:
        tex = to_latex(table, args.digits, args.edits, args.caption, args.label)
        print()
        print(tex)
        if args.tex_out:
            args.tex_out.parent.mkdir(parents=True, exist_ok=True)
            args.tex_out.write_text(tex + "\n")
            print(f"\n[success_rate_table] wrote {args.tex_out}", flush=True)


if __name__ == "__main__":
    main()
