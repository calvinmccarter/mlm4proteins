"""Scatter the approximate and Gumbel-perturbed PLLs against the true PLL.

Reads the combined ``results/pll_one_edit_h3.parquet`` (produced by
``pll_one_edit_h3.py``, one ``model_key`` per model) and writes a figure with
one row per model and two columns:

  - left column:  approximate (wild-type-marginal) PLL vs true PLL
  - right column: Gumbel-perturbed PLL vs true PLL

Every panel shares the true PLL on the x-axis and draws the y = x identity
line. Limits are shared within a model's row (so the two panels are directly
comparable) but not across rows, because the structure-conditioned model's PLL
lives on a different scale than the mask-marginal MLMs. Each panel reports both
the Pearson and Spearman correlation of its y-quantity against the true PLL.
One point per single-edit CDR-H3 variant.

Also writes a LaTeX ``tabular`` (default alongside the figure, same stem with
a ``.tex`` extension) with one row per model and the same four correlations
(Approx Pearson/Spearman, Gumbel Pearson/Spearman) shown in the panels.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

EXPERIMENT_DIR = Path(__file__).resolve().parent
SAVE_DPI = 240

# Preferred row order and pretty labels.
MODEL_ORDER = ["esm2", "ablang2", "proteinmpnn", "abmpnn"]
MODEL_PRETTY = {
    "esm2": "ESM2-650M",
    "ablang2": "AbLang2",
    "proteinmpnn": "ProteinMPNN",
    "abmpnn": "AbMPNN",
}

COLUMNS = [
    ("approx_pll", "Approximate PLL"),
    ("gumbel_pll", "Gumbel-perturbed PLL"),
]


def _write_latex_table(summary: list[dict], out_path: Path, tau: float) -> None:
    """Write the Approx/Gumbel Pearson+Spearman summary as a LaTeX tabular."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Approximation} & \multicolumn{2}{c}{Approximation + Gumbel} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r"Model & Pearson & Spearman & Pearson & Spearman \\",
        r"\midrule",
    ]
    for row in summary:
        lines.append(
            f"{row['Model']} & {row['Approx_Pearson']:.3f} & "
            f"{row['Approx_Spearman']:.3f} & {row['Gumbel_Pearson']:.3f} & "
            f"{row['Gumbel_Spearman']:.3f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{Approximate-vs-true PLL correlation over all 190 one-edit "
        rf"CDR-H3 variants of Trastuzumab ($\tau={tau:g}$).}}",
        r"\label{tab:pll_one_edit_h3}",
        r"\end{table}",
    ]
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet", type=Path,
        default=EXPERIMENT_DIR / "results" / "pll_one_edit_h3.parquet",
    )
    parser.add_argument(
        "--out", type=Path,
        default=EXPERIMENT_DIR / "results" / "pll_one_edit_h3.png",
    )
    parser.add_argument(
        "--tex-out", type=Path, default=None,
        help="Where to write the LaTeX summary table (default: --out with a "
             ".tex extension instead of .png).",
    )
    args = parser.parse_args()
    tex_out = args.tex_out or args.out.with_suffix(".tex")

    df = pd.read_parquet(args.parquet)
    # `model_key` is written by the current script; fall back to `model` for
    # older single-model parquets.
    key_col = "model_key" if "model_key" in df.columns else "model"
    present = list(df[key_col].unique())
    models = [m for m in MODEL_ORDER if m in present]
    models += [m for m in present if m not in models]  # keep any extras
    n = len(models)

    fig, axes = plt.subplots(n, 2, figsize=(6.6, 3.3 * n), squeeze=False)
    summary = []
    for r, model_key in enumerate(models):
        sub = df[df[key_col] == model_key]
        x = sub["true_pll"].to_numpy()
        row_summary = {"Model": MODEL_PRETTY.get(model_key, model_key)}

        # Shared limits across this model's row (span true + both y-quantities).
        row_vals = sub[["true_pll", "approx_pll", "gumbel_pll"]].to_numpy()
        lo, hi = float(row_vals.min()), float(row_vals.max())
        pad = 0.03 * (hi - lo)
        line = np.array([lo - pad, hi + pad])

        for c, (col, col_label) in enumerate(COLUMNS):
            ax = axes[r][c]
            y = sub[col].to_numpy()

            ax.plot(line, line, color="0.5", lw=1.0, ls="--", zorder=1, label="y = x")
            ax.scatter(x, y, s=22, alpha=0.7, edgecolor="none",
                       color="#3b6fb0", zorder=2)

            pear = pearsonr(x, y).statistic
            spear = spearmanr(x, y).statistic
            tag = "Approx" if col == "approx_pll" else "Gumbel"
            row_summary[f"{tag}_Pearson"] = float(pear)
            row_summary[f"{tag}_Spearman"] = float(spear)
            ax.text(
                0.97, 0.03,
                f"Pearson: {pear:.3f}\nSpearman: {spear:.3f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7"),
            )

            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(line)
            ax.set_ylim(line)
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.legend(loc="upper left", frameon=False, fontsize=8)

            if r == 0:
                ax.set_title(col_label)
            if r == n - 1:
                ax.set_xlabel("True PLL")
            if c == 0:
                # Left column carries the model name as the row label.
                ax.set_ylabel(f"{MODEL_PRETTY.get(model_key, model_key)}")

        summary.append(row_summary)

    fig.tight_layout()
    fig.savefig(args.out, dpi=SAVE_DPI, bbox_inches="tight")
    print(f"wrote {args.out}")

    tau = float(df["temperature"].iloc[0]) if "temperature" in df.columns else float("nan")
    _write_latex_table(summary, tex_out, tau)


if __name__ == "__main__":
    main()
