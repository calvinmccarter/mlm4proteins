"""Compact grouped-by-model bar version of timings_by_method.png.

Same data and filtering as ``plot_timings.py`` (reads ``results/timings.csv``,
excludes the OASis-guided methods by default -- see that module's docstring
for why -- and refuses a mixed-``gpu_procs_at_start`` file), plus the
strong-oracle-guided methods (Beam + STS (oracle), Gibbs + PoE (oracle)) are
also excluded by default here: they're tautologically dominant on the same
oracle used to score them, so they add two more bars without a fair
comparison behind them. A different picture besides: one panel per model,
one horizontal bar per method within a panel,
on a *raw* (linear, not log) wallclock-seconds axis. Each panel gets its own
x-scale rather than a shared one -- the ~70x spread that makes a single shared
linear axis unreadable (Beam ~16-59s vs. Gibbs+PoE on AbMPNN ~1107s) is mostly
*across* models, not within one, so a per-model raw scale is a reasonable
compromise (worst within-model spread is AbMPNN's ~25x). No per-bar value
labels and no device/process annotation -- small and plain on purpose, sized
to sit comfortably within a single page.

    python experiments/trastuzumab_cdrh3/plot_timings_bar.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR))

from analyze import METHOD_ORDER, METHOD_PRETTY, MODEL_PRETTY, SAVE_DPI  # noqa: E402
from extra_diversity import _color_map  # noqa: E402

OASIS_GUIDED = {"beam_oasis_guided", "beam_oasis_weakguided"}
STRONG_ORACLE_GUIDED = {"beam_guided", "gibbs_guided"}
MODEL_ORDER = ["ESM2-650M", "AbLang2", "ProteinMPNN", "AbMPNN"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timings", type=Path,
                        default=EXPERIMENT_DIR / "results" / "timings.csv")
    parser.add_argument("--out", type=Path,
                        default=EXPERIMENT_DIR / "results" / "timings_by_method_bar.png")
    parser.add_argument("--include-oasis-guided", action="store_true",
                        help="Plot the OASis-guided methods too. Their wall time "
                             "measures cache warmth, not method cost -- see "
                             "plot_timings.py's module docstring before using this.")
    parser.add_argument("--include-oracle-guided", action="store_true",
                        help="Plot the strong-oracle-guided methods too (Beam + STS "
                             "(oracle), Gibbs + PoE (oracle)) -- tautologically "
                             "dominant on the same oracle used to score them, so "
                             "excluded by default alongside the OASis-guided pair.")
    args = parser.parse_args()

    df = pd.read_csv(args.timings)
    procs = sorted(df["gpu_procs_at_start"].dropna().unique())
    if len(procs) > 1:
        raise SystemExit(
            f"timings span multiple contention levels (gpu_procs_at_start={procs}); "
            "wall times are not comparable across them"
        )
    if not args.include_oasis_guided:
        df = df[~df["method"].isin(OASIS_GUIDED)]
    if not args.include_oracle_guided:
        df = df[~df["method"].isin(STRONG_ORACLE_GUIDED)]

    df["Method"] = df["method"].map(METHOD_PRETTY)
    df["Model"] = df["model"].map(MODEL_PRETTY).fillna(df["model"])
    total = df.groupby(["Model", "Method"], as_index=False)["seconds"].sum()

    order = [m for m in METHOD_ORDER if m in set(total["Method"])]
    models = [m for m in MODEL_ORDER if m in set(total["Model"])]
    colors = _color_map()

    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(
        1, len(models), figsize=(2.2 * len(models), 0.42 * len(order) + 1.1),
        squeeze=False,  # no sharex: each panel gets its own raw-seconds scale
    )
    axes = axes.flatten()

    for ax, model in zip(axes, models):
        sub = total[total["Model"] == model].set_index("Method").reindex(order)
        for i, m in enumerate(order):
            v = sub["seconds"].get(m)
            if pd.isna(v):
                continue
            ax.barh(i, v, height=0.62, color=colors.get(m, "0.5"), zorder=3)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order if ax is axes[0] else [], fontsize=10)
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_title(model, fontsize=11)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", alpha=0.3)
        ax.tick_params(axis="x", labelsize=8)
        ax.margins(x=0.08)

    fig.supxlabel("Wallclock seconds", fontsize=10, y=-0.04)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
