"""Appendix figure: wall-clock cost of each sampling method, per MLM.

Reads ``results/timings.csv`` (written by ``run.py``) and plots seconds summed
over the edit-budget sweep, one panel per model. Method names sit on the y axis,
so colour is redundant with the labels and only carries family identity — matched
to the palette in :mod:`analyze` so this figure reads as part of the same set.

The x axis is log-scaled because the range spans ~70x, and every bar is directly
labelled since values are hard to read off a log axis.

``beam_oasis_guided`` / ``beam_oasis_weakguided`` are excluded. Their wall time is
dominated by OASis database lookups and therefore by how warm the on-disk cache
happened to be, not by the method: in the run this figure is built from they cost
88 s for ProteinMPNN/AbMPNN, whose candidates were already cached, against
5747-8332 s for AbLang2/ESM2, whose were not. Including them would invite a
comparison the numbers cannot support.

Only meaningful when every cell ran under the same contention, so the script
refuses to plot a mixed-``gpu_procs_at_start`` timings file.
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

# Search geometry, needed to count how many candidate sequences each method
# actually scored. Must match run.py's defaults for the sweep being plotted.
NPOS, NAA, BEAM_FORWARD, GIBBS_N = 10, 19, 20, 200
BUDGETS = range(2, 8)

# Measured per-call costs on a solo A10G (see the benchmark in the session that
# produced timings.csv). Only used to infer Gibbs's stochastic update count.
SECONDS_PER_1POS_CALL = {
    "AbLang2": 0.0140, "ESM2-650M": 0.0229,
    "ProteinMPNN": 0.0108, "AbMPNN": 0.0108,
}


def candidates_scored(method: str, model: str, seconds: float) -> tuple[float, str]:
    """Candidate sequences each method scored across the E=2..7 sweep.

    A "candidate" is a full sequence assigned a score the method could rank on.

      * Beam expands one template per ``template_logits`` call and, via the
        wild-type marginal, scores all ``NPOS * NAA`` of its single-substitution
        neighbours from that one call. Exact, from the search geometry.
      * Denoising masks E positions and each call returns conditionals at every
        still-masked position, so iteration i scores ``(E-i+1) * NAA``. Exact.
      * Gibbs masks one position per call and scores the ``NAA`` alternatives
        there, discarding all but the sampled one. Its call count is the chain's
        hitting time, which is stochastic, so it is inferred as
        ``seconds / SECONDS_PER_1POS_CALL``. Validated against independently
        measured hitting times: 44 inferred vs 42 measured updates per chain for
        ESM2, 342 vs 361 for AbMPNN — within ~5%.
    """
    if method == "beam":
        n = sum(NPOS * NAA * (1 + BEAM_FORWARD * (E - 1)) for E in BUDGETS)
        return float(n), "exact"
    if method == "denoising":
        per_chain = sum(NAA * E * (E + 1) // 2 for E in BUDGETS)
        return float(GIBBS_N * per_chain), "exact"
    if method == "gibbs":
        return round(seconds / SECONDS_PER_1POS_CALL[model]) * NAA, "inferred"
    raise ValueError(f"no candidate-count model for method {method!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timings", type=Path,
                        default=EXPERIMENT_DIR / "results" / "timings.csv")
    parser.add_argument("--out", type=Path,
                        default=EXPERIMENT_DIR / "results" / "timings_by_method.png")
    parser.add_argument("--designs", type=Path,
                        default=EXPERIMENT_DIR / "results" / "all_sequences.csv",
                        help="Used for the per-candidate figure's design counts.")
    parser.add_argument("--include-oasis-guided", action="store_true",
                        help="Plot the OASis-guided methods too. Their wall time "
                             "measures cache warmth, not method cost -- see the "
                             "module docstring before using this.")
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

    df["Method"] = df["method"].map(METHOD_PRETTY)
    df["Model"] = df["model"].map(MODEL_PRETTY).fillna(df["model"])
    total = df.groupby(["Model", "Method"], as_index=False)["seconds"].sum()

    order = [m for m in METHOD_ORDER if m in set(total["Method"])]
    colors = _color_map()
    models = sorted(total["Model"].unique())

    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(
        1, len(models), figsize=(4.1 * len(models), 0.42 * len(order) + 2.1),
        sharex=True, squeeze=False,
    )
    axes = axes.flatten()

    lo, hi = total["seconds"].min(), total["seconds"].max()
    for ax, model in zip(axes, models):
        sub = total[total["Model"] == model].set_index("Method").reindex(order)
        # Dot plot, not bars: the x axis is log, where a bar's length would be
        # measured from an arbitrary baseline and encode nothing. A dot encodes
        # position, which is well defined on a log scale.
        for i, m in enumerate(order):
            v = sub["seconds"].get(m)
            ax.axhline(i, color="0.88", linewidth=0.8, zorder=0)
            if pd.isna(v):
                continue
            ax.plot(v, i, "o", color=colors.get(m, "0.5"), markersize=8,
                    markeredgecolor="white", markeredgewidth=1.4, zorder=3)
            ax.text(v * 1.22, i, f"{v:,.0f}", va="center", ha="left",
                    fontsize=7.5, color="0.25")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order if ax is axes[0] else [])
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_xscale("log")
        ax.set_xlim(lo * 0.55, hi * 3.4)
        ax.set_title(model)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", alpha=0.3)

    device = df["device"].dropna().iloc[0] if df["device"].notna().any() else "unknown"
    fig.suptitle(
        f"Cost of one edit-budget sweep per method  ·  {device}, "
        f"{int(procs[0])} process on the GPU",
        y=1.0, fontsize=10,
    )
    fig.supxlabel("Wall-clock seconds for E=2..7 (log scale)", fontsize=9, y=-0.02)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")

    piv = total.pivot(index="Method", columns="Model", values="seconds").reindex(order)
    print("\nseconds, summed over E=2..7:")
    print(piv.round(1).to_string())

    _plot_per_candidate(df, colors, args)


def _plot_per_candidate(df: pd.DataFrame, colors: dict, args) -> None:
    """Second figure: scoring throughput vs selectivity, for the unguided trio.

    Separates two things a wall-clock total conflates. Cost *per candidate
    scored* is a throughput property of the wrapper's batching; candidates
    scored *per design emitted* is how efficiently a method converts that
    scoring into output. Beam's advantage lives entirely in the second.

    Restricted to beam / gibbs / denoising: the guided variants' wall time
    includes CNN oracle scoring, and guided Gibbs's update count cannot be
    recovered from it, so their candidate counts are not recoverable.
    """
    from analyze import METHOD_PRETTY as MP

    plain = ["beam", "gibbs", "denoising"]
    sub = df[df["method"].isin(plain)]
    if sub.empty:
        return
    tot = sub.groupby(["Model", "method"], as_index=False)["seconds"].sum()

    designs = pd.read_csv(args.designs)
    designs = designs[designs["edits"] > 0]
    from analyze import dedupe_across_budgets
    designs, _ = dedupe_across_budgets(designs)
    designs["Model"] = designs["model"].map(MODEL_PRETTY).fillna(designs["model"])
    n_out = designs.groupby(["Model", "method"]).size().rename("designs")

    rows = []
    for r in tot.itertuples():
        cand, kind = candidates_scored(r.method, r.Model, r.seconds)
        rows.append({
            "Model": r.Model, "Method": MP[r.method], "kind": kind,
            "ms_per_candidate": 1000 * r.seconds / cand,
            "cand_per_design": cand / n_out.loc[(r.Model, r.method)],
        })
    m = pd.DataFrame(rows)

    order = [MP[k] for k in ("denoising", "gibbs", "beam")]
    models = sorted(m["Model"].unique())
    panels = [
        ("cand_per_design", "Candidates scored per design emitted  (selectivity)"),
        ("ms_per_candidate", "Milliseconds per candidate scored  (throughput)"),
    ]
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(len(panels), len(models),
                             figsize=(3.5 * len(models), 4.4), squeeze=False)
    for row, (col, xlabel) in enumerate(panels):
        lo, hi = m[col].min(), m[col].max()
        for j, model in enumerate(models):
            ax = axes[row][j]
            s = m[m["Model"] == model].set_index("Method").reindex(order)
            for i, meth in enumerate(order):
                v = s[col].get(meth)
                ax.axhline(i, color="0.88", linewidth=0.8, zorder=0)
                if pd.isna(v):
                    continue
                ax.plot(v, i, "o", color=colors.get(meth, "0.5"), markersize=8,
                        markeredgecolor="white", markeredgewidth=1.4, zorder=3)
                lbl = f"{v:,.0f}" if col == "cand_per_design" else f"{v:.2f}"
                ax.text(v * 1.25, i, lbl, va="center", ha="left",
                        fontsize=7.5, color="0.25")
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels(order if j == 0 else [])
            ax.set_ylim(len(order) - 0.5, -0.5)
            ax.set_xscale("log")
            ax.set_xlim(lo * 0.5, hi * 4.0)
            ax.grid(axis="y", visible=False)
            ax.grid(axis="x", alpha=0.3)
            if row == 0:
                ax.set_title(model)
            ax.set_xlabel(xlabel, fontsize=7.5)

    fig.suptitle(
        "Beam's advantage is selectivity, not scoring throughput  ·  "
        "unguided methods, E=2..7  ·  lower is better in both rows",
        y=1.02, fontsize=10,
    )
    fig.tight_layout()
    out = args.out.with_name("timings_per_candidate.png")
    fig.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    print("\nper-candidate view (gibbs counts inferred from wall time):")
    print(m.set_index(["Model", "Method"])[
        ["cand_per_design", "ms_per_candidate", "kind"]].round(2).to_string())


if __name__ == "__main__":
    main()
