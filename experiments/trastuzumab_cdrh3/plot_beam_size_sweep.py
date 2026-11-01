"""Plot p(binder) and pairwise H3 diversity as a function of beam_forward.

Reads ``results/beam_size_sweep*/all_sequences.csv`` produced by one or
more ``sweep_beam_size.py`` runs (the default 3-method realistic-guidance
sweep and the disjoint 2-method strong-oracle sweep) and writes three
figures into ``results/beam_size_sweep/``:

  - ``beam_size_pbinder.png``  — mean oracle p(binder) vs beam_forward
    (log-x), one line per method, faceted by model.
  - ``beam_size_diversity.png`` — mean per-sequence pairwise H3 distance
    to siblings vs beam_forward, same layout.
  - ``beam_size_pareto.png`` — mean p(binder) vs mean H3 diversity, with
    one path per method connecting adjacent beam sizes (smallest → largest)
    so the quality / diversity trajectory is visible directly.

In all three, sequences are pooled across realized edit counts to keep
the figures legible (the per-E trend is visible in the headline summary
figures already). Error bars in the first two are 95% bootstrap CIs of
the per-method mean within each (Model, Method, beam_forward) cell.

Use ``--models "ESM2-650M"`` to restrict to one model (single panel), and
``--slide`` for a compact, large-text version for presentations (writes with a
``_slide`` suffix so it doesn't clobber the paper figures).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR))

from analyze import (  # noqa: E402
    METHOD_PRETTY, METHOD_ORDER, MODEL_PRETTY, SAVE_DPI,
    _bootstrap_ci, _linestyle_for, _pairwise_diversity_per_seq,
)


def _color_map() -> dict[str, tuple[float, float, float]]:
    """Method colors mirror the headline summary palette: Mason-CNN
    variants take the dark shade of their (red / blue) pair, strong-
    oracle variants take the light shade."""
    paired = sns.color_palette("Paired", 12)
    return {
        "Beam":                                paired[11],  # dark brown
        "Beam + STS (Mason CNN)":              paired[5],   # dark red
        "Beam + STS (oracle)":                 paired[4],   # light red
        "Beam + STS (Mason CNN + OASis)":      paired[1],   # dark blue
        "Beam + STS (oracle + OASis)":         paired[0],   # light blue
    }


def _per_method_pairwise_mean(group: pd.DataFrame) -> float:
    """Mean of per-sequence pairwise H3 distance within ``group``."""
    h3s = group["h3"].tolist()
    if len(h3s) < 2:
        return float("nan")
    return float(np.nanmean(_pairwise_diversity_per_seq(h3s)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csvs", type=Path, nargs="+",
        default=[
            EXPERIMENT_DIR / "results" / "beam_size_sweep" / "all_sequences.csv.gz",
            EXPERIMENT_DIR / "results" / "beam_size_sweep_strong" / "all_sequences.csv.gz",
        ],
        help="One or more sweep CSVs (gz-compressed by default) to concatenate.",
    )
    parser.add_argument("--out-dir", type=Path,
                        default=EXPERIMENT_DIR / "results" / "beam_size_sweep")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Restrict to these (pretty) model names, e.g. "
                             "'ESM2-650M'. Default: all models present.")
    parser.add_argument("--slide", action="store_true",
                        help="Compact figure + larger fonts for slides; writes "
                             "with a '_slide' suffix.")
    parser.add_argument("--suffix", default=None,
                        help="Override the output filename suffix.")
    parser.add_argument("--drop-oracle", action="store_true",
                        help="Drop the strong-oracle-guided methods "
                             "(Beam + STS (oracle) and its +OASis variant).")
    args = parser.parse_args()

    frames = []
    for p in args.csvs:
        if not p.exists():
            print(f"[skip] {p} does not exist", flush=True)
            continue
        frames.append(pd.read_csv(p))
        print(f"  read {p}: {len(frames[-1])} rows", flush=True)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["edits"] > 0].copy()
    df["Method"] = df["method"].map(METHOD_PRETTY)
    df["Model"] = df["model"].map(MODEL_PRETTY).fillna(df["model"])

    if args.models:
        keep = set(args.models)
        df = df[df["Model"].isin(keep)].copy()
        if df.empty:
            raise SystemExit(f"no rows for models {sorted(keep)}; "
                             f"available: {sorted(df['Model'].unique())}")

    color_map = _color_map()
    methods = [m for m in METHOD_ORDER if m in color_map and m in df["Method"].unique()]
    if args.drop_oracle:
        methods = [m for m in methods if "oracle" not in m.lower()]

    # Slide preset: smaller panels + larger relative text.
    if args.slide:
        sns.set_theme(style="whitegrid", context="talk")
        panel_w, panel_h, legend_fs, annot_fs, marker = 4.6, 3.9, 11, 8, 5
    else:
        sns.set_theme(style="whitegrid", context="paper")
        panel_w, panel_h, legend_fs, annot_fs, marker = 5.5, 4.0, 9, 7, 4
    suffix = args.suffix if args.suffix is not None else ("_slide" if args.slide else "")

    # Shorter axis labels on slides so the larger font doesn't overrun the frame.
    if args.slide:
        xlab_bf, ylab_p, lab_div = "Beam frontier width", "Oracle p(binder)", "Pairwise H3 distance"
    else:
        xlab_bf, ylab_p = "Beam frontier width (beam_forward)", "Mean oracle p(binder)"
        lab_div = "Mean pairwise H3 distance to siblings"

    # ---- Figure 1: p(binder) vs beam_forward ---------------------------
    rows = []
    for (model, method, bf), sub in df.groupby(["Model", "Method", "beam_forward"]):
        if method not in methods:
            continue
        p = sub["score_tz_cnn_p_binder"].dropna().values
        if p.size == 0:
            continue
        mean, lo, hi = _bootstrap_ci(p)
        rows.append({"Model": model, "Method": method, "bf": int(bf),
                     "mean": mean, "lo": lo, "hi": hi, "n": int(p.size)})
    pb_df = pd.DataFrame(rows)

    n_models = pb_df["Model"].nunique()
    fig, axes = plt.subplots(1, n_models, figsize=(panel_w * n_models, panel_h),
                             sharey=True, squeeze=False)
    axes = axes.flatten()
    for ax, (model, sub) in zip(axes, pb_df.groupby("Model")):
        for method in methods:
            m = sub[sub["Method"] == method].sort_values("bf")
            if m.empty:
                continue
            ax.errorbar(
                m["bf"], m["mean"],
                yerr=[(m["mean"] - m["lo"]).clip(lower=0),
                      (m["hi"] - m["mean"]).clip(lower=0)],
                marker="o", capsize=3, label=method, markersize=marker,
                linewidth=1.6, alpha=0.8,
                linestyle=_linestyle_for(method),
                color=color_map[method],
            )
        ax.set_xscale("log")
        ax.set_xlabel(xlab_bf)
        ax.set_ylabel(ylab_p)
        ax.set_title(model)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(loc="lower right", frameon=False, fontsize=legend_fs)
    fig.tight_layout()
    fig.savefig(args.out_dir / f"beam_size_pbinder{suffix}.png",
                dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: pairwise H3 diversity vs beam_forward ----------------
    rows = []
    for (model, method, bf), sub in df.groupby(["Model", "Method", "beam_forward"]):
        if method not in methods:
            continue
        h3s = sub["h3"].tolist()
        if len(h3s) < 2:
            continue
        diversity = _pairwise_diversity_per_seq(h3s)
        diversity = diversity[~np.isnan(diversity)]
        if diversity.size == 0:
            continue
        mean, lo, hi = _bootstrap_ci(diversity)
        rows.append({"Model": model, "Method": method, "bf": int(bf),
                     "mean": mean, "lo": lo, "hi": hi, "n": int(diversity.size)})
    div_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, n_models, figsize=(panel_w * n_models, panel_h),
                             sharey=True, squeeze=False)
    axes = axes.flatten()
    for ax, (model, sub) in zip(axes, div_df.groupby("Model")):
        for method in methods:
            m = sub[sub["Method"] == method].sort_values("bf")
            if m.empty:
                continue
            ax.errorbar(
                m["bf"], m["mean"],
                yerr=[(m["mean"] - m["lo"]).clip(lower=0),
                      (m["hi"] - m["mean"]).clip(lower=0)],
                marker="o", capsize=3, label=method, markersize=marker,
                linewidth=1.6, alpha=0.8,
                linestyle=_linestyle_for(method),
                color=color_map[method],
            )
        ax.set_xscale("log")
        ax.set_xlabel(xlab_bf)
        ax.set_ylabel(lab_div)
        ax.set_title(model)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(loc="upper right", frameon=False, fontsize=legend_fs)
    fig.tight_layout()
    fig.savefig(args.out_dir / f"beam_size_diversity{suffix}.png",
                dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 3: p(binder) vs diversity, beam-size paths --------------
    pareto_df = pb_df.merge(
        div_df, on=["Model", "Method", "bf"], suffixes=("_p", "_d")
    )

    fig, axes = plt.subplots(1, n_models, figsize=(panel_w * n_models, panel_h),
                             sharey=True, squeeze=False)
    axes = axes.flatten()
    for ax, (model, sub) in zip(axes, pareto_df.groupby("Model")):
        for method in methods:
            m = sub[sub["Method"] == method].sort_values("bf")
            if m.empty:
                continue
            ax.plot(
                m["mean_d"], m["mean_p"],
                marker="o", label=method, markersize=marker,
                linewidth=1.6, alpha=0.8,
                linestyle=_linestyle_for(method),
                color=color_map[method],
            )
            for _, row in m.iterrows():
                ax.annotate(
                    str(row["bf"]),
                    xy=(row["mean_d"], row["mean_p"]),
                    xytext=(3, 3), textcoords="offset points",
                    fontsize=annot_fs, color=color_map[method], alpha=0.9,
                )
        ax.set_xlabel(lab_div)
        ax.set_ylabel(ylab_p)
        ax.set_title(model)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(loc="lower left", frameon=False, fontsize=legend_fs)
    fig.tight_layout()
    fig.savefig(args.out_dir / f"beam_size_pareto{suffix}.png",
                dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 4: OASis percentile vs diversity, beam-size paths -------
    # Same layout as the p(binder) pareto but with humanness on the y-axis.
    # Only methods with OASis percentiles populated appear (in the sweep that is
    # the OASis-guided method(s)).
    rows = []
    for (model, method, bf), sub in df.groupby(["Model", "Method", "beam_forward"]):
        if method not in methods:
            continue
        o = sub["score_oasis_percentile"].dropna().values
        if o.size == 0:
            continue
        mean, lo, hi = _bootstrap_ci(o)
        rows.append({"Model": model, "Method": method, "bf": int(bf),
                     "mean": mean, "lo": lo, "hi": hi, "n": int(o.size)})
    oa_df = pd.DataFrame(rows)
    if oa_df.empty:
        print("[oasis pareto] no OASis percentiles for the selected methods; skipping")
    else:
        oa_pareto = oa_df.merge(div_df, on=["Model", "Method", "bf"], suffixes=("_o", "_d"))
        n_o = oa_pareto["Model"].nunique()
        fig, axes = plt.subplots(1, n_o, figsize=(panel_w * n_o, panel_h),
                                 sharey=True, squeeze=False)
        axes = axes.flatten()
        for ax, (model, sub) in zip(axes, oa_pareto.groupby("Model")):
            for method in methods:
                m = sub[sub["Method"] == method].sort_values("bf")
                if m.empty:
                    continue
                ax.plot(m["mean_d"], m["mean_o"], marker="o", label=method,
                        markersize=marker, linewidth=1.6, alpha=0.8,
                        linestyle=_linestyle_for(method), color=color_map[method])
                for _, row in m.iterrows():
                    ax.annotate(str(row["bf"]), xy=(row["mean_d"], row["mean_o"]),
                                xytext=(3, 3), textcoords="offset points",
                                fontsize=annot_fs, color=color_map[method], alpha=0.9)
            ax.set_xlabel(lab_div)
            ax.set_ylabel("OASis percentile")
            ax.set_title(model)
            ax.grid(alpha=0.3, which="both")
        axes[0].legend(loc="best", frameon=False, fontsize=legend_fs)
        fig.tight_layout()
        fig.savefig(args.out_dir / f"beam_size_pareto_oasis{suffix}.png",
                    dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)

    print(f"plots written to {args.out_dir}"
          + (f" (suffix '{suffix}')" if suffix else ""))


if __name__ == "__main__":
    main()
