"""Plot p(binder) and pairwise H3 diversity as a function of temperature.

Reads ``results/temperature_sweep/all_sequences.csv`` produced by
``sweep_temperature.py`` and writes three figures into the same
directory:

  - ``temperature_sweep_pbinder.png``  — mean oracle p(binder) vs
    temperature (log-x), one line per method, faceted by model.
  - ``temperature_sweep_diversity.png`` — mean per-sequence pairwise H3
    distance to siblings vs temperature, same layout.
  - ``temperature_sweep_pareto.png`` — mean p(binder) vs mean H3
    diversity with one path per method connecting adjacent temperatures
    (smallest → largest) so the quality / diversity trajectory is
    visible directly.

In all three, sequences are pooled across realized edit counts to keep
the figures legible. Error bars in the first two are 95% bootstrap CIs
of the per-method mean within each (Model, Method, temperature) cell.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csvs", type=Path, nargs="+",
        default=[
            EXPERIMENT_DIR / "results" / "temperature_sweep" / "all_sequences.csv.gz",
        ],
        help="One or more sweep CSVs (gz-compressed by default) to concatenate.",
    )
    parser.add_argument("--out-dir", type=Path,
                        default=EXPERIMENT_DIR / "results" / "temperature_sweep")
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

    color_map = _color_map()
    methods = [m for m in METHOD_ORDER
               if m in color_map and m in df["Method"].unique()]

    sns.set_theme(style="whitegrid", context="paper")

    # ---- Figure 1: p(binder) vs temperature -----------------------------
    rows = []
    for (model, method, T), sub in df.groupby(["Model", "Method", "temperature"]):
        if method not in methods:
            continue
        p = sub["score_tz_cnn_p_binder"].dropna().values
        if p.size == 0:
            continue
        mean, lo, hi = _bootstrap_ci(p)
        rows.append({"Model": model, "Method": method, "T": float(T),
                     "mean": mean, "lo": lo, "hi": hi, "n": int(p.size)})
    pb_df = pd.DataFrame(rows)

    n_models = pb_df["Model"].nunique()
    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 4),
                             sharey=True, squeeze=False)
    axes = axes.flatten()
    for ax, (model, sub) in zip(axes, pb_df.groupby("Model")):
        for method in methods:
            m = sub[sub["Method"] == method].sort_values("T")
            if m.empty:
                continue
            ax.errorbar(
                m["T"], m["mean"],
                yerr=[(m["mean"] - m["lo"]).clip(lower=0),
                      (m["hi"] - m["mean"]).clip(lower=0)],
                marker="o", capsize=3, label=method,
                linewidth=1.6, alpha=0.8,
                linestyle=_linestyle_for(method),
                color=color_map[method],
            )
        ax.set_xscale("log")
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Mean oracle p(binder)")
        ax.set_title(model)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(loc="lower left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out_dir / "temperature_sweep_pbinder.png",
                dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: pairwise H3 diversity vs temperature -----------------
    rows = []
    for (model, method, T), sub in df.groupby(["Model", "Method", "temperature"]):
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
        rows.append({"Model": model, "Method": method, "T": float(T),
                     "mean": mean, "lo": lo, "hi": hi, "n": int(diversity.size)})
    div_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 4),
                             sharey=True, squeeze=False)
    axes = axes.flatten()
    for ax, (model, sub) in zip(axes, div_df.groupby("Model")):
        for method in methods:
            m = sub[sub["Method"] == method].sort_values("T")
            if m.empty:
                continue
            ax.errorbar(
                m["T"], m["mean"],
                yerr=[(m["mean"] - m["lo"]).clip(lower=0),
                      (m["hi"] - m["mean"]).clip(lower=0)],
                marker="o", capsize=3, label=method,
                linewidth=1.6, alpha=0.8,
                linestyle=_linestyle_for(method),
                color=color_map[method],
            )
        ax.set_xscale("log")
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Mean pairwise H3 distance to siblings")
        ax.set_title(model)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out_dir / "temperature_sweep_diversity.png",
                dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 3: p(binder) vs diversity, temperature paths ------------
    pareto_df = pb_df.merge(
        div_df, on=["Model", "Method", "T"], suffixes=("_p", "_d")
    )

    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 4),
                             sharey=True, squeeze=False)
    axes = axes.flatten()
    for ax, (model, sub) in zip(axes, pareto_df.groupby("Model")):
        for method in methods:
            m = sub[sub["Method"] == method].sort_values("T")
            if m.empty:
                continue
            ax.plot(
                m["mean_d"], m["mean_p"],
                marker="o", label=method,
                linewidth=1.6, alpha=0.8,
                linestyle=_linestyle_for(method),
                color=color_map[method],
            )
            for _, row in m.iterrows():
                ax.annotate(
                    f"{row['T']:g}",
                    xy=(row["mean_d"], row["mean_p"]),
                    xytext=(3, 3), textcoords="offset points",
                    fontsize=7, color=color_map[method], alpha=0.9,
                )
        ax.set_xlabel("Mean pairwise H3 distance to siblings")
        ax.set_ylabel("Mean oracle p(binder)")
        ax.set_title(model)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(loc="lower left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out_dir / "temperature_sweep_pareto.png",
                dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"plots written to {args.out_dir}")


if __name__ == "__main__":
    main()
