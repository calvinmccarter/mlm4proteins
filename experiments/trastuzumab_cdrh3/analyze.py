"""Plot summaries of the trastuzumab CDR-H3 experiment.

Reads ``results/all_sequences.csv`` and produces:
  - results/method_vs_E.png        : mean oracle p(binder) by method as a function of edit budget
  - results/success_rate_vs_E.png  : fraction with p > 0.5 by method as a function of edit budget
  - results/p_binder_violin.png    : full distributions per (model, method, E)
  - results/diversity.png          : unique H3 count per method
  - results/edits_distribution.png : edit-count histograms
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


METHOD_PRETTY = {
    "beam": "Beam",
    "gibbs": "Gibbs",
    "denoising": "Denoising",
    "beam_guided": "Beam + STS (oracle)",
    "beam_weakguided": "Beam + STS (Mason CNN)",
    "gibbs_guided": "Gibbs + PoE (oracle)",
    "gibbs_weakguided": "Gibbs + PoE (Mason CNN)",
    "beam_oasis_guided": "Beam + STS (oracle + OASis)",
    "beam_oasis_weakguided": "Beam + STS (Mason CNN + OASis)",
}
METHOD_ORDER = [
    "Denoising",
    "Gibbs",
    "Beam",
    "Gibbs + PoE (Mason CNN)",
    "Beam + STS (Mason CNN)",
    "Beam + STS (Mason CNN + OASis)",
    "Gibbs + PoE (oracle)",
    "Beam + STS (oracle)",
    "Beam + STS (oracle + OASis)",
]
MODEL_PRETTY = {
    "facebook/esm2_t33_650M_UR50D": "ESM2-650M",
    "ablang2": "AbLang2",
    "proteinmpnn": "ProteinMPNN",
    "abmpnn": "AbMPNN",
    # Apo variants: same weights and backbone, but the HER2 chain dropped from
    # the fixed context, so the model scores H3 without the epitope. Held in a
    # separate CSV (results/apo/) and rendered only as their own appendix
    # summary figures -- they are deliberately absent from the main figure set.
    "proteinmpnn_apo": "ProteinMPNN-apo",
    "abmpnn_apo": "AbMPNN-apo",
}

# Method partitioning by guidance source. We split methods into two
# overlapping sets that share the unguided baselines (denoising, gibbs, beam)
# but differ on the supervised oracle used for guidance:
#
#   - Strong-oracle methods use ``cnn.pt`` (the same model used to score the
#     output) — tautologically dominant on binding metrics.
#   - Mason-CNN methods use ``cnn_mason.pt`` (trained on the smaller Mason
#     dataset) — realistic surrogate guidance.
#
# Plots that show "what's actually achievable" use only the Mason variant
# (``STRONG_ORACLE_GUIDED_METHODS`` are filtered out). The per-model summary
# figure is the exception: ``summary_<model>.png`` renders both variants in a
# single figure, pairing each Mason-CNN method with its strong-oracle
# counterpart in a lighter shade of the same hue.
STRONG_ORACLE_GUIDED_METHODS = {
    "beam_guided",
    "gibbs_guided",
    "beam_oasis_guided",
}
MASON_GUIDED_METHODS = {
    "beam_weakguided",
    "gibbs_weakguided",
    "beam_oasis_weakguided",
}

# Role mapping: maps each pretty-printed method name to a "role" identifier
# so that the Mason-CNN and strong-oracle variants of the *same* method play
# (e.g. "Beam + STS guided") share a color across both summary figures.
# The role order below is the order in which colors are assigned from the
# palette.
METHOD_ROLES = {
    "Denoising":                              "denoising",
    "Gibbs":                                  "gibbs",
    "Beam":                                   "beam",
    "Gibbs + PoE (Mason CNN)":                "gibbs_poe",
    "Gibbs + PoE (oracle)":                   "gibbs_poe",
    "Beam + STS (Mason CNN)":                 "beam_sts",
    "Beam + STS (oracle)":                    "beam_sts",
    "Beam + STS (Mason CNN + OASis)":         "beam_sts_oasis",
    "Beam + STS (oracle + OASis)":            "beam_sts_oasis",
}
ROLE_ORDER = [
    "denoising", "gibbs", "beam",
    "gibbs_poe", "beam_sts", "beam_sts_oasis",
]


SAVE_DPI = 240

# OASis percentile of the unmodified Trastuzumab VH (full chain, medium
# 50%-subjects threshold). Computed once via BioPhi and stamped here as a
# constant so plots don't need a live OASis lookup at render time.
TRASTUZUMAB_SEED_OASIS = 0.2455


def _linestyle_for(method_pretty: str) -> str:
    """Family-level linestyle: beam=solid, gibbs=dashed, denoising=dotted.

    Color (per method) and linestyle (per family) are mutually distinct
    visual axes, so the legend communicates both "which specific method"
    and "which methodology family" at a glance.
    """
    if method_pretty.startswith("Denoising"):
        return ":"
    if method_pretty.startswith("Gibbs"):
        return "--"
    if method_pretty.startswith("Beam"):
        return "-"
    return "-"


def _pareto_frontier(oasis: np.ndarray, p_binder: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the upper-right Pareto frontier of (oasis, p_binder).

    A point is on the frontier if no other point has strictly greater oasis
    AND strictly greater p_binder. Returns the frontier sorted by oasis
    ascending, with p_binder monotonically non-increasing — appropriate for
    a step plot showing the maximum achievable p_binder at each OASis level
    (or stricter humanness).
    """
    if len(oasis) == 0:
        return np.empty(0), np.empty(0)
    # Walk from highest oasis down; keep points whose p_binder strictly
    # exceeds anything seen so far.
    order = np.argsort(-oasis, kind="stable")
    fx, fy = [], []
    best_p = -np.inf
    for i in order:
        if p_binder[i] > best_p:
            fx.append(oasis[i])
            fy.append(p_binder[i])
            best_p = p_binder[i]
    return np.array(fx[::-1]), np.array(fy[::-1])


def _pairwise_diversity_per_seq(h3s: list[str]) -> np.ndarray:
    """For each H3 in ``h3s`` return the mean Hamming distance to all other H3s.

    Pairs are formed within the same group (e.g. (Model, Method, edits)). All
    H3s must share the same length. Returns ``np.array`` of length ``len(h3s)``;
    all NaN when there are fewer than two sequences in the group.
    """
    n = len(h3s)
    if n < 2:
        return np.full(n, np.nan)
    arr = np.array([list(s) for s in h3s], dtype="U1")
    diff = (arr[:, None, :] != arr[None, :, :]).sum(axis=-1)
    np.fill_diagonal(diff, 0)
    return diff.sum(axis=1) / (n - 1)


def _bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                  seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    boot = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)


def _render_pbinder_vs_edits(ax, df_for_model: pd.DataFrame,
                              methods: list[str],
                              color_map: dict | None = None) -> None:
    """Render the mean p_binder vs realized edits line plot on ``ax``."""
    rows = []
    for (method, E), sub in df_for_model.groupby(["Method", "edits"]):
        p = sub["score_tz_cnn_p_binder"].dropna().values
        if p.size == 0:
            continue
        mean, lo, hi = _bootstrap_ci(p)
        rows.append({"method": method, "edits": E, "mean_p": mean,
                     "lo": lo, "hi": hi})
    perf = pd.DataFrame(rows)
    for method in methods:
        m = perf[perf["method"] == method].sort_values("edits")
        if m.empty:
            continue
        kwargs = dict(
            marker="o", capsize=3, label=method, linewidth=1.6,
            linestyle=_linestyle_for(method), alpha=0.8,
        )
        if color_map is not None and method in color_map:
            kwargs["color"] = color_map[method]
        ax.errorbar(
            m["edits"], m["mean_p"],
            yerr=[(m["mean_p"] - m["lo"]).clip(lower=0),
                  (m["hi"] - m["mean_p"]).clip(lower=0)],
            **kwargs,
        )
    ax.set_xlabel("Edits from seed")
    ax.set_ylabel("Mean oracle p(binder)")
    ax.set_ylim(0, 1.02)
    if not perf.empty:
        ax.set_xticks(range(int(perf["edits"].min()),
                            int(perf["edits"].max()) + 1))
    ax.grid(alpha=0.3)


def _render_pbinder_vs_oasis(
    ax, df_subset: pd.DataFrame, methods: list[str], color_map: dict,
    statistic: str = "mean", min_n: int = 3,
    x_range: tuple[float, float] | None = None,
) -> None:
    """For each method, line-plot a per-OASis-value summary statistic of
    p(binder) — the mean or 90th-percentile across all sequences with that
    (quantized) OASis percentile, pooling across E. Skips OASis buckets
    with fewer than ``min_n`` samples per method.
    """
    rows = []
    for method, sub in df_subset.groupby("Method"):
        if method not in methods:
            continue
        for oasis_val, group in sub.groupby("score_oasis_percentile"):
            n = len(group)
            if n < min_n:
                continue
            if statistic == "mean":
                y = float(group["score_tz_cnn_p_binder"].mean())
            elif statistic == "p90":
                y = float(group["score_tz_cnn_p_binder"].quantile(0.9))
            else:
                raise ValueError(f"unknown statistic {statistic!r}")
            rows.append({"Method": method, "oasis": float(oasis_val),
                         "y": y, "n": n})
    stat_df = pd.DataFrame(rows)

    # Render in reversed order so the first-listed methods (the unguided
    # baselines) end up on top of the visually-dominant guided lines —
    # otherwise they get drawn first and covered.
    for method in reversed(list(methods)):
        m = stat_df[stat_df["Method"] == method].sort_values("oasis")
        if m.empty:
            continue
        ax.plot(m["oasis"], m["y"],
                color=color_map[method],
                linestyle=_linestyle_for(method),
                marker="o", markersize=4,
                linewidth=1.6, alpha=0.8, label=method)
    ylab = "Mean p(binder)" if statistic == "mean" else "p90 p(binder)"
    ax.set_xlabel("OASis percentile (humanness)")
    ax.set_ylabel(ylab)
    if x_range is not None:
        ax.set_xlim(*x_range)
    ax.set_xticks([0.20, 0.25, 0.30, 0.35])
    if statistic == "p90":
        # Zoom the p90 axis: every method sits in [0.95, 1.0] for most
        # OASis values, so the default [0, 1] range crushes the entire
        # signal against the top edge.
        ax.set_ylim(0.94, 1.005)
    else:
        ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)


def _render_pareto(ax, df_at_E: pd.DataFrame, methods: list[str],
                   color_map: dict, jitter_scale: float = 0.0015,
                   x_range: tuple[float, float] | None = None) -> None:
    """Render OASis × p_binder Pareto frontier plus faded scatter on ``ax``."""
    rng = np.random.default_rng(0)
    for method in methods:
        rows = df_at_E[df_at_E["Method"] == method]
        if rows.empty:
            continue
        oasis = rows["score_oasis_percentile"].values
        pb = rows["score_tz_cnn_p_binder"].values
        jitter = rng.uniform(-jitter_scale, jitter_scale, size=oasis.shape)
        ax.scatter(oasis + jitter, pb, s=3, alpha=0.12,
                   color=color_map[method], linewidth=0)
    # Frontier lines: draw in reversed order so the first methods listed
    # (the unguided baselines) end up on top instead of being hidden by
    # the dominant guided lines.
    for method in reversed(list(methods)):
        rows = df_at_E[df_at_E["Method"] == method]
        if rows.empty:
            continue
        fx, fy = _pareto_frontier(
            rows["score_oasis_percentile"].values,
            rows["score_tz_cnn_p_binder"].values,
        )
        if fx.size:
            ax.plot(fx, fy, color=color_map[method], linewidth=1.8,
                    linestyle=_linestyle_for(method), alpha=0.8,
                    label=method)
            ax.scatter(fx, fy, color=color_map[method], s=14,
                       edgecolor="white", linewidth=0.5, alpha=0.8,
                       zorder=3)
    ax.set_xlabel("OASis percentile (humanness)")
    ax.set_ylabel("Oracle p(binder)")
    if x_range is not None:
        ax.set_xlim(*x_range)
    ax.set_xticks([0.20, 0.25, 0.30, 0.35])
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)


def dedupe_across_budgets(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Collapse rows that repeat the same design across edit budgets.

    ``run.py`` sweeps ``--edit-budgets 2 3 ... 7`` as separate cells, but
    ``beam_search`` seeds its RNG per call and ``edit_budget`` only controls the
    loop count — so a run at budget E is an exact *prefix* of the run at E+1 and
    emits the same sequence once per budget that reaches it. On the committed
    results each ``(model, method, vh)`` recurs under ~2.4 budgets, and e.g. the
    190 one-edit ESM2 beam designs at E=2 are a 190/190 subset of those at E=7.

    Every plot here groups by *realized* edits, not by budget, so leaving the
    repeats in counts each design once per budget it appeared under: it inflates
    n, weights means toward designs the search reached early (those recur under
    the most budgets), and shrinks bootstrap CIs that resample duplicated points.

    Keeps the row from the largest budget — the deepest search that produced it.
    Everything plotted (p(binder), OASis percentile, edits, h3) is a function of
    the sequence alone, so the choice only affects ``budget`` / ``pll`` /
    ``sort_key``. ``seed`` is part of the key because different seeds are
    genuinely independent draws, not duplicates.
    """
    key = [c for c in ("model", "method", "seed", "vh") if c in df.columns]
    before = len(df)
    ordered = (
        df.sort_values("edit_budget", kind="mergesort")
        if "edit_budget" in df.columns else df
    )
    out = ordered.drop_duplicates(subset=key, keep="last").sort_index()
    return out.copy(), before - len(out)


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    halfw = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p, centre - halfw, centre + halfw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--keep-duplicate-budgets", action="store_true",
                        help="Skip cross-budget de-duplication. Only for "
                             "reproducing figures made before it was added; "
                             "see dedupe_across_budgets for why it is wrong.")
    args = parser.parse_args()

    df = pd.read_csv(args.results_dir / "all_sequences.csv")
    # Pretty-printed columns are named with capital letters so seaborn uses
    # them as legend / axis titles directly (e.g. "Method" rather than the
    # raw column name).
    df["Method"] = df["method"].map(METHOD_PRETTY)
    df["Model"] = df["model"].map(MODEL_PRETTY).fillna(df["model"])
    # Drop seed-equivalent rows (edits == 0) — these only appear when Gibbs
    # sampled the same residue at all selected positions; nothing to compare.
    df = df[df["edits"] > 0].copy()
    if not args.keep_duplicate_budgets:
        df, n_dupes = dedupe_across_budgets(df)
        print(f"de-duplicated {n_dupes} rows repeated across edit budgets; "
              f"{len(df)} distinct designs remain")
    # ``df_all`` retains every method (Mason CNN-guided + oracle-guided +
    # unguided baselines) for the combined per-model summary figure.
    # ``df`` is the Mason-CNN view used by every other plot in the main
    # text; the oracle-guided methods are filtered out as tautological
    # because the same model is used for both guidance and scoring.
    df_all = df.copy()
    df = df[~df["method"].isin(STRONG_ORACLE_GUIDED_METHODS)].copy()

    sns.set_theme(style="whitegrid", context="paper")

    # 1) Method vs E — mean p(binder) ± 95% bootstrap CI, faceted by model.
    rows = []
    for (model, method, E), sub in df.groupby(["Model", "Method", "edits"]):
        p = sub["score_tz_cnn_p_binder"].dropna().values
        if p.size == 0:
            continue
        mean = float(p.mean())
        # Bootstrap 95% CI for the mean
        rng = np.random.default_rng(0)
        boot = rng.choice(p, size=(2000, p.size), replace=True).mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({
            "model": model, "method": method, "edits": E,
            "mean_p": mean, "lo": lo, "hi": hi, "n": int(p.size),
        })
    perf = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, perf["model"].nunique(),
                             figsize=(5.5 * perf["model"].nunique(), 4.5),
                             sharey=True, squeeze=False)
    axes = axes.flatten()
    for ax, (model, sub) in zip(axes, perf.groupby("model")):
        for method in METHOD_ORDER:
            m = sub[sub["method"] == method].sort_values("edits")
            if m.empty:
                continue
            ax.errorbar(
                m["edits"], m["mean_p"],
                yerr=[(m["mean_p"] - m["lo"]).clip(lower=0), (m["hi"] - m["mean_p"]).clip(lower=0)],
                marker="o", capsize=3, label=method, linewidth=1.6, linestyle=_linestyle_for(method),
            )
        ax.set_title(model)
        ax.set_xlabel("Edits from seed")
        ax.set_ylabel("Mean oracle p(binder)")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="lower left", frameon=True)
    fig.tight_layout()
    fig.savefig(args.results_dir / "method_vs_E.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)

    # 2) Success rate (p > 0.5) vs E with Wilson CI.
    rows = []
    for (model, method, E), sub in df.groupby(["Model", "Method", "edits"]):
        successes = (sub["score_tz_cnn_p_binder"] > 0.5).sum()
        n = sub["score_tz_cnn_p_binder"].count()
        rate, lo, hi = _wilson_ci(int(successes), int(n))
        rows.append({"model": model, "method": method, "edits": E,
                     "rate": rate, "lo": lo, "hi": hi, "n": n})
    succ = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, succ["model"].nunique(),
                             figsize=(5.5 * succ["model"].nunique(), 4.5),
                             sharey=True, squeeze=False)
    axes = axes.flatten()
    for ax, (model, sub) in zip(axes, succ.groupby("model")):
        for method in METHOD_ORDER:
            m = sub[sub["method"] == method].sort_values("edits")
            if m.empty:
                continue
            ax.errorbar(
                m["edits"], m["rate"],
                yerr=[(m["rate"] - m["lo"]).clip(lower=0), (m["hi"] - m["rate"]).clip(lower=0)],
                marker="o", capsize=3, label=method, linewidth=1.6, linestyle=_linestyle_for(method),
            )
        ax.set_title(model)
        ax.set_xlabel("Edits from seed")
        ax.set_ylabel("Success rate (p_binder > 0.5)")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="lower left", frameon=True)
    fig.tight_layout()
    fig.savefig(args.results_dir / "success_rate_vs_E.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)

    # 3) Full p_binder violins (small multiples)
    g = sns.catplot(
        data=df, kind="violin", x="Method", y="score_tz_cnn_p_binder",
        col="edits", row="Model", height=3.5, aspect=1.3,
        cut=0, inner="quartile", order=METHOD_ORDER,
    )
    g.set_xticklabels(rotation=20, ha="right")
    g.set_axis_labels("", "Oracle p(binder)")
    g.savefig(args.results_dir / "p_binder_violin.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()

    # 3b) OASis percentile vs E (mean ± bootstrap CI), if available.
    if "score_oasis_percentile" in df.columns and df["score_oasis_percentile"].notna().any():
        rows = []
        for (model, method, E), sub in df.groupby(["Model", "Method", "edits"]):
            v = sub["score_oasis_percentile"].dropna().values
            if v.size == 0:
                continue
            rng = np.random.default_rng(0)
            boot = rng.choice(v, size=(2000, v.size), replace=True).mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append({
                "model": model, "method": method, "edits": E,
                "mean_o": float(v.mean()), "lo": lo, "hi": hi, "n": int(v.size),
            })
        oas = pd.DataFrame(rows)
        if not oas.empty:
            fig, axes = plt.subplots(1, oas["model"].nunique(),
                                     figsize=(5.5 * oas["model"].nunique(), 4.5),
                                     sharey=True, squeeze=False)
            axes = axes.flatten()
            for ax, (model, sub) in zip(axes, oas.groupby("model")):
                for method in METHOD_ORDER:
                    m = sub[sub["method"] == method].sort_values("edits")
                    if m.empty:
                        continue
                    ax.errorbar(
                        m["edits"], m["mean_o"],
                        yerr=[(m["mean_o"] - m["lo"]).clip(lower=0), (m["hi"] - m["mean_o"]).clip(lower=0)],
                        marker="o", capsize=3, label=method, linewidth=1.6, linestyle=_linestyle_for(method),
                    )
                ax.set_title(model)
                ax.set_xlabel("Edits from seed")
                ax.set_ylabel("Mean OASis percentile (humanness)")
                # Adaptive y-limits with a small margin around the actual range.
                lo_all = oas["lo"].min()
                hi_all = oas["hi"].max()
                pad = max(0.005, 0.05 * (hi_all - lo_all))
                ax.set_ylim(lo_all - pad, hi_all + pad)
                ax.grid(alpha=0.3)
            axes[0].legend(loc="lower left", frameon=True, fontsize="small")
            fig.tight_layout()
            fig.savefig(args.results_dir / "oasis_vs_E.png", dpi=SAVE_DPI, bbox_inches="tight")
            plt.close(fig)

        # Joint OASis × p_binder view at realized edits = 6 (pinned because
        # every method has ≥ 50 samples in this bucket; edits=7 has as few
        # as 3 for some Gibbs variants). We show:
        #   - faded scatter of every (oasis, p_binder) point per method
        #     (low alpha so the strip structure is visible but not overwhelming)
        #   - bold step line of each method's Pareto frontier (max p_binder
        #     achievable at each OASis level)
        # so both density and dominance are legible at a glance.
        max_E = 6
        sub = df[df["edits"] == max_E].copy()
        if sub["score_oasis_percentile"].notna().any():
            present = [m for m in METHOD_ORDER if m in sub["Method"].unique()]
            palette = sns.color_palette(n_colors=len(present))
            color_map = dict(zip(present, palette))

            n_models = sub["Model"].nunique()
            fig, axes = plt.subplots(
                1, n_models, figsize=(6 * n_models, 5), sharey=True, squeeze=False,
            )
            axes = axes.flatten()
            for ax, (model, mod_sub) in zip(axes, sub.groupby("Model")):
                # Faded scatter: jitter the OASis x slightly so the strips
                # don't fully overlap.
                rng = np.random.default_rng(0)
                for method in present:
                    rows = mod_sub[mod_sub["Method"] == method]
                    if rows.empty:
                        continue
                    oasis = rows["score_oasis_percentile"].values
                    pb = rows["score_tz_cnn_p_binder"].values
                    jitter = rng.uniform(-0.0015, 0.0015, size=oasis.shape)
                    ax.scatter(
                        oasis + jitter, pb,
                        s=6, alpha=0.12, color=color_map[method],
                        linewidth=0,
                    )
                # Pareto frontier per method, on top of the scatter.
                for method in present:
                    rows = mod_sub[mod_sub["Method"] == method]
                    if rows.empty:
                        continue
                    fx, fy = _pareto_frontier(
                        rows["score_oasis_percentile"].values,
                        rows["score_tz_cnn_p_binder"].values,
                    )
                    if fx.size:
                        ax.plot(
                            fx, fy,
                            color=color_map[method], linewidth=1.8,
                            linestyle=_linestyle_for(method),
                            label=method,
                        )
                        ax.scatter(
                            fx, fy, color=color_map[method],
                            s=24, edgecolor="white", linewidth=0.6, zorder=3,
                        )
                ax.set_title(f"{model} (E={max_E})")
                ax.set_xlabel("OASis percentile (humanness)")
                ax.set_ylabel("Oracle p(binder)")
                xlo = sub["score_oasis_percentile"].min()
                xhi = sub["score_oasis_percentile"].max()
                xpad = max(0.005, 0.05 * (xhi - xlo))
                ax.set_xlim(xlo - xpad, xhi + xpad)
                ax.set_ylim(0, 1.02)
                ax.grid(alpha=0.3)
            axes[0].legend(loc="lower left", frameon=True, fontsize="small",
                           title="Method (Pareto frontier)")
            fig.tight_layout()
            fig.savefig(args.results_dir / "oasis_vs_pbinder.png",
                        dpi=SAVE_DPI, bbox_inches="tight")
            plt.close(fig)

    # 4a) Pairwise diversity: per-child mean Hamming distance to siblings
    # within the same (Model, Method, edits) group. Each child is one sample,
    # so each method × edits gets a distribution shown as a violin.
    pw_rows = []
    for (model, method, edits), sub in df.groupby(["Model", "Method", "edits"]):
        h3s = sub["h3"].tolist()
        means = _pairwise_diversity_per_seq(h3s)
        if not np.isfinite(means).any():
            continue
        for v in means:
            if not np.isnan(v):
                pw_rows.append({
                    "Model": model, "Method": method,
                    "edits": int(edits), "pairwise_diversity": float(v),
                })
    if pw_rows:
        pw_df = pd.DataFrame(pw_rows)
        present = [m for m in METHOD_ORDER if m in pw_df["Method"].unique()]
        g = sns.catplot(
            data=pw_df, kind="violin", x="edits", y="pairwise_diversity",
            col="Model", hue="Method", hue_order=present,
            cut=0, inner="quartile", split=False, density_norm="width",
            height=5, aspect=1.4,
        )
        g.set_axis_labels("Edits from seed",
                          "Mean pairwise H3 distance to siblings")
        for ax in g.axes.flat:
            ax.grid(alpha=0.3, axis="y")
        g.savefig(args.results_dir / "pairwise_diversity.png",
                  dpi=SAVE_DPI, bbox_inches="tight")
        plt.close()

    # 4) Diversity: unique H3 sequences per (model, method, E)
    div = df.groupby(["Model", "Method", "edits"])["h3"].nunique().reset_index(name="unique_h3")
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(
        data=div, x="edits", y="unique_h3",
        hue=div["Model"] + " · " + div["Method"], ax=ax,
    )
    ax.set_ylabel("Unique H3 sequences")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=True)
    fig.tight_layout()
    fig.savefig(args.results_dir / "diversity.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()

    # 5) Realized-edits distribution per (Model, Method, original budget).
    if "budget" in df.columns:
        g = sns.catplot(
            data=df, kind="box", x="Method", y="edits",
            col="budget", row="Model", height=3, aspect=1.2,
            order=METHOD_ORDER,
        )
        g.set_xticklabels(rotation=20, ha="right")
        g.set_axis_labels("", "Edits from seed")
        g.savefig(args.results_dir / "edits_distribution.png",
                  dpi=SAVE_DPI, bbox_inches="tight")
        plt.close()

    # 6) Per-model summary figures: 3 panels each
    #    (mean p(binder) vs edits | Pareto at E=3 | Pareto at E=6).
    if "score_oasis_percentile" in df_all.columns and df_all["score_oasis_percentile"].notna().any():
        # Use seaborn's "Paired" palette so each Mason-CNN method shares a
        # hue with its strong-oracle counterpart (one light, one dark).
        # Paired's first three pairs are blue/green/red; we re-order so the
        # three "guided method types" (Beam+STS, Gibbs+PoE, Beam+STS+OASis)
        # get red/green/blue. Baselines pull distinct hues from later in
        # the palette.
        paired = sns.color_palette("Paired", 12)
        # Mason-CNN methods take the *dark* shade (the realistic, main-story
        # variant); strong-oracle counterparts take the lighter shade so
        # they're visibly subordinate.
        color_map = {
            # Pair 1 — red: Beam + STS variants
            "Beam + STS (Mason CNN)":              paired[5],   # dark red
            "Beam + STS (oracle)":                 paired[4],   # light red
            # Pair 2 — green: Gibbs + PoE variants
            "Gibbs + PoE (Mason CNN)":             paired[3],   # dark green
            "Gibbs + PoE (oracle)":                paired[2],   # light green
            # Pair 3 — blue: Beam + STS + OASis variants
            "Beam + STS (Mason CNN + OASis)":      paired[1],   # dark blue
            "Beam + STS (oracle + OASis)":         paired[0],   # light blue
            # Unguided baselines — distinct hues from later in Paired.
            "Denoising":                           paired[7],   # dark orange
            "Gibbs":                               paired[9],   # dark purple
            "Beam":                                paired[11],  # dark brown
        }

        oasis_lo = df_all["score_oasis_percentile"].min()
        oasis_hi = df_all["score_oasis_percentile"].max()
        oasis_pad = max(0.005, 0.05 * (oasis_hi - oasis_lo))
        x_range = (oasis_lo - oasis_pad, oasis_hi + oasis_pad)

        def _slug(model_pretty: str) -> str:
            return (model_pretty.lower()
                    .replace(" ", "_").replace("-", "").replace("/", "_"))

        baselines = {"Denoising", "Gibbs", "Beam"}
        mason_pretty = {METHOD_PRETTY[m] for m in MASON_GUIDED_METHODS}
        oracle_pretty = {METHOD_PRETTY[m] for m in STRONG_ORACLE_GUIDED_METHODS}
        # Build per-variant lists by filtering METHOD_ORDER, so the order
        # is stable and matches the merged-left legend exactly. The raw
        # MASON_GUIDED_METHODS / STRONG_ORACLE_GUIDED_METHODS sets have
        # non-deterministic iteration order, which previously scrambled
        # the bottom-right violin's row order.
        present_in_data = set(df_all["Method"].unique())
        mason_methods_present = [
            m for m in METHOD_ORDER
            if (m in baselines or m in mason_pretty) and m in present_in_data
        ]
        oracle_methods_present = [
            m for m in METHOD_ORDER
            if (m in baselines or m in oracle_pretty) and m in present_in_data
        ]
        all_methods_present = [m for m in METHOD_ORDER if m in present_in_data]

        for model_pretty, df_model in df_all.groupby("Model"):
            fig = plt.figure(figsize=(11, 5), constrained_layout=True)
            gs = fig.add_gridspec(2, 3)
            ax_left    = fig.add_subplot(gs[:, 0])
            ax_top_mid = fig.add_subplot(gs[0, 1])
            ax_bot_mid = fig.add_subplot(gs[1, 1])
            ax_top_rt  = fig.add_subplot(gs[0, 2])
            ax_bot_rt  = fig.add_subplot(gs[1, 2])

            # Left (merged across both rows): p(binder) vs edits, with all
            # 9 methods. Mason / oracle pairs share a hue (dark / light).
            _render_pbinder_vs_edits(
                ax_left, df_model, all_methods_present, color_map,
            )
            ax_left.set_title("p(binder) vs edits")
            ax_left.legend(loc="lower left", frameon=False, fontsize=10)

            # Middle column — Mason-only Pareto frontiers at E=3 and E=6.
            _render_pareto(
                ax_top_mid, df_model[df_model["edits"] == 3],
                mason_methods_present, color_map, x_range=x_range,
            )
            ax_top_mid.set_title("Pareto frontier at E=3")

            _render_pareto(
                ax_bot_mid, df_model[df_model["edits"] == 6],
                mason_methods_present, color_map, x_range=x_range,
            )
            ax_bot_mid.set_title("Pareto frontier at E=6")

            # Top right — Mason-only mean p(binder) at each (quantized)
            # OASis percentile, pooling all E.
            df_mason = df_model[df_model["Method"].isin(mason_methods_present)]
            _render_pbinder_vs_oasis(
                ax_top_rt, df_mason, mason_methods_present, color_map,
                statistic="mean", x_range=x_range,
            )
            ax_top_rt.set_title("Mean p(binder) vs OASis")

            # Bottom right — pairwise H3 diversity among Mason-only
            # "high-quality" variants (p(binder) > 0.5 AND OASis above
            # the Trastuzumab-seed baseline; all E pooled). Horizontal
            # violins; order matches the merged-left legend so the
            # individual method labels can be omitted (legend supplies
            # them).
            kept = df_mason[
                (df_mason["score_tz_cnn_p_binder"] > 0.5)
                & (df_mason["score_oasis_percentile"] > TRASTUZUMAB_SEED_OASIS)
            ]
            pw_rows = []
            for method, group in kept.groupby("Method"):
                h3s = group["h3"].tolist()
                if len(h3s) < 2:
                    continue
                means = _pairwise_diversity_per_seq(h3s)
                for v in means:
                    if not np.isnan(v):
                        pw_rows.append({"Method": method,
                                        "pairwise_diversity": float(v)})
            if pw_rows:
                pw_df = pd.DataFrame(pw_rows)
                violin_order = [m for m in mason_methods_present
                                if m in pw_df["Method"].unique()]
                sns.violinplot(
                    data=pw_df, y="Method", x="pairwise_diversity",
                    order=violin_order, palette=color_map,
                    cut=0, inner="quartile", ax=ax_bot_rt,
                )
                ax_bot_rt.set_yticklabels([])
                ax_bot_rt.set_ylabel("")
                ax_bot_rt.set_xlabel("Mean pairwise H3 distance to siblings")
                ax_bot_rt.set_title(
                    "Diversity at p(binder) > 0.5 AND OASis > Tras",
                    fontsize=10,
                )
                ax_bot_rt.set_xlim(left=0)
                ax_bot_rt.grid(alpha=0.3, axis="x")

            fname = f"summary_{_slug(model_pretty)}.png"
            fig.savefig(args.results_dir / fname,
                        dpi=SAVE_DPI, bbox_inches="tight")
            plt.close(fig)

        # (The pooled per-sequence pairwise-distance violins and per-E
        # faceted lineplots were superseded by extra_diversity.py's
        # diversity_grid_<model>.png, which reports set-level Vendi and
        # entropy in a 3x2 grid per MLM with E aggregated.)

    print(f"plots written to {args.results_dir}")


if __name__ == "__main__":
    main()
