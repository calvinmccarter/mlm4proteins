"""Tidy tables behind the beam-size and temperature sweep plots.

Reproduces the per-cell numbers plotted by ``plot_beam_size_sweep.py`` and
``plot_temperature_sweep.py`` as plain-text tables: for each
(Model, Method, sweep value) cell it reports the mean oracle p(binder) and the
mean per-sequence pairwise H3 diversity, each with a 95% bootstrap CI, plus the
cell size ``n``. Sequences are pooled across realized edit counts (edits > 0),
exactly as the plots do. Restricted to the realistic Mason-CNN-guided method
set (the strong-oracle guided variants are omitted).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR))

from analyze import (  # noqa: E402
    METHOD_PRETTY, MODEL_PRETTY, _bootstrap_ci, _pairwise_diversity_per_seq,
)

# Realistic (Mason-CNN) method set, in plot order; short labels for the table.
METHODS = ["beam", "beam_weakguided"]
METHOD_SHORT = {
    "Beam": "Beam",
    "Beam + STS (Mason CNN)": "Beam+STS(Mason)",
}
MODEL_ORDER = ["ESM2-650M", "AbLang2"]

SWEEPS = {
    "BEAM SIZE SWEEP": {
        "csvs": ["results/beam_size_sweep/all_sequences.csv.gz",
                 "results/beam_size_sweep_strong/all_sequences.csv.gz"],
        "col": "beam_forward",
        "label": "BeamFwd",
        "fmt": lambda v: str(int(v)),
    },
    "TEMPERATURE SWEEP": {
        "csvs": ["results/temperature_sweep/all_sequences.csv.gz"],
        "col": "temperature",
        "label": "Temp",
        "fmt": lambda v: f"{v:g}",
    },
}


def _load(csvs: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(EXPERIMENT_DIR / c) for c in csvs
              if (EXPERIMENT_DIR / c).exists()]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["edits"] > 0].copy()
    df["Method"] = df["method"].map(METHOD_PRETTY)
    df["Model"] = df["model"].map(MODEL_PRETTY).fillna(df["model"])
    return df


def _cell(sub: pd.DataFrame) -> dict:
    p = sub["score_tz_cnn_p_binder"].dropna().to_numpy()
    pm, plo, phi = _bootstrap_ci(p) if p.size else (np.nan, np.nan, np.nan)
    div = _pairwise_diversity_per_seq(sub["h3"].tolist())
    div = div[~np.isnan(div)]
    dm, dlo, dhi = _bootstrap_ci(div) if div.size else (np.nan, np.nan, np.nan)
    return {
        "pbinder": f"{pm:.3f} [{plo:.3f}, {phi:.3f}]",
        "diversity": f"{dm:.2f} [{dlo:.2f}, {dhi:.2f}]",
    }


def _print_padded(headers: list[str], rows: list[list[str]]) -> None:
    w = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]
    def fmt(cells):
        # left-align the first three (labels), right-align the numeric cells.
        return "  ".join(
            c.ljust(w[i]) if i < 3 else c.rjust(w[i]) for i, c in enumerate(cells)
        )
    print(fmt(headers))
    for r in rows:
        print(fmt(r))


def main() -> None:
    pretty_by_key = {v: k for k, v in METHOD_PRETTY.items()}
    for title, spec in SWEEPS.items():
        df = _load(spec["csvs"])
        col, label = spec["col"], spec["label"]
        headers = ["Model", "Method", label,
                   "p(binder) [95% CI]", "H3 diversity [95% CI]"]
        rows = []
        for model in MODEL_ORDER:
            for mkey in METHODS:
                pretty = METHOD_PRETTY[mkey]
                short = METHOD_SHORT[pretty]
                sel = df[(df["Model"] == model) & (df["Method"] == pretty)]
                for v in sorted(sel[col].dropna().unique()):
                    c = _cell(sel[sel[col] == v])
                    rows.append([model, short, spec["fmt"](v),
                                 c["pbinder"], c["diversity"]])
        print(f"\n===== {title} =====")
        _print_padded(headers, rows)


if __name__ == "__main__":
    main()
