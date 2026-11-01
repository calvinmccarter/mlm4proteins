"""Set-level H3 diversity metrics for the Trastuzumab CDR-H3 results.

Computes three metrics per ``(Model, Method, edits)`` cell from
``results/all_sequences.csv`` (the per-cell numbers are written to
``extra_diversity_summary.csv``):

  - ``pos_entropy``: mean Shannon entropy (bits) across the 10 H3
    positions of the per-position amino-acid distribution. Same quantity
    plotted in a sequence logo, averaged across positions. Maximum
    achievable = log2(20) ~= 4.32 (uniform over canonical AAs).
  - ``vendi_hamming``: Vendi score (Friedman & Dieng 2023) of the
    normalized Hamming kernel ``K_ij = 1 - d_H(h_i, h_j) / L``. PSD
    because each H3 position contributes an indicator kernel
    ``1[h_i[p] == h_j[p]]`` (a sum of one-hot rank-1 kernels), and a
    sum of PSDs is PSD. Reports the effective number of distinct H3s
    under this similarity. Since all H3s here are length 10 with no
    indels, Hamming = edit distance.
  - ``vendi_esm2``: Vendi score of the cosine kernel ``K_ij = e_i.e_j``
    where ``e_i`` is ESM2-650M's mean-pooled embedding of the H3
    positions of the full VH (framework provides context). Cached by
    H3 string in ``cache/esm2_h3_embeddings.pkl``. Only computed when
    ``--with-esm2`` is passed (requires GPU; ~5-10 min for ~20k H3s).

Plots a 3x2 ``diversity_grid_<model>.png`` per MLM with E aggregated:
3 rows (one per metric) x 2 columns (all variants / filtered to
p(binder) > 0.5 AND OASis > Trastuzumab-seed baseline). All 9 methods
present in the input are shown as horizontal bars colored to match the
headline summary palette.
"""
from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR))

from analyze import (  # noqa: E402
    METHOD_PRETTY, METHOD_ORDER, MODEL_PRETTY, SAVE_DPI,
    TRASTUZUMAB_SEED_OASIS, _pairwise_diversity_per_seq,
    dedupe_across_budgets,
)
from run import TRASTUZUMAB_VH, TRASTUZUMAB_H3  # noqa: E402

H3_START = TRASTUZUMAB_VH.index(TRASTUZUMAB_H3)
H3_END = H3_START + len(TRASTUZUMAB_H3)
H3_LEN = len(TRASTUZUMAB_H3)
LOG2 = math.log(2.0)


# ---- Color palette (mirrors analyze.py headline figures) -----------------
def _color_map() -> dict[str, tuple[float, float, float]]:
    import seaborn as sns
    paired = sns.color_palette("Paired", 12)
    return {
        "Beam + STS (Mason CNN)":              paired[5],
        "Beam + STS (oracle)":                 paired[4],
        "Gibbs + PoE (Mason CNN)":             paired[3],
        "Gibbs + PoE (oracle)":                paired[2],
        "Beam + STS (Mason CNN + OASis)":      paired[1],
        "Beam + STS (oracle + OASis)":         paired[0],
        "Denoising":                           paired[7],
        "Gibbs":                               paired[9],
        "Beam":                                paired[11],
    }


# ---- Metric computations -------------------------------------------------
def position_entropy_bits(h3s: list[str]) -> float:
    """Mean Shannon entropy (bits) across the L H3 positions."""
    L = len(h3s[0])
    n = len(h3s)
    arr = np.array([list(s) for s in h3s])
    total = 0.0
    for pos in range(L):
        _, counts = np.unique(arr[:, pos], return_counts=True)
        p = counts / n
        # log(0) handled by p > 0 mask; here p > 0 by construction
        total += float(-(p * np.log(p)).sum()) / LOG2
    return total / L


def vendi_from_kernel(K: np.ndarray) -> float:
    """Vendi score = exp(Shannon entropy of normalized eigenvalues of K/n)."""
    n = K.shape[0]
    if n <= 1:
        return float(n)
    K = (K + K.T) / 2.0
    w = np.linalg.eigvalsh(K / n)
    w = np.clip(w, 0.0, None)
    w = w[w > 1e-12]
    if w.size == 0:
        return 1.0
    w = w / w.sum()
    H = float(-(w * np.log(w)).sum())
    return float(math.exp(H))


def hamming_kernel(h3s: list[str]) -> np.ndarray:
    n = len(h3s)
    L = len(h3s[0])
    arr = np.frombuffer("".join(h3s).encode("ascii"), dtype=np.uint8).reshape(n, L)
    # Pairwise mismatch count via per-position equality, summed:
    D = np.zeros((n, n), dtype=np.float32)
    for pos in range(L):
        col = arr[:, pos]
        D += (col[:, None] != col[None, :]).astype(np.float32)
    return 1.0 - D / float(L)


def vendi_hamming(h3s: list[str]) -> float:
    if len(h3s) <= 1:
        return float(len(h3s))
    K = hamming_kernel(h3s)
    return vendi_from_kernel(K)


def vendi_esm2(h3s: list[str], emb_by_h3: dict[str, np.ndarray]) -> float:
    if len(h3s) <= 1:
        return float(len(h3s))
    E = np.stack([emb_by_h3[h] for h in h3s], axis=0).astype(np.float64)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    En = E / norms
    K = En @ En.T
    return vendi_from_kernel(K)


# ---- ESM2 embedding cache ------------------------------------------------
def compute_esm2_h3_embeddings(
    unique_h3s: list[str],
    cache_path: Path,
    batch_size: int = 8,
    device: str | None = None,
) -> dict[str, np.ndarray]:
    """Mean-pool ESM2-650M hidden states over the H3 positions of the full VH.

    Framework is constant (TRASTUZUMAB_VH minus the H3 region), so the cache
    is keyed by the H3 string alone.
    """
    cache: dict[str, np.ndarray] = {}
    if cache_path.exists():
        with cache_path.open("rb") as f:
            cache = pickle.load(f)
    needed = [h for h in unique_h3s if h not in cache]
    print(f"ESM2 cache: {len(cache)} cached, {len(needed)} new", flush=True)
    if not needed:
        return cache

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    model_name = "facebook/esm2_t33_650M_UR50D"
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {model_name} on {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device).eval()

    framework_pre = TRASTUZUMAB_VH[:H3_START]
    framework_post = TRASTUZUMAB_VH[H3_END:]

    with torch.inference_mode():
        for i in range(0, len(needed), batch_size):
            batch = needed[i:i + batch_size]
            seqs = [framework_pre + h + framework_post for h in batch]
            enc = tokenizer(seqs, return_tensors="pt",
                            padding=True, add_special_tokens=True)
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attn,
                        output_hidden_states=True)
            # ESM2 prepends <cls> (token 0), so seq pos p -> token pos p+1.
            last = out.hidden_states[-1]  # [B, L+2, D]
            h3_token_idx = list(range(H3_START + 1, H3_END + 1))
            h3_emb = last[:, h3_token_idx, :].mean(dim=1).cpu().numpy()
            for h, e in zip(batch, h3_emb):
                cache[h] = e.astype(np.float32)
            if (i // batch_size) % 20 == 0:
                print(f"  {i + len(batch)}/{len(needed)}", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(cache, f)
    print(f"wrote ESM2 cache: {cache_path} ({len(cache)} entries)", flush=True)
    return cache


# ---- Bootstrap helper ----------------------------------------------------
def bootstrap_ci(values_or_fn, n: int, n_boot: int = 200,
                 alpha: float = 0.05, seed: int = 0,
                 ) -> tuple[float, float, float]:
    """Bootstrap CI for either an array of per-sample values OR a callable
    that takes a list of resampled indices and returns the scalar metric.

    For position entropy: callable receives indices into the H3 list and
    re-computes entropy on the resampled set. Same for Vendi metrics.
    Returns ``(point_estimate, lo, hi)``.
    """
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[b] = float(values_or_fn(idx))
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(samples.mean()), float(lo), float(hi)


# ---- Main ----------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", type=Path,
        default=EXPERIMENT_DIR / "results" / "all_sequences.csv",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=EXPERIMENT_DIR / "results",
    )
    parser.add_argument(
        "--esm2-cache", type=Path,
        default=EXPERIMENT_DIR / "cache" / "esm2_h3_embeddings.pkl",
    )
    parser.add_argument("--with-esm2", action="store_true",
                        help="Compute (or update) ESM2 embeddings and "
                             "the vendi_esm2 metric. Requires GPU.")
    parser.add_argument("--n-boot", type=int, default=200,
                        help="Bootstrap iterations for CIs.")
    parser.add_argument("--min-cell-n", type=int, default=5,
                        help="Skip (Model, Method, edits) cells with fewer "
                             "sequences than this.")
    parser.add_argument("--summary-csv", type=Path, default=None,
                        help="Path to write the per-cell metric table.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["edits"] > 0].copy()
    # Matters more here than for the mean-based plots in analyze.py: an exact
    # repeat is an identical point, so leaving cross-budget duplicates in
    # deflates position-wise entropy and both Vendi scores directly.
    df, n_dupes = dedupe_across_budgets(df)
    print(f"de-duplicated {n_dupes} rows repeated across edit budgets; "
          f"{len(df)} distinct designs remain")
    df["Method"] = df["method"].map(METHOD_PRETTY)
    df["Model"] = df["model"].map(MODEL_PRETTY).fillna(df["model"])

    color_map = _color_map()
    methods_present = [m for m in METHOD_ORDER if m in df["Method"].unique()]

    # Pre-compute ESM2 embeddings if requested ----------------------------
    emb_by_h3: dict[str, np.ndarray] | None = None
    if args.with_esm2:
        unique_h3s = sorted(df["h3"].unique())
        emb_by_h3 = compute_esm2_h3_embeddings(unique_h3s, args.esm2_cache)

    # Per-cell metrics ----------------------------------------------------
    # Position entropy is cheap, so we report a bootstrap CI for it.
    # Vendi is a deterministic functional of the kernel; bootstrap with
    # replacement injects duplicate rows that collapse eigenvalues and
    # systematically deflate the score, so we report a single point
    # estimate (no CI). Including organic duplicates from the sampler in
    # the kernel is correct: Vendi naturally treats them as zero-distance
    # pairs and adjusts the effective count accordingly.
    rows = []
    for (model, method, edits), group in df.groupby(["Model", "Method", "edits"]):
        h3s = group["h3"].tolist()
        n = len(h3s)
        if n < args.min_cell_n:
            continue

        h3_arr = h3s
        def _pe(idx, _h3=h3_arr):
            return position_entropy_bits([_h3[i] for i in idx])
        pe_mean, pe_lo, pe_hi = bootstrap_ci(_pe, n=n, n_boot=args.n_boot)

        vh_point = vendi_hamming(h3s)

        row = {
            "Model": model, "Method": method, "edits": int(edits), "n": n,
            "pos_entropy": pe_mean,
            "pos_entropy_lo": pe_lo, "pos_entropy_hi": pe_hi,
            "vendi_hamming": vh_point,
        }
        if emb_by_h3 is not None:
            row["vendi_esm2"] = vendi_esm2(h3s, emb_by_h3)
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary_csv = args.summary_csv or (
        args.out_dir / "extra_diversity_summary.csv"
    )
    summary.to_csv(summary_csv, index=False)
    print(f"wrote per-cell metrics: {summary_csv} ({len(summary)} rows)",
          flush=True)

    # ---- 3x2 diversity grid per MLM, E aggregated -----------------------
    for model_pretty, df_model in df.groupby("Model"):
        render_diversity_grid(
            df_model=df_model,
            model_pretty=str(model_pretty),
            color_map=color_map,
            methods_order=methods_present,
            emb_by_h3=emb_by_h3,
            out_path=args.out_dir / f"diversity_grid_{_slug(model_pretty)}.png",
        )

    print(f"plots written to {args.out_dir}", flush=True)


def _slug(model_pretty: str) -> str:
    return (model_pretty.lower()
            .replace(" ", "_").replace("-", "").replace("/", "_"))


def render_diversity_grid(
    df_model: pd.DataFrame,
    model_pretty: str,
    color_map: dict,
    methods_order: list[str],
    emb_by_h3: dict | None,
    out_path: Path,
):
    """3 rows x 2 cols (no filter / quality filter) for one MLM. Row 1
    is a per-sequence horizontal violin (mean pairwise H3 distance to
    siblings within the same method's pool) so the within-method spread
    is visible. Rows 2-3 are horizontal bars of set-level scalars:
    Vendi-ESM2 cosine and position-wise H3 entropy. E is aggregated
    within each (Method, filter) pool before the metrics are computed.
    """
    import seaborn as sns

    filters = [
        ("All variants", lambda d: d),
        (f"p(binder) > 0.5 AND OASis > {TRASTUZUMAB_SEED_OASIS:.2f}",
         lambda d: d[(d["score_tz_cnn_p_binder"] > 0.5)
                    & (d["score_oasis_percentile"] > TRASTUZUMAB_SEED_OASIS)]),
    ]
    metrics = [
        ("Mean pairwise H3 distance to siblings", "pairwise",     "violin"),
        ("Vendi (ESM2 cosine kernel)",            "vendi_esm2",   "bar"),
        ("Position-wise H3 entropy (bits)",       "pos_entropy",  "bar"),
    ]

    cell_data: dict = {}
    for f_idx, (_, f_fn) in enumerate(filters):
        df_f = f_fn(df_model)
        for method in methods_order:
            h3s = df_f[df_f["Method"] == method]["h3"].tolist()
            if len(h3s) < 2:
                cell_data[(f_idx, method, "pairwise")]    = np.array([])
                cell_data[(f_idx, method, "vendi_esm2")]  = np.nan
                cell_data[(f_idx, method, "pos_entropy")] = np.nan
                continue
            dist = _pairwise_diversity_per_seq(h3s)
            dist = dist[~np.isnan(dist)]
            cell_data[(f_idx, method, "pairwise")]    = dist
            cell_data[(f_idx, method, "pos_entropy")] = position_entropy_bits(h3s)
            cell_data[(f_idx, method, "vendi_esm2")] = (
                vendi_esm2(h3s, emb_by_h3) if emb_by_h3 is not None else np.nan
            )

    fig, axes = plt.subplots(
        len(metrics), len(filters),
        figsize=(6.5 * len(filters), 1.05 * len(methods_order) + 0.6),
        squeeze=False, constrained_layout=True,
    )

    for r, (m_label, m_key, m_kind) in enumerate(metrics):
        for c, (f_name, _) in enumerate(filters):
            ax = axes[r, c]
            if m_kind == "violin":
                rows = []
                for method in methods_order:
                    for v in cell_data[(c, method, m_key)]:
                        rows.append({"Method": method, "Value": float(v)})
                if rows:
                    long_df = pd.DataFrame(rows)
                    sns.violinplot(
                        data=long_df, x="Value", y="Method", hue="Method",
                        order=methods_order, palette=color_map,
                        inner="quartile", cut=0, legend=False, ax=ax,
                    )
            else:
                values = [cell_data[(c, method, m_key)] for method in methods_order]
                colors = [color_map.get(m, "0.7") for m in methods_order]
                y = np.arange(len(methods_order))
                ax.barh(y, values, color=colors, alpha=0.85, edgecolor="none")
                ax.set_yticks(y)
                ax.invert_yaxis()
            ax.set_yticks(np.arange(len(methods_order)))
            if c == 0:
                ax.set_yticklabels(methods_order, fontsize=8)
                ax.set_ylabel(m_label, fontsize=9)
            else:
                ax.set_yticklabels([])
                ax.set_ylabel("")
            ax.set_xlabel("")
            if r == 0:
                ax.set_title(f_name, fontsize=10)
            ax.set_xlim(left=0)
            ax.grid(alpha=0.3, axis="x")

    fig.suptitle(f"H3 diversity ({model_pretty})", fontsize=12)
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
