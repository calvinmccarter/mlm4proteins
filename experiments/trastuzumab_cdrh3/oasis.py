"""OASis humanness scorer (BioPhi-backed, Pool-parallel, disk-cached).

Wraps the BioPhi ``get_chain_humanness(...).get_oasis_percentile(0.5)`` flow
from Prihoda et al. 2022, with three practical performance choices:

  * a per-call ``multiprocessing.Pool`` (workers persist across calls), since
    each underlying lookup against the 22 GB ``OASis_9mers_v1.db`` is ~1.5 s
    on a single core;
  * a sequence-keyed in-memory cache, since beam search re-evaluates many of
    the same templates as the search progresses outward from the seed;
  * a pickle-backed on-disk cache that persists across process restarts and is
    flushed on every batch, so an interrupted run doesn't lose work.

Both AbNumber and BioPhi must be importable. We add ``$ABNUMBER_DIR`` /
``$BIOPHI_DIR`` to ``sys.path`` if they are set, since neither is on PyPI;
unset, they default to ``$PROJECTS_DIR/{AbNumber,BioPhi}`` (``~/projects``).
The database path likewise defaults to ``$OASIS_DB_PATH``, else
``$PROJECTS_DIR/OASis_9mers_v1.db``.

Returns OASis percentile in [0, 1]. Direction is ``"maximize"`` — higher is
more human-like at the medium 50% subject threshold (matching the paper's
medium threshold).
"""
from __future__ import annotations

import os
import pickle
import sys
import tempfile
from multiprocessing import Pool, get_context
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


# AbNumber, BioPhi and the 22 GB OASis database all live outside the repo (the
# first two are not on PyPI). We look for them under ``$PROJECTS_DIR``, falling
# back to ``~/projects`` -- the layout the paper's sweeps were run under. Each
# path also has its own override below, so a scattered install needs no symlinks.
PROJECTS_DIR = Path(os.environ.get("PROJECTS_DIR") or Path.home() / "projects")
DEFAULT_ABNUMBER_DIR = PROJECTS_DIR / "AbNumber"
DEFAULT_BIOPHI_DIR = PROJECTS_DIR / "BioPhi"
DEFAULT_OASIS_DB = Path(
    os.environ.get("OASIS_DB_PATH") or PROJECTS_DIR / "OASis_9mers_v1.db"
)


def _ensure_imports():
    for env_var, default in [
        ("ABNUMBER_DIR", DEFAULT_ABNUMBER_DIR),
        ("BIOPHI_DIR", DEFAULT_BIOPHI_DIR),
    ]:
        path = os.environ.get(env_var) or str(default)
        if path and path not in sys.path:
            sys.path.insert(0, path)


_DB_PATH: str | None = None
_PARAMS = None


def _worker_init(db_path: str):
    """Pool initializer: import BioPhi/AbNumber once and warm OASisParams."""
    _ensure_imports()
    global _DB_PATH, _PARAMS
    from biophi.humanization.methods.humanness import OASisParams
    _DB_PATH = db_path
    _PARAMS = OASisParams(oasis_db_path=db_path, min_fraction_subjects=0.5)


def _oasis_one(sequence: str) -> float:
    """Worker function: compute OASis percentile for one VH or VL."""
    from abnumber import Chain
    from biophi.humanization.methods.humanness import get_chain_humanness
    try:
        chain = Chain(sequence, scheme="kabat", cdr_definition="kabat", name="x")
        hs = get_chain_humanness(chain, _PARAMS)
        return float(hs.get_oasis_percentile(0.5))
    except Exception:
        return 0.0


DEFAULT_CACHE_PATH = (
    Path(__file__).resolve().parent / "cache" / "oasis_cache.pkl"
)


def _atomic_pickle_dump(obj, path: Path) -> None:
    """Write pickle atomically: tmp file in same dir, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


class OASisHumannessScorer:
    """``mmmlm.scoring.base.ScoringFunction`` for VH OASis humanness percentile.

    The in-memory ``str -> float`` cache is mirrored to ``cache_path`` (a pickle
    file) and re-loaded on construction so cache hits survive process restarts.
    The on-disk cache is flushed after every batch of new computations so that
    progress is preserved if the process is killed mid-run.
    """

    name = "oasis_percentile"
    direction = "maximize"

    def __init__(
        self,
        db_path: str = str(DEFAULT_OASIS_DB),
        n_workers: int | None = None,
        cache: dict[str, float] | None = None,
        cache_path: str | Path | None = DEFAULT_CACHE_PATH,
    ):
        self.db_path = db_path
        self.n_workers = n_workers if n_workers is not None else max(1, (os.cpu_count() or 4) // 2)
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self._cache: dict[str, float] = {}
        if cache is not None:
            self._cache.update(cache)
        # Disk cache merges on top of any caller-provided cache — disk wins
        # for entries with a finite stored value.
        if self.cache_path is not None and self.cache_path.exists():
            try:
                with open(self.cache_path, "rb") as f:
                    disk = pickle.load(f)
                if isinstance(disk, dict):
                    self._cache.update(disk)
                    print(f"[OASis] loaded {len(disk)} entries from {self.cache_path}",
                          flush=True)
            except Exception as exc:
                print(f"[OASis] WARNING: could not load cache {self.cache_path}: {exc}",
                      flush=True)
        self._pool: Pool | None = None

    def warm_from_csv(self, csv_path: str | Path,
                      seq_col: str = "vh", value_col: str = "score_oasis_percentile") -> int:
        """Seed the cache from a CSV that already has scored sequences."""
        import pandas as pd
        df = pd.read_csv(csv_path)
        added = 0
        for s, v in zip(df[seq_col], df[value_col]):
            if isinstance(s, str) and isinstance(v, (int, float)) and not np.isnan(v):
                if s not in self._cache:
                    self._cache[s] = float(v)
                    added += 1
        if added and self.cache_path is not None:
            self._flush()
        print(f"[OASis] warmed {added} entries from {csv_path}", flush=True)
        return added

    def _flush(self) -> None:
        if self.cache_path is None:
            return
        _atomic_pickle_dump(self._cache, self.cache_path)

    def _ensure_pool(self):
        if self._pool is None:
            ctx = get_context("spawn")
            self._pool = ctx.Pool(
                self.n_workers,
                initializer=_worker_init,
                initargs=(self.db_path,),
            )
        return self._pool

    def __call__(self, sequences: Sequence[str]) -> np.ndarray:
        sequences = list(sequences)
        # Find unique sequences that need fresh computation.
        unseen_set: set[str] = set()
        for s in sequences:
            if s not in self._cache:
                unseen_set.add(s)
        if unseen_set:
            unseen = list(unseen_set)
            pool = self._ensure_pool()
            results = pool.map(_oasis_one, unseen)
            for s, r in zip(unseen, results):
                self._cache[s] = float(r)
            self._flush()
        return np.array([self._cache[s] for s in sequences], dtype=np.float64)

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
        # Final flush in case a previous flush failed silently.
        self._flush()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def score_oasis_serial(sequences: Iterable[str], db_path: str) -> np.ndarray:
    """Convenience: batch score with default settings, single shot."""
    scorer = OASisHumannessScorer(db_path=db_path)
    try:
        return scorer(list(sequences))
    finally:
        scorer.close()
