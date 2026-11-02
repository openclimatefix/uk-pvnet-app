"""Equivalence oracle: the harness reproduces the numbers of the original ad hoc analysis.

Needs the local export directories and the cached floored deltas; skipped (and reported as
skipped) when either is absent. Tolerance is +-0.05 MW over the full year; the match is exact.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from replay_harness import cv, evaluate, io, metrics, replay

pytest.importorskip("pyarrow")  # reported as skipped where parquet cannot be read

ROOT = Path(__file__).resolve().parents[2]
DIRS = ["adjuster_data_prod_year", "adjuster_data_prod_ext", "adjuster_data_prod"]
CACHE = ROOT / "adjuster_data" / "replayed_d8_floor_year.parquet"

pytestmark = pytest.mark.skipif(
    not (CACHE.exists() and all((ROOT / d / "forecasts.parquet").exists() for d in DIRS)),
    reason="local adjuster export directories or floored-delta cache not present",
)

# name -> metric -> expected MW (mae, me: per-row capacity; mae_fixed, jerk, run2run: final 22.5 GW)
EXPECTED = {
    "raw": {"mae": 263.74, "mae_fixed": 276.77, "me": 62.17, "jerk": 93.94, "run2run": 81.31},
    "current": {"mae": 248.39, "mae_fixed": 260.41, "me": 8.88, "jerk": 100.63, "run2run": 71.10},
    "w9": {"mae": 247.55, "mae_fixed": 259.55, "me": 8.21, "jerk": 92.13, "run2run": 72.89},
    "w9_gain08": {"mae": 245.65, "mae_fixed": 257.59, "me": 15.83, "jerk": 92.37, "run2run": 72.60},
    "w9_gain08+run_blend": {"mae": 242.85, "mae_fixed": 254.64, "run2run": 34.73},
    "longer_memory+w9+run_blend": {"mae": 237.80, "mae_fixed": 249.40, "run2run": 34.16},
}


@pytest.fixture(scope="module")
def replayed():
    args = argparse.Namespace(
        data=",".join(str(ROOT / d) for d in DIRS),
        observer="pvlive_day_after",
        base="blend",
        intersect="blend_adjust",
        warmup_days=14,
        lookback_days=8,
    )
    return evaluate.prepare(args)


def test_floored_deltas_match_the_cache_under_the_original_dedup():
    """The cache predates the deterministic dedup: it kept the first row in directory order."""
    frames = [pd.read_parquet(ROOT / d / "forecasts.parquet") for d in DIRS]
    legacy = pd.concat(frames).drop_duplicates(
        subset=["forecaster", "init_time_utc", "horizon_mins"]
    )
    legacy["forecaster"] = legacy["forecaster"].str.split(":").str[0]
    obs = io.clean_observations(
        pd.concat([pd.read_parquet(ROOT / d / "observations.parquet") for d in DIRS]),
        "pvlive_day_after",
    )
    base, blend_adjust = io.pivot(legacy, "blend"), io.pivot(legacy, "blend_adjust")
    inits = base.index[base.index >= base.index.min() + pd.Timedelta(days=14)]
    inits = inits.intersection(blend_adjust.index)
    deltas, _ = replay.replay_deltas(base, replay.apply_floor(obs), inits, 8)
    cache = pd.read_parquet(CACHE)
    cache.columns = cache.columns.astype(int)
    rows = deltas.index.intersection(cache.index)
    assert len(rows) > 16000
    diff = (deltas.loc[rows] - cache.loc[rows]).abs().to_numpy()
    assert np.nanmax(diff) < 1e-9
    assert (deltas.loc[rows].isna().to_numpy() == cache.loc[rows].isna().to_numpy()).all()


def test_w9_matches_pandas_triangular_rolling(replayed):
    pytest.importorskip("scipy")
    _, _, deltas, _ = replayed
    pandas_w9 = deltas.T.rolling(9, center=True, min_periods=1, win_type="triang").mean().T
    ours = replay.smooth_across_horizon(deltas, 9).to_numpy()
    assert np.nanmax(np.abs(ours - pandas_w9.to_numpy())) < 1e-12


def test_canonical_folds_every_fold_win(replayed):
    base, obs, deltas, final_mw = replayed
    mats = evaluate.variants(["current", "w9", "w9_gain08"], base, deltas, final_mw)
    folds = cv.folds_from_dates(cv.CANONICAL_UK_2026)
    assert len(folds) == 5
    table = cv.fold_table(mats, obs, final_mw, folds)
    assert (table.loc["w9"] < table.loc["current"]).all()
    assert (table.loc["w9_gain08"] < table.loc["w9"]).all()


def test_headline_numbers(replayed):
    base, obs, deltas, final_mw = replayed
    assert final_mw == pytest.approx(22528, abs=2)  # the 22.5 GW basis
    names = [n for n in EXPECTED if n != "raw"]
    table = metrics.score(evaluate.variants(names, base, deltas, final_mw), obs, final_mw)
    for name, expected in EXPECTED.items():
        for metric, value in expected.items():
            assert table.loc[name, metric] == pytest.approx(value, abs=0.05), (name, metric)
