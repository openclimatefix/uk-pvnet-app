"""Synthetic-fixture tests: matched-point scoring, the two smoothness metrics, capacity basis,
fold helpers, export layout and the input guards."""

import argparse
import asyncio
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from replay_harness import cv, evaluate, export, io, metrics

T0 = pd.Timestamp("2026-03-01", tz="UTC")


def obs_frame(times, value=0.4, capacity_watts=100e6):
    return pd.DataFrame({"value_fraction": value, "capacity_watts": capacity_watts}, index=times)


def small_case():
    inits = pd.DatetimeIndex([T0 + pd.Timedelta(minutes=30 * i) for i in range(4)])
    mat = pd.DataFrame(0.5, index=inits, columns=[0, 30])
    obs = obs_frame(pd.date_range(T0, periods=8, freq="30min"))
    return mat, obs


def test_jerkiness_is_mean_abs_second_difference_along_a_run():
    assert metrics.jerkiness(np.array([[0.0, 1.0, 4.0, 9.0]])) == pytest.approx(2.0)
    assert metrics.jerkiness(np.array([[0.0, 1.0, 4.0, 9.0], [0.0] * 4])) == pytest.approx(1.0)


def test_run_to_run_compares_consecutive_inits_at_shared_target_times():
    idx = pd.DatetimeIndex(
        ["2026-03-01 00:00", "2026-03-01 00:30", "2026-03-01 01:00", "2026-03-01 03:00"],
        tz="UTC",
    )
    m = np.array([[10, 20, 30], [20, 35, 50], [35, 50, 70], [100, 100, 130]], dtype=float)
    # (00:30 vs 00:00): |20-20|, |35-30|; (01:00 vs 00:30): |35-35|, |50-50|; 03:00 not consecutive
    assert metrics.run_to_run(m, idx, pd.Index([0, 30, 60])) == pytest.approx(1.25)
    assert metrics.jerkiness(m) == pytest.approx(8.75)  # the two metrics measure different things


def test_matched_points_blank_cells_missing_in_any_variant():
    mat, _ = small_case()
    holes = mat.copy()
    holes.iloc[1, 1] = np.nan
    out = metrics.matched({"a": mat, "b": holes})
    assert out["a"].isna().sum().sum() == 1
    assert np.isnan(out["a"].iloc[1, 1])


def test_a_variant_cannot_gain_by_dropping_rows():
    mat, obs = small_case()
    bad = mat.copy()
    bad.iloc[1, 1] = 0.9  # a large error that variant "b" avoids by not publishing the cell
    dropped = bad.copy()
    dropped.iloc[1, 1] = np.nan
    alone = metrics.score({"a": bad}, obs, 100.0)["mae"]["a"]
    paired = metrics.score({"a": bad, "b": dropped}, obs, 100.0)["mae"]
    assert alone > paired["a"]  # "a" is scored on the cells "b" has, so the comparison is fair
    assert paired["a"] == pytest.approx(paired["b"])
    short = metrics.matched({"a": bad, "b": mat.iloc[:2]})  # fewer inits restricts the others
    assert len(short["a"]) == 2


def test_capacity_basis_per_row_vs_fixed():
    mat, _ = small_case()
    times = pd.date_range(T0, periods=8, freq="30min")
    obs = obs_frame(times, capacity_watts=50e6)  # obs capacity 50 MW, final capacity 100 MW
    table = metrics.score({"a": mat}, obs, final_mw=100.0)
    assert table["mae"]["a"] == pytest.approx(0.1 * 50)
    assert table["mae_fixed"]["a"] == pytest.approx(0.1 * 100)
    unknown = obs.assign(capacity_watts=np.nan)  # missing obs capacity falls back to final
    assert metrics.score({"a": mat}, unknown, 100.0)["mae"]["a"] == pytest.approx(10.0)


def test_report_order_and_bias_by_month():
    mat, obs = small_case()
    cols = list(metrics.score({"raw": mat}, obs, 100.0).columns)
    assert cols[:4] == ["mae", "mae_fixed", "p99_err", "worst_decile"]  # gate, then tails
    assert cols.index("jerk") < cols.index("run2run") < cols.index("me")
    assert "me_worst_month" not in cols
    idx = pd.DatetimeIndex(["2026-01-15", "2026-02-15"], tz="UTC")
    pooled_zero = np.array([[1.0], [-1.0]])  # opposite-signed months cancel in the pooled figure
    assert metrics.bias(pooled_zero) == 0.0
    assert metrics.bias_by_month(pooled_zero, idx).tolist() == [1.0, -1.0]
    table = metrics.monthly_bias({"raw": mat}, obs, 100.0)
    assert table.loc["raw"].tolist() == pytest.approx([10.0])  # 0.1 error x 100 MW, one month


def test_tail_metrics():
    e = np.array([[1.0, -1.0]] * 9 + [[10.0, 10.0]])  # nine easy days, one bad day
    idx = pd.DatetimeIndex([T0 + pd.Timedelta(days=d) for d in range(10)])
    assert metrics.worst_decile_daily_mae(e, idx) == pytest.approx(10.0)
    assert metrics.p99_abs_error(e) == pytest.approx(10.0)


def test_folds_are_contiguous_and_fold_dates_are_validated():
    idx = pd.DatetimeIndex(
        [T0 + pd.Timedelta(days=d, hours=h) for d in range(100) for h in (0, 12)],
    )
    folds = cv.make_folds(idx, 5)
    assert len(folds) == 5
    assert folds[0][0] == T0
    assert all(a[1] == b[0] for a, b in pairwise(folds))
    assert len(cv.folds_from_dates("2026-02-01,2026-04-01,2026-06-01")) == 2
    for bad in ("2026-04-01,2026-02-01", "2026-02-01,2026-02-01", "2026-02-01"):
        with pytest.raises(ValueError, match="fold-dates"):
            cv.folds_from_dates(bad)
    with pytest.raises(ValueError, match="cannot make"):
        cv.make_folds(idx[:4], 5)


def test_export_windows_cover_the_range_under_the_platform_limit():
    start, end = datetime(2025, 10, 31, tzinfo=UTC), datetime(2026, 10, 17, tzinfo=UTC)
    windows = export.chunks(start, end)
    assert windows[0][0] == start
    assert windows[-1][1] == end
    assert all(a[1] == b[0] for a, b in pairwise(windows))
    assert all((w[1] - w[0]).days <= 30 for w in windows)


async def _as_coro(df):
    return df


def test_export_layout_is_what_load_reads(tmp_path):
    """Chunk files are the resume mechanism; the two canonical files are what io.load reads."""
    pytest.importorskip("pyarrow")
    start, end = datetime(2026, 3, 1, tzinfo=UTC), datetime(2026, 5, 1, tzinfo=UTC)
    (tmp_path / "chunks").mkdir()
    f_frames, o_frames = [], []
    for i, w in enumerate(export.chunks(start, end)):
        init = T0 + pd.Timedelta(days=30 * i)
        forecast = pd.DataFrame(
            {
                "forecaster": ["blend:1.3.0"],
                "init_time_utc": [init],
                "horizon_mins": [0],
                "p50_fraction": [0.1],
            },
        )
        obs = pd.DataFrame(
            {
                "observer": ["pvlive_day_after"],
                "timestamp_utc": [init],
                "value_fraction": [0.2],
                "created_time_utc": [init],
            },
        )
        f_path = export.chunk_path(tmp_path, "forecasts", "uuid-1", "blend", w)
        o_path = export.chunk_path(tmp_path, "observations", "uuid-1", "pvlive_day_after", w)
        f_frames.append(asyncio.run(export.cached(f_path, lambda f=forecast: _as_coro(f))))
        o_frames.append(asyncio.run(export.cached(o_path, lambda o=obs: _as_coro(o))))
    export.assemble(tmp_path, f_frames, o_frames)
    assert len(list((tmp_path / "chunks").glob("*.parquet"))) == 6  # 3 windows x 2 kinds
    forecasts, observations = io.load(str(tmp_path))
    assert len(forecasts) == len(f_frames)
    assert forecasts["forecaster"].unique().tolist() == ["blend"]
    assert len(observations) == len(o_frames)
    assert observations.index.is_unique


def test_empty_window_is_announced_and_warned_about(tmp_path, capsys, caplog):
    pytest.importorskip("pyarrow")

    async def fetch():
        return pd.DataFrame({"x": []})

    w = (datetime(2026, 3, 1, tzinfo=UTC), datetime(2026, 3, 31, tzinfo=UTC))
    label = export.window_label(w, "blend")
    asyncio.run(export.cached(tmp_path / "c.parquet", fetch, label))
    assert "requesting 2026-03-01..2026-03-31 (blend)" in capsys.readouterr().out
    assert "returned 0 rows" in caplog.text


def test_short_samples_warn_and_default_paths_have_one_slash():
    short = pd.date_range(T0, periods=10, freq="D")
    assert "indicative only" in evaluate.short_sample_warning(short)
    assert evaluate.short_sample_warning(pd.date_range(T0, periods=90, freq="D")) is None
    assert str(evaluate.default_out("data/,other", "cv.csv")) == "data/cv.csv"


def test_export_chunk_is_fetched_once(tmp_path):
    pytest.importorskip("pyarrow")
    calls = []

    async def fetch():
        calls.append(1)
        return pd.DataFrame({"x": [1, 2]})

    path = tmp_path / "chunk.parquet"
    first = asyncio.run(export.cached(path, fetch))
    second = asyncio.run(export.cached(path, fetch))
    assert len(calls) == 1
    assert first.equals(second)


def test_chunk_files_are_keyed_by_location_and_source(tmp_path):
    w = export.chunks(datetime(2026, 3, 1, tzinfo=UTC), datetime(2026, 4, 1, tzinfo=UTC))[0]
    paths = {
        export.chunk_path(tmp_path, "forecasts", loc, name, w)
        for loc in ("uuid-1", "uuid-2")
        for name in ("blend", "blend-blend_adjust")
    } | {
        export.chunk_path(tmp_path, "observations", "uuid-1", obs, w)
        for obs in ("pvlive_day_after", "pvlive_in_day")
    }
    assert len(paths) == 6


def test_fold_table_handles_one_init_folds_and_rejects_empty_ones():
    mat, obs = small_case()
    mats = {"raw": mat, "a": mat + 0.1}
    one = (T0, T0 + pd.Timedelta(minutes=30))  # a single init
    table = cv.fold_table(mats, obs, 100.0, [one])
    assert table.iloc[:, 0].to_dict() == pytest.approx({"raw": 10.0, "a": 20.0})
    outside = (T0 + pd.Timedelta(days=5), T0 + pd.Timedelta(days=6))
    with pytest.raises(ValueError, match="no scored inits"):
        cv.fold_table(mats, obs, 100.0, [one, outside])


def write_export(
    path: Path,
    days: range,
    capacity_watts: float,
    horizons=(0, 30, 60),
    p50: float = 0.5,
    created_offset_hours: int = 0,
    observer: str = "pvlive_day_after",
) -> None:
    """A small export directory: daily 06:00 inits of `blend` (plus an `other` forecaster)."""
    path.mkdir()
    rows = []
    for d in days:
        init = T0 + pd.Timedelta(days=d, hours=6)
        for name, cap in (("blend:1.0.0", capacity_watts), ("other:1.0.0", 99e9)):
            rows += [
                {
                    "forecaster": name,
                    "init_time_utc": init,
                    "horizon_mins": h,
                    "p50_fraction": p50,
                    "created_time_utc": init + pd.Timedelta(hours=created_offset_hours),
                    "capacity_watts": cap,
                }
                for h in horizons
            ]
    times = pd.date_range(
        T0 + pd.Timedelta(days=days.start), periods=48 * (len(days) + 2), freq="30min"
    )
    obs = pd.DataFrame(
        {
            "observer": observer,
            "timestamp_utc": times,
            "value_fraction": 0.4,
            "capacity_watts": capacity_watts,
            "created_time_utc": times + pd.Timedelta(hours=created_offset_hours),
        },
    )
    pd.DataFrame(rows).to_parquet(path / "forecasts.parquet", index=False)
    obs.to_parquet(path / "observations.parquet", index=False)


def args_for(data: str, **kwargs) -> argparse.Namespace:
    return argparse.Namespace(
        **{
            "data": data,
            "observer": "pvlive_day_after",
            "base": "blend",
            "intersect": "",
            "warmup_days": 0,
            "lookback_days": 3,
        }
        | kwargs,
    )


def test_scores_do_not_depend_on_data_directory_order(tmp_path):
    """Overlapping directories disagree: the later-created re-run wins in either order."""
    pytest.importorskip("pyarrow")
    a, b = tmp_path / "a", tmp_path / "b"
    write_export(a, range(0, 10), 20e9)
    write_export(b, range(5, 15), 22e9, p50=0.52, created_offset_hours=1)  # re-run of days 5-9
    tables = []
    for data in (f"{a},{b}", f"{b},{a}"):
        base, obs, deltas, final_mw = evaluate.prepare(args_for(data))
        assert final_mw == pytest.approx(22000.0)  # the base forecaster's latest init, not `other`
        assert (base.iloc[5:] == 0.52).all().all()  # overlap days 5-9 and b's own days 10-14
        assert (base.iloc[:5] == 0.5).all().all()
        mats = evaluate.variants(["current", "w9"], base, deltas, final_mw)
        tables.append(metrics.score(mats, obs, final_mw))
    pd.testing.assert_frame_equal(tables[0], tables[1])


def test_prepare_rejects_a_missing_observer_and_stray_commas(tmp_path):
    pytest.importorskip("pyarrow")
    write_export(tmp_path / "in_day", range(0, 10), 20e9, observer="pvlive_in_day")
    with pytest.raises(ValueError, match=r"pvlive_day_after.*pvlive_in_day"):
        evaluate.prepare(args_for(str(tmp_path / "in_day")))
    base, *_ = evaluate.prepare(args_for(str(tmp_path / "in_day"), observer="pvlive_in_day"))
    assert len(base) == 10
    with pytest.raises(ValueError, match="empty entry"):
        evaluate.prepare(args_for(f"{tmp_path / 'in_day'},"))


def test_prepare_rejects_unknown_forecasters_and_non_uniform_horizons(tmp_path):
    pytest.importorskip("pyarrow")
    write_export(tmp_path / "ok", range(0, 10), 20e9)
    with pytest.raises(ValueError, match="'nope'"):
        evaluate.prepare(args_for(str(tmp_path / "ok"), base="nope"))
    with pytest.raises(ValueError, match="'nope'"):
        evaluate.prepare(args_for(str(tmp_path / "ok"), intersect="nope"))
    write_export(tmp_path / "gappy", range(0, 10), 20e9, horizons=(0, 30, 90))
    with pytest.raises(ValueError, match="uniformly spaced"):
        evaluate.prepare(args_for(str(tmp_path / "gappy")))


def test_observation_only_directory_merges(tmp_path):
    """A backfill directory may hold observations.parquet alone."""
    pytest.importorskip("pyarrow")
    write_export(tmp_path / "main", range(0, 10), 20e9)
    extra = tmp_path / "extra"
    extra.mkdir()
    obs = pd.read_parquet(tmp_path / "main" / "observations.parquet")
    shifted = obs.copy()
    shifted["timestamp_utc"] = shifted["timestamp_utc"] - pd.Timedelta(days=400)
    shifted.to_parquet(extra / "observations.parquet", index=False)
    _, observations = io.load(f"{tmp_path / 'main'},{extra}")
    assert observations.index.min() == obs["timestamp_utc"].min() - pd.Timedelta(days=400)


def test_timezone_naive_file_is_rejected_by_name(tmp_path):
    """One naive datetime column fails loudly at load, naming the file and column."""
    pytest.importorskip("pyarrow")
    write_export(tmp_path / "main", range(0, 10), 20e9)
    bad = tmp_path / "bad"
    bad.mkdir()
    obs = pd.read_parquet(tmp_path / "main" / "observations.parquet")
    obs["created_time_utc"] = obs["created_time_utc"].dt.tz_localize(None)
    obs.to_parquet(bad / "observations.parquet", index=False)
    with pytest.raises(ValueError, match=r"created_time_utc.+timezone-naive"):
        io.load(f"{tmp_path / 'main'},{bad}")
