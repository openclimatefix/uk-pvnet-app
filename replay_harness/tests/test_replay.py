"""Synthetic-fixture tests for the replay invariants: floor, no lookahead, caps, smoothing."""

import numpy as np
import pandas as pd
import pytest

from replay_harness import candidates, io, metrics, replay

D0 = pd.Timestamp("2026-03-01", tz="UTC")
HORIZONS = [0, 30]


def make_inputs(days: int = 12, obs_value=0.4, forecast=0.5):
    """Daily 06:00 inits, constant forecast, half-hourly obs with stored created = target time."""
    inits = pd.DatetimeIndex([D0 + pd.Timedelta(days=d, hours=6) for d in range(days)])
    mat = pd.DataFrame(forecast, index=inits, columns=HORIZONS)
    times = pd.date_range(D0, D0 + pd.Timedelta(days=days + 1), freq="30min")
    obs = pd.DataFrame(
        {"value_fraction": obs_value, "capacity_watts": 1e11, "created_time_utc": times},
        index=times,
    )
    return mat, obs, inits


def test_floor_raises_early_created_times_and_keeps_late_ones():
    _, obs, _ = make_inputs()
    day4 = D0 + pd.Timedelta(days=4)
    obs.loc[day4 + pd.Timedelta(hours=10), "created_time_utc"] = day4 + pd.Timedelta(
        days=2, hours=3
    )  # late
    floored = replay.apply_floor(obs)["created_time_utc"]
    assert floored[day4 + pd.Timedelta(hours=10)] == day4 + pd.Timedelta(days=2, hours=3)
    assert floored[day4 + pd.Timedelta(hours=2)] == day4 + pd.Timedelta(
        days=1, hours=12
    )  # D+1 12:00
    assert (floored >= obs.index.normalize() + pd.Timedelta(hours=36)).all()


def test_floor_excludes_previous_day_obs_from_morning_init():
    mat, obs, inits = make_inputs()
    obs.loc[obs.index.normalize() == D0 + pd.Timedelta(days=9), "value_fraction"] = 0.0  # err 0.5
    t0 = inits[10]  # D10 06:00; D9 obs only exist from D10 12:00
    unfloored, n_raw = replay.replay_deltas(mat, obs, inits[[10]])
    floored, n_floor = replay.replay_deltas(mat, replay.apply_floor(obs), inits[[10]])
    assert n_raw.loc[t0, 0] == 8 and n_floor.loc[t0, 0] == 7
    assert floored.loc[t0, 0] == pytest.approx(0.1)
    assert unfloored.loc[t0, 0] == pytest.approx((7 * 0.1 + 0.5) / 8)


def test_no_lookahead_and_created_time_boundary_is_inclusive():
    mat, obs, inits = make_inputs()
    obs = replay.apply_floor(obs)
    t0 = inits[10]
    base, _ = replay.replay_deltas(mat, obs, inits[[10]])

    future = obs.copy()  # observations whose floored created time is after t0 must not matter
    future.loc[future["created_time_utc"] > t0, "value_fraction"] = 99.0
    assert replay.replay_deltas(mat, future, inits[[10]])[0].equals(base)

    past = obs.copy()  # but an available one must
    past.loc[past.index.normalize() == D0 + pd.Timedelta(days=5), "value_fraction"] = 0.0
    assert not replay.replay_deltas(mat, past, inits[[10]])[0].equals(base)

    on_time, late = obs.copy(), obs.copy()
    day8 = obs.index.normalize() == D0 + pd.Timedelta(days=8)
    on_time.loc[day8, "created_time_utc"] = t0
    late.loc[day8, "created_time_utc"] = t0 + pd.Timedelta(seconds=1)
    assert replay.replay_deltas(mat, on_time, inits[[10]])[1].loc[t0, 0] == 7
    assert replay.replay_deltas(mat, late, inits[[10]])[1].loc[t0, 0] == 6


def test_caps_order_and_clip():
    f = pd.DataFrame([[0.5, 0.05, 0.02, 0.0]])
    d = pd.DataFrame([[0.5, 0.5, -0.5, np.nan]])
    # capacity 10 GW -> absolute cap 0.1; then +-10% of forecast: 0.05, 0.005, 0.002
    out = replay.apply_adjuster(f, d, 10e9)
    assert out.iloc[0].tolist() == pytest.approx([0.45, 0.045, 0.022, 0.0])
    # capacity 20 GW -> absolute cap 0.05 binds before the 10% cap of 0.08
    assert replay.apply_adjuster(pd.DataFrame([[0.8]]), pd.DataFrame([[0.5]]), 20e9).iloc[
        0, 0
    ] == pytest.approx(0.75)


def test_gain_is_applied_before_the_caps():
    cols = list(range(0, 270, 30))
    f = pd.DataFrame([[0.8] * 9], columns=cols)
    huge = pd.DataFrame([[1.0] * 9], columns=cols)
    out = replay.apply_adjuster(f, candidates.w9_gain08(huge), 20e9)
    assert out.iloc[0, 4] == pytest.approx(0.75)  # 0.8 * 1.0 still exceeds the 0.05 cap
    small = pd.DataFrame([[0.04] * 9], columns=cols)
    assert replay.apply_adjuster(f, candidates.w9_gain08(small), 20e9).iloc[0, 4] == pytest.approx(
        0.8 - 0.032
    )


def test_smoothing_weights_are_triangular():
    impulse = pd.DataFrame([[0, 0, 0, 0, 9.0, 0, 0, 0, 0]])
    out = replay.smooth_across_horizon(impulse, 5).iloc[0].tolist()
    assert out == pytest.approx([0, 0, 1, 2, 3, 2, 1, 0, 0])  # weights 1,2,3,2,1 over a total of 9


def test_smoothing_ignores_nans_and_renormalises_at_edges():
    out = replay.smooth_across_horizon(pd.DataFrame([[np.nan, 1.0, 1.0]]), 3).iloc[0].tolist()
    assert out == pytest.approx([1.0, 1.0, 1.0])  # NaN centre filled from its neighbours
    assert replay.smooth_across_horizon(pd.DataFrame([[np.nan, np.nan]]), 3).isna().all().all()
    ramp = pd.DataFrame([[0.0, 1.0, 2.0, 3.0]])
    assert replay.smooth_across_horizon(ramp, 3).iloc[0, 0] == pytest.approx(
        (2 * 0 + 1 * 1) / 3
    )  # edge: weights 2,1


def test_window_bins_follow_horizon_spacing():
    assert replay.window_bins(pd.Index(range(0, 300, 30))) == 9
    assert replay.window_bins(pd.Index(range(0, 300, 15))) == 17
    assert replay.window_bins(pd.Index(range(0, 300, 60))) == 5


def test_run_blend_recursion_on_shared_target_times():
    idx = pd.DatetimeIndex(["2026-03-01 00:00", "2026-03-01 00:30", "2026-03-01 03:00"], tz="UTC")
    f = pd.DataFrame([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], index=idx, columns=HORIZONS)
    out = candidates.run_blend(f, alpha=0.5)
    assert out.loc[idx[1], 0] == pytest.approx(
        0.5 * 30 + 0.5 * 20
    )  # previous run's value at the same valid time
    assert out.loc[idx[1], 30] == 40  # no previous value at that valid time
    assert out.loc[idx[2]].tolist() == [50.0, 60.0]  # non-consecutive init: untouched


def test_longer_memory_uses_only_previous_occurrences():
    idx = pd.DatetimeIndex([D0 + pd.Timedelta(days=d, hours=6) for d in range(20)])
    d = pd.DataFrame(1.0, index=idx, columns=HORIZONS)
    d.iloc[-1] = 2.0
    out = candidates.longer_memory(d)
    assert out.iloc[-1, 0] == pytest.approx(
        0.4 * 2.0 + 0.6 * 1.0
    )  # trailing mean excludes the current delta
    assert out.iloc[0, 0] == 1.0  # no history: falls back to the latest delta


def test_build_stacks_and_rejects_bad_order():
    mat, _, inits = make_inputs()
    deltas = pd.DataFrame(0.01, index=inits, columns=HORIZONS)
    assert candidates.build("w9_gain08+run_blend", mat, deltas, 1e11).shape == deltas.shape
    with pytest.raises(ValueError):
        candidates.build("run_blend+w9", mat, deltas, 1e11)
    with pytest.raises(ValueError, match="unknown candidate 'nope'"):
        candidates.build("nope", mat, deltas, 1e11)


def test_run_blend_damps_against_the_blended_previous_row():
    idx = pd.DatetimeIndex(
        ["2026-03-01 00:00", "2026-03-01 00:30", "2026-03-01 01:00"],
        tz="UTC",
    )
    f = pd.DataFrame(
        [[10.0, 20.0, 30.0], [30.0, 40.0, 50.0], [50.0, 60.0, 70.0]],
        index=idx,
        columns=[0, 30, 60],
    )
    out = candidates.run_blend(f, alpha=0.5)
    assert out.iloc[1].tolist() == [25.0, 35.0, 50.0]
    # row 3 blends against row 2 AFTER damping (35), not its raw value (40)
    assert out.iloc[2, 0] == pytest.approx(0.5 * 50 + 0.5 * 35)


def test_longer_memory_window_min_periods_and_time_of_day_grouping():
    idx = pd.DatetimeIndex([D0 + pd.Timedelta(days=d, hours=6) for d in range(40)])
    ramp = pd.DataFrame({0: np.arange(40.0), 30: np.arange(40.0)}, index=idx)
    out = candidates.longer_memory(ramp)
    # previous 15 values are 24..38 (mean 31); a 30-value window would give a different answer
    assert out.iloc[-1, 0] == pytest.approx(0.4 * 39 + 0.6 * 31)
    assert out.iloc[7, 0] == pytest.approx(0.4 * 7 + 0.6 * 3)  # 7 prior values: just enough
    assert out.iloc[6, 0] == pytest.approx(
        6.0
    )  # 6 prior values: below min_periods, falls back to the delta

    two = pd.DatetimeIndex(
        [D0 + pd.Timedelta(days=d, hours=h) for d in range(20) for h in (6, 18)],
    )
    d = pd.DataFrame({0: np.where(two.hour == 6, 1.0, 5.0), 30: 0.0}, index=two)
    d.iloc[-2, 0] = 2.0  # last 06:00 init
    assert candidates.longer_memory(d).iloc[-2, 0] == pytest.approx(0.4 * 2.0 + 0.6 * 1.0)


def test_versions_of_one_forecaster_writing_one_cell_are_averaged():
    t = pd.Timestamp("2026-03-01", tz="UTC")
    row = {"init_time_utc": t, "horizon_mins": 0}
    df = pd.DataFrame(
        [
            {"forecaster": "blend:1.2.3", **row, "p50_fraction": 0.1},
            {"forecaster": "blend:1.3.0", **row, "p50_fraction": 0.3},
        ],
    )
    out = io.clean_forecasts(df)
    assert len(out) == 2  # not deduplicated across versions
    assert io.pivot(out, "blend").iloc[0, 0] == pytest.approx(0.2)


def test_check_grid_rejects_non_uniform_horizons_and_bad_init_step():
    idx = pd.date_range("2026-03-01", periods=4, freq="30min", tz="UTC")
    ok = pd.DataFrame(0.1, index=idx, columns=[0, 30, 60])
    metrics.check_grid(ok, "blend")
    with pytest.raises(ValueError, match=r"uniformly spaced.*\[30, 60\]"):
        metrics.check_grid(pd.DataFrame(0.1, index=idx, columns=[0, 30, 90]), "blend")
    quarter = pd.date_range("2026-03-01", periods=4, freq="15min", tz="UTC")
    with pytest.raises(ValueError, match=r"multiple of the horizon spacing"):
        metrics.check_grid(pd.DataFrame(0.1, index=quarter, columns=[0, 30, 60]), "blend")
    with pytest.raises(ValueError, match="at least two horizons"):
        metrics.check_grid(pd.DataFrame(0.1, index=idx, columns=[0]), "blend")


def test_pivot_names_the_available_forecasters_when_one_is_missing():
    df = pd.DataFrame(
        {
            "forecaster": ["blend"],
            "init_time_utc": [D0],
            "horizon_mins": [0],
            "p50_fraction": [0.1],
        },
    )
    with pytest.raises(ValueError, match=r"'nope'.*\['blend'\]"):
        io.pivot(df, "nope")


def test_warmup_inits_have_no_deltas_and_equal_the_raw_forecast():
    mat, obs, inits = make_inputs()
    deltas, counts = replay.replay_deltas(mat, replay.apply_floor(obs), inits[:2])
    assert deltas.isna().all().all()
    assert (counts == 0).all().all()
    adjusted = replay.apply_adjuster(mat, deltas, 22e9)
    pd.testing.assert_frame_equal(adjusted, mat.loc[deltas.index])


def test_duplicate_cells_keep_the_latest_created_row_whatever_the_input_order():
    t = pd.Timestamp("2026-03-01", tz="UTC")
    rows = [
        {"p50_fraction": 0.1, "created_time_utc": t + pd.Timedelta(hours=1)},
        {"p50_fraction": 0.3, "created_time_utc": t + pd.Timedelta(hours=3)},  # latest
        {"p50_fraction": 0.2, "created_time_utc": t + pd.Timedelta(hours=2)},
    ]
    for order in ([0, 1, 2], [2, 1, 0], [1, 0, 2]):
        df = pd.DataFrame(
            [
                {"forecaster": "blend:1.3.0", "init_time_utc": t, "horizon_mins": 0, **rows[i]}
                for i in order
            ],
        )
        assert io.pivot(io.clean_forecasts(df), "blend").iloc[0, 0] == pytest.approx(0.3)
    tied = pd.DataFrame(
        [
            {"forecaster": "blend:1.3.0", "init_time_utc": t, "horizon_mins": 0, "p50_fraction": v}
            | {"created_time_utc": t}
            for v in (0.1, 0.2)
        ],
    )
    assert io.pivot(io.clean_forecasts(tied), "blend").iloc[0, 0] == pytest.approx(0.2)  # later row


def test_a_late_backfill_loses_to_the_operational_row():
    t = pd.Timestamp("2026-03-10", tz="UTC")

    def pick(rows):
        frames = [
            {"forecaster": "blend:1.3.0", "init_time_utc": t, "horizon_mins": 0}
            | {"p50_fraction": value, "created_time_utc": t + delay}
            for value, delay in rows
        ]
        out = io.clean_forecasts(pd.DataFrame(frames))
        return io.pivot(out, "blend").iloc[0, 0]

    minute = pd.Timedelta(minutes=1)
    backfill = pd.Timedelta(days=35)
    for order in ([0, 1, 2], [2, 1, 0], [1, 2, 0]):
        rows = [(0.4, 10 * minute), (0.9, backfill), (0.2, 5 * minute)]
        assert pick([rows[i] for i in order]) == pytest.approx(0.4)  # latest operational row
        close = [(0.5, -minute), (0.6, minute), (0.9, backfill)]
        assert pick([close[i] for i in order]) == pytest.approx(0.6)  # 1 min later wins
        late_only = [(0.7, pd.Timedelta(days=10)), (0.8, pd.Timedelta(days=20)), (0.9, backfill)]
        assert pick([late_only[i] for i in order]) == pytest.approx(0.7)  # none operational
    edge = [(0.3, pd.Timedelta(hours=6)), (0.8, pd.Timedelta(hours=6, minutes=1))]
    assert pick(edge) == pytest.approx(0.3)  # exactly init + 6 h still counts as operational


def test_dedup_without_created_time_warns_and_keeps_the_first_row(caplog):
    t = pd.Timestamp("2026-03-01", tz="UTC")
    df = pd.DataFrame(
        [
            {"forecaster": "blend:1.3.0", "init_time_utc": t, "horizon_mins": 0, "p50_fraction": v}
            for v in (0.1, 0.9)
        ],
    )
    out = io.clean_forecasts(df)
    assert out["p50_fraction"].tolist() == [0.1]
    assert "created_time_utc" in caplog.text


def test_missing_observer_and_empty_data_entries_are_rejected(tmp_path):
    obs = pd.DataFrame(
        {
            "observer": ["pvlive_in_day"],
            "timestamp_utc": [D0],
            "value_fraction": [0.2],
            "created_time_utc": [D0],
        },
    )
    with pytest.raises(ValueError, match=r"pvlive_day_after.*\['pvlive_in_day'\]"):
        io.clean_observations(obs, "pvlive_day_after")
    for bad in ("a,,b", ",a", "a,"):
        with pytest.raises(ValueError, match="empty entry"):
            io.split_dirs(bad)
    with pytest.raises(ValueError, match="not found"):
        io.split_dirs(str(tmp_path / "nope"))


def test_dedup_and_version_collapse():
    t = pd.Timestamp("2026-03-01", tz="UTC")
    row = {"init_time_utc": t, "horizon_mins": 0, "p50_fraction": 0.1}
    df = pd.DataFrame(
        [
            {"forecaster": "blend:1.3.0", **row},
            {"forecaster": "blend:1.3.0", **row, "p50_fraction": 0.9},  # duplicate: first kept
            {"forecaster": "blend_adjust:1.2.3", **row},
        ],
    )
    out = io.clean_forecasts(df)
    assert sorted(out["forecaster"]) == ["blend", "blend_adjust"]
    assert io.pivot(out, "blend").iloc[0, 0] == 0.1

    obs = pd.DataFrame(
        {
            "observer": "pvlive_day_after",
            "timestamp_utc": [t, t],
            "value_fraction": [0.2, 0.3],
            "created_time_utc": [t + pd.Timedelta(hours=40), t + pd.Timedelta(hours=36)],
        },
    )
    kept = io.clean_observations(obs, "pvlive_day_after")
    assert len(kept) == 1 and kept["value_fraction"].iloc[0] == 0.2  # latest created wins
