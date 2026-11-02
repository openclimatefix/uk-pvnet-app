"""Metrics on matched points. Forecast matrices: index = init time, columns = horizon minutes.

Capacity basis: errors are converted to MW either per row (capacity of the observation at the
valid time, `capacity_grid`) or with one fixed final capacity (`final_mw`). `score` reports both
for MAE; bias and tails use per row; the smoothness metrics use the fixed basis.
"""

import numpy as np
import pandas as pd


def init_step(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Spacing between consecutive inits (the most common gap)."""
    return index.to_series().diff().mode().iloc[0]


def check_grid(mat: pd.DataFrame, name: str) -> None:
    """Raise ValueError unless horizons are uniformly spaced and the init step is a multiple.

    Run-to-run, run blending and jerkiness shift or difference by column position, which is only
    correct on a uniform horizon grid whose spacing divides the init step.
    """
    if len(mat) < 2:
        raise ValueError(f"{name!r}: need at least two init times, got {len(mat)}")
    horizons = np.asarray(mat.columns, dtype=int)
    if len(horizons) < 2:
        raise ValueError(f"{name!r}: need at least two horizons, got {horizons.tolist()}")
    gaps = np.unique(np.diff(horizons))
    if len(gaps) != 1:
        raise ValueError(
            f"{name!r}: horizons must be uniformly spaced, found gaps of {gaps.tolist()} minutes "
            f"(a missing horizon in the export?)",
        )
    step_mins = init_step(mat.index) / pd.Timedelta(minutes=1)
    if step_mins % gaps[0] != 0:
        raise ValueError(
            f"{name!r}: init step ({step_mins:g} min) must be a multiple of the horizon spacing "
            f"({gaps[0]} min)",
        )


def run_shift(index: pd.DatetimeIndex, horizons: pd.Index) -> int:
    """Columns between the same valid time in consecutive runs (init step / horizon spacing)."""
    return int(init_step(index) / pd.Timedelta(minutes=int(horizons[1] - horizons[0])))


def matched(mats: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Restrict every matrix to the common inits and blank any cell missing in any matrix."""
    index = None
    for m in mats.values():
        index = m.index if index is None else index.intersection(m.index)
    cut = {k: m.loc[index] for k, m in mats.items()}
    hole = np.zeros(next(iter(cut.values())).shape, dtype=bool)
    for m in cut.values():
        hole |= m.isna().to_numpy()
    return {k: m.mask(hole) for k, m in cut.items()}


def error_grid(mat: pd.DataFrame, obs_value: pd.Series) -> np.ndarray:
    """Forecast - observation as a fraction of capacity, shape [init, horizon]; NaN if no obs."""
    cols = [
        obs_value.reindex(mat.index + pd.Timedelta(minutes=int(h))).to_numpy() for h in mat.columns
    ]
    return mat.to_numpy(dtype=float) - np.stack(cols, axis=1)


def capacity_grid(mat: pd.DataFrame, obs: pd.DataFrame, final_mw: float) -> np.ndarray:
    """Per-row capacity in MW (observation capacity at the valid time), `final_mw` if unknown."""
    cap = obs["capacity_watts"] / 1e6
    cols = [cap.reindex(mat.index + pd.Timedelta(minutes=int(h))).to_numpy() for h in mat.columns]
    grid = np.stack(cols, axis=1)
    return np.where(np.isnan(grid), final_mw, grid)


def errors_mw(
    mat: pd.DataFrame, obs: pd.DataFrame, final_mw: float
) -> tuple[np.ndarray, np.ndarray]:
    """Errors in MW on the (per-row capacity, fixed final capacity) bases."""
    frac = error_grid(mat, obs["value_fraction"])
    return frac * capacity_grid(mat, obs, final_mw), frac * final_mw


def scored_inits(mat: pd.DataFrame, obs: pd.DataFrame) -> pd.DatetimeIndex:
    """Inits with at least one scorable (forecast and observation) cell."""
    finite = np.isfinite(error_grid(mat, obs["value_fraction"])).any(axis=1)
    return mat.index[finite]


def mae(e_mw: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.nanmean(np.abs(e_mw)))


def bias(e_mw: np.ndarray) -> float:
    """Mean signed error (forecast minus observation)."""
    return float(np.nanmean(e_mw))


def p99_abs_error(e_mw: np.ndarray) -> float:
    """99th percentile of the absolute error."""
    return float(np.nanquantile(np.abs(e_mw), 0.99))


def _group_mean(values: np.ndarray, index: pd.DatetimeIndex, key: object) -> pd.Series:
    """Mean of all finite cells per group of inits."""
    total = pd.Series(np.nansum(values, axis=1), index=index).groupby(key).sum()
    count = pd.Series(np.isfinite(values).sum(axis=1), index=index).groupby(key).sum()
    return (total / count.where(count > 0)).dropna()


def daily_mae(e_mw: np.ndarray, index: pd.DatetimeIndex) -> pd.Series:
    """MAE per init date."""
    return _group_mean(np.abs(e_mw), index, index.date)


def worst_decile_daily_mae(e_mw: np.ndarray, index: pd.DatetimeIndex) -> float:
    """Mean of the highest 10% of daily MAEs (at least one day)."""
    daily = daily_mae(e_mw, index)
    return float(daily.nlargest(max(len(daily) // 10, 1)).mean())


def bias_by_month(e_mw: np.ndarray, index: pd.DatetimeIndex) -> pd.Series:
    """Mean signed error per init month; pooled bias can hide opposite-signed months."""
    return _group_mean(e_mw, index, index.tz_localize(None).to_period("M"))


def jerkiness(mat_mw: np.ndarray) -> float:
    """Within-run: mean |second difference| along each forecast trajectory."""
    return float(np.nanmean(np.abs(np.diff(mat_mw, n=2, axis=1))))


def run_to_run(mat_mw: np.ndarray, index: pd.DatetimeIndex, horizons: pd.Index) -> float:
    """Run-to-run: mean |f_t(v) - f_{t-step}(v)| at shared target times v, consecutive inits."""
    shift = run_shift(index, horizons)
    consecutive = (index.to_series().diff() == init_step(index)).to_numpy()[1:]
    revision = np.abs(mat_mw[1:, :-shift] - mat_mw[:-1, shift:])[consecutive]
    return float(np.nanmean(revision))


def monthly_bias(
    mats: dict[str, pd.DataFrame],
    obs: pd.DataFrame,
    final_mw: float,
) -> pd.DataFrame:
    """Per-row-capacity bias (MW) by init month: one row per variant, matched points."""
    return pd.DataFrame(
        {
            k: bias_by_month(errors_mw(m, obs, final_mw)[0], m.index)
            for k, m in matched(mats).items()
        },
    ).T


def score(
    mats: dict[str, pd.DataFrame],
    obs: pd.DataFrame,
    final_mw: float,
    reference: str = "raw",
) -> pd.DataFrame:
    """One row per variant, scored on matched points, columns in the reporting hierarchy.

    mae (per-row capacity) and mae_fixed (final capacity) are the gate; p99_err and worst_decile
    are tail guardrails; jerk (objective) and run2run (hold) use the fixed basis; me is the
    pooled bias (see `monthly_bias` for the by-month view).
    """
    rows = {}
    for name, mat in matched(mats).items():
        e_row, e_fix = errors_mw(mat, obs, final_mw)
        fixed = mat.to_numpy(dtype=float) * final_mw
        rows[name] = {
            "mae": mae(e_row),
            "mae_fixed": mae(e_fix),
            "p99_err": p99_abs_error(e_row),
            "worst_decile": worst_decile_daily_mae(e_row, mat.index),
            "jerk": jerkiness(fixed),
            "run2run": run_to_run(fixed, mat.index, mat.columns),
            "me": bias(e_row),
        }
    table = pd.DataFrame.from_dict(rows, orient="index")
    if reference in table.index:
        table["mae_vs_ref_%"] = (1 - table["mae"] / table.loc[reference, "mae"]) * 100
    return table
