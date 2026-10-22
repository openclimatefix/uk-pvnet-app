"""Load, dedup and pivot exported parquet directories."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

BACKFILL_AFTER = pd.Timedelta(hours=6)  # a row created later than init + this is a backfill


def split_dirs(data: str) -> list[str]:
    """Split comma-separated export directories, rejecting empty entries and missing paths."""
    dirs = data.split(",")
    if not all(dirs):
        raise ValueError(f"--data has an empty entry (stray comma?): {data!r}")
    missing = [d for d in dirs if not Path(d).is_dir()]
    if missing:
        raise ValueError(f"--data directory not found: {missing}")
    return dirs


def clean_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    """Dedup on (versioned forecaster, init, horizon), then collapse `name:x.y.z` to `name`.

    When a versioned cell was written more than once, the row with the latest created_time_utc
    at or before init + BACKFILL_AFTER (6 hours) wins (ties: the later row after a stable sort);
    if every row was written later than that, the earliest row wins. A late rewrite is a
    backfill, not what production published, and using it would leak future values into later
    lookbacks. The result does not depend on row or directory order. Without a created_time_utc
    column the first row wins and a warning is logged. Dedup runs before the collapse, so the same
    (init, horizon) written by two versions of one forecaster survives as two rows and `pivot`
    averages them.
    """
    key = ["forecaster", "init_time_utc", "horizon_mins"]
    if "created_time_utc" in df.columns:
        df = df.sort_values("created_time_utc", kind="stable", na_position="first")
        operational = ~(df["created_time_utc"] > df["init_time_utc"] + BACKFILL_AFTER)
        has_operational = operational.groupby([df[k] for k in key], sort=False).transform("any")
        latest = df[operational].drop_duplicates(subset=key, keep="last")
        earliest = df[~has_operational].drop_duplicates(subset=key, keep="first")
        df = pd.concat([latest, earliest])
    else:
        logger.warning("forecasts have no created_time_utc column: duplicates keep the first row")
        df = df.drop_duplicates(subset=key)
    df = df.copy()
    df["forecaster"] = df["forecaster"].str.split(":").str[0]
    return df


def clean_observations(df: pd.DataFrame, observer: str) -> pd.DataFrame:
    """One observer, deduped on timestamp keeping the latest created, indexed by timestamp."""
    obs = df[df["observer"] == observer]
    if obs.empty:
        raise ValueError(
            f"no observations from observer {observer!r}; the data has "
            f"{sorted(df['observer'].unique())} (see --observer)",
        )
    return (
        obs.sort_values("created_time_utc", kind="stable")
        .drop_duplicates("timestamp_utc", keep="last")
        .set_index("timestamp_utc")
        .sort_index()
    )


def _read_utc(path: Path) -> pd.DataFrame:
    """Read one parquet file, requiring every datetime column to be tz-aware.

    A tz-naive column mixed into tz-aware data makes pandas fail much later with an
    opaque comparison error, so the mismatch is named here, at the file that has it.
    """
    df = pd.read_parquet(path)
    naive = [
        col
        for col in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[col])
        and getattr(df[col].dtype, "tz", None) is None
    ]
    if naive:
        raise ValueError(
            f"{path}: column(s) {naive} are timezone-naive; exports store tz-aware UTC "
            f"timestamps throughout — localise these to UTC before loading",
        )
    return df


def load(data: str, observer: str = "pvlive_day_after") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one or several (comma-separated) export directories.

    A directory may hold only one of the two files (e.g. an observation-only backfill).
    Returns (forecasts, observations): forecasts is long-format with unversioned forecaster
    names; observations is indexed by timestamp_utc for the chosen observer.
    """
    dirs = split_dirs(data)
    frames: dict[str, list[pd.DataFrame]] = {"forecasts": [], "observations": []}
    for d in dirs:
        for kind, found in frames.items():
            path = Path(d) / f"{kind}.parquet"
            if path.exists():
                found.append(_read_utc(path))
    for kind, found in frames.items():
        if not found:
            raise ValueError(f"no {kind}.parquet in any of the --data directories: {dirs}")
    forecasts = pd.concat(frames["forecasts"])
    observations = pd.concat(frames["observations"])
    return clean_forecasts(forecasts), clean_observations(observations, observer)


def pivot(forecasts: pd.DataFrame, forecaster: str) -> pd.DataFrame:
    """P50 matrix for one forecaster: index init_time_utc, columns horizon_mins."""
    sub = forecasts[forecasts["forecaster"] == forecaster]
    if sub.empty:
        known = sorted(forecasts["forecaster"].unique())
        raise ValueError(f"no forecasts for forecaster {forecaster!r}; the data has {known}")
    return sub.pivot_table(
        index="init_time_utc",
        columns="horizon_mins",
        values="p50_fraction",
    ).sort_index()


def latest_capacity_mw(forecasts: pd.DataFrame, forecaster: str) -> float:
    """Capacity (MW) on the forecaster's latest init: independent of directory order."""
    sub = forecasts[forecasts["forecaster"] == forecaster]
    latest = sub.loc[sub["init_time_utc"] == sub["init_time_utc"].max(), "capacity_watts"]
    return float(latest.max()) / 1e6
