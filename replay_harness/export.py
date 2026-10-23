r"""Export forecasts and observations from the Data Platform to local parquet.

    DATA_PLATFORM_HOST=localhost DATA_PLATFORM_PORT=50051 \
    uv run --with pyarrow --with "dp-sdk @ <0.40.5 wheel URL>" python -m replay_harness.export \
        --forecasters blend,blend_adjust --location national \
        --start 2025-10-31 --end 2026-10-17 --out data/

Writes <out>/forecasts.parquet and <out>/observations.parquet, assembled from one parquet per
30-day request window under <out>/chunks/. A chunk that already exists is skipped, so an
interrupted export resumes and a repeated one is a no-op. Every version of each forecaster name
is requested. Observations carry created_time_utc (needs dp-sdk >= 0.40).

--location is "national" (the UK national solar location) or a location uuid.
"""

import argparse
import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

CHUNK = timedelta(days=30)  # the platform rejects windows of 31 days or more


Window = tuple[datetime, datetime]


def chunks(start: datetime, end: datetime) -> list[Window]:
    """Consecutive [start, end) windows of at most CHUNK covering [start, end)."""
    out, t = [], start
    while t < end:
        out.append((t, min(t + CHUNK, end)))
        t += CHUNK
    return out


def chunk_path(out: Path, kind: str, location_uuid: str, name: str, window: Window) -> Path:
    """File holding one request window's rows for one location and forecaster set or observer."""
    stem = f"{kind}_{location_uuid}_{name}_{window[0]:%Y%m%d}_{window[1]:%Y%m%d}"
    return out / "chunks" / f"{stem}.parquet"


async def resolve_location(client: Any, location: str) -> str:  # noqa: ANN401
    """Location uuid: the UK national solar location for "national", else the value itself."""
    if location != "national":
        return location
    from ocf import dp

    found = (
        await client.list_locations(
            dp.ListLocationsRequest(
                location_type_filter=dp.LocationType.NATION,
                energy_source_filter=dp.EnergySource.SOLAR,
                location_names_filter=["uk"],
            ),
        )
    ).locations
    if len(found) != 1:
        raise ValueError(f"Expected exactly one UK national location, got: {found}")
    return found[0].location_uuid


async def fetch_forecasts(
    client: Any,  # noqa: ANN401
    location_uuid: str,
    forecasters: list,
    window: Window,
) -> pd.DataFrame:
    """Stream one window of forecasts (all requested forecaster versions)."""
    from ocf import dp

    request = dp.StreamForecastDataRequest(
        location_uuids=[location_uuid],
        energy_source=dp.EnergySource.SOLAR,
        time_window=dp.TimeWindow(start_timestamp_utc=window[0], end_timestamp_utc=window[1]),
        forecasters=forecasters,
    )
    rows = []
    async for chunk in client.stream_forecast_data(request):
        for v in chunk.values:
            row = {
                "forecaster": v.forecaster_fullname,
                "init_time_utc": v.init_timestamp,
                "horizon_mins": v.horizon_mins,
                "p50_fraction": v.p50_fraction,
                "created_time_utc": v.created_timestamp_utc,
                "capacity_watts": v.effective_capacity_watts,
            }
            row.update({f"{k}_fraction": val for k, val in v.other_statistics_fractions.items()})
            rows.append(row)
    return pd.DataFrame(rows)


async def fetch_observations(
    client: Any,  # noqa: ANN401
    location_uuid: str,
    observer: str,
    window: Window,
) -> pd.DataFrame:
    """Fetch one window of one observer's observations, with created times."""
    from ocf import dp

    response = await client.get_observations_as_timeseries(
        dp.GetObservationsAsTimeseriesRequest(
            location_uuid=location_uuid,
            observer_name=observer,
            energy_source=dp.EnergySource.SOLAR,
            time_window=dp.TimeWindow(start_timestamp_utc=window[0], end_timestamp_utc=window[1]),
        ),
    )
    return pd.DataFrame(
        [
            {
                "observer": observer,
                "timestamp_utc": v.timestamp_utc,
                "value_fraction": v.value_fraction,
                "capacity_watts": v.effective_capacity_watts,
                "created_time_utc": getattr(v, "created_timestamp_utc", None),
            }
            for v in response.values
        ],
    )


async def cached(
    path: Path,
    fetch: Callable[[], Awaitable[pd.DataFrame]],
    label: str = "",
) -> pd.DataFrame:
    """Return the saved chunk if present, else fetch it and save; `label` names the request."""
    if path.exists():
        print(f"exists, skipping {path.name}")
        return pd.read_parquet(path)
    print(f"requesting {label}", flush=True)
    df = await fetch()
    df.to_parquet(path, index=False)
    print(f"  written {path.name}: {len(df)} rows", flush=True)
    if df.empty:
        logger.warning("WARNING: %s returned 0 rows (wrong tunnel, location or dates?)", label)
    return df


def window_label(window: Window, what: str) -> str:
    """`2026-03-01..2026-03-31 (blend)`: the request window (end exclusive) and its source."""
    return f"{window[0]:%Y-%m-%d}..{window[1]:%Y-%m-%d} ({what})"


def assemble(
    out: Path, forecast_frames: list[pd.DataFrame], observation_frames: list[pd.DataFrame]
) -> None:
    """Merge the per-window frames into the two files `io.load` reads (rewritten on every run)."""
    pd.concat(forecast_frames, ignore_index=True).to_parquet(out / "forecasts.parquet", index=False)
    pd.concat(observation_frames, ignore_index=True).to_parquet(
        out / "observations.parquet", index=False
    )


async def export(args: argparse.Namespace) -> None:
    """Fetch every missing window, then assemble the two canonical files."""
    from grpclib.client import Channel
    from ocf import dp

    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end = datetime.fromisoformat(args.end).replace(tzinfo=UTC)
    out = Path(args.out)
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    windows = chunks(start, end)

    host = os.environ.get("DATA_PLATFORM_HOST", "localhost")
    port = int(os.environ.get("DATA_PLATFORM_PORT", "50051"))
    async with Channel(host, port) as channel:
        client = dp.DataPlatformDataServiceStub(channel)
        location_uuid = await resolve_location(client, args.location)

        forecasters = []
        for name in args.forecasters.split(","):
            found = (
                await client.list_forecasters(
                    dp.ListForecastersRequest(forecaster_names_filter=[name])
                )
            ).forecasters
            if not found:
                raise ValueError(f"No forecaster named {name!r}")
            forecasters.extend(found)  # every version; empty ones contribute no rows

        tag = args.forecasters.replace(",", "-")
        forecast_frames = [
            await cached(
                chunk_path(out, "forecasts", location_uuid, tag, w),
                lambda w=w: fetch_forecasts(client, location_uuid, forecasters, w),
                window_label(w, args.forecasters),
            )
            for w in windows
        ]
        observation_frames = [
            await cached(
                chunk_path(out, "observations", location_uuid, obs, w),
                lambda w=w, obs=obs: fetch_observations(client, location_uuid, obs, w),
                window_label(w, obs),
            )
            for obs in args.observers.split(",")
            for w in windows
        ]
        assemble(out, forecast_frames, observation_frames)
    print(f"done: {out.resolve()}")


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--forecasters",
        default="blend,blend_adjust",
        help="comma-separated names, all versions fetched (default: blend,blend_adjust)",
    )
    p.add_argument(
        "--observers",
        default="pvlive_day_after,pvlive_in_day",
        help="comma-separated names (default: pvlive_day_after,pvlive_in_day)",
    )
    p.add_argument(
        "--location",
        default="national",
        help='"national" or a location uuid (default: national)',
    )
    p.add_argument("--start", required=True, help="UTC date, e.g. 2025-10-31 (required)")
    p.add_argument("--end", required=True, help="UTC date, exclusive, e.g. 2026-10-17 (required)")
    p.add_argument("--out", default="./data", help="output directory (default: ./data)")
    asyncio.run(export(p.parse_args()))


if __name__ == "__main__":
    main()
