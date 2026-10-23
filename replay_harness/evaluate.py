"""Score candidates on replayed history and print one table (the ladder format).

uv run --with pyarrow python -m replay_harness.evaluate --data data/ --candidates w9,w9_gain08
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from replay_harness import candidates, io, metrics, replay

SHORT_SAMPLE_DAYS = 60


def add_data_args(p: argparse.ArgumentParser) -> None:
    """Arguments shared by the evaluate and cv CLIs."""
    p.add_argument(
        "--data", required=True, help="export dir(s), comma-separated, merged (required)"
    )
    p.add_argument(
        "--candidates",
        default="current,w9,w9_gain08",
        help="names, stack with + (default: current,w9,w9_gain08)",
    )
    p.add_argument(
        "--observer",
        default="pvlive_day_after",
        help="observation source to score on (default: pvlive_day_after)",
    )
    p.add_argument(
        "--base",
        default="blend",
        help="forecaster whose output is adjusted (default: blend)",
    )
    p.add_argument(
        "--intersect",
        default="",
        help="forecasters whose inits must also exist (default: none)",
    )
    p.add_argument(
        "--warmup-days",
        type=int,
        default=14,
        help="skip this long after the first init (default: 14)",
    )
    p.add_argument(
        "--lookback-days",
        type=int,
        default=8,
        help="days of history in the week-average deltas (default: 8)",
    )


def prepare(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """Load data and replay the floored deltas. Returns (base matrix, obs, deltas, final MW).

    Final capacity is the base forecaster's capacity on its latest init. Raises ValueError for an
    unknown forecaster, a non-uniform horizon grid or no inits left after the warmup.
    """
    forecasts, obs = io.load(args.data, args.observer)
    base = io.pivot(forecasts, args.base)
    metrics.check_grid(base, args.base)
    inits = base.index[base.index >= base.index.min() + pd.Timedelta(days=args.warmup_days)]
    for name in filter(None, args.intersect.split(",")):
        inits = inits.intersection(io.pivot(forecasts, name).index)
    if len(inits) < 2:
        raise ValueError(f"only {len(inits)} init(s) left after --warmup-days and --intersect")
    deltas, _ = replay.replay_deltas(base, replay.apply_floor(obs), inits, args.lookback_days)
    return base, obs, deltas, io.latest_capacity_mw(forecasts, args.base)


def variants(
    names: list[str],
    base: pd.DataFrame,
    deltas: pd.DataFrame,
    final_mw: float,
) -> dict[str, pd.DataFrame]:
    """The raw base forecast plus one adjusted matrix per candidate name."""
    out = {"raw": base.loc[deltas.index]}
    for name in names:
        out[name] = candidates.build(name, base, deltas, final_mw * 1e6)
    return out


def default_out(data: str, name: str) -> Path:
    """`<first --data dir>/<name>`."""
    return Path(data.split(",")[0]) / name


def short_sample_warning(scored: pd.DatetimeIndex, min_days: int = SHORT_SAMPLE_DAYS) -> str | None:
    """A warning when the scored inits span fewer than `min_days` days, else None."""
    span = (scored.max() - scored.min()) / pd.Timedelta(days=1) if len(scored) else 0.0
    if span >= min_days:
        return None
    return (
        f"WARNING: scored inits span only {span:.0f} days (under {min_days}): results on short "
        f"exports are indicative only; the warmup consumes the first --warmup-days"
    )


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    add_data_args(p)
    p.add_argument(
        "--out", default=None, help="CSV path (default: <first --data dir>/evaluate.csv)"
    )
    args = p.parse_args()

    try:
        names = args.candidates.split(",")
        for name in names:
            candidates.check(name)
        base, obs, deltas, final_mw = prepare(args)
        mats = variants(names, base, deltas, final_mw)
    except ValueError as e:
        p.error(str(e))
    table = metrics.score(mats, obs, final_mw)
    monthly = metrics.monthly_bias(mats, obs, final_mw)

    pd.set_option("display.width", 250)
    if warning := short_sample_warning(metrics.scored_inits(mats["raw"], obs)):
        print(warning, file=sys.stderr)
    print(f"final capacity {final_mw:.0f} MW; {len(deltas)} inits; MW, matched points\n")
    print(table.round(1).to_string())
    print("\nbias by init month (MW, per-row capacity)")
    print(monthly.round(1).to_string())

    out = Path(args.out) if args.out else default_out(args.data, "evaluate.csv")
    table.to_csv(out)
    monthly.to_csv(out.with_name(f"{out.stem}_bias_by_month.csv"))
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
