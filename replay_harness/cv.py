"""Blocked seasonal cross-validation: per-fold MAE for the same candidates as evaluate.

    uv run --with pyarrow python -m replay_harness.cv --data data/ --candidates w9,w9_gain08

Folds are contiguous blocks of init dates. No shipped candidate fits parameters, so scoring needs
no gap between folds; a candidate that fits something must fit outside the scored fold, with a
gap of days on either side.
"""

import argparse
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from replay_harness import candidates, metrics
from replay_harness.evaluate import (
    add_data_args,
    default_out,
    prepare,
    short_sample_warning,
    variants,
)

MIN_FOLD_DAYS = 3
CANONICAL_UK_2026 = "2026-02-01,2026-04-01,2026-06-01,2026-08-01,2026-09-09,2026-10-17"

Fold = tuple[pd.Timestamp, pd.Timestamp]


def make_folds(index: pd.DatetimeIndex, n: int = 5) -> list[Fold]:
    """N contiguous [start, end) blocks of init dates with (nearly) equal numbers of days."""
    days = index.normalize().unique().sort_values()
    if n < 1 or len(days) < n:
        raise ValueError(f"cannot make {n} folds from {len(days)} scorable init days")
    blocks = np.array_split(np.arange(len(days)), n)
    return [(days[b[0]], days[b[-1]] + pd.Timedelta(days=1)) for b in blocks]


def folds_from_dates(dates: str) -> list[Fold]:
    """Comma-separated UTC boundary dates (n+1, strictly increasing) -> n consecutive folds."""
    edges = [pd.Timestamp(d, tz="UTC") for d in dates.split(",")]
    if len(edges) < 2:
        raise ValueError("--fold-dates needs at least two dates (n+1 boundaries for n folds)")
    if any(b <= a for a, b in pairwise(edges)):
        raise ValueError(f"--fold-dates must be strictly increasing, got {dates}")
    return list(pairwise(edges))


def fold_table(
    mats: dict[str, pd.DataFrame],
    obs: pd.DataFrame,
    final_mw: float,
    folds: list[Fold],
) -> pd.DataFrame:
    """MAE (per-row capacity) per fold, one row per variant, scored on matched points.

    Raises ValueError if a fold contains no scored init.
    """
    mats = metrics.matched(mats)
    errors = {k: metrics.errors_mw(m, obs, final_mw)[0] for k, m in mats.items()}
    index = next(iter(mats.values())).index
    scored = np.isfinite(next(iter(errors.values()))).any(axis=1)
    out = {}
    for start, end in folds:
        rows = (index >= start) & (index < end)
        label = f"{start:%Y-%m-%d}..{end:%Y-%m-%d}"
        if not (rows & scored).any():
            first, last = index[scored].min(), index[scored].max()
            raise ValueError(
                f"fold {label} has no scored inits (they span {first:%Y-%m-%d}..{last:%Y-%m-%d})",
            )
        out[f"fold {label}"] = {k: metrics.mae(e[rows]) for k, e in errors.items()}
    return pd.DataFrame(out)


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description=__doc__,
        epilog=f"Canonical UK-2026 folds (those the published adjuster claims were validated "
        f"on): --fold-dates {CANONICAL_UK_2026}",
    )
    add_data_args(p)
    p.add_argument(
        "--reference",
        default="current",
        help="candidate others must beat in each fold (default: current)",
    )
    p.add_argument(
        "--folds",
        type=int,
        default=5,
        help="number of automatic equal-day blocks (default: 5)",
    )
    p.add_argument(
        "--fold-dates",
        default=None,
        help="n+1 UTC boundary dates; overrides --folds (default: automatic blocks)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="CSV path (default: <first --data dir>/cv.csv)",
    )
    args = p.parse_args()

    try:
        names = args.candidates.split(",")
        if args.reference not in names:
            names = [args.reference, *names]
        for name in names:
            candidates.check(name)
        base, obs, deltas, final_mw = prepare(args)
        mats = variants(names, base, deltas, final_mw)
        scorable = metrics.scored_inits(mats["raw"], obs)
        if args.fold_dates:
            folds = folds_from_dates(args.fold_dates)
        else:  # blocks over inits that have observations
            folds = make_folds(scorable, args.folds)
        table = fold_table(mats, obs, final_mw, folds)
    except ValueError as e:
        p.error(str(e))

    gain = table.loc[args.reference] - table  # positive = better than the reference
    table["wins_all_folds"] = (gain > 0).all(axis=1) & (table.index != args.reference)
    pd.set_option("display.width", 250)
    if warning := short_sample_warning(scorable):
        print(warning, file=sys.stderr)
    for start, end in folds:
        days = scorable[(scorable >= start) & (scorable < end)].normalize().nunique()
        if days < MIN_FOLD_DAYS:
            print(
                f"WARNING: fold {start:%Y-%m-%d}..{end:%Y-%m-%d} has only {days} day(s) of "
                f"scored inits (under {MIN_FOLD_DAYS}): its MAE is noisy",
                file=sys.stderr,
            )
    print(f"MAE (MW, per-row capacity); reference = {args.reference}")
    print("fold columns are start..end with the end date exclusive\n")
    print(table.round(1).to_string())
    print("\nreference MAE minus candidate MAE per fold (positive = candidate better)")
    print(gain.round(2).to_string())

    out = Path(args.out) if args.out else default_out(args.data, "cv.csv")
    table.to_csv(out)
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
