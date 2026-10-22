# replay_harness

Offline evaluation of forecast post-processing against replayed production history.

The question it answers: *if the published forecast had been produced with mechanism X instead of
mechanism Y, over a long stretch of real history, which would have been better, on which metrics,
and with what confidence?* Nothing is deployed to find out.

It does four things:

1. **Export** forecasts (any forecasters, any location) and observations from the Data Platform
   to local parquet. Resumable (per-window chunk cache; delete chunks to force a refetch).
2. **Replay** a mechanism offline as production would have run it, using only information
   available at each init time.
3. **Score** a fixed metric set on matched points, so every variant is judged on identical
   (init, target) pairs.
4. **Validate** with blocked seasonal folds.

Pure pandas, numpy and pyarrow. Only `export.py` needs the Data Platform SDK. Nothing imports
`pvnet_app`, so the package can move to its own repository unchanged.

## Quickstart

```bash
# 1. open the Data Platform tunnel (dev and prod both end at localhost:50051: check which is up)
export DATA_PLATFORM_HOST=localhost DATA_PLATFORM_PORT=50051
# 2. export; the repo pins dp-sdk 0.31, which lacks observation created times, so overlay 0.40.5
DP="dp-sdk @ https://github.com/openclimatefix/data-platform/releases/download/v0.40.5/dp_sdk-0.40.5-py3-none-any.whl"
uv run --with pyarrow --with "$DP" python -m replay_harness.export \
    --forecasters blend,blend_adjust --location national \
    --start 2025-10-31 --end 2026-10-17 --out data/
# 3. score candidates; prints the table and writes data/evaluate.csv
uv run --with pyarrow python -m replay_harness.evaluate --data data/ --candidates current,w9,w9_gain08
# 4. per-fold table, and which candidates win every fold
uv run --with pyarrow python -m replay_harness.cv --data data/ --candidates w9,w9_gain08
# 5. tests (synthetic fixtures always; the equivalence test only when the local caches exist)
uv run --with pyarrow pytest replay_harness/tests -q
```

The first export run needs outbound access to GitHub: `uv` downloads the dp-sdk wheel from the
release URL above. Evaluate and cv print a warning (they never fail) when the scored inits span
under 60 days, and cv when a fold has under 3 days of scored inits.

### Options and files

- `--data`: one or several export directories, comma-separated (no empty entries); they are
  merged and deduplicated (see the Dedup invariant).
- `--location` (export): `national` (the UK national solar location) or a location uuid. Horizon
  spacing, capacity and init cadence are read from the data, so other locations work unchanged
  provided day-after observations exist (`--observer` selects the name).
- `--observer` (evaluate, cv): the observation source scored against. Default
  `pvlive_day_after`. It is a hard error if the data holds no observations from it; the message
  names the observers present. Export fetches `--observers pvlive_day_after,pvlive_in_day`.
- `--base` (evaluate, cv): the forecaster whose output is adjusted. Default `blend`.
- `--intersect` (evaluate, cv): comma-separated forecasters whose init times must also exist, for
  comparing against what production published (e.g. `blend_adjust`). Default none.
- `--lookback-days`: days of history in the week-average deltas. Default 8.
- `--warmup-days`: inits earlier than this after the first init are not scored. Default 14.
- `--candidates`: comma-separated candidate names. A candidate's CLI name is its function name in
  `candidates.py`.
- `--out`: CSV path. Default `<first --data dir>/evaluate.csv` (evaluate) or `.../cv.csv` (cv).
  Evaluate also writes `<out stem>_bias_by_month.csv`: bias per variant per init month.

**Export layout.** `--out` receives `forecasts.parquet` and `observations.parquet`, the two files
the loader reads, rewritten at the end of every run. Each 30-day request window is also saved
under `<out>/chunks/` (one parquet per window); that is the resume mechanism, and a window that
already exists there is not requested again. Chunk file names carry the resolved location uuid
and the forecaster set or observer, so a different location or forecaster list never reuses
another's chunks. Windows are 30 days long and start at `--start`, so their boundaries are fixed
by `--start` alone: extending `--end` reuses every full window and fetches only the new ones,
and the old partial last window (its file name carries its end date) is fetched again as part of
a new window. The canonical files are assembled from the current run's windows only, so there is
no overlap to remove. Changing `--start` moves every boundary and refetches everything. Each
request prints `requesting <start>..<end> (<source>)` (end exclusive) and a row count, and a
window that returns 0 rows logs a warning. A cached window is frozen as it was fetched: a window exported before its
day-after observations were published is not refreshed by re-running with the same `--end`;
delete its chunk file to refetch. Chunk files from before names carried the location are
unsafe to reuse: delete `<out>/chunks/` (the two canonical files are rewritten on every run
anyway).

## What is known at t0

```
 day D-8 ......... D-2        D-1                  D (today)
 runs at the same  |           |                   |
 time of day  ---->|---------->|------------------>|  t0 = init time of the run being adjusted

 forecast f(init, h) targets init + h.  Its error is f - observation(init + h).

 observation for day X (day-after PVLive):
     usable from  max(stored created time, X 00:00 + 36h)  =  X+1 12:00 UTC

 at t0 the adjuster may use only errors whose observation is usable at or before t0:
   - runs from t0-1d ... t0-8d (t0's own run does not exist yet)
   - for each horizon h, average (f - obs) over those runs  ->  delta[t0, h]

 published forecast = clip( f(t0, h) - cap( smooth(delta[t0, :])[h] * gain ), 0 )
```

The init cadence is read from the data; the example's daily same-time-of-day runs are
illustrative (the lookback always uses the run at the same time of day on earlier days).

Worked example, t0 = day D 08:00 UTC. Observations for day X are usable from X+1 12:00, so at
D 08:00 the days D-2 and earlier are usable and D-1 is not (it appears at D 12:00).

| Run (init) | Targets it covers (up to 37 h) | Usable pairs |
|---|---|---|
| D-1 08:00 | D-1 08:00 ... D 21:00 | none: every target is on D-1 or D |
| D-2 08:00 | D-2 08:00 ... D-1 21:00 | only targets on D-2 (horizons below 16 h) |
| D-3 08:00 | D-3 08:00 ... D-2 21:00 | all |
| D-4 ... D-8 08:00 | earlier days | all |

So a morning t0 gets 6 or 7 contributing days per horizon, not 8, and the `counts` returned by
`replay.replay_deltas` show it. Averaged over all init times of day the mean is about 6.5, because
an afternoon t0 (after 12:00) can already use D-1. Production behaves the same way, which is why
the replay needs the floor.

## Invariants

Each has a test in `tests/`, except the fitting-gap rule under Cross-validation, which is
guidance for future fitted candidates.

- **Observation availability floor.** Day-after PVLive is usable only from 12:00 UTC the next
  day. Replays set `created_time = max(stored_created_time, obs_day + 36h)` before any delta is
  computed. Stored created times are backfilled for part of the history, so without the floor a
  replay reads data production could not have had.
- **No lookahead.** A delta for init t0 uses only observations whose floored created time is
  <= t0 (inclusive). Otherwise a candidate looks better by seeing the answer.
- **Matched points.** Every variant in a comparison is scored on the intersection of available
  (init, target) cells. A variant that fails to publish a hard cell must not score better for it.
- **Dedup.** Forecast rows are deduplicated on (versioned forecaster, init, horizon), keeping the
  latest row created at or before init + 6 h (the earliest row if none qualifies; ties: the later
  row after a stable sort), because a late rewrite is a backfill and not what production
  published, and scoring it would also leak future values into later lookbacks. The result does
  not depend on row or directory order. Then versions `name:x.y.z` are collapsed to `name` (a
  cell written by two versions is averaged); all versions are requested at export. Observations
  are deduplicated on timestamp, keeping the latest created. Forecasts without a
  `created_time_utc` column fall back to keeping the first row, with a logged warning. The
  original ad hoc analysis kept the first row in directory order; on the shipped data this policy
  differs from it by at most 0.3 MW on any pinned number, and no ranking changes.
- **Capacity basis is explicit.** MW figures are either *per-row* (the observation's capacity at
  the valid time) or *fixed* (the latest capacity in the data). `mae`, `p99_err`, `worst_decile`
  and `me` use per-row; `mae_fixed`, `jerk` and `run2run` use fixed. Per-row falls back to fixed
  where the observation has no capacity. The final capacity is the base forecaster's capacity on
  its latest init, so it does not depend on the order of `--data`. The absolute 1 GW cap uses this
  final capacity for the whole replay, whereas production recomputes it per run from the
  capacity current at the time. This is inherited from the audited analysis and its effect is
  small (about 0.05 MW on per-row MAE for a 2.6% capacity change).
- **Warmup artefact.** The first weeks of any export have no error history, so the replay makes
  no adjustment there (deltas empty) and every adjusted variant equals the raw forecast. That
  depresses adjusted variants' performance in those weeks and can dominate the largest-magnitude
  month in `*_bias_by_month.csv`: read that file with the warmup month in mind. The
  `--warmup-days` default of 14 only trims inits; it does not remove this artefact.
- **Uniform horizon grid.** Run-to-run, run blending and jerkiness work by column position, so
  `evaluate` and `cv` refuse a base forecaster whose horizons are not uniformly spaced or whose
  init step is not a multiple of the horizon spacing, naming the offending values.
- **Two smoothness metrics, never conflated** (below).
- **Metric hierarchy**, in table order: MAE (the gate), then tails (`p99_err`, `worst_decile`,
  guardrails), then `jerk` (the objective), then `run2run` (hold), then bias, reported pooled
  (`me`) and by month (`*_bias_by_month.csv`) because opposite months cancel in the pooled figure.
- **Cross-validation.** Contiguous folds of init dates. A candidate wins only if it beats the
  reference in every fold. A candidate that fits parameters must fit them outside the scored
  fold, with a gap of about 8 days around it (guidance for future fitted candidates; the shipped
  candidates fit nothing, so nothing enforces or tests it).

## Reading the evaluate table

```
             mae  mae_fixed  p99_err  worst_decile   jerk  run2run    me  mae_vs_ref_%
raw        263.7      276.8   2013.9         475.4   93.9     81.3  62.1           0.0
current    248.4      260.4   1962.1         461.9  100.6     71.1   8.9           5.8
w9         247.6      259.5   1952.2         461.5   92.1     72.9   8.2           6.1
w9_gain08  245.6      257.6   1933.5         456.1   92.4     72.6  15.8           6.9
```

This table comes from the legacy local export directories (forecasts from 2025-10-17,
observations from 2026-01-01), run as `--data ./adjuster_data_prod_year,./adjuster_data_prod_ext,
./adjuster_data_prod --candidates current,w9,w9_gain08`. A fresh export over other dates gives
different numbers. Evaluate also prints bias by init month and writes it to
`*_bias_by_month.csv`.

All figures are MW on matched points. `raw` is the unadjusted base forecast and is always the
first row.

- `mae`, `mae_fixed`: mean absolute error, per-row and fixed capacity. The gate.
- `p99_err`: 99th percentile of |error|. `worst_decile`: mean of the worst 10% of daily MAEs.
- `jerk`, `run2run`: the two smoothness metrics below. Lower is smoother.
- `me`: mean signed error, forecast minus observation. Positive means over-forecast.
- `mae_vs_ref_%`: percentage MAE improvement over the reference row (`raw`). Positive is better.
- The current adjuster's `jerk` exceeding the raw forecast's is expected: the week-average deltas
  are noisy across horizon, and that jaggedness is what smoothing removes.

## Candidates

A candidate is a registered function in `candidates.py`. A `deltas` candidate maps the replayed
deltas to new deltas (the production caps are applied afterwards); a `forecast` candidate maps the
adjusted forecast matrix to a new one.

```python
@candidate("deltas")
def w9(deltas):
    return smooth_across_horizon(deltas, window_bins(deltas.columns))
```

Shipped: `current` (replayed production), `w9` (triangular smoothing over 270 minutes, 9 bins at
30-minute spacing), `w9_gain08` (smoothing then 80% gain, before the caps), `run_blend` (blend
each run with the previous run at shared target times, alpha 0.5), `longer_memory`
(0.4 x latest delta + 0.6 x mean of the previous 15 deltas at the same time of day).

Names stack with `+`, deltas steps first: `--candidates w9_gain08+run_blend`,
`longer_memory+w9+run_blend`.

The replay mirrors `src/pvnet_app/adjuster.py`: NaN-aware triangular smoothing across horizon,
gain, an absolute cap of 1e9 W (as a fraction of capacity), a cap of +-10% of the forecast p50,
subtract, clip at zero. Gain applies before the caps.

## Cross-validation

`cv` prints one MAE column per fold, then `wins_all_folds`. Fold dates must be strictly
increasing and every fold must contain scored inits.

- **Folds.** Contiguous blocks of init dates. By default the scorable init dates are split into
  `--folds` (5) equal-day blocks, which works for any location. `--fold-dates` takes n+1
  comma-separated UTC boundary dates and overrides the blocks. The canonical UK-2026 folds, which
  the published adjuster claims were validated on, are
  `--fold-dates 2026-02-01,2026-04-01,2026-06-01,2026-08-01,2026-09-09,2026-10-17`. They exclude
  the December/January warmup. Without `--fold-dates` the folds are non-canonical, and the
  default equal blocks include the warmup period (the first fold starts before any delta
  exists, so there every adjusted variant equals `raw` and the fold-1 margin is diluted).
- **Fold choice matters.** Different boundaries move points between folds, and a thin margin can
  flip. Measured on the UK 2026 data: under the default equal blocks `w9_gain08` loses fold 3 to
  `current` by 0.8 MW, while under the canonical dates it wins all five folds; `w9` wins every
  fold under both. That is a property of thin margins. Quote per-fold claims together with the
  fold dates they were measured on.
- **`wins_all_folds`** is true when the candidate's MAE is strictly lower than the reference's in
  every fold. The reference is `--reference` (default `current`, added automatically). It is
  consistency evidence, not statistical significance, and it covers MAE only, not tails or
  smoothness.

## The two smoothness metrics

- **Within-run jerkiness** (`jerk`): the mean absolute second difference along one forecast
  trajectory, `mean |f(h+2) - 2 f(h+1) + f(h)|`. It measures how jagged a single published run is.
- **Run-to-run revision** (`run2run`): the mean `|f_t(v) - f_{t-step}(v)|` at shared target
  times `v`, between consecutive inits only. It measures how much the forecast for a given moment
  moves from one run to the next.

A mechanism can lower one and raise the other: smoothing across horizon lowers `jerk` and raises
`run2run` slightly (71.4 to 73.2 MW, about 2.5%); blending with the previous run lowers `run2run`
a great deal (to about 35 MW), also lowers `jerk` slightly, and costs about 1.4 MW on
`worst_decile`, so one tail guardrail gets a little worse.

## Why the replay is trusted

The replay is NOT expected to match live production cell-by-cell: live ran older code before
June 2026 and had roughly 19 deploy/incident days. Reconciled at the year level, live MAE differs
from the floored replay by ~3 MW on ~260, consistent with those causes plus observation
publication timing (which the floor models); a residual signed gap of ~1.4 MW outside the
incident days remains unattributed. Outside the incident days the daily MAE gap has median ~4 MW.
What makes the replay trustworthy is (a) the availability floor, without which it reads
observations production could not have had, and (b) exact equivalence to the independently
audited analysis, which test_equivalence.py enforces.

`tests/test_equivalence.py` checks the pipeline against the original ad hoc analysis. It needs
two things on disk, and skips itself when either is missing:

- the export directories `adjuster_data_prod_year`, `adjuster_data_prod_ext`, `adjuster_data_prod`
  (each with `forecasts.parquet` and `observations.parquet`) at the repository root;
- the cache `adjuster_data/replayed_d8_floor_year.parquet`, the floored deltas written by the
  original script. Regenerate it with
  `uv run --with pyarrow --with matplotlib python adjuster_authentic_charts.py` from the
  repository root (the script builds the cache only when the file is absent).

It asserts that the deltas regenerated from raw parquet, under the original first-row dedup the
cache was built with, match the cache (max absolute difference below 1e-9 on shared cells), that `w9` matches pandas triangular rolling, that on the canonical
folds `w9` beats `current` and `w9_gain08` beats `w9` in all five, and these headline figures over
the scored span (matched points, which day-after availability limits to Jan–Oct 2026) with
`--intersect blend_adjust` (tolerance +-0.05 MW; `mae` and `me` per-row
capacity, the others at the fixed 22.5 GW final capacity):

| Variant | `mae` | `mae_fixed` | `me` | `jerk` | `run2run` |
|---|---|---|---|---|---|
| raw | 263.74 | 276.77 | 62.17 | 93.94 | 81.31 |
| current | 248.39 | 260.41 | 8.88 | 100.63 | 71.10 |
| w9 | 247.55 | 259.55 | 8.21 | 92.13 | 72.89 |
| w9_gain08 | 245.65 | 257.59 | 15.83 | 92.37 | 72.60 |
| w9_gain08+run_blend | 242.85 | 254.64 | | | 34.73 |
| longer_memory+w9+run_blend | 237.80 | 249.40 | | | 34.16 |

The suite runs in CI (synthetic tests; the tests that write parquet are skipped where pyarrow is
absent, and the equivalence tests are reported as skipped without the local data).

## Layout

```
export.py      CLI: Data Platform -> parquet (30-day windows, one file per window, exists-skip)
io.py          load / dedup / pivot, comma-separated directories merged
replay.py      floor, week-average deltas, smoothing, caps (the production mechanism)
candidates.py  registry of named transforms; build() stacks them
metrics.py     the metric set, matched-point scoring
evaluate.py    CLI: one table + CSV
cv.py          CLI: per-fold table, wins-in-every-fold
tests/         synthetic fixtures for every invariant, plus the equivalence test
```
