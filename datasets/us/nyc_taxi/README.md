# datasets/us/nyc_taxi - NYC yellow taxi trip records

One row per taxi trip in New York City: when and where it started and ended,
how far it went, and the fare broken down into its parts. Published monthly by
the **NYC Taxi & Limousine Commission** since 2009.

This is the source for the
[drift-detector-overlap](../../../experiments/drift-detector-overlap/)
experiment, which uses it because the drift in it is real, dated and not of
our choosing.

> **Retrieved 2026-09-05.** The URL pattern, the file layout and the columns
> documented below were valid on that date. The TLC reissues months when it
> finds errors, so a file downloaded later may differ from the one that
> produced the published results. `MANIFEST.sha256` records the exact bytes;
> `python load.py --validate` checks what you have against it.

## How to download

Public, unauthenticated, no key. The loader does it:

```bash
python load.py                      # the six pre-registered months
python load.py --months 2024-06     # any single month
python load.py --inspect            # one month's raw schema, uncleaned
python load.py --validate           # check cached files against the manifest
```

Files land in `raw/` and are **gitignored** - six months is about 330 MB. What
is committed is `MANIFEST.sha256`, so the provenance is versioned without the
bulk bytes.

```
https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_YYYY-MM.parquet
```

**License.** Published by the TLC for public use through NYC Open Data. The
TLC provides the records as-is and states it did not create them; verify the
current terms on the TLC trip-record page before redistributing.

## The six months

Chosen for documented reasons and frozen in `CRITERIA.md` before any detector
ran, so that no pair could be swapped later for one that gave a nicer result.

| Month | Raw rows | Kept | Why it is in the set |
|---|---|---|---|
| 2019-06 | 6,971,560 | 96.4% | pre-pandemic baseline |
| 2020-03 | 3,007,687 | 95.1% | the month the shutdown began |
| 2020-04 | **238,073** | 86.1% | the collapse; the file is 4 MB against 60-100 MB |
| 2023-06 | 3,307,234 | 93.2% | one year before the endpoint, same season |
| 2024-05 | 3,723,833 | 85.7% | adjacent to the endpoint |
| 2024-06 | 3,539,193 | 84.9% | the reference month |

April 2020 is worth a second look: **238,073 trips against 6.97 million in
June 2019**. The pandemic is visible in the size of the file before anything
is parsed.

## Columns

The loader returns seven features, chosen because they exist in every month in
the range:

| Feature | Unit | Source |
|---|---|---|
| `trip_distance` | miles | as reported by the taximeter |
| `fare_amount` | USD | metered fare, before extras |
| `tip_amount` | USD | card tips only; cash tips are not recorded |
| `total_amount` | USD | fare plus all surcharges and tolls |
| `trip_duration_min` | minutes | derived: dropoff − pickup |
| `passenger_count` | count | driver-entered, not measured |
| `pickup_hour` | 0-23 | derived from the pickup timestamp |

Two columns in the raw files are **deliberately excluded**:
`congestion_surcharge` and `airport_fee` do not exist across the whole range.
That absence is recorded rather than patched over, because it is itself an
instance of the problem the experiment studies.

`tip_amount` deserves its own warning: it captures **card tips only**. Cash
tips are never recorded. Since the cash share halved between 2019 and 2024,
part of any apparent rise in tipping is a change in what gets measured rather
than in what passengers do.

## What this data will do to you if you are careless

Four traps, all verified on the files themselves before being written here.

**1. Schema drift, the silent kind.** The airport fee column is `airport_fee`
in the 2019 file and `Airport_fee` in the 2024 one. Same field, different
capitalisation. Concatenate the two without normalising and you get **two
columns, each about half null, with no error raised**. The loader lowercases
every column name on read. That one line is the entire fix, and needing it is
the point.

**2. Files are not confined to their month.** The 2019-06 file contains trips
dated as far back as **2001**; the 2024-06 file contains one dated
**2026-06-26**, which is in the future relative to the file. It is a tiny
fraction - 534 and 51 rows - but a min/max over the timestamp column will find
them, and a time-series split will silently misplace them.

**3. Coverage changed, and it looks exactly like demand collapsing.** Yellow
taxi volume fell from 6.97 M trips in 2019-06 to 3.54 M in 2024-06. **A large
part of that is not New Yorkers travelling less**; it is that high-volume
for-hire trips - Uber, Lyft - are published in separate files. Any statement
about "taxi demand" drawn from this file alone is that fact being ignored.

**4. Data quality drifts too.** Rows failing the passenger-count rule grew
from 156,472 in 2019-06 to 421,304 in 2024-06 - from 2% of the file to 12%,
mostly nulls. The rate at which a field goes missing is itself a series that
moves, and a detector watching feature values will not see it.

## Cleaning

Frozen in [`CRITERIA.md`](../../../experiments/drift-detector-overlap/CRITERIA.md)
and duplicated in `load.py` as constants, so a change in one shows up as a
diff against the other.

| Rule | Interval |
|---|---|
| pickup month equals the file's nominal month | - |
| `trip_distance` | `(0, 50]` miles |
| `fare_amount` | `(0, 300]` USD |
| `trip_duration_min` | `(0, 180]` minutes |
| `passenger_count` | `[1, 6]` |

The intervals are **not uniform**, and that mattered: four are half-open but
passenger count is closed at both ends. Applying the half-open convention to
all five excludes every single-passenger trip - 78% of the file - and the run
completes normally, reporting a clean-looking dataset with most of its rows
gone. The pre-registered row counts are what caught it, which is the argument
for writing them down first.

Every rule reports what it removed on each run.
