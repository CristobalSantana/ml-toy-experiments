# datasets/de/opsd_solar - German solar generation, measured hourly

One row per hour: how much solar power Germany actually produced, how much
capacity was installed at the time, and where the sun was.

This is the source for the
[solar-forecast-skill](../../../experiments/solar-forecast-skill/) experiment.

> **Measured, not simulated.** These are settlement figures reported by the
> four German transmission system operators - 50Hertz, Amprion, TenneT and
> TransnetBW - and aggregated by Open Power System Data. That distinction
> matters for a forecasting experiment: predicting the output of a simulator
> is a different and far easier problem, and a result obtained that way would
> not transfer.

> **Retrieved 2026-09-05, OPSD release 2020-10-06.** OPSD versions its
> releases by date in the URL, so this is a fixed artefact rather than a
> moving target. `MANIFEST.sha256` records the exact bytes.

## How to get it

Public, no key, one file:

```bash
python load.py               # download, clean, write the manifest
python load.py --inspect     # what columns the release contains
python load.py --validate    # check the cached file against the manifest
python load.py --check-sun   # verify the solar geometry against PVGIS
```

The raw CSV is about 130 MB and is **gitignored**; what is committed is the
manifest, so the provenance is versioned without the bulk bytes.

**Licence.** Open Power System Data publishes the time-series package openly
and attributes the underlying series to their sources - here the four German
TSOs via ENTSO-E Transparency. Check the package's own licence page before
redistributing.

## What the loader returns

| Column | Meaning |
|---|---|
| `time` | hour, UTC |
| `generation_mw` | measured solar generation, megawatts |
| `capacity_mw` | installed solar capacity at that time |
| `cf` | **capacity factor** - generation divided by capacity, the modelling target |
| `elevation` | sun elevation in degrees, computed here |
| `is_day` | elevation above 5 degrees |

**43,696 hours**, 2015-01-01 to 2019-12-30. Generation is reported past that,
but capacity is not, and without capacity the target cannot be normalised.

### Why capacity factor and not megawatts

Installed capacity grew **36% over the period**, from 37.2 to 50.5 GW. A model
fitted to raw megawatts would be rewarded for learning that build-out, which
is a trend anyone can look up, not a forecast. Dividing by capacity removes it.

A handful of hours have a capacity factor above 1, because capacity is
reported with a reporting lag while generation is not. They are clipped to 1
and the count is printed on every run.

## The sun

Solar elevation is **computed, not downloaded**, using the NOAA solar position
algorithm. It is in this dataset rather than in the experiment because it is
the part of solar output that is knowable centuries in advance, and therefore
the thing any forecast has to be measured against rather than credited for.

`python load.py --check-sun` compares it against PVGIS, which reports its own
independently implemented `H_sun`, over a full year:

```
solar elevation vs PVGIS, 8,784 hours at (51.16, 10.45)
  daylight hours compared : 4,192
  mean absolute error     : 0.126 deg
  90th percentile         : 0.228 deg
  worst                   : 0.426 deg
```

Two independent implementations agreeing is worth more than one looking
plausible.

## What this data will do to you if you are careless

**A national aggregate has no single location.** German panels run from 47.3°N
to 55.1°N - about 860 km. Elevation is computed at the country centroid
(51.16°N, 10.45°E), so sunrise and sunset are wrong by up to roughly twenty
minutes at the extremes. Fine for separating day from night and for a
clear-sky envelope; not fine for anything that turns on the exact minute.

**Aggregation hides the hard part.** Clouds over Bavaria do not cover
Schleswig-Holstein, so summing 800 km of installations averages away most of
the variance that makes single-site forecasting difficult. Any forecast skill
measured here is an **upper bound** on what one rooftop would see.

**54.8% of hours have the sun below 5 degrees, and 43.4% have output of
exactly zero.** More than half of this dataset is trivially predictable. Any
error metric computed over all hours is dominated by getting the night right,
which is not a forecasting achievement. This is the whole subject of the
experiment that uses it.

**Two summer months are missing from 2018** in the underlying TSO reporting
and appear as gaps rather than zeros. The loader drops rows without both
generation and capacity rather than filling them; imputing would invent
observations nobody made.

## Sanity figures

| | |
|---|---|
| Mean capacity factor | 0.102 |
| Maximum capacity factor | 0.687 |
| Hours at exactly zero | 43.4% |
| Hours with the sun below 5° | 54.8% |
