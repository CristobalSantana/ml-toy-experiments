"""
test_leakage.py -- Prove the feature builder cannot see the future.

    python test_leakage.py

A forecasting experiment has exactly one way to produce a spectacular and
worthless result: let a feature carry information from after the forecast was
issued. `df["cf"].shift(-1)` does that and looks identical to
`df["cf"].shift(1)`, which does not.

The check that matters is causal, not statistical. Corrupt every observation
strictly after the issue time and rebuild: a feature that changes was reading
ahead. This is stronger than comparing correlations, and it does not need a
"ceiling" to compare against - an earlier version of this file tried that and
failed on `cs_now`, which tracks the target better than the last observation
does and is entirely legitimate, because the sun's position is known in
advance.

Five checks. Two of them build the broken version on purpose.
run_all.py stops if any fail.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

import features as F
from models import ClearSky

SEED = 20260905
LAGS = [1, 2, 3, 24]
FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if detail:
        print(f"        {detail}")
    if not ok:
        FAILURES.append(label)


def _toy(n: int = 3000) -> pd.DataFrame:
    """A small synthetic series shaped like the real one: a diurnal cycle
    multiplied by slowly drifting cloud."""
    rng = np.random.default_rng(SEED)
    t = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    elev = 40 * np.sin(2 * np.pi * (t.hour - 6) / 24) - 5
    clear = np.clip(elev / 45.0, 0, None)
    cloud = np.clip(0.55 + 0.45 * np.sin(np.cumsum(rng.normal(0, 0.25, n)) / 9), 0, 1)
    df = pd.DataFrame({"time": t, "elevation": elev,
                       "cf": np.clip(clear * cloud, 0, 1)})
    df["is_day"] = df["elevation"] > 5
    cs = ClearSky().fit(df["elevation"].to_numpy(), df["cf"].to_numpy())
    df["cs"] = cs.predict(df["elevation"].to_numpy())
    df["k"] = cs.index(df["elevation"].to_numpy(), df["cf"].to_numpy())
    return df


def _causal_violations(build_fn, df: pd.DataFrame, horizon: int,
                       n_probe: int = 20) -> list[str]:
    """Rebuild with the future corrupted; report any feature that moved.

    Only the observations are corrupted - capacity factor and clear-sky index.
    Elevation and clear-sky output are geometry, known centuries ahead, and a
    feature depending on the target hour's geometry is not leakage.
    """
    rng = np.random.default_rng(SEED + 1)
    base = build_fn(df, horizon, LAGS)
    cols = [c for c in F.feature_columns(base) if not c.startswith("target_")]
    probes = np.linspace(len(base) // 4, len(base) - horizon - 5, n_probe).astype(int)

    offenders: set[str] = set()
    for i in probes:
        t = base["time"].iloc[i]
        d2 = df.copy()
        after = d2["time"] > t
        d2.loc[after, "cf"] = rng.random(int(after.sum()))
        d2.loc[after, "k"] = rng.random(int(after.sum()))

        alt = build_fn(d2, horizon, LAGS)
        row = alt.index[alt["time"] == t]
        if len(row) == 0:
            continue
        j = row[0]
        for c in cols:
            a, b = base[c].iloc[i], alt[c].iloc[j]
            if not np.isclose(float(a), float(b), rtol=1e-9, atol=1e-12):
                offenders.add(c)
    return sorted(offenders)


def _leaky_build(df, horizon, cf_lags, **kw):
    """The bug, written out: one lag shifted the wrong way.

    `shift(-horizon)` instead of `shift(horizon - 1)`. Off by one character in
    the source, and the "lag" now holds the hour being predicted.
    """
    frame = F.build(df, horizon, cf_lags, **kw)
    d = df.sort_values("time").reset_index(drop=True)
    leaked = d[["time", "cf"]].copy()
    leaked["time"] = leaked["time"] - pd.Timedelta(hours=horizon)
    leaked = leaked.rename(columns={"cf": "cf_lag1"})
    # merged on the issue time, so the alignment is right and the value really
    # is the target hour's output - an earlier version reindexed by position
    # and silently compared unrelated hours
    return frame.drop(columns=["cf_lag1"]).merge(leaked, on="time", how="left").dropna()


def test_static_no_backward_shifts() -> None:
    """The target may read ahead. Nothing else may."""
    src = open("features.py", encoding="utf-8").read()
    body = src.split("def build(")[1].split("def feature_columns")[0]
    bad = [ln.strip() for ln in body.splitlines()
           if ".shift(-" in ln and 'out["y"]' not in ln and "target_" not in ln]
    check("only the target and the sun's position read ahead", not bad,
          "; ".join(bad) if bad else
          "every observation column uses a forward shift")


def test_causal_h1() -> None:
    bad = _causal_violations(F.build, _toy(), horizon=1)
    check("1 hour ahead: no feature changes when the future is corrupted",
          not bad, f"offending: {bad}" if bad else
          "all features identical across 20 corrupted rebuilds")


def test_causal_h24() -> None:
    bad = _causal_violations(F.build, _toy(), horizon=24)
    check("24 hours ahead: no feature changes when the future is corrupted",
          not bad, f"offending: {bad}" if bad else
          "all features identical across 20 corrupted rebuilds")


def test_leaky_builder_is_caught() -> None:
    """The check has to fire on a builder that is actually broken."""
    bad = _causal_violations(_leaky_build, _toy(), horizon=24)
    check("a lag shifted the wrong way is caught", "cf_lag1" in bad,
          f"flagged: {bad}" if bad else "NOT CAUGHT - the check is useless")


def test_leak_would_have_looked_like_success() -> None:
    """What the leak buys, so the cost of missing it is on the record."""
    df = _toy()
    honest = F.build(df, 24, LAGS)
    leaky = _leaky_build(df, 24, LAGS)
    cols = [c for c in F.feature_columns(honest)]

    def fit_score(frame):
        y = frame["y"].to_numpy()
        n = int(len(frame) * 0.7)
        m = Ridge(alpha=1.0).fit(frame[cols].iloc[:n], y[:n])
        return r2_score(y[n:], m.predict(frame[cols].iloc[n:]))

    a, b = fit_score(honest), fit_score(leaky)
    # Measured as the share of remaining error the leak removes, not as a gain
    # in R2. An earlier version demanded +0.10 R2, which is unreachable when
    # the honest model already sits at 0.92 and only 0.08 is left to win.
    removed = (1 - a - (1 - b)) / (1 - a)
    check("the leak would have looked like a far better forecast",
          removed > 0.30,
          f"R2 {a:.3f} -> {b:.3f}: the leak removes {removed:.0%} of the error "
          f"the honest model could not explain, 24 hours ahead")


def main() -> None:
    print("Leakage checks\n")
    test_static_no_backward_shifts()
    test_causal_h1()
    test_causal_h24()
    test_leaky_builder_is_caught()
    test_leak_would_have_looked_like_success()
    print()
    if FAILURES:
        sys.exit(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
    print("All checks passed. Features see only the past and the sun.")


if __name__ == "__main__":
    main()
