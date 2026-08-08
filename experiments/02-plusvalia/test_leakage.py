"""
test_leakage.py -- Prove the leakage checks actually fire.

A safety net that is never exercised is decoration. These tests inject leaks
into the real feature matrix and assert that each check aborts, plus one
control asserting a clean split does *not* abort (a check that rejects
everything is equally useless).

Test 2 is the one that matters most: it re-adds the real `avaluo_exento` leak
under an innocuous name, so it can only be caught empirically. That test
failed on the first implementation - a global-R² detector missed it, because
the column reconstructs the target exactly in 51% of rows and differs in the
rest, which dilutes a global fit. The partial-leak (constant-residual)
detector was added in response.

    python test_leakage.py        # prints one line per check
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import features  # noqa: E402
import leakage  # noqa: E402


def _expect_abort(name: str, fn) -> bool:
    try:
        fn()
    except leakage.LeakageError as e:
        print(f"  PASS  {name}: aborted -> {str(e).splitlines()[0][:90]}")
        return True
    print(f"  FAIL  {name}: did NOT abort (the check is useless)")
    return False


def main() -> int:
    cfg = features.load_config()
    df = features.clean(features.load_raw(cfg), cfg, verbose=False)
    X, y, groups, area = features.build_features(df, cfg)

    print(f"baseline: {len(X)} rows, {X.shape[1]} features")
    leakage.run_all_cross_sectional(X, y, area, groups, cfg, verbose=False)
    print("  PASS  clean feature matrix passes every check")

    ok = [True]

    # 1. forbidden by name
    Xa = X.copy(); Xa["avaluo_exento"] = df["avaluo_exento"].values
    ok.append(_expect_abort("forbidden column by name",
        lambda: leakage.run_all_cross_sectional(Xa, y, area, groups, cfg, verbose=False)))

    # 2. the same leak, renamed -> must be caught empirically (partial leak)
    Xb = X.copy(); Xb["indicador_fiscal_z"] = df["avaluo_exento"].values
    ok.append(_expect_abort("partial value reconstruction (renamed avaluo_exento)",
        lambda: leakage.run_all_cross_sectional(Xb, y, area, groups, cfg, verbose=False)))

    # 3. monotone transform of the target -> Pearson would miss it, Spearman must not
    Xc = X.copy(); Xc["score_raro"] = (y.values ** 3) + 7
    ok.append(_expect_abort("monotone function of the target",
        lambda: leakage.run_all_cross_sectional(Xc, y, area, groups, cfg, verbose=False)))

    # 4. a manzana on both sides of a split
    idx = np.arange(len(groups))
    ok.append(_expect_abort("manzana split across train/test",
        lambda: leakage.check_group_disjoint(groups, idx[:1000], idx[500:1500])))

    # 5. as-of violation
    fut = pd.Series(pd.to_datetime(["2024-01-01"] * 5))
    own = pd.Series(pd.to_datetime(["2023-01-01"] * 5))
    ok.append(_expect_abort("as-of violation",
        lambda: leakage.check_asof(fut, own, "ipv_lag")))

    # 6. control: a correct grouped split must NOT abort
    held = set(np.sort(groups.unique())[:500])
    mask = groups.isin(held)
    try:
        leakage.check_group_disjoint(groups, np.where(~mask)[0], np.where(mask)[0])
        print("  PASS  control: a correct grouped split does not abort")
    except leakage.LeakageError as e:
        print(f"  FAIL  control: false positive -> {e}")
        ok.append(False)

    passed = all(ok)
    print(f"\n{'ALL CHECKS BEHAVE CORRECTLY' if passed else 'SOME CHECKS ARE BROKEN'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
