"""
leakage.py -- Fail-loud leakage checks for Experiment 02.

CRITERIA.md commits to three rules, checked here *before* any model is fit.
Each raises LeakageError (which the caller lets crash) rather than warning,
because a leak that only prints a warning gets ignored and silently inflates
every number downstream.

  1. No feature may be a deterministic function of the target. Because the
     target is value *per m²*, this is not hypothetical in this dataset:
     `avaluo_exento` equals `avaluo_fiscal_total` in 51% of rows, so
     exento / built_area / UF reconstructs the target exactly (measured
     correlation 1.000000). We check by name *and* empirically, since the next
     leak will have a different name.
  2. Rolling/aggregate features must use a strict as-of cut: no future period
     may contribute to a row's features.
  3. Rows from the same manzana must never span train and test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


class LeakageError(AssertionError):
    """Raised when a leakage rule is violated. Not catchable by intent."""


def check_forbidden_columns(features: pd.DataFrame, forbidden: list[str]) -> None:
    """Rule 1a (by name): columns known to reconstruct the target are absent."""
    present = sorted(set(features.columns) & set(forbidden))
    if present:
        raise LeakageError(
            f"Forbidden column(s) present in the feature matrix: {present}. "
            f"These reconstruct the target exactly (see config.yaml -> features.forbidden)."
        )


def check_no_deterministic_feature(
    features: pd.DataFrame, target: pd.Series, max_abs_spearman: float = 0.999,
    sample: int = 50_000, seed: int = 0,
) -> pd.DataFrame:
    """Rule 1b (empirical): no single numeric feature is a monotone
    reconstruction of the target.

    Spearman (not Pearson) because a leak survives any monotone transform -
    a feature equal to `target ** 3` or `log(target)` is just as fatal and
    Pearson would miss it.
    """
    num = features.select_dtypes(include=[np.number])
    if num.empty:
        return pd.DataFrame(columns=["feature", "abs_spearman"])

    idx = num.index
    if len(idx) > sample:
        idx = pd.Index(np.random.default_rng(seed).choice(idx, sample, replace=False))

    rows = []
    for col in num.columns:
        x, y = num.loc[idx, col], target.loc[idx]
        ok = x.notna() & y.notna()
        if ok.sum() < 100 or x[ok].nunique() < 2:
            continue
        rho = stats.spearmanr(x[ok], y[ok]).statistic
        rows.append({"feature": col, "abs_spearman": abs(float(rho))})

    report = pd.DataFrame(rows).sort_values("abs_spearman", ascending=False)
    bad = report[report["abs_spearman"] > max_abs_spearman]
    if not bad.empty:
        raise LeakageError(
            "Feature(s) are a deterministic (monotone) function of the target:\n"
            + bad.to_string(index=False)
            + f"\n(|Spearman| > {max_abs_spearman})"
        )
    return report


def check_no_value_reconstruction(
    features: pd.DataFrame, target_log10: pd.Series, built_area: pd.Series,
    max_r2: float = 0.99, max_exact_frac: float = 0.005,
) -> pd.DataFrame:
    """Rule 1c (empirical, dataset-specific): no feature is the assessed value
    in disguise.

    The target is log10(avalúo / area / UF), so any feature f proportional to
    the avalúo satisfies log10(f) - (target_log10 + log10(area)) = const.

    Two detectors, because leaks come in two shapes:

    * `reconstruction_r2` - a global fit, which catches a feature that is
      proportional to the avalúo across *all* rows.
    * `exact_frac` - the share of rows collapsing onto a single constant
      residual, which catches a *partial* leak. This one is not hypothetical:
      `avaluo_exento` equals `avaluo_fiscal_total` in 51% of rows and differs
      in the rest, so its global R² stays unremarkable while half the dataset
      is exactly reconstructible. A model happily learns that subset, so a
      global-fit-only check gives false reassurance.
    """
    implied_log_value = target_log10 + np.log10(built_area.replace(0, np.nan))
    num = features.select_dtypes(include=[np.number])

    rows = []
    for col in num.columns:
        f = num[col]
        ok = (f > 0) & implied_log_value.notna() & f.notna()
        if ok.sum() < 100:
            continue
        x, y = np.log10(f[ok]), implied_log_value[ok]
        if x.nunique() < 2:
            continue
        r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
        # a constant residual means f = const * avalúo on those rows
        resid = np.round(x - y, 6)
        exact_frac = float(resid.value_counts(normalize=True).iloc[0]) if len(resid) else 0.0
        rows.append({"feature": col, "reconstruction_r2": r2, "exact_frac": exact_frac})

    report = (pd.DataFrame(rows)
              .sort_values(["exact_frac", "reconstruction_r2"], ascending=False))
    bad = report[(report["reconstruction_r2"] > max_r2)
                 | (report["exact_frac"] > max_exact_frac)]
    if not bad.empty:
        raise LeakageError(
            "Feature(s) reconstruct the assessed value (target x area):\n"
            + bad.to_string(index=False)
            + f"\n(limits: R² > {max_r2}, or exact_frac > {max_exact_frac} of rows "
              f"sharing one constant residual)"
        )
    return report


def check_group_disjoint(groups: pd.Series, train_idx, test_idx, label: str = "split") -> None:
    """Rule 3: no manzana appears on both sides of a split."""
    overlap = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
    if overlap:
        raise LeakageError(
            f"{label}: {len(overlap)} manzana(s) appear in BOTH train and test, "
            f"e.g. {sorted(overlap)[:5]}. Grouped splitting is broken."
        )


def check_asof(feature_period: pd.Series, row_period: pd.Series, name: str) -> None:
    """Rule 2: a rolling/aggregate feature may only use periods strictly before
    the row's own period. Call this for every temporal aggregate built in the
    IPV arm; the cross-sectional arm has no time axis and so cannot violate it.
    """
    violations = int((feature_period >= row_period).sum())
    if violations:
        raise LeakageError(
            f"As-of violation in '{name}': {violations} row(s) use a period at or "
            f"after their own. Aggregates must use strictly past periods only."
        )


def run_all_cross_sectional(
    features: pd.DataFrame, target_log10: pd.Series, built_area: pd.Series,
    groups: pd.Series, cfg: dict, verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Every pre-fit check for the cross-sectional arm. Raises on any violation."""
    forbidden = cfg["features"]["forbidden"]
    lk = cfg["leakage"]

    check_forbidden_columns(features, forbidden)
    spearman = check_no_deterministic_feature(
        features, target_log10, lk["max_abs_spearman"], seed=cfg["seed"])
    recon = check_no_value_reconstruction(
        features, target_log10, built_area, lk["max_reconstruction_r2"],
        lk["max_exact_reconstruction_frac"])

    if verbose:
        print("  leakage: forbidden columns absent OK")
        print(f"  leakage: max |Spearman| vs target = {spearman['abs_spearman'].max():.4f} "
              f"(limit {lk['max_abs_spearman']}) -- "
              f"top: {spearman.iloc[0]['feature']}")
        print(f"  leakage: max value-reconstruction R2 = {recon['reconstruction_r2'].max():.4f} "
              f"(limit {lk['max_reconstruction_r2']}), max exact-residual share = "
              f"{recon['exact_frac'].max():.4f} (limit {lk['max_exact_reconstruction_frac']})")
        print(f"  leakage: {groups.nunique()} distinct manzanas available for grouped CV")

    return {"spearman": spearman, "reconstruction": recon}
