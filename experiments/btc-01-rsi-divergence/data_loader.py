"""
data_loader.py -- BTC/USDT candles from Binance's public REST API, at any
supported interval.

No authentication, no key. Binance caps klines at 1000 candles per request, so
the history is paginated and then cached: a rerun reads the parquet/csv and
never touches the network again, which is what makes the backtest reproducible
rather than dependent on when it happened to run.

    python data_loader.py            # download (or reuse) and summarise
    python data_loader.py --refresh  # force a re-download
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"



SYMBOL = "BTCUSDT"
START = "2017-08-17"          # first day Binance lists BTC/USDT
# Bars per year, needed to annualise Sharpe correctly. Getting this wrong
# scales the ratio by sqrt(24) between timeframes and makes them incomparable.
BARS_PER_YEAR = {"1d": 365, "4h": 365 * 6, "1h": 365 * 24}
BASE = "https://api.binance.com/api/v3/klines"
LIMIT = 1000                  # Binance's per-request maximum

# Binance returns 12 fields per candle; these are the ones a backtest needs.
COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
           "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]


def _fetch(start_ms: int, interval: str) -> list:
    url = f"{BASE}?symbol={SYMBOL}&interval={interval}&startTime={start_ms}&limit={LIMIT}"
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def download(interval: str) -> pd.DataFrame:
    """Page through the full history, oldest first."""
    start_ms = int(pd.Timestamp(START, tz="UTC").timestamp() * 1000)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    rows: list = []

    while start_ms < now_ms:
        batch = _fetch(start_ms, interval)
        if not batch:
            break
        rows.extend(batch)
        last_open = batch[-1][0]
        if len(batch) < LIMIT:
            break
        start_ms = last_open + 1
        time.sleep(0.25)          # stay well inside Binance's rate limits
        print(f"  {len(rows):>5} candles ... "
              f"{pd.to_datetime(last_open, unit='ms').date()}", flush=True)

    df = pd.DataFrame(rows, columns=COLUMNS)
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df = (df[["date", "open", "high", "low", "close", "volume"]]
          .drop_duplicates(subset="date")
          .sort_values("date")
          .reset_index(drop=True))
    return df


def validate(df: pd.DataFrame) -> None:
    """Fail loudly on the ways price data is usually broken."""
    problems = []
    if df["date"].duplicated().any():
        problems.append("duplicate dates")
    if not df["date"].is_monotonic_increasing:
        problems.append("dates not sorted")
    # Gaps are a fact about exchange history, not necessarily a fault: Binance
    # has maintenance windows, and nine years of hourly candles contain a few.
    # Report them always, and only abort when enough of the series is missing
    # to change conclusions.
    step = df["date"].diff().dropna()
    expected = step.mode().iloc[0] if len(step) else None
    if expected is not None:
        gaps = step[step > expected]
        missing = int(((gaps - expected) / expected).sum())
        frac = missing / (len(df) + missing)
        if len(gaps):
            print(f"  note: {len(gaps)} gap(s), {missing} missing bar(s) "
                  f"({frac*100:.3f}% of the series), largest {gaps.max()}")
        if frac > 0.01:
            problems.append(f"{frac*100:.1f}% of bars missing across {len(gaps)} gap(s) "
                            f"- too much to treat as exchange downtime")
    bad_ohlc = ((df["high"] < df["low"]) |
                (df["high"] < df[["open", "close"]].max(axis=1)) |
                (df["low"] > df[["open", "close"]].min(axis=1)))
    if bad_ohlc.any():
        problems.append(f"{int(bad_ohlc.sum())} row(s) with inconsistent OHLC")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        problems.append("non-positive prices")

    if problems:
        raise SystemExit("Data validation failed:\n  - " + "\n  - ".join(problems))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(refresh: bool = False, interval: str = "1d") -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv = DATA_DIR / f"btcusdt_{interval}.csv"
    manifest = DATA_DIR / f"manifest_{interval}.json"
    if csv.exists() and not refresh:
        df = pd.read_csv(csv, parse_dates=["date"])
        print(f"Using cached {csv.name} ({len(df):,} candles)")
        # The candles are committed, so anyone can rerun this offline. The
        # checksum is what makes that worth something: if a re-download ever
        # returns different bytes - a backfilled gap, a revised candle - the
        # run says so instead of quietly producing different numbers under the
        # same write-up.
        if manifest.exists():
            recorded = json.loads(manifest.read_text(encoding="utf-8")).get("sha256")
            actual = _sha256(csv)
            if recorded and recorded != actual:
                print(f"  WARNING: {csv.name} does not match the manifest.\n"
                      f"    recorded {recorded}\n    actual   {actual}\n"
                      f"    The published results were produced from the "
                      f"recorded file.")
    else:
        print(f"Downloading {SYMBOL} {interval} from {START} ...")
        df = download(interval)
        df.to_csv(csv, index=False)
        manifest.write_text(json.dumps({
            "symbol": SYMBOL, "interval": interval, "source": BASE,
            "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "first_candle": str(df["date"].min().date()),
            "last_candle": str(df["date"].max().date()),
            "n_candles": int(len(df)),
            "sha256": _sha256(csv),
        }, indent=2), encoding="utf-8")
    validate(df)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--refresh", action="store_true", help="ignore the cache and re-download")
    ap.add_argument("--interval", default="1d", choices=list(BARS_PER_YEAR),
                    help="candle size; each is cached under its own file")
    a = ap.parse_args()
    df = load(a.refresh, a.interval)

    print(f"\n{len(df):,} {a.interval} candles  "
          f"{df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"  close: {df['close'].min():,.0f} .. {df['close'].max():,.0f} USDT")
    bh = df["close"].iloc[-1] / df["close"].iloc[0] - 1
    print(f"  buy-and-hold over the whole span: {bh*100:,.0f}%")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
