"""Download funding, price and open-interest history.

Sources (all public, no key):
  * Binance USDT-M futures: funding rate history, daily perp klines
  * Binance spot: daily klines
  * Bybit linear perps: daily open interest, funding history (cross-check)

Run:  python -m perp_risk.fetch
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd
import requests

from . import config

DAY_MS = 86_400_000
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "perp-risk-study"


def _ms(date: str) -> int:
    return int(datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def _get(url: str, params: dict):
    for attempt in range(5):
        resp = SESSION.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        time.sleep(2 ** attempt)
    resp.raise_for_status()


def binance_funding(symbol: str, start: str, end: str) -> pd.DataFrame:
    rows, cursor, stop = [], _ms(start), _ms(end) + DAY_MS
    while cursor < stop:
        batch = _get("https://fapi.binance.com/fapi/v1/fundingRate",
                     {"symbol": symbol, "startTime": cursor, "endTime": stop - 1, "limit": 1000})
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1]["fundingTime"] + 1
        time.sleep(0.25)
    df = pd.DataFrame(rows)
    df["funding_time"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
    df["rate"] = df["fundingRate"].astype(float)
    return df[["funding_time", "rate"]].drop_duplicates("funding_time")


def binance_klines(symbol: str, start: str, end: str, futures: bool, mark: bool = False) -> pd.DataFrame:
    """Daily candles. `mark=True` gives the perp's mark price, which is what
    the exchange liquidates on (last-trade candles include flash wicks)."""
    url = ("https://fapi.binance.com/fapi/v1/markPriceKlines" if mark
           else "https://fapi.binance.com/fapi/v1/klines" if futures
           else "https://api.binance.com/api/v3/klines")
    rows, cursor, stop = [], _ms(start), _ms(end) + DAY_MS
    while cursor < stop:
        batch = _get(url, {"symbol": symbol, "interval": "1d", "startTime": cursor,
                           "endTime": stop - 1, "limit": 1000})
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + DAY_MS
        time.sleep(0.25)
    df = pd.DataFrame(rows).iloc[:, :8]
    df.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume"]
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    for c in ("open", "high", "low", "close", "volume", "quote_volume"):
        df[c] = df[c].astype(float)
    return df[["date", "open", "high", "low", "close", "volume", "quote_volume"]].drop_duplicates("date")


def _bybit_paged(path: str, params: dict) -> list:
    rows, cursor = [], ""
    for _ in range(200):
        res = _get(f"https://api.bybit.com/v5/market/{path}", {**params, "cursor": cursor})["result"]
        rows.extend(res["list"])
        cursor = res.get("nextPageCursor", "")
        if not cursor or not res["list"]:
            break
        time.sleep(0.2)
    return rows


def bybit_open_interest(symbol: str) -> pd.DataFrame:
    rows = _bybit_paged("open-interest", {"category": "linear", "symbol": symbol,
                                          "intervalTime": "1d", "limit": 200})
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    df["open_interest"] = df["openInterest"].astype(float)   # in coins
    return df[["date", "open_interest"]].drop_duplicates("date").sort_values("date")


def bybit_funding(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Bybit pages backwards from endTime, 200 rows at a time."""
    rows, end_ms, floor = [], _ms(end) + DAY_MS - 1, _ms(start)
    while end_ms > floor:
        batch = _get("https://api.bybit.com/v5/market/funding/history",
                     {"category": "linear", "symbol": symbol, "endTime": end_ms, "limit": 200})["result"]["list"]
        if not batch:
            break
        rows.extend(batch)
        end_ms = min(int(b["fundingRateTimestamp"]) for b in batch) - 1
        time.sleep(0.2)
    df = pd.DataFrame(rows)
    df["funding_time"] = pd.to_datetime(df["fundingRateTimestamp"].astype("int64"), unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
    df["rate"] = df["fundingRate"].astype(float)
    df = df[df.funding_time >= start]
    return df[["funding_time", "rate"]].drop_duplicates("funding_time").sort_values("funding_time")


def fred_tbill(start: str, end: str) -> pd.DataFrame:
    """3-month US T-bill yield (DTB3, % a year): the cash alternative."""
    from io import StringIO
    resp = SESSION.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                       params={"id": "DTB3", "cosd": start, "coed": end},
                       headers={"User-Agent": "curl/8.7.1"}, timeout=90)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    df.columns = ["date", "tbill_pct"]
    return df.dropna()


def main() -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    s, e = config.START_DATE, config.END_DATE
    # OI snapshots are taken at 00:00 UTC, so the reading for day e's close is
    # stamped e + 1 day.
    oi_end = (pd.Timestamp(e) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    for asset, sym in config.SYMBOLS.items():
        jobs = {
            f"binance_funding_{asset}.csv": lambda: binance_funding(sym, s, e),
            f"binance_perp_{asset}.csv": lambda: binance_klines(sym, s, e, futures=True),
            f"binance_spot_{asset}.csv": lambda: binance_klines(sym, s, e, futures=False),
            f"binance_mark_{asset}.csv": lambda: binance_klines(sym, s, e, futures=True, mark=True),
            f"bybit_oi_{asset}.csv": lambda: (lambda d: d[d.date <= oi_end])(bybit_open_interest(sym)),
            f"bybit_funding_{asset}.csv": lambda: bybit_funding(sym, s, e),
        }
        for name, job in jobs.items():
            df = job()
            df.to_csv(config.RAW_DIR / name, index=False)
            first = df.iloc[0, 0]; last = df.iloc[-1, 0]
            print(f"{name:28s} {len(df):6d} rows  {first} → {last}")
    fred_tbill(s, e).to_csv(config.RAW_DIR / "fred_tbill.csv", index=False)


if __name__ == "__main__":
    main()
