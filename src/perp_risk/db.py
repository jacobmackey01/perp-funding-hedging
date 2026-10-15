"""Load the raw CSVs into SQLite, build the SQL feature tables, read them back.

Run:  python -m perp_risk.db
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from . import config


def connect(path: Path = config.DB_PATH) -> sqlite3.Connection:
    return sqlite3.connect(path)


def _insert(con, table: str, df: pd.DataFrame) -> None:
    cols = list(df.columns)
    con.executemany(f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                    df.itertuples(index=False, name=None))


def load_raw(con: sqlite3.Connection) -> None:
    raw = config.RAW_DIR
    for asset in config.ASSETS:
        for source in ("binance", "bybit"):
            df = pd.read_csv(raw / f"{source}_funding_{asset}.csv")
            _insert(con, "funding_events", df.assign(source=source, asset=asset))
        for market in ("spot", "perp", "mark"):
            df = pd.read_csv(raw / f"binance_{market}_{asset}.csv")
            _insert(con, "klines", df.assign(market=market, asset=asset))
        oi = pd.read_csv(raw / f"bybit_oi_{asset}.csv").rename(columns={"open_interest": "oi_coins"})
        _insert(con, "open_interest", oi.assign(asset=asset))
    _insert(con, "tbill", pd.read_csv(raw / "fred_tbill.csv"))
    con.commit()


def build(path: Path = config.DB_PATH) -> None:
    path.unlink(missing_ok=True)
    with connect(path) as con:
        con.executescript((config.SQL_DIR / "schema.sql").read_text())
        load_raw(con)
        con.executescript((config.SQL_DIR / "features.sql").read_text())


def query(sql: str, params: dict | None = None, path: Path = config.DB_PATH) -> pd.DataFrame:
    with connect(path) as con:
        return pd.read_sql_query(sql, con, params=params)


def query_file(name: str, params: dict | None = None) -> pd.DataFrame:
    return query((config.SQL_DIR / "queries" / name).read_text(), params)


def load_panel() -> pd.DataFrame:
    """Long panel (asset, date) with features and forward outcomes."""
    df = query("""
        SELECT p.*, s.funding_7d_ann, s.funding_30d_ann, s.funding_pos_share_30d, s.basis_7d,
               s.oi_change_7d, s.spot_ret_7d, s.vol_30d, s.sma_50, s.sma_100, s.sma_200,
               f.fwd_ret_7d, f.fwd_ret_30d, f.fwd_worst_7d
        FROM panel p
        JOIN signals s USING (asset, date)
        JOIN forward_outcomes f USING (asset, date)
        ORDER BY asset, date
    """)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_tbill() -> pd.Series:
    """Daily cash yield (decimal, per year), forward-filled over non-trading days."""
    t = query("SELECT date, tbill_pct FROM tbill ORDER BY date")
    s = pd.Series(t.tbill_pct.to_numpy() / 100, index=pd.to_datetime(t.date))
    return s.reindex(pd.date_range(s.index[0], config.END_DATE)).ffill()


def wide(panel: pd.DataFrame, column: str) -> pd.DataFrame:
    return panel.pivot(index="date", columns="asset", values=column)


if __name__ == "__main__":
    build()
    with connect() as con:
        for t in ("funding_events", "klines", "open_interest", "tbill", "funding_daily", "panel", "signals"):
            (n,) = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()
            print(f"{t:16s} {n:>8,d} rows")
