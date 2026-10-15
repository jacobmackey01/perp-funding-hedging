-- Raw downloads, one row per observation exactly as the venues report them.
DROP TABLE IF EXISTS funding_events;
CREATE TABLE funding_events (
    source        TEXT NOT NULL,   -- 'binance' (primary) | 'bybit' (cross-check)
    asset         TEXT NOT NULL,   -- 'BTC' | 'ETH' | 'SOL'
    funding_time  TEXT NOT NULL,   -- UTC 'YYYY-MM-DD HH:MM:SS'
    rate          REAL NOT NULL,   -- paid by longs to shorts when positive
    PRIMARY KEY (source, asset, funding_time)
);

DROP TABLE IF EXISTS klines;
CREATE TABLE klines (
    market        TEXT NOT NULL,   -- 'spot' | 'perp' | 'mark' (Binance perp mark price)
    asset         TEXT NOT NULL,
    date          TEXT NOT NULL,   -- UTC day the candle opens
    open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL,
    close REAL NOT NULL CHECK (close > 0),
    volume        REAL NOT NULL,   -- coins
    quote_volume  REAL NOT NULL,   -- USDT
    PRIMARY KEY (market, asset, date)
);

DROP TABLE IF EXISTS open_interest;
CREATE TABLE open_interest (
    asset         TEXT NOT NULL,
    date          TEXT NOT NULL,   -- Bybit daily snapshot, 00:00 UTC
    oi_coins      REAL NOT NULL,
    PRIMARY KEY (asset, date)
);

DROP TABLE IF EXISTS tbill;
CREATE TABLE tbill (
    date       TEXT PRIMARY KEY,
    tbill_pct  REAL NOT NULL        -- 3-month T-bill yield, % a year (FRED DTB3)
);
