-- Daily derivatives panel. Every column on date d uses information up to the
-- close of d (00:00 UTC of d+1) only; strategies add a one-day lag on top.

-- 1. Funding summed per UTC day. A day normally has three 8-hour payments;
--    venues shorten the interval in stressed markets, so count them too.
DROP TABLE IF EXISTS funding_daily;
CREATE TABLE funding_daily AS
SELECT source, asset, SUBSTR(funding_time, 1, 10) AS date,
       SUM(rate)   AS funding,
       COUNT(*)    AS payments,
       MAX(rate)   AS max_rate,
       MIN(rate)   AS min_rate
FROM funding_events
GROUP BY source, asset, date;

-- 2. One row per asset per day: spot and perp prices, basis, funding,
--    returns, open interest and volume. Only days where both markets traded.
DROP TABLE IF EXISTS panel;
CREATE TABLE panel AS
WITH joined AS (
    SELECT s.asset, s.date,
           s.close AS spot_close, s.high AS spot_high, s.low AS spot_low,
           p.close AS perp_close, p.high AS perp_high, p.low AS perp_low,
           -- Liquidations run on the mark price. Before Binance's mark history
           -- starts (Dec 2019) use the spot high scaled by the closing basis.
           COALESCE(m.high, s.high * p.close / s.close) AS mark_high,
           m.high IS NULL                               AS mark_imputed,
           s.quote_volume AS spot_volume_usd, p.quote_volume AS perp_volume_usd,
           COALESCE(f.funding, 0) AS funding, COALESCE(f.payments, 0) AS payments,
           o.oi_coins
    FROM klines s
    JOIN klines p ON p.market = 'perp' AND p.asset = s.asset AND p.date = s.date
    LEFT JOIN klines m ON m.market = 'mark' AND m.asset = s.asset AND m.date = s.date
    LEFT JOIN funding_daily f ON f.source = 'binance' AND f.asset = s.asset AND f.date = s.date
    LEFT JOIN open_interest o ON o.asset = s.asset AND o.date = DATE(s.date, '+1 day')
    WHERE s.market = 'spot'
)
SELECT *,
       perp_close / spot_close - 1                              AS basis,
       spot_close / LAG(spot_close) OVER w - 1                  AS spot_ret,
       perp_close / LAG(perp_close) OVER w - 1                  AS perp_ret,
       LN(spot_close / LAG(spot_close) OVER w)                  AS spot_log,
       oi_coins * perp_close                                    AS oi_usd,
       perp_volume_usd / NULLIF(spot_volume_usd, 0)             AS perp_spot_volume_ratio
FROM joined
WINDOW w AS (PARTITION BY asset ORDER BY date);

-- 3. Rolling features used as signals (trailing windows, full windows only).
DROP TABLE IF EXISTS signals;
CREATE TABLE signals AS
WITH numbered AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY asset ORDER BY date) AS i FROM panel
)
SELECT asset, date,
       CASE WHEN COUNT(*) OVER w7  = 7  THEN AVG(funding) OVER w7  * 365 END AS funding_7d_ann,
       CASE WHEN COUNT(*) OVER w30 = 30 THEN AVG(funding) OVER w30 * 365 END AS funding_30d_ann,
       CASE WHEN COUNT(*) OVER w30 = 30 THEN AVG(funding > 0) OVER w30 END   AS funding_pos_share_30d,
       CASE WHEN COUNT(*) OVER w7  = 7  THEN AVG(basis) OVER w7 END          AS basis_7d,
       oi_usd / LAG(oi_usd, 7) OVER w - 1                                    AS oi_change_7d,
       spot_close / LAG(spot_close, 7) OVER w - 1                            AS spot_ret_7d,
       CASE WHEN COUNT(*) OVER w30 = 30
            THEN SQRT((SUM(spot_log * spot_log) OVER w30
                       - SUM(spot_log) OVER w30 * SUM(spot_log) OVER w30 / 30) / 29) * SQRT(365)
       END                                                                   AS vol_30d,
       CASE WHEN COUNT(*) OVER w200 = 200 THEN AVG(spot_close) OVER w200 END AS sma_200,
       CASE WHEN COUNT(*) OVER w100 = 100 THEN AVG(spot_close) OVER w100 END AS sma_100,
       CASE WHEN COUNT(*) OVER w50  = 50  THEN AVG(spot_close) OVER w50  END AS sma_50
FROM numbered
WINDOW w    AS (PARTITION BY asset ORDER BY date),
       w7   AS (PARTITION BY asset ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW),
       w30  AS (PARTITION BY asset ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW),
       w50  AS (PARTITION BY asset ORDER BY date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW),
       w100 AS (PARTITION BY asset ORDER BY date ROWS BETWEEN 99 PRECEDING AND CURRENT ROW),
       w200 AS (PARTITION BY asset ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW);

-- 4. Forward outcomes for signal research. These look AHEAD by design and
--    are only ever used as the thing being predicted, never as an input.
DROP TABLE IF EXISTS forward_outcomes;
CREATE TABLE forward_outcomes AS
SELECT asset, date,
       LEAD(spot_close, 7)  OVER w / spot_close - 1 AS fwd_ret_7d,
       LEAD(spot_close, 30) OVER w / spot_close - 1 AS fwd_ret_30d,
       MIN(spot_low) OVER (PARTITION BY asset ORDER BY date
                           ROWS BETWEEN 1 FOLLOWING AND 7 FOLLOWING) / spot_close - 1 AS fwd_worst_7d
FROM panel
WINDOW w AS (PARTITION BY asset ORDER BY date);
