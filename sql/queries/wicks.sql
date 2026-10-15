-- Largest intraday rises against a short perp: last-trade high vs mark-price
-- high vs spot high, as a move from the previous perp close. The gap is why
-- liquidation risk must be measured on the mark price.
WITH moves AS (
    SELECT asset, date,
           perp_high / LAG(perp_close) OVER w - 1 AS last_trade_up,
           mark_high / LAG(perp_close) OVER w - 1 AS mark_up,
           spot_high / LAG(spot_close) OVER w - 1 AS spot_up,
           mark_imputed
    FROM panel
    WINDOW w AS (PARTITION BY asset ORDER BY date)
)
SELECT * FROM moves
WHERE last_trade_up IS NOT NULL
ORDER BY last_trade_up DESC
LIMIT 10;
