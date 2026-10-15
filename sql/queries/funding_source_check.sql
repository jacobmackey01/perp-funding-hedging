-- Binance vs Bybit daily funding on overlapping days: does the primary venue
-- represent the market, or is the carry venue-specific?
WITH pairs AS (
    SELECT b.asset, b.date, b.funding AS bn, y.funding AS bb
    FROM funding_daily b
    JOIN funding_daily y ON y.source = 'bybit' AND y.asset = b.asset AND y.date = b.date
    WHERE b.source = 'binance'
),
s AS (
    SELECT asset, COUNT(*) n, MIN(date) first_day, AVG(bn) * 365 bn_ann, AVG(bb) * 365 bb_ann,
           SUM(bn) sx, SUM(bb) sy, SUM(bn*bn) sxx, SUM(bb*bb) syy, SUM(bn*bb) sxy,
           AVG((bn > 0) = (bb > 0)) same_sign
    FROM pairs GROUP BY asset
)
SELECT asset, n AS overlapping_days, first_day, bn_ann AS binance_ann, bb_ann AS bybit_ann, same_sign,
       (n*sxy - sx*sy) / SQRT((n*sxx - sx*sx) * (n*syy - sy*sy)) AS correlation
FROM s ORDER BY asset;
