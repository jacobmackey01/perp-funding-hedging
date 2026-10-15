-- Latest derivatives snapshot per asset, with each reading's percentile
-- against that asset's own history (the "market tracker" table).
WITH latest AS (
    SELECT p.asset, MAX(p.date) AS date FROM panel p GROUP BY p.asset
),
cur AS (
    SELECT p.asset, p.date, p.spot_close, p.basis, s.funding_7d_ann, s.funding_30d_ann,
           s.oi_change_7d, p.oi_usd, s.vol_30d, p.spot_close / s.sma_200 - 1 AS vs_sma_200
    FROM panel p JOIN signals s USING (asset, date) JOIN latest l USING (asset, date)
)
SELECT c.*,
       (SELECT AVG(s.funding_7d_ann <= c.funding_7d_ann) FROM signals s
         WHERE s.asset = c.asset AND s.funding_7d_ann IS NOT NULL) AS funding_7d_pctile,
       (SELECT AVG(s.oi_change_7d <= c.oi_change_7d) FROM signals s
         WHERE s.asset = c.asset AND s.oi_change_7d IS NOT NULL)   AS oi_change_pctile,
       (SELECT AVG(s.vol_30d <= c.vol_30d) FROM signals s
         WHERE s.asset = c.asset AND s.vol_30d IS NOT NULL)        AS vol_pctile
FROM cur c ORDER BY c.asset;
