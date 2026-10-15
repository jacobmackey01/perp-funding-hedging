-- Funding regime by asset and year: what a short perp position was paid.
SELECT asset, STRFTIME('%Y', date) AS year,
       COUNT(*)                         AS days,
       SUM(funding)                     AS funding_total,
       AVG(funding) * 365               AS funding_ann,
       AVG(funding > 0)                 AS share_days_positive,
       MAX(funding)                     AS best_day,
       MIN(funding)                     AS worst_day,
       AVG(basis)                       AS avg_basis,
       AVG(oi_usd) / 1e9                AS avg_oi_usd_bn
FROM panel
GROUP BY asset, year
ORDER BY asset, year;
