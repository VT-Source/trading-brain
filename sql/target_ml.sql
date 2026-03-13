-- Reset
UPDATE actions_prix_historique SET target_ml = NULL;

-- Recalcul sur TOUTES les lignes
UPDATE actions_prix_historique a
SET target_ml = CASE
    WHEN sub.future_price >= a.prix_ajuste * 1.05 THEN 1
    ELSE 0
END
FROM (
    SELECT ticker, date,
        LEAD(prix_ajuste, 10) OVER (
            PARTITION BY ticker ORDER BY date ASC
        ) AS future_price
    FROM actions_prix_historique
    WHERE prix_ajuste IS NOT NULL AND prix_ajuste > 0
) sub
WHERE a.ticker = sub.ticker
  AND a.date   = sub.date
  AND sub.future_price IS NOT NULL;

-- Vérification
SELECT 
    target_ml,
    COUNT(*) AS nb,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
FROM actions_prix_historique
WHERE target_ml IS NOT NULL
GROUP BY target_ml;



