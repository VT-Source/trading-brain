-- ============================================================
-- couverture_par_place.sql — mesure du garde-fou de fraîcheur (roadmap #28)
-- ============================================================
-- Objectif : compter, pour chaque séance des 120 derniers jours et chaque
-- place de cotation, combien de tickers ont effectivement une barre en base.
--
-- Sert à mesurer le taux de FAUSSES ALERTES de la règle avant de la brancher :
--   pct = 0    → la place n'a pas coté (férié local) : comportement attendu
--   0 < pct<80 → couverture partielle : c'est ce que la règle déclare cassé
--   pct >= 80  → place exploitable
--
-- Ne sont retournées que les lignes < 100 %, pour tenir dans un export lisible.
-- ============================================================

WITH univers AS (
    SELECT DISTINCT
        ticker,
        CASE WHEN ticker NOT LIKE '%.%' THEN 'US'
             ELSE upper(split_part(ticker, '.', 2))
        END AS place
    FROM actions_prix_historique
    WHERE date >= CURRENT_DATE - INTERVAL '120 days'
      AND prix_ajuste IS NOT NULL
),
effectifs AS (
    SELECT place, COUNT(*) AS nb_tickers
    FROM univers
    GROUP BY place
),
couverture AS (
    SELECT h.date, u.place, COUNT(DISTINCT h.ticker) AS nb_barres
    FROM actions_prix_historique h
    JOIN univers u ON u.ticker = h.ticker
    WHERE h.date >= CURRENT_DATE - INTERVAL '120 days'
      AND h.prix_ajuste IS NOT NULL
    GROUP BY h.date, u.place
),
-- Toutes les combinaisons date × place, pour que les places TOTALEMENT
-- absentes un jour donné apparaissent bien avec 0 (un LEFT JOIN sur
-- couverture seule les ferait disparaître, et ce sont justement les fériés).
grille AS (
    SELECT d.date, e.place, e.nb_tickers
    FROM (SELECT DISTINCT date FROM couverture) d
    CROSS JOIN effectifs e
)
SELECT
    g.date,
    g.place,
    COALESCE(c.nb_barres, 0) AS nb_barres,
    g.nb_tickers,
    ROUND(100.0 * COALESCE(c.nb_barres, 0) / g.nb_tickers, 1) AS pct
FROM grille g
LEFT JOIN couverture c ON c.date = g.date AND c.place = g.place
WHERE COALESCE(c.nb_barres, 0) < g.nb_tickers
ORDER BY g.date, g.place;
