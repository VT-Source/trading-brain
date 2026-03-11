-- ============================================================
-- target_ml.sql — Définition de la cible ML v3.1
-- Trading Brain | VT-Source
-- ============================================================
-- Principe : target_ml = 1 si un signal AT a abouti à +5%
--            dans les 10 jours ouvrés suivants
--            target_ml = 0 sinon
--
-- ⚠️  À exécuter APRÈS que signal_achat soit correctement
--     calculé avec les 4 conditions AT complètes dans main.py
--
-- Ordre d'exécution recommandé :
--   1. run-analysis  (calcule signal_achat avec 4 conditions AT)
--   2. Ce script     (calcule target_ml sur les signaux AT)
--   3. train-model   (entraîne le ML sur les signaux AT)
-- ============================================================

-- Étape 1 : Reset target_ml sur toutes les lignes
UPDATE actions_prix_historique
SET target_ml = NULL;

-- Étape 2 : Calculer target_ml uniquement sur signal_achat = TRUE
--           en regardant le prix 10 jours ouvrés plus tard
UPDATE actions_prix_historique a
SET target_ml = CASE
    WHEN sub.future_price >= a.prix_ajuste * 1.05 THEN 1
    ELSE 0
END
FROM (
    SELECT
        ticker,
        date,
        LEAD(prix_ajuste, 10) OVER (
            PARTITION BY ticker
            ORDER BY date ASC
        ) AS future_price
    FROM actions_prix_historique
    WHERE signal_achat = TRUE
      AND prix_ajuste IS NOT NULL
      AND prix_ajuste > 0
) sub
WHERE a.ticker = sub.ticker
  AND a.date   = sub.date
  AND a.signal_achat = TRUE
  AND sub.future_price IS NOT NULL;  -- On ignore les 10 derniers jours (pas de futur)

-- Étape 3 : Rapport de vérification
SELECT
    COUNT(*)                                          AS total_signaux_AT,
    COUNT(target_ml)                                  AS signaux_avec_target,
    SUM(CASE WHEN target_ml = 1 THEN 1 ELSE 0 END)   AS succes_plus5pct,
    SUM(CASE WHEN target_ml = 0 THEN 1 ELSE 0 END)   AS echecs,
    ROUND(
        100.0 * SUM(CASE WHEN target_ml = 1 THEN 1 ELSE 0 END)
        / NULLIF(COUNT(target_ml), 0), 2
    )                                                 AS taux_succes_pct
FROM actions_prix_historique
WHERE signal_achat = TRUE;

-- ============================================================
-- Résultat attendu :
--   taux_succes_pct entre 15% et 40% = normal et sain
--   taux_succes_pct > 60%            = suspect, vérifier signal_achat
--   taux_succes_pct < 10%            = seuils AT trop permissifs
-- ============================================================






