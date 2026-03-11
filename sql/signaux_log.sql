-- ============================================================
-- signaux_log.sql — Traçabilité des décisions v3.1
-- Trading Brain | VT-Source
-- ============================================================
-- Principe : Chaque signal détecté par le système est loggué,
--            qu'il soit retenu ou rejeté. Permet de mesurer
--            la qualité réelle des décisions dans le temps.
-- ============================================================

CREATE TABLE IF NOT EXISTS signaux_log (
    id               SERIAL PRIMARY KEY,

    -- Identification du signal
    ticker           VARCHAR(20)  NOT NULL,
    date             DATE         NOT NULL,

    -- Couche 1 : Analyse Technique
    rsi_14           NUMERIC,
    sma_200          NUMERIC,
    vol_avg_20       NUMERIC,
    bb_lower         NUMERIC,
    signal_achat     BOOLEAN      NOT NULL DEFAULT FALSE,

    -- Couche 2 : Score ML
    confiance_ml     NUMERIC,     -- NULL si signal_achat = FALSE

    -- Couche 3 : Sentiment NLP
    sentiment_label  VARCHAR(20), -- NULL si confiance_ml <= 0.6
                                  -- 'positif' / 'neutre' / 'négatif'

    -- Décision finale du système
    decision_finale  VARCHAR(20)  NOT NULL
                     CHECK (decision_finale IN (
                         'SUGGESTION',    -- AT + ML + NLP OK → affiché dashboard
                         'REJETÉ_ML',     -- signal_achat=TRUE mais confiance_ml <= 0.6
                         'REJETÉ_NLP',    -- AT+ML OK mais sentiment négatif
                         'IGNORÉ'         -- signal_achat = FALSE
                     )),

    -- Prix au moment du signal (J0)
    prix_j0          NUMERIC,

    -- Résultat mesuré a posteriori (rempli automatiquement à J+10)
    prix_j10         NUMERIC,     -- NULL jusqu'à J+10
    resultat_pct     NUMERIC,     -- NULL jusqu'à J+10

    -- Métadonnées
    logged_at        TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
);

-- Index pour accès rapide
CREATE INDEX IF NOT EXISTS idx_signaux_log_date
    ON signaux_log (date DESC);

CREATE INDEX IF NOT EXISTS idx_signaux_log_ticker
    ON signaux_log (ticker, date DESC);

CREATE INDEX IF NOT EXISTS idx_signaux_log_decision
    ON signaux_log (decision_finale, date DESC);

-- ============================================================
-- Vue : Métriques de santé du système (Onglet 3 dashboard)
-- ============================================================
CREATE OR REPLACE VIEW v_sante_systeme AS
SELECT
    -- Fenêtre temporelle
    MIN(date)                                          AS depuis,
    MAX(date)                                          AS jusqu_au,
    COUNT(DISTINCT date)                               AS nb_jours,

    -- Volume de signaux
    COUNT(*)                                           AS total_analyses,
    SUM(CASE WHEN signal_achat THEN 1 ELSE 0 END)      AS signaux_AT,
    SUM(CASE WHEN decision_finale = 'SUGGESTION' THEN 1 ELSE 0 END)
                                                       AS suggestions_envoyees,
    SUM(CASE WHEN decision_finale = 'REJETÉ_ML'  THEN 1 ELSE 0 END)
                                                       AS rejets_ml,
    SUM(CASE WHEN decision_finale = 'REJETÉ_NLP' THEN 1 ELSE 0 END)
                                                       AS rejets_nlp,

    -- Qualité ML (rolling 30 jours, signaux avec résultat connu)
    ROUND(
        100.0 * SUM(
            CASE WHEN decision_finale = 'SUGGESTION'
                 AND resultat_pct >= 5 THEN 1 ELSE 0 END
        ) / NULLIF(
            SUM(CASE WHEN decision_finale = 'SUGGESTION'
                     AND resultat_pct IS NOT NULL THEN 1 ELSE 0 END), 0
        ), 2
    )                                                  AS taux_reussite_pct

FROM signaux_log
WHERE date >= CURRENT_DATE - INTERVAL '30 days';

-- ============================================================
-- Job : Mise à jour automatique des résultats à J+10
-- À appeler depuis scheduler.py chaque soir
-- ============================================================
CREATE OR REPLACE FUNCTION update_signaux_log_resultats()
RETURNS INTEGER AS $$
DECLARE
    updated_count INTEGER;
BEGIN
    UPDATE signaux_log sl
    SET
        prix_j10     = aph.prix_ajuste,
        resultat_pct = ROUND(
            100.0 * (aph.prix_ajuste - sl.prix_j0) / NULLIF(sl.prix_j0, 0),
            2
        )
    FROM actions_prix_historique aph
    WHERE aph.ticker = sl.ticker
      AND aph.date   = sl.date + INTERVAL '10 days'
      AND sl.prix_j10 IS NULL          -- Pas encore rempli
      AND sl.prix_j0  IS NOT NULL      -- Prix initial connu
      AND sl.date <= CURRENT_DATE - INTERVAL '10 days';  -- 10 jours écoulés

    GET DIAGNOSTICS updated_count = ROW_COUNT;
    RETURN updated_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- Commentaire : Intégration dans main.py
-- ============================================================
-- À appeler dans run_analysis_logic() pour chaque ticker analysé :
--
-- INSERT INTO signaux_log
--   (ticker, date, rsi_14, sma_200, vol_avg_20, bb_lower,
--    signal_achat, confiance_ml, sentiment_label,
--    decision_finale, prix_j0)
-- VALUES
--   (:ticker, CURRENT_DATE, :rsi_14, :sma_200, :vol_avg_20, :bb_lower,
--    :signal_achat, :confiance_ml, :sentiment_label,
--    :decision_finale, :prix_actuel)
-- ON CONFLICT (ticker, date) DO NOTHING;
-- ============================================================
