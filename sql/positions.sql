-- ============================================================
-- positions.sql — Journal de suivi du portefeuille v3.1
-- Trading Brain | VT-Source
-- ============================================================
-- Principe : Saisie 100% manuelle par VT-Source après
--            exécution sur broker externe (Degiro, IBKR...)
--            Le système ne place aucun ordre automatiquement.
-- ============================================================

CREATE TABLE IF NOT EXISTS positions (
    id               SERIAL PRIMARY KEY,

    -- Identification
    ticker           VARCHAR(20)  NOT NULL,

    -- Entrée (saisie manuelle après achat sur broker)
    date_achat       DATE         NOT NULL,
    prix_achat       NUMERIC      NOT NULL,
    montant_investi  NUMERIC      NOT NULL,  -- € investis réellement

    -- Calculé automatiquement à partir de montant_investi / prix_achat
    quantite         NUMERIC GENERATED ALWAYS AS
                     (ROUND(montant_investi / NULLIF(prix_achat, 0), 4))
                     STORED,

    -- Statut
    statut           VARCHAR(10)  NOT NULL DEFAULT 'OUVERT'
                     CHECK (statut IN ('OUVERT', 'FERMÉ')),

    -- Sortie (saisie manuelle après vente sur broker)
    date_vente       DATE,
    prix_vente       NUMERIC,
    raison_vente     VARCHAR(30)
                     CHECK (raison_vente IN (
                         'OBJECTIF',    -- +5% atteint
                         'STOP_LOSS',   -- -7% atteint
                         'RSI',         -- RSI > 65
                         'TEMPOREL',    -- J+15 dépassé
                         'MANUEL'       -- décision libre VT-Source
                     )),

    -- Résultats (calculés automatiquement à la fermeture)
    resultat_eur     NUMERIC GENERATED ALWAYS AS (
                         ROUND(
                             (prix_vente - prix_achat)
                             * (montant_investi / NULLIF(prix_achat, 0)),
                         2)
                     ) STORED,
    resultat_pct     NUMERIC GENERATED ALWAYS AS (
                         ROUND(
                             100.0 * (prix_vente - prix_achat)
                             / NULLIF(prix_achat, 0),
                         2)
                     ) STORED,

    -- Métadonnées
    created_at       TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    updated_at       TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
);

-- Index pour accès rapide aux positions ouvertes
CREATE INDEX IF NOT EXISTS idx_positions_statut
    ON positions (statut);

CREATE INDEX IF NOT EXISTS idx_positions_ticker
    ON positions (ticker, statut);

-- ============================================================
-- Vue pratique pour le dashboard Streamlit
-- ============================================================
CREATE OR REPLACE VIEW v_positions_ouvertes AS
SELECT
    p.id,
    p.ticker,
    p.date_achat,
    p.prix_achat,
    p.montant_investi,
    p.quantite,
    -- Prix actuel récupéré depuis la dernière analyse
    aph.prix_ajuste                                  AS prix_actuel,
    -- P&L en temps réel
    ROUND((aph.prix_ajuste - p.prix_achat) * p.quantite, 2)  AS pl_eur,
    ROUND(100.0 * (aph.prix_ajuste - p.prix_achat)
          / NULLIF(p.prix_achat, 0), 2)              AS pl_pct,
    -- Jours de détention
    CURRENT_DATE - p.date_achat                      AS j_detention,
    -- Indicateurs techniques actuels
    aph.rsi_14,
    -- Alertes de vente
    CASE
        WHEN aph.prix_ajuste >= p.prix_achat * 1.05  THEN 'OBJECTIF ✅'
        WHEN aph.prix_ajuste <= p.prix_achat * 0.93  THEN 'STOP LOSS 🛑'
        WHEN aph.rsi_14 > 65                          THEN 'RSI ÉLEVÉ ⚠️'
        WHEN (CURRENT_DATE - p.date_achat) >= 15      THEN 'TEMPOREL ⏱️'
        ELSE 'Conserver 🟡'
    END                                              AS alerte_vente
FROM positions p
LEFT JOIN actions_prix_historique aph
    ON aph.ticker = p.ticker
    AND aph.date  = (
        SELECT MAX(date)
        FROM actions_prix_historique
        WHERE ticker = p.ticker
    )
WHERE p.statut = 'OUVERT';

-- ============================================================
-- Commentaire : Exemples d'utilisation
-- ============================================================
-- Ajouter une position après achat sur broker :
--   INSERT INTO positions (ticker, date_achat, prix_achat, montant_investi)
--   VALUES ('NVDA', '2026-03-03', 124.50, 1000.00);
--
-- Fermer une position après vente sur broker :
--   UPDATE positions
--   SET statut='FERMÉ', date_vente='2026-03-11',
--       prix_vente=130.75, raison_vente='OBJECTIF'
--   WHERE ticker='NVDA' AND statut='OUVERT';
--
-- Voir toutes les positions ouvertes avec alertes :
--   SELECT * FROM v_positions_ouvertes ORDER BY j_detention DESC;
-- ============================================================
