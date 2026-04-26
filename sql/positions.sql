-- ============================================================
-- positions.sql — Journal de suivi du portefeuille v4.1
-- Trading Brain | VT-Source
-- ============================================================
-- Principe : Saisie 100% manuelle par VT-Source après
--            exécution sur broker externe (Degiro, IBKR...)
--            Le système ne place aucun ordre automatiquement.
-- ============================================================
-- ⚠️  Ce fichier reflète la structure RÉELLE en production
--     (vérifiée le 2026-04-26 via information_schema + pg_constraint).
--     Ne pas modifier la base sans mettre à jour ce fichier.
-- ============================================================

CREATE TABLE IF NOT EXISTS positions (
    id               SERIAL PRIMARY KEY,

    -- Identification
    ticker           VARCHAR(20)  NOT NULL,

    -- Entrée (saisie manuelle après achat sur broker)
    date_achat       DATE         NOT NULL,
    prix_achat       NUMERIC      NOT NULL,
    quantite         NUMERIC      NOT NULL,

    -- Montant calculé automatiquement
    montant_investi  NUMERIC GENERATED ALWAYS AS
                     (ROUND(quantite * prix_achat, 2))
                     STORED,

    -- Statut
    statut           VARCHAR(10)  NOT NULL DEFAULT 'OUVERT'
                     CHECK (statut IN ('OUVERT', 'FERMÉ')),

    -- Sortie (saisie manuelle après vente sur broker)
    date_vente       DATE,
    prix_vente       NUMERIC,
    raison_vente     VARCHAR(30)
                     CHECK (raison_vente IN (
                         'TRAILING_STOP',
                         'TREND_BROKEN',
                         'MOMENTUM_LOST',
                         'SECTOR_WEAK',
                         'MACRO_BEARISH',
                         'MANUEL'
                     )),

    -- Résultats calculés automatiquement à la fermeture
    resultat_eur     NUMERIC GENERATED ALWAYS AS (
                         ROUND(
                             (prix_vente - prix_achat) * quantite,
                         2)
                     ) STORED,
    resultat_pct     NUMERIC GENERATED ALWAYS AS (
                         ROUND(
                             (100.0 * (prix_vente - prix_achat)
                             / NULLIF(prix_achat, 0))::numeric,
                         2)
                     ) STORED,

    -- Traçabilité
    decision_id      INTEGER,
    source           VARCHAR(20)  NOT NULL DEFAULT 'ranking'
                     CHECK (source IN ('ranking', 'manuel')),
    commentaire      TEXT,

    -- Métadonnées
    created_at       TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    updated_at       TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
);

-- Index
CREATE INDEX IF NOT EXISTS idx_positions_statut
    ON positions (statut);

CREATE INDEX IF NOT EXISTS idx_positions_ticker
    ON positions (ticker, statut);

-- ============================================================
-- Vue pour le dashboard Streamlit
-- ============================================================
CREATE OR REPLACE VIEW v_positions_ouvertes AS
SELECT
    p.id,
    p.ticker,
    p.date_achat,
    p.prix_achat,
    p.montant_investi,
    p.quantite,
    aph.prix_ajuste                                       AS prix_actuel,
    ROUND(
        ((aph.prix_ajuste - p.prix_achat) * p.quantite)::numeric,
    2)                                                    AS pl_eur,
    ROUND(
        (100.0 * (aph.prix_ajuste - p.prix_achat)
        / NULLIF(p.prix_achat, 0))::numeric,
    2)                                                    AS pl_pct,
    (CURRENT_DATE - p.date_achat)                         AS j_detention,
    aph.rsi_14
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
-- Exemples d'utilisation
-- ============================================================
-- Ouvrir une position après achat sur broker :
--   INSERT INTO positions (ticker, date_achat, prix_achat, quantite, source)
--   VALUES ('NVDA', '2026-03-03', 124.50, 8, 'ranking');
--
-- Fermer une position après vente sur broker :
--   UPDATE positions
--   SET statut='FERMÉ', date_vente='2026-03-11',
--       prix_vente=130.75, raison_vente='TRAILING_STOP'
--   WHERE ticker='NVDA' AND statut='OUVERT';
--
-- Voir toutes les positions ouvertes :
--   SELECT * FROM v_positions_ouvertes ORDER BY j_detention DESC;
-- ============================================================
