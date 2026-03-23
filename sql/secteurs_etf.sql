-- ============================================================
-- secteurs_etf.sql — Infrastructure Force Relative Sectorielle v1.0
-- Trading Brain | VT-Source
-- ============================================================
-- Principe :
--   - secteurs_etf        : référentiel des ETF sectoriels + mapping Yahoo
--   - secteurs_etf_prix   : OHLCV + force relative calculée quotidiennement
--   - v_secteurs_en_force : vue prête à l'emploi pour main.py (Niveau 2)
--
-- Indice de référence :
--   US  → ^GSPC  (S&P 500)
--   EU  → ^STOXX (STOXX Europe 600) — ticker Yahoo : ^STOXX
--   BE  → ^BFX   (BEL 20)
-- ============================================================

-- ============================================================
-- TABLE 1 : Référentiel ETF sectoriels
-- ============================================================
CREATE TABLE IF NOT EXISTS secteurs_etf (
    id               SERIAL PRIMARY KEY,
    ticker_etf       VARCHAR(20)  NOT NULL UNIQUE,   -- ex: XLK, TNOW.DE
    nom_etf          VARCHAR(100) NOT NULL,
    secteur_yahoo    VARCHAR(50)  NOT NULL,           -- doit matcher tickers_info.secteur
    zone             VARCHAR(10)  NOT NULL            -- 'US', 'EU', 'BE'
                     CHECK (zone IN ('US', 'EU', 'BE')),
    indice_reference VARCHAR(20)  NOT NULL,           -- ^GSPC, ^STOXX, ^BFX
    actif            BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at       TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- TABLE 2 : Historique OHLCV + force relative ETF sectoriels
-- ============================================================
CREATE TABLE IF NOT EXISTS secteurs_etf_prix (
    id                    SERIAL PRIMARY KEY,
    ticker_etf            VARCHAR(20)  NOT NULL,
    date                  DATE         NOT NULL,

    -- OHLCV ETF
    prix_cloture          NUMERIC,
    prix_ajuste           NUMERIC,
    volume                BIGINT,

    -- Prix de l'indice de référence ce jour-là
    prix_indice           NUMERIC,

    -- Force relative calculée
    ratio_force_relative  NUMERIC,     -- prix_ajuste / prix_indice
    ratio_vs_mm50         NUMERIC,     -- ratio / MM50(ratio) — signal Niveau 2
    en_force_relative     BOOLEAN,     -- TRUE si ratio_vs_mm50 > 1.0

    -- Métadonnées
    updated_at            TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (ticker_etf, date)
);

-- ============================================================
-- INDEX
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_secteurs_etf_prix_date
    ON secteurs_etf_prix (date DESC);

CREATE INDEX IF NOT EXISTS idx_secteurs_etf_prix_ticker_date
    ON secteurs_etf_prix (ticker_etf, date DESC);

CREATE INDEX IF NOT EXISTS idx_secteurs_etf_prix_force
    ON secteurs_etf_prix (en_force_relative, date DESC);

-- ============================================================
-- DONNÉES DE RÉFÉRENCE — ETF US (11 secteurs SPDR S&P 500)
-- ============================================================
INSERT INTO secteurs_etf (ticker_etf, nom_etf, secteur_yahoo, zone, indice_reference) VALUES
    ('XLK',  'Technology Select Sector SPDR',              'Technology',             'US', '^GSPC'),
    ('XLF',  'Financial Select Sector SPDR',               'Financial Services',     'US', '^GSPC'),
    ('XLV',  'Health Care Select Sector SPDR',             'Healthcare',             'US', '^GSPC'),
    ('XLI',  'Industrial Select Sector SPDR',              'Industrials',            'US', '^GSPC'),
    ('XLE',  'Energy Select Sector SPDR',                  'Energy',                 'US', '^GSPC'),
    ('XLY',  'Consumer Discretionary Select Sector SPDR',  'Consumer Cyclical',      'US', '^GSPC'),
    ('XLP',  'Consumer Staples Select Sector SPDR',        'Consumer Defensive',     'US', '^GSPC'),
    ('XLC',  'Communication Services Select Sector SPDR',  'Communication Services', 'US', '^GSPC'),
    ('XLB',  'Materials Select Sector SPDR',               'Basic Materials',        'US', '^GSPC'),
    ('XLRE', 'Real Estate Select Sector SPDR',             'Real Estate',            'US', '^GSPC'),
    ('XLU',  'Utilities Select Sector SPDR',               'Utilities',              'US', '^GSPC')
ON CONFLICT (ticker_etf) DO NOTHING;

-- ============================================================
-- DONNÉES DE RÉFÉRENCE — ETF EU (iShares / Xtrackers STOXX 600)
-- Tous cotés sur Xetra (DE) — liquides, couverts en EUR
-- Indice de référence : ^STOXX (STOXX Europe 600)
-- ============================================================
INSERT INTO secteurs_etf (ticker_etf, nom_etf, secteur_yahoo, zone, indice_reference) VALUES
    ('TNOW.DE', 'iShares STOXX Europe 600 Technology',              'Technology',             'EU', '^STOXX'),
    ('EXV1.DE', 'iShares STOXX Europe 600 Banks',                   'Financial Services',     'EU', '^STOXX'),
    ('EXV4.DE', 'iShares STOXX Europe 600 Health Care',             'Healthcare',             'EU', '^STOXX'),
    ('EXV2.DE', 'iShares STOXX Europe 600 Industrial G&S',          'Industrials',            'EU', '^STOXX'),
    ('EXV6.DE', 'iShares STOXX Europe 600 Oil & Gas',               'Energy',                 'EU', '^STOXX'),
    ('EXH2.DE', 'iShares STOXX Europe 600 Retail',                  'Consumer Cyclical',      'EU', '^STOXX'),
    ('EXH1.DE', 'iShares STOXX Europe 600 Food & Beverage',         'Consumer Defensive',     'EU', '^STOXX'),
    ('EXV3.DE', 'iShares STOXX Europe 600 Telecom',                 'Communication Services', 'EU', '^STOXX'),
    ('EXV5.DE', 'iShares STOXX Europe 600 Basic Resources',         'Basic Materials',        'EU', '^STOXX'),
    ('EXI6.DE', 'iShares STOXX Europe 600 Real Estate',             'Real Estate',            'EU', '^STOXX'),
    ('EXV7.DE', 'iShares STOXX Europe 600 Utilities',               'Utilities',              'EU', '^STOXX'),
    -- Secteur Insurance — présent dans le portefeuille BE (KBC, ARGX)
    ('EXH4.DE', 'iShares STOXX Europe 600 Insurance',               'Financial Services',     'EU', '^STOXX'),
    -- Secteur Pharma — complément Healthcare (UCB, SOLB présents en BE)
    ('EXH6.DE', 'iShares STOXX Europe 600 Pharma & Biotech',        'Healthcare',             'EU', '^STOXX')
ON CONFLICT (ticker_etf) DO NOTHING;

-- ============================================================
-- DONNÉES DE RÉFÉRENCE — ETF Belgique
-- Remarque : pas d'ETF sectoriel BEL 20 liquide par secteur.
-- Stratégie retenue : les actions belges utilisent les ETF EU
-- correspondant à leur secteur Yahoo (même mapping).
-- On ajoute uniquement un ETF BEL 20 global pour le contexte macro BE.
-- ============================================================
INSERT INTO secteurs_etf (ticker_etf, nom_etf, secteur_yahoo, zone, indice_reference) VALUES
    ('EWK',  'iShares MSCI Belgium ETF (contexte macro BE)',  'ALL', 'BE', '^BFX')
ON CONFLICT (ticker_etf) DO NOTHING;

-- ============================================================
-- TABLE 3 : Indices de référence (OHLCV quotidien)
-- Centralisé ici pour éviter d'appeler Yahoo N fois par jour
-- ============================================================
CREATE TABLE IF NOT EXISTS indices_prix (
    id            SERIAL PRIMARY KEY,
    ticker_indice VARCHAR(20)  NOT NULL,    -- ^GSPC, ^STOXX, ^BFX
    date          DATE         NOT NULL,
    prix_cloture  NUMERIC,
    prix_ajuste   NUMERIC,
    UNIQUE (ticker_indice, date)
);

CREATE INDEX IF NOT EXISTS idx_indices_prix_date
    ON indices_prix (ticker_indice, date DESC);

-- ============================================================
-- VUE : Secteurs en force relative — prête pour main.py
-- Retourne les secteurs actifs à J (dernière date disponible)
-- ============================================================
CREATE OR REPLACE VIEW v_secteurs_en_force AS
SELECT
    e.secteur_yahoo,
    e.zone,
    e.ticker_etf,
    e.indice_reference,
    p.date,
    p.ratio_force_relative,
    p.ratio_vs_mm50,
    p.en_force_relative
FROM secteurs_etf e
JOIN secteurs_etf_prix p
    ON p.ticker_etf = e.ticker_etf
    AND p.date = (
        SELECT MAX(date)
        FROM secteurs_etf_prix
        WHERE ticker_etf = e.ticker_etf
    )
WHERE e.actif = TRUE
  AND p.en_force_relative = TRUE;

-- ============================================================
-- VUE : Dashboard — état de tous les secteurs (Onglet 3)
-- ============================================================
CREATE OR REPLACE VIEW v_secteurs_dashboard AS
SELECT
    e.zone,
    e.secteur_yahoo,
    e.ticker_etf,
    e.nom_etf,
    p.date,
    ROUND(p.ratio_force_relative::numeric, 6)  AS ratio_fr,
    ROUND(p.ratio_vs_mm50::numeric, 4)         AS ratio_vs_mm50,
    p.en_force_relative,
    CASE
        WHEN p.en_force_relative THEN '✅ En force'
        ELSE '❌ Hors force'
    END AS statut
FROM secteurs_etf e
LEFT JOIN secteurs_etf_prix p
    ON p.ticker_etf = e.ticker_etf
    AND p.date = (
        SELECT MAX(date)
        FROM secteurs_etf_prix
        WHERE ticker_etf = e.ticker_etf
    )
WHERE e.actif = TRUE
ORDER BY e.zone, p.en_force_relative DESC, p.ratio_vs_mm50 DESC;

-- ============================================================
-- Commentaires d'utilisation
-- ============================================================
-- Voir les secteurs actifs aujourd'hui :
--   SELECT * FROM v_secteurs_en_force ORDER BY zone, ratio_vs_mm50 DESC;
--
-- Dashboard complet :
--   SELECT * FROM v_secteurs_dashboard;
--
-- Initialisation (appeler manuellement après déploiement) :
--   GET /sync-etf-sectoriels   → charge 5 ans d'historique
--
-- Mise à jour quotidienne (scheduler 06h15 lun-ven) :
--   GET /sync-etf-sectoriels   → mode incrémental (30 derniers jours)
-- ============================================================
