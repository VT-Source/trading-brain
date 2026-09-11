-- ============================================================
-- analyse_sonde_publication.sql — heure de publication Yahoo (roadmap #35)
-- ============================================================
-- Objectif : pour chaque instrument sondé et chaque séance J, encadrer
-- l'heure UTC à laquelle Yahoo a commencé à renvoyer J avec une clôture
-- exploitable — c'est-à-dire l'heure à partir de laquelle sync_prix l'aurait
-- stockée.
--
--   encore_absente_a : dernier passage où la dernière séance valide était < J
--   presente_des     : premier passage où elle était >= J
--   h_apres_minuit_J : presente_des, en heures après 00h00 UTC du jour J
--                      (24 = minuit suivant ; le pipeline tourne à 25 h, soit
--                      01h00 UTC de J+1)
--
-- Les horodatages s'affichent dans le fuseau de la session (suffixe TZ) ;
-- le calcul en heures, lui, ne dépend d'aucun fuseau (secondes epoch).
-- Pas de AT TIME ZONE : le connecteur MCP Postgres le refuse à la validation.
--
-- La publication a eu lieu entre les deux bornes. Résolution : 1 h, sauf
-- autour du pipeline (le passage de 01h20 se saute pendant qu'il tourne).
-- La première séance vue par un instrument est exclue : elle était déjà
-- publiée avant le début de la mesure.
--
-- Requête 2 (en bas) : synthèse par place, pour décider.
-- ============================================================

-- 1. Détail par instrument et par séance
WITH passages AS (
    SELECT ticker, place, type_instrument, sonde_at, derniere_date_valide
    FROM sonde_publication
    WHERE erreur IS NULL AND derniere_date_valide IS NOT NULL
),
seances AS (
    SELECT DISTINCT ticker, derniere_date_valide AS seance FROM passages
),
encadrement AS (
    SELECT s.ticker, s.seance,
           MAX(p.sonde_at) FILTER (WHERE p.derniere_date_valide <  s.seance) AS encore_absente_a,
           MIN(p.sonde_at) FILTER (WHERE p.derniere_date_valide >= s.seance) AS presente_des
    FROM seances s
    JOIN passages p ON p.ticker = s.ticker
    GROUP BY s.ticker, s.seance
),
instruments AS (
    SELECT DISTINCT ticker, place, type_instrument FROM sonde_publication
)
SELECT i.place, i.type_instrument, e.ticker, e.seance,
       to_char(e.encore_absente_a, 'YYYY-MM-DD HH24:MI TZ') AS encore_absente_a,
       to_char(e.presente_des,     'YYYY-MM-DD HH24:MI TZ') AS presente_des,
       ROUND(((EXTRACT(EPOCH FROM e.presente_des) - (e.seance - DATE '1970-01-01') * 86400) / 3600)::numeric, 1)
           AS h_apres_minuit_j
FROM encadrement e
JOIN instruments i ON i.ticker = e.ticker
WHERE e.encore_absente_a IS NOT NULL
ORDER BY i.place, e.ticker, e.seance;


-- 2. Synthèse par place : quand la séance J est-elle disponible ?
--    h_max <= 25 → le pipeline de 01h00 UTC la verrait chaque nuit.
WITH passages AS (
    SELECT ticker, place, sonde_at, derniere_date_valide
    FROM sonde_publication
    WHERE erreur IS NULL AND derniere_date_valide IS NOT NULL
),
seances AS (
    SELECT DISTINCT ticker, place, derniere_date_valide AS seance FROM passages
),
encadrement AS (
    SELECT s.place, s.ticker, s.seance,
           MAX(p.sonde_at) FILTER (WHERE p.derniere_date_valide <  s.seance) AS encore_absente_a,
           MIN(p.sonde_at) FILTER (WHERE p.derniere_date_valide >= s.seance) AS presente_des
    FROM seances s
    JOIN passages p ON p.ticker = s.ticker
    GROUP BY s.place, s.ticker, s.seance
),
heures AS (
    SELECT place, ticker, seance,
           (EXTRACT(EPOCH FROM presente_des) - (seance - DATE '1970-01-01') * 86400) / 3600 AS h
    FROM encadrement
    WHERE encore_absente_a IS NOT NULL
)
SELECT place,
       COUNT(*)                                                         AS nb_mesures,
       COUNT(DISTINCT ticker)                                           AS nb_instruments,
       ROUND(MIN(h)::numeric, 1)                                        AS h_min,
       ROUND((percentile_cont(0.5) WITHIN GROUP (ORDER BY h))::numeric, 1) AS h_mediane,
       ROUND(MAX(h)::numeric, 1)                                        AS h_max,
       COUNT(*) FILTER (WHERE h <= 25)                                  AS vues_par_le_run_de_01h
FROM heures
GROUP BY place
ORDER BY h_mediane DESC NULLS LAST;
