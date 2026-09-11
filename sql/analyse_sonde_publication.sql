-- ============================================================
-- analyse_sonde_publication.sql — heure de publication Yahoo (roadmap #35)
-- ============================================================
-- Objectif : pour chaque instrument sondé et chaque séance J, encadrer
-- l'heure UTC à partir de laquelle sync_prix aurait pu stocker J.
--
-- On raisonne sur `derniere_date_close` (sonde.py v1.1, #35b) : la dernière
-- séance à clôture valide ET close selon la règle de sync_prix. Jamais sur la
-- dernière ligne brute : pendant qu'une place cote, yfinance renvoie la barre
-- EN COURS du jour, qui masque l'état de la veille — et qui semble ensuite
-- disparaître après la clôture (J absent à 01h00 UTC le lendemain).
--
--   encore_absente_a : DERNIER passage où la séance close était < J
--   presente_des     : premier passage APRÈS celui-ci où elle était >= J
--                      (NULL : J n'est pas revenue depuis — cf. le 07/09 EU)
--   h_absente / h_presente : ces deux instants en heures après 00h00 UTC du
--                      jour J (24 = minuit suivant ; le pipeline tourne à 25 h,
--                      soit 01h00 UTC de J+1)
--
-- La publication définitive a eu lieu entre les deux bornes. Résolution :
-- 1 h, sauf autour du pipeline (le passage de 01h20 se saute pendant qu'il
-- tourne). Une séance déjà publiée au premier passage de la mesure n'a pas
-- de borne basse et n'apparaît pas.
--
-- Hors de portée : un trou AU MILIEU de l'historique (une séance absente
-- alors que la suivante est présente) — la sonde ne regarde que la fin.
--
-- Les horodatages s'affichent dans le fuseau de la session (suffixe TZ) ;
-- les heures, elles, ne dépendent d'aucun fuseau (secondes epoch).
-- Pas de AT TIME ZONE : le connecteur MCP Postgres le refuse à la validation.
--
-- Requête 2 (en bas) : synthèse par place, pour décider.
-- ============================================================

-- 1. Détail par instrument et par séance
WITH passages AS (
    SELECT ticker, place, type_instrument, sonde_at, derniere_date_close
    FROM sonde_publication
    WHERE erreur IS NULL AND derniere_date_close IS NOT NULL
),
seances AS (
    SELECT DISTINCT ticker, derniere_date_close AS seance FROM passages
),
derniere_absence AS (
    SELECT s.ticker, s.seance,
           MAX(p.sonde_at) FILTER (WHERE p.derniere_date_close < s.seance) AS encore_absente_a
    FROM seances s
    JOIN passages p ON p.ticker = s.ticker
    GROUP BY s.ticker, s.seance
),
encadrement AS (
    SELECT d.ticker, d.seance, d.encore_absente_a, MIN(p.sonde_at) AS presente_des
    FROM derniere_absence d
    LEFT JOIN passages p ON p.ticker = d.ticker
                        AND p.sonde_at > d.encore_absente_a
                        AND p.derniere_date_close >= d.seance
    WHERE d.encore_absente_a IS NOT NULL
    GROUP BY d.ticker, d.seance, d.encore_absente_a
),
instruments AS (
    SELECT DISTINCT ticker, place, type_instrument FROM sonde_publication
)
SELECT i.place, i.type_instrument, e.ticker, e.seance,
       to_char(e.encore_absente_a, 'YYYY-MM-DD HH24:MI TZ') AS encore_absente_a,
       to_char(e.presente_des,     'YYYY-MM-DD HH24:MI TZ') AS presente_des,
       ROUND(((EXTRACT(EPOCH FROM e.encore_absente_a) - (e.seance - DATE '1970-01-01') * 86400) / 3600)::numeric, 1)
           AS h_absente,
       ROUND(((EXTRACT(EPOCH FROM e.presente_des) - (e.seance - DATE '1970-01-01') * 86400) / 3600)::numeric, 1)
           AS h_presente,
       CASE WHEN e.presente_des IS NULL THEN 'absente depuis' ELSE 'publiée' END AS statut
FROM encadrement e
JOIN instruments i ON i.ticker = e.ticker
ORDER BY i.place, e.ticker, e.seance;


-- 2. Synthèse par place : la séance J est-elle là pour le run de 01h00 UTC ?
--    vues_par_run_01h    : présente au plus tard à 25 h (01h00 de J+1)
--    manquees_par_run_01h : encore absente à 25 h ou après
--    le reste tombe dans la zone d'incertitude d'un passage.
WITH passages AS (
    SELECT ticker, place, sonde_at, derniere_date_close
    FROM sonde_publication
    WHERE erreur IS NULL AND derniere_date_close IS NOT NULL
),
seances AS (
    SELECT DISTINCT ticker, place, derniere_date_close AS seance FROM passages
),
derniere_absence AS (
    SELECT s.place, s.ticker, s.seance,
           MAX(p.sonde_at) FILTER (WHERE p.derniere_date_close < s.seance) AS encore_absente_a
    FROM seances s
    JOIN passages p ON p.ticker = s.ticker
    GROUP BY s.place, s.ticker, s.seance
),
encadrement AS (
    SELECT d.place, d.ticker, d.seance, d.encore_absente_a, MIN(p.sonde_at) AS presente_des
    FROM derniere_absence d
    LEFT JOIN passages p ON p.ticker = d.ticker
                        AND p.sonde_at > d.encore_absente_a
                        AND p.derniere_date_close >= d.seance
    WHERE d.encore_absente_a IS NOT NULL
    GROUP BY d.place, d.ticker, d.seance, d.encore_absente_a
),
heures AS (
    SELECT place, ticker, seance,
           (EXTRACT(EPOCH FROM encore_absente_a) - (seance - DATE '1970-01-01') * 86400) / 3600 AS h_absente,
           (EXTRACT(EPOCH FROM presente_des)     - (seance - DATE '1970-01-01') * 86400) / 3600 AS h_presente
    FROM encadrement
)
SELECT place,
       COUNT(*)                                                                  AS nb_seances,
       COUNT(DISTINCT ticker)                                                    AS nb_instruments,
       COUNT(*) FILTER (WHERE h_presente IS NULL)                                AS absentes_depuis,
       ROUND(MIN(h_presente)::numeric, 1)                                        AS h_presente_min,
       ROUND((percentile_cont(0.5) WITHIN GROUP (ORDER BY h_presente))::numeric, 1) AS h_presente_mediane,
       ROUND(MAX(h_presente)::numeric, 1)                                        AS h_presente_max,
       COUNT(*) FILTER (WHERE h_presente <= 25)                                  AS vues_par_run_01h,
       COUNT(*) FILTER (WHERE h_absente >= 25)                                   AS manquees_par_run_01h
FROM heures
GROUP BY place
ORDER BY h_presente_mediane DESC NULLS LAST;
