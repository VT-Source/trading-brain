# ============================================================
# ranking.py — Trading Brain — v1.1
# ============================================================
# Calcul et persistance du ranking momentum journalier.
#
# v1.1 (2026-09-08) — RATTRAPAGE D'UNE PLAGE DE DATES (roadmap #31)
#   Le trou de ranking du 24 au 31/08 (job figé, #25) laisse 7 jours sans
#   ligne dans ranking_hebdo. Aucun endpoint ne savait le combler :
#   `/sync-prix` rapatrie des prix (ils étaient déjà là), `/run-analysis`
#   écrit dans actions_prix_historique (que le ranking ne lit pas), et
#   `/compute-ranking` écrivait TOUJOURS `date_calcul = date.today()`.
#   Ce n'était pas une donnée manquante, c'était une capacité absente.
#
#   Le scoring était déjà entièrement paramétré par la date
#   (`compute_composite_score`, `get_macro_regime`,
#   `get_secteur_force_for_ticker` prennent tous une date) : seules deux
#   valeurs étaient figées sur « maintenant ». Le lot les paramètre et
#   n'ajoute rien d'autre au chemin quotidien, qui reste iso-comportement.
#
# Extrait de main.py (roadmap #5, 2026-09-08). Le lot est à
# ISO-COMPORTEMENT : aucune règle n'est modifiée, seul le découpage change.
#
# Deux raisons à l'extraction, dans cet ordre d'importance :
#   1. Le branchement des 4 politiques de fraîcheur décidait du sort d'un
#      ranking depuis l'intérieur d'une fonction de 170 lignes qui touche la
#      DB — donc intestable. `appliquer_politique_fraicheur` est désormais
#      une fonction PURE, verrouillée par tests (motif `zone_priority_for`,
#      puis `scheduling.py`, puis `freshness.py`).
#   2. main.py avait franchi les 100 Ko, au-delà de la limite d'écriture de
#      l'API GitHub.
#
# `engine` est INJECTÉ et non importé : un `from main import engine` créerait
# un import circulaire. Précédent dans le code : `_poll_ai_opinions_with_alert`
# et `_pipeline_prix_puis_analyse` prennent déjà `engine` en paramètre.
#
# ⚠️ DÉPLOIEMENT COUPLÉ : ranking.py + analysis.py + main.py, un seul commit.
#    Import NON gardé côté main.py, comme scheduling.py et freshness.py — un
#    module qui porte une décision de trade ne doit pas disparaître en silence.
# ============================================================

import os
import json
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import text

from freshness import diagnostic_fraicheur, derniere_seance_commune

# --- Alerting Telegram (roadmap #19) ---
# Import tolérant, identique à celui de main.py : un alerting.py absent ou
# cassé ne doit jamais empêcher un ranking de se calculer.
try:
    from alerting import alert_ranking_composition, alert_fraicheur_places
except Exception as _e_alert:  # pragma: no cover
    print(f"⚠️ alerting.py indisponible ({_e_alert}) — alertes ranking désactivées")
    def alert_ranking_composition(*a, **k): return {"sent": False, "reason": "module_absent"}
    def alert_fraicheur_places(*a, **k):    return {"sent": False, "reason": "module_absent"}

try:
    from backtest_ranking import (
        load_all_tickers,
        load_all_price_data,
        compute_all_indicators,
        compute_composite_score,
        compute_adaptive_k,
        load_secteur_mapping,
        load_all_secteur_force,
        load_macro_data,
        get_macro_regime,
        get_ticker_zone,
        MIN_HISTORY,
    )
except ImportError:  # pragma: no cover
    load_all_tickers = None
    load_all_price_data = None
    compute_all_indicators = None
    compute_composite_score = None
    compute_adaptive_k = None
    load_secteur_mapping = None
    load_all_secteur_force = None
    load_macro_data = None
    get_macro_regime = None
    get_ticker_zone = None
    MIN_HISTORY = 260

RANKING_VERSION = "1.1"

# Jours où le ranking tourne : lundi(0) à samedi(5). Le scheduler est en
# `day_of_week="mon-sat"` — un rattrapage doit produire exactement les mêmes
# jours, sinon il fabrique des lignes que la prod n'aurait jamais écrites.
JOURS_RANKING = (0, 1, 2, 3, 4, 5)

# Garde-fou du rattrapage : au-delà, on refuse. Une plage large n'est presque
# jamais un rattrapage d'incident, c'est une faute de frappe — et avec
# overwrite=true elle réécrirait des mois d'historique avec la version
# courante de l'algorithme.
BACKFILL_MAX_JOURS = 120

# ============================================================
# GARDE-FOU DE FRAÎCHEUR PAR PLACE DE COTATION (roadmap #28)
# ============================================================
# Que faire quand une place de cotation n'est pas exploitable à la date
# de classement (couverture partielle, ou absence de plus d'une séance) :
#   observer — diagnostiquer, tracer, alerter, mais classer quand même.
#              Mode de déploiement : on mesure le taux de fausses alertes
#              sur données de prod avant de laisser la règle décider.
#   exclure  — écarter les tickers des places douteuses et classer le reste.
#   aligner  — écarter les places douteuses PUIS classer tout le monde à la
#              dernière séance commune aux places saines. Seul mode qui
#              supprime vraiment la comparaison inter-dates : un férié local
#              (US fermés, Corée ouverte) laisse sinon le score comparer du
#              04/09 américain à du 07/09 coréen, ce qui est légitime côté
#              données mais reste une comparaison décalée d'une séance.
#              Coût : le ranking recule d'une séance chaque jour férié local.
#   refuser  — ne pas écrire de ranking du tout ce jour-là.
# Piloté par variable d'environnement : basculer ne demande ni commit ni
# redéploiement. Motif d'origine (08/09) : main.py pesait alors plus de 100 Ko
# et chaque bascule aurait coûté un upload web. L'extraction (#5) a levé cette
# contrainte, mais la raison de fond demeure — un comportement encore à
# arbitrer se livre derrière une variable, pas derrière un commit.
POLITIQUE_FRAICHEUR = os.getenv("FRAICHEUR_POLITIQUE", "observer").strip().lower()
if POLITIQUE_FRAICHEUR not in ("observer", "exclure", "aligner", "refuser"):
    print(f"⚠️ FRAICHEUR_POLITIQUE={POLITIQUE_FRAICHEUR!r} inconnue — repli sur 'observer'")
    POLITIQUE_FRAICHEUR = "observer"

# ============================================================
# RÈGLE PURE — application de la politique de fraîcheur
# ============================================================

def appliquer_politique_fraicheur(diagnostic: dict, politique: str,
                                  date_classement: date) -> dict:
    """
    Traduit un diagnostic de fraîcheur (freshness.diagnostic_fraicheur) en
    décision, sans aucun accès DB, réseau ou I/O.

    Args:
        diagnostic      : sortie de freshness.diagnostic_fraicheur
        politique       : observer | exclure | aligner | refuser
        date_classement : dernière séance disponible (datetime.date)

    Returns:
        {
          "action"          : "classer" | "refuser",
          "alerter"         : bool  — déclencheur (e), vrai ssi le diagnostic
                              n'est pas ok, indépendamment de la politique
          "ecarter"         : bool  — vrai ssi la politique demande d'écarter
                                      des tickers (la liste peut être vide)
          "tickers_exclus"  : list — tickers à écarter avant scoring ([] en
                              mode observer, même quand le diagnostic alerte)
          "date_classement" : date — reculée à la dernière séance commune en
                              mode aligner, inchangée sinon
          "aligne"          : bool  — vrai ssi la date a effectivement reculé
        }

    ⚠️ Le recul `aligner` s'applique MÊME quand le diagnostic est ok : un férié
    local (US fermés, Corée ouverte) produit un diagnostic sain tout en laissant
    le score comparer du 04/09 américain à du 07/09 coréen. C'est précisément
    ce que ce mode existe pour supprimer.
    """
    ok = bool(diagnostic.get("ok"))
    alerter = not ok

    if not ok and politique == "refuser":
        return {"action": "refuser", "alerter": True, "ecarter": False,
                "tickers_exclus": [], "date_classement": date_classement,
                "aligne": False}

    ecarter = (not ok) and politique in ("exclure", "aligner")
    tickers_exclus = list(diagnostic.get("tickers_exclus", [])) if ecarter else []

    aligne = False
    if politique == "aligner":
        commune = derniere_seance_commune(diagnostic)
        if commune and commune < date_classement:
            date_classement = commune
            aligne = True

    return {"action": "classer", "alerter": alerter, "ecarter": ecarter,
            "tickers_exclus": tickers_exclus,
            "date_classement": date_classement, "aligne": aligne}


# ============================================================
# RÈGLES PURES — sélection des dates d'un rattrapage
# ============================================================

def dates_a_backfiller(debut: date, fin: date, dates_existantes=(),
                       overwrite: bool = False,
                       jours_actifs=JOURS_RANKING) -> list:
    """
    Dates pour lesquelles il faut (re)calculer un ranking, sans aucun accès
    DB, réseau ou I/O.

    Deux filtres, et ils ne servent pas la même chose :
      - `jours_actifs` reproduit le calendrier du scheduler (lun-sam). Le
        rattrapage ne doit pas inventer des dimanches que la prod n'aurait
        jamais produits, sinon il rend le calendrier de `ranking_hebdo`
        incohérent avec lui-même — exactement le genre de trou qu'on répare.
      - `dates_existantes` protège l'historique : par défaut on ne repasse
        JAMAIS sur une date déjà classée. `overwrite=True` est réservé à
        deux usages : réparer un rattrapage raté, et surtout alimenter le
        `dry_run` de calibration (recalculer des dates connues pour comparer
        au stocké) — dans ce cas rien n'est écrit.

    Args:
        debut, fin        : bornes incluses
        dates_existantes  : dates déjà présentes dans ranking_hebdo
        overwrite         : ignorer le filtre `dates_existantes`
        jours_actifs      : indices weekday() autorisés (0 = lundi)

    Returns:
        liste triée de `datetime.date`
    """
    if debut is None or fin is None or debut > fin:
        return []

    existantes = set(dates_existantes or ())
    resultat, jour = [], debut
    while jour <= fin:
        if jour.weekday() in jours_actifs and (overwrite or jour not in existantes):
            resultat.append(jour)
        jour += timedelta(days=1)
    return resultat


def date_de_scoring(dates_disponibles, cible: date = None,
                    inclure_seance_du_jour: bool = False):
    """
    Dernière séance de marché utilisable pour classer à la date `cible`.
    Pure : ne lit que la liste de dates qu'on lui passe.

    `cible=None` → `max(dates_disponibles)`. C'est le comportement littéral du
    job quotidien, qui classe sur la donnée la plus récente en base.

    `cible` fournie (rattrapage) → dernière séance **strictement antérieure**
    à la cible. C'est le point le plus important de tout ce lot, et il mérite
    d'être justifié :

      Un ranking daté du 25/08 est une décision que le système prend au petit
      matin du 25/08 : à cet instant, la séance du 25 n'a pas encore eu lieu.
      Reconstruire ce ranking en y incluant la clôture du 25 fabriquerait une
      ligne que la prod n'aurait jamais pu produire, et surtout une ligne qui
      « sait » ce qui allait se passer. Toute analyse qui mesure un rendement
      à partir de `date_calcul` (durée dans le classement, rendement futur du
      panier) serait alors biaisée sur les seules dates rattrapées — un biais
      invisible, concentré, et du mauvais côté : il flatte le rattrapage.

      C'est aussi la convention que la prod applique depuis `sync.py` v1.7
      (2026-09-07), qui refuse toute barre datée du jour courant. Les lignes
      antérieures à cette date affichent souvent `data_date == date_calcul` :
      c'est le bug de la barre partielle, pas une convention à reproduire.

    `inclure_seance_du_jour=True` restaure l'ancienne convention (séance de la
    cible incluse). À n'utiliser que pour comparer un rattrapage aux lignes
    d'avant le 07/09, jamais pour produire des lignes destinées à l'analyse.

    Returns:
        `datetime.date`, ou None si aucune séance ne convient.
    """
    dates = [d for d in dates_disponibles if d is not None]
    if not dates:
        return None
    if cible is None:
        return max(dates)

    if inclure_seance_du_jour:
        eligibles = [d for d in dates if d <= cible]
    else:
        eligibles = [d for d in dates if d < cible]
    return max(eligibles) if eligibles else None


# ============================================================
# HELPER : Secteurs en force relative
# ============================================================

def get_secteurs_en_force(engine) -> list[dict]:
    """
    Retourne la liste des secteurs Yahoo actuellement en force relative.
    Interroge la vue v_secteurs_en_force (dernière date disponible).
    Si la table est vide ou absente → retourne [] sans planter l'appelant.
    """
    if engine is None:
        return []
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT secteur_yahoo, zone, ticker_etf, indice_reference,
                       date, ratio_force_relative, ratio_vs_mm50
                FROM v_secteurs_en_force
                ORDER BY zone, ratio_vs_mm50 DESC
            """)).fetchall()

        return [
            {
                "secteur_yahoo"       : r[0],
                "zone"                : r[1],
                "ticker_etf"          : r[2],
                "indice_reference"    : r[3],
                "date"                : str(r[4]),
                "ratio_force_relative": float(r[5]) if r[5] is not None else None,
                "ratio_vs_mm50"       : float(r[6]) if r[6] is not None else None,
            }
            for r in rows
        ]
    except Exception as e:
        print(f"⚠️ get_secteurs_en_force : {e} — retour liste vide.")
        return []

def _charger_contexte(engine) -> dict:
    """
    Charge UNE FOIS tout ce dont le scoring a besoin : prix, indicateurs,
    mapping sectoriel, force relative, macro, et le calendrier des séances.

    Extrait des étapes 1 à 4 de `compute_and_store_ranking`, à
    iso-comportement. C'est ce qui rend le rattrapage d'une plage entière
    aussi coûteux qu'un seul jour : les ~4 min sont ici, le scoring d'une
    date supplémentaire est négligeable.

    ⚠️ Ce partage n'est légitime que parce que `compute_all_indicators` est
    intégralement CAUSAL : `rolling(...)` pour sma_200/sma_150/rvol/atr_14,
    `diff()` pour obv_slope, et pour mom_r2 une boucle qui ne lit que
    `log_prices[i-252:i]`. Aucun indicateur d'une ligne ne dépend d'une ligne
    postérieure — la valeur au 25/08 est donc la même que l'historique
    s'arrête au 25/08 ou coure jusqu'à aujourd'hui. C'est ce qui autorise à
    calculer les indicateurs sur l'historique complet puis à scorer une date
    passée. `tests/test_ranking_backfill.py` verrouille cette propriété : si
    un indicateur non causal apparaît un jour, le test tombe avant que le
    rattrapage ne produise des lignes silencieusement fausses.
    """
    all_tickers = load_all_tickers()
    ticker_data = load_all_price_data(all_tickers)

    for ticker in list(ticker_data.keys()):
        ticker_data[ticker] = compute_all_indicators(ticker_data[ticker])

    seances = sorted({d.date() for df in ticker_data.values() for d in df.index})

    return {
        "ticker_data":     ticker_data,
        "secteur_mapping": load_secteur_mapping(),
        "force_data":      load_all_secteur_force(),
        "macro_data":      load_macro_data(),
        "seances":         seances,
    }


def _construire_records(contexte: dict, date_calcul: date, scoring_date: date,
                        top_n: int, politique: str, alerting: bool = True):
    """
    Cœur commun au job quotidien et au rattrapage : diagnostic de fraîcheur,
    application de la politique, scoring, mise en forme des lignes.
    Ne touche JAMAIS la DB — l'écriture est dans `_persister_records`.

    Retourne (records, meta). `records` vaut None quand la politique refuse
    de classer ; `meta` porte toujours de quoi tracer la décision.

    `alerting=False` coupe les alertes (c) et (e) : un rattrapage de 7 jours
    ne doit pas pousser 7 alertes Telegram rétroactives décrivant l'état du
    marché d'il y a deux semaines.
    """
    secteur_mapping = contexte["secteur_mapping"]
    force_data      = contexte["force_data"]
    macro_data      = contexte["macro_data"]

    ts_scoring = pd.Timestamp(scoring_date)

    # Univers tel qu'il se présentait à la date de scoring. La coupe
    # `df.loc[:ts]` est une tranche contiguë sur un index trié : pas de copie
    # des données, et elle garantit qu'aucune barre postérieure n'entre dans
    # le scoring même si `compute_composite_score` évoluait.
    # Le seuil MIN_HISTORY est réappliqué à cette date : un ticker ajouté
    # récemment passe le filtre aujourd'hui mais ne le passait pas il y a
    # trois semaines — le rattrapage ne doit pas le faire apparaître dans un
    # classement où il n'avait pas sa place.
    # En quotidien, `scoring_date` est le maximum du calendrier : la tranche
    # rend le DataFrame entier et le seuil est déjà garanti par
    # `load_all_price_data`. Les deux lignes sont donc sans effet — c'est
    # voulu, le chemin quotidien reste iso-comportement.
    univers = {}
    for ticker, df in contexte["ticker_data"].items():
        if df.empty:
            continue
        vue = df.loc[:ts_scoring]
        if len(vue) < MIN_HISTORY:
            continue
        univers[ticker] = vue

    if not univers:
        return None, {"action": "refuser", "motif": "aucun ticker exploitable",
                      "resume": "aucun ticker exploitable"}

    # Garde-fou de fraîcheur par place de cotation (roadmap #28)
    #
    # compute_composite_score accepte pour chaque ticker sa dernière barre
    # disponible jusqu'à 5 jours d'écart, en silence (backtest_ranking.py
    # l. 386-388). Le score composite compare donc des tickers arrêtés à
    # des dates différentes — et comme la normalisation min-max est
    # calculée sur l'ensemble des candidats, une place décalée ne fausse
    # pas seulement ses propres lignes : elle déplace le score de tout le
    # monde. Deux incidents en deux jours viennent de là (Corée trop tôt,
    # Europe trop tard).
    #
    # La règle est pure et testée → freshness.py. Elle raisonne par PLACE
    # et non par zone : une place porte un calendrier de bourse, pas une
    # zone. Sans ce découpage, le 14 juillet (Paris fermé, Amsterdam et
    # Bruxelles ouverts) deviendrait une fausse alerte annuelle.
    dernieres_dates = {
        t: df.index.max().date()
        for t, df in univers.items() if not df.empty
    }
    diagnostic = diagnostic_fraicheur(
        dernieres_dates,
        calendrier=[d for d in contexte["seances"] if d <= scoring_date],
        zones={t: get_ticker_zone(t, secteur_mapping) for t in univers},
    )
    print(f"🕐 Fraîcheur : {diagnostic['resume']}")

    # Le branchement des 4 politiques est une règle PURE et testée
    # → appliquer_politique_fraicheur (haut de ce module). Elle décide,
    # cette fonction exécute : alerte, pop des tickers, écriture.
    decision = appliquer_politique_fraicheur(diagnostic, politique, scoring_date)

    if decision["alerter"] and alerting:
        # (e) Alerte push — l'alerte (b) sur tickers en retard a un seuil
        # de 3 jours, sous lequel une séance manquante passait.
        alert_fraicheur_places(diagnostic)

    if decision["action"] == "refuser":
        print("⏭️  Ranking non calculé — places arrêtées à des séances différentes")
        return None, {"action": "refuser", "motif": diagnostic["resume"],
                      "resume": diagnostic["resume"], "diagnostic": diagnostic,
                      "scoring_date": scoring_date}

    if decision["ecarter"]:
        for t in decision["tickers_exclus"]:
            univers.pop(t, None)
        print(f"✂️  {len(decision['tickers_exclus'])} tickers écartés — "
              f"places : {', '.join(diagnostic['places_douteuses'])}")
        if not univers:
            return None, {"action": "refuser", "motif": "Aucune place exploitable",
                          "resume": "Aucune place exploitable",
                          "scoring_date": scoring_date}

    if decision["aligne"]:
        # Reculée jusqu'à la dernière séance commune aux places saines.
        print(f"↩️  Alignement : classement au {decision['date_classement']} "
              f"(dernière séance commune) au lieu du {scoring_date}")
        scoring_date = decision["date_classement"]
        ts_scoring = pd.Timestamp(scoring_date)

    ranking = compute_composite_score(
        univers, ts_scoring, secteur_mapping, force_data, sma_period=200
    )

    macro_regime = get_macro_regime(macro_data, ts_scoring)

    records = []
    for r in ranking[:top_n]:
        ticker = r["ticker"]
        zone = get_ticker_zone(ticker, secteur_mapping)
        secteur = secteur_mapping.get(ticker, {}).get("secteur", "—")
        k = compute_adaptive_k(r["atr_14"], r["prix"]) if r["atr_14"] > 0 else 3.0

        records.append({
            "date_calcul":  date_calcul,
            "rank":         r.get("rank", 0),
            "ticker":       ticker,
            "score":        round(r["score"], 4),
            "mom_r2":       round(r["mom_r2"], 4),
            "rvol":         round(r["rvol"], 2),
            "obv_slope":    round(r["obv_slope"], 2),
            "prix":         round(r["prix"], 2),
            "sma_200":      round(r["sma_200"], 2),
            "atr_14":       round(r["atr_14"], 2),
            "k_adaptatif":  k,
            "zone":         zone,
            "secteur":      secteur,
            "macro_regime": json.dumps(macro_regime),
            "nb_eligible":  len(ranking),
            "nb_total":     len(univers),
            "data_date":    scoring_date,
        })

    meta = {"action": "classer", "scoring_date": scoring_date,
            "nb_eligible": len(ranking), "nb_total": len(univers),
            "resume": diagnostic["resume"], "diagnostic": diagnostic,
            "zones": [r["zone"] for r in records]}
    return records, meta


def _persister_records(engine, date_calcul: date, records: list) -> None:
    """
    Écrit les lignes d'UNE date, en remplaçant ce qui s'y trouvait.
    Le DELETE rend l'opération idempotente : relancer un rattrapage sur la
    même date ne duplique rien.
    """
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM ranking_hebdo WHERE date_calcul = :d"),
                     {"d": date_calcul})
        for rec in records:
            conn.execute(text("""
                INSERT INTO ranking_hebdo
                    (date_calcul, rank, ticker, score, mom_r2, rvol, obv_slope,
                     prix, sma_200, atr_14, k_adaptatif, zone, secteur,
                     macro_regime, nb_eligible, nb_total, data_date)
                VALUES
                    (:date_calcul, :rank, :ticker, :score, :mom_r2, :rvol, :obv_slope,
                     :prix, :sma_200, :atr_14, :k_adaptatif, :zone, :secteur,
                     CAST(:macro_regime AS jsonb), :nb_eligible, :nb_total, :data_date)
            """), rec)


def _ranking_existant(engine, debut: date, fin: date) -> dict:
    """
    {date_calcul: [tickers dans l'ordre du rang]} pour la plage donnée.
    Sert à deux choses : ne pas réécrire une date déjà classée, et comparer
    un recalcul au stocké en `dry_run`.
    """
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT date_calcul, ticker
            FROM ranking_hebdo
            WHERE date_calcul BETWEEN :d1 AND :d2
            ORDER BY date_calcul, rank
        """), {"d1": debut, "d2": fin}).fetchall()

    existant = {}
    for date_calcul, ticker in rows:
        existant.setdefault(date_calcul, []).append(ticker)
    return existant


def compute_and_store_ranking(engine, top_n: int = 20, politique: str = None):
    """
    Calcule le ranking momentum sur tous les tickers et le persiste
    dans ranking_hebdo. Appelé par le scheduler (lun-sam 02h15)
    et par l'endpoint /compute-ranking.

    NB : malgré le nom historique 'ranking_hebdo', la table contient
    désormais un ranking journalier (1 calcul/jour ouvré + samedi).
    Chaque ligne est identifiée par (date_calcul, ticker). Les avis IA
    et décisions humaines restent agrégés par semaine via la clé
    'semaine' (lundi), indépendante de date_calcul.

    Args:
        engine    : SQLAlchemy engine, injecté par main.py (pas d'import
                    circulaire — cf. entête du module)
        top_n     : nombre de lignes persistées dans ranking_hebdo
        politique : surcharge de POLITIQUE_FRAICHEUR. None = variable Railway.
                    Le paramètre existe pour les tests ; la prod passe par
                    l'environnement, pour pouvoir basculer sans redéployer.

    Durée estimée : 3-4 min sur 400 tickers.
    """
    if politique is None:
        politique = POLITIQUE_FRAICHEUR

    if compute_composite_score is None:
        print("❌ compute_and_store_ranking : backtest_ranking.py non disponible")
        return {"error": "backtest_ranking.py non disponible"}
    if engine is None:
        print("❌ compute_and_store_ranking : engine non connecté")
        return {"error": "engine non connecté"}

    try:
        print("📊 Calcul ranking journalier...")

        contexte = _charger_contexte(engine)
        if not contexte["seances"]:
            return {"error": "Aucune donnée disponible"}

        # `cible=None` → `max(seances)`, soit littéralement l'ancien
        # `latest_date = max(all_dates)`. Le chemin quotidien ne connaît pas
        # la notion de date cible : il classe sur la donnée la plus récente.
        aujourd_hui = date.today()
        scoring_date = date_de_scoring(contexte["seances"])

        records, meta = _construire_records(
            contexte, date_calcul=aujourd_hui, scoring_date=scoring_date,
            top_n=top_n, politique=politique, alerting=True,
        )

        if records is None:
            motif = meta.get("motif", "")
            if motif in ("Aucune place exploitable", "aucun ticker exploitable"):
                return {"error": "Aucune place exploitable"}
            return {"status": "skipped", "message": motif,
                    "fraicheur": motif,
                    "data_date": str(meta.get("scoring_date", scoring_date))}

        _persister_records(engine, aujourd_hui, records)

        print(f"✅ Ranking sauvegardé : {len(records)} tickers, "
              f"date données {meta['scoring_date']}")

        # (c) Alerte composition — filet en aval de l'audit post-sync.
        # Le 16/05, 59 tickers EU non synchronisés ont produit un ranking
        # 100 % US sans qu'aucune erreur ne soit levée.
        alert_ranking_composition(meta["zones"],
                                  nb_eligible=meta["nb_eligible"],
                                  data_date=meta["scoring_date"])

        return {"status": "ok", "nb_ranked": len(records),
                "data_date": str(meta["scoring_date"]),
                "fraicheur": meta["resume"],
                "nb_exclus": len(meta["diagnostic"]["tickers_exclus"])
                             if politique in ("exclure", "aligner")
                             else 0}

    except Exception as e:
        print(f"❌ Erreur compute_and_store_ranking : {e}")
        return {"error": str(e)}


def backfill_ranking(engine, debut: date, fin: date, top_n: int = 20,
                     politique: str = None, overwrite: bool = False,
                     dry_run: bool = True,
                     inclure_seance_du_jour: bool = False) -> dict:
    """
    Recalcule le ranking pour une plage de dates PASSÉES et le persiste dans
    ranking_hebdo. Appelé par l'endpoint /backfill-ranking.

    Motif (roadmap #31) : le trou du 24 au 31/08 (job figé, #25) laisse 7
    jours sans ranking. Les prix de ces séances sont en base — c'est le
    ranking dérivé qui manque, et rien ne savait le reconstruire.

    Trois garde-fous, dans cet ordre d'importance :

      1. `dry_run=True` PAR DÉFAUT. Une fonction qui peut réécrire
         l'historique d'un système de trading ne doit pas écrire parce qu'on
         a oublié un paramètre. Il faut demander l'écriture explicitement.
      2. `overwrite=False` par défaut : les dates déjà classées sont
         ignorées. Sélection et écriture sont deux choses distinctes —
         `overwrite=True, dry_run=True` est le mode CALIBRATION : on
         recalcule des dates connues pour comparer au stocké, sans rien
         toucher. C'est la seule façon honnête de savoir ce que vaut un
         rattrapage avant de l'écrire.
      3. Aucune alerte poussée : un rattrapage de 7 jours ne doit pas
         envoyer 7 alertes Telegram sur l'état du marché d'il y a 15 jours.

    ⚠️ Un rattrapage n'est PAS une reconstitution à l'identique. Il calcule
    ce que le ranking d'une date aurait été *avec l'historique de prix tel
    qu'il est aujourd'hui* : barres complètes (celles que la prod voyait
    partielles avant `sync.py` v1.7 ont été écrasées depuis) et prix ajustés
    révisés rétroactivement par les splits/dividendes (R5). Les lignes
    rattrapées sont donc plus propres que leurs voisines, pas identiques.
    `created_at` (posté à l'insertion) les distingue de `date_calcul` : un
    écart de plusieurs jours entre les deux signale une ligne rattrapée.

    Returns:
        dict de rapport — `message` est un résumé d'une ligne, repris tel
        quel par `_summarize_result` dans /health-jobs.
    """
    if politique is None:
        politique = POLITIQUE_FRAICHEUR

    if compute_composite_score is None:
        return {"status": "error", "error": "backtest_ranking.py non disponible",
                "message": "backfill impossible : backtest_ranking.py non disponible"}
    if engine is None:
        return {"status": "error", "error": "engine non connecté",
                "message": "backfill impossible : engine non connecté"}
    if debut is None or fin is None or debut > fin:
        return {"status": "error", "error": "plage invalide",
                "message": f"plage invalide : {debut} → {fin}"}
    if (fin - debut).days > BACKFILL_MAX_JOURS:
        return {"status": "error", "error": "plage trop large",
                "message": (f"plage de {(fin - debut).days} jours refusée "
                            f"(maximum {BACKFILL_MAX_JOURS})")}
    if fin >= date.today():
        # Classer aujourd'hui, c'est le travail du job quotidien ; classer
        # demain n'a pas de sens. La borne évite aussi d'écraser la ligne du
        # jour avec une version calculée à un autre moment de la journée.
        return {"status": "error", "error": "fin dans le futur",
                "message": f"fin doit être antérieure à aujourd'hui ({date.today()})"}

    try:
        existant = _ranking_existant(engine, debut, fin)
        cibles = dates_a_backfiller(debut, fin, existant.keys(), overwrite=overwrite)

        if not cibles:
            return {"status": "ok", "mode": "dry_run" if dry_run else "ecriture",
                    "nb_dates": 0, "dates": [],
                    "message": (f"aucune date à traiter entre {debut} et {fin} "
                                f"({len(existant)} déjà classée(s), overwrite={overwrite})")}

        print(f"🔁 Backfill ranking {debut} → {fin} : {len(cibles)} date(s) — "
              f"mode {'DRY-RUN' if dry_run else 'ÉCRITURE'}, politique={politique}")

        contexte = _charger_contexte(engine)
        if not contexte["seances"]:
            return {"status": "error", "error": "Aucune donnée disponible",
                    "message": "aucun prix en base"}

        rapport, nb_ecrites = [], 0
        for cible in cibles:
            scoring_date = date_de_scoring(
                contexte["seances"], cible,
                inclure_seance_du_jour=inclure_seance_du_jour,
            )
            if scoring_date is None:
                rapport.append({"date_calcul": str(cible), "statut": "ignoree",
                                "motif": "aucune séance disponible avant cette date"})
                continue

            print(f"   • {cible} — scoring sur la séance du {scoring_date}")
            records, meta = _construire_records(
                contexte, date_calcul=cible, scoring_date=scoring_date,
                top_n=top_n, politique=politique, alerting=False,
            )

            if records is None:
                rapport.append({"date_calcul": str(cible), "statut": "refusee",
                                "motif": meta.get("motif", ""),
                                "data_date": str(scoring_date)})
                continue

            ligne = {
                "date_calcul": str(cible),
                "data_date":   str(meta["scoring_date"]),
                "nb_ranked":   len(records),
                "nb_eligible": meta["nb_eligible"],
                "nb_total":    meta["nb_total"],
                "fraicheur":   meta["resume"],
                "top5":        [r["ticker"] for r in records[:5]],
            }

            # Calibration : quand la date existe déjà, on compare le recalcul
            # au stocké. Un recouvrement élevé sur des dates connues est ce
            # qui autorise à faire confiance aux dates inconnues ; un
            # recouvrement faible dit l'inverse, avant d'avoir rien écrit.
            if cible in existant:
                recalcule = [r["ticker"] for r in records]
                stocke = existant[cible]
                communs = set(recalcule) & set(stocke)
                ligne["comparaison"] = {
                    "nb_stocke":     len(stocke),
                    "nb_recalcule":  len(recalcule),
                    "nb_communs":    len(communs),
                    "rangs_identiques": sum(1 for i, t in enumerate(recalcule)
                                            if i < len(stocke) and stocke[i] == t),
                    "stocke_top5":   stocke[:5],
                }

            if dry_run:
                ligne["statut"] = "simulee"
            else:
                _persister_records(engine, cible, records)
                ligne["statut"] = "ecrite"
                nb_ecrites += 1

            rapport.append(ligne)

        traitees = [l for l in rapport if l["statut"] in ("simulee", "ecrite")]
        message = (f"{debut}→{fin} : {len(traitees)}/{len(cibles)} date(s) "
                   f"{'simulées' if dry_run else 'écrites'}"
                   + (f", {len(cibles) - len(traitees)} refusée(s)"
                      if len(traitees) < len(cibles) else ""))
        print(f"✅ Backfill terminé — {message}")

        return {"status": "ok",
                "mode": "dry_run" if dry_run else "ecriture",
                "nb_dates": len(cibles),
                "nb_ranked": nb_ecrites,
                "message": message,
                "seance_du_jour_incluse": inclure_seance_du_jour,
                "dates": rapport}

    except Exception as e:
        print(f"❌ Erreur backfill_ranking : {e}")
        return {"status": "error", "error": str(e),
                "message": f"backfill interrompu : {e}"}
