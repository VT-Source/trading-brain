# ============================================================
# ranking.py — Trading Brain — v1.0
# ============================================================
# Calcul et persistance du ranking momentum journalier.
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
from datetime import date

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

RANKING_VERSION = "1.0"

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

        # 1. Charger les tickers et données
        all_tickers = load_all_tickers()
        ticker_data = load_all_price_data(all_tickers)

        # 2. Calculer les indicateurs pour chaque ticker
        for ticker in list(ticker_data.keys()):
            ticker_data[ticker] = compute_all_indicators(ticker_data[ticker])

        # 3. Charger contexte sectoriel et macro
        secteur_mapping = load_secteur_mapping()
        force_data = load_all_secteur_force()
        macro_data = load_macro_data()

        # 4. Trouver le dernier jour de trading disponible
        all_dates = set()
        for df in ticker_data.values():
            all_dates.update(df.index)
        if not all_dates:
            return {"error": "Aucune donnée disponible"}

        latest_date = max(all_dates)

        # 4bis. Garde-fou de fraîcheur par place de cotation (roadmap #28)
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
            for t, df in ticker_data.items() if not df.empty
        }
        diagnostic = diagnostic_fraicheur(
            dernieres_dates,
            calendrier={d.date() for d in all_dates},
            zones={t: get_ticker_zone(t, secteur_mapping) for t in ticker_data},
        )
        print(f"🕐 Fraîcheur : {diagnostic['resume']}")

        # Le branchement des 4 politiques est une règle PURE et testée
        # → appliquer_politique_fraicheur (haut de ce module). Elle décide,
        # cette fonction exécute : alerte, pop des tickers, écriture.
        decision = appliquer_politique_fraicheur(
            diagnostic, politique, latest_date.date()
        )

        if decision["alerter"]:
            # (e) Alerte push — l'alerte (b) sur tickers en retard a un seuil
            # de 3 jours, sous lequel une séance manquante passait.
            alert_fraicheur_places(diagnostic)

        if decision["action"] == "refuser":
            print("⏭️  Ranking non calculé — places arrêtées à des séances différentes")
            return {"status": "skipped", "message": diagnostic["resume"],
                    "fraicheur": diagnostic["resume"],
                    "data_date": str(latest_date.date())}

        if decision["ecarter"]:
            for t in decision["tickers_exclus"]:
                ticker_data.pop(t, None)
            print(f"✂️  {len(decision['tickers_exclus'])} tickers écartés — "
                  f"places : {', '.join(diagnostic['places_douteuses'])}")
            if not ticker_data:
                return {"error": "Aucune place exploitable"}

        if decision["aligne"]:
            # Reculée jusqu'à la dernière séance commune aux places saines.
            print(f"↩️  Alignement : classement au {decision['date_classement']} "
                  f"(dernière séance commune) au lieu du {latest_date.date()}")
            latest_date = pd.Timestamp(decision["date_classement"])

        # 5. Calculer le ranking
        ranking = compute_composite_score(
            ticker_data, latest_date, secteur_mapping, force_data, sma_period=200
        )

        # 6. Enrichir et préparer les records
        macro_regime = get_macro_regime(macro_data, latest_date)
        today = date.today()

        records = []
        for r in ranking[:top_n]:
            ticker = r["ticker"]
            zone = get_ticker_zone(ticker, secteur_mapping)
            secteur = secteur_mapping.get(ticker, {}).get("secteur", "—")
            k = compute_adaptive_k(r["atr_14"], r["prix"]) if r["atr_14"] > 0 else 3.0

            records.append({
                "date_calcul":  today,
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
                "nb_total":     len(ticker_data),
                "data_date":    latest_date.date(),
            })

        # 7. Supprimer l'ancien ranking du jour et insérer
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM ranking_hebdo WHERE date_calcul = :d"), {"d": today})
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

        print(f"✅ Ranking sauvegardé : {len(records)} tickers, date données {latest_date.date()}")

        # (c) Alerte composition — filet en aval de l'audit post-sync.
        # Le 16/05, 59 tickers EU non synchronisés ont produit un ranking
        # 100 % US sans qu'aucune erreur ne soit levée.
        alert_ranking_composition([r["zone"] for r in records],
                                  nb_eligible=len(ranking),
                                  data_date=latest_date.date())

        return {"status": "ok", "nb_ranked": len(records),
                "data_date": str(latest_date.date()),
                "fraicheur": diagnostic["resume"],
                "nb_exclus": len(diagnostic["tickers_exclus"])
                             if politique in ("exclure", "aligner")
                             else 0}

    except Exception as e:
        print(f"❌ Erreur compute_and_store_ranking : {e}")
        return {"error": str(e)}
