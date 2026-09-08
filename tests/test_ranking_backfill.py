# ============================================================
# tests/test_ranking_backfill.py — Trading Brain
# ============================================================
# Verrouille le rattrapage d'une plage de dates (roadmap #31) : les deux
# règles pures de sélection, la propriété de causalité sur laquelle tout
# repose, et les garde-fous d'écriture.
#
# Pourquoi ces tests existent : `backfill_ranking` est la seule fonction du
# projet capable de RÉÉCRIRE de l'historique de ranking. Une erreur y est
# silencieuse par nature — des lignes plausibles, datées du bon jour, mais
# fausses — et elle contaminerait ensuite toute analyse rétrospective. Les
# trois risques couverts ici, par ordre de gravité :
#
#   1. Le look-ahead. Un ranking daté du 25/08 qui connaîtrait la clôture du
#      25/08 est une ligne que la prod n'aurait jamais pu produire, et qui
#      flatte exactement les mesures qu'on veut faire dessus.
#   2. La non-causalité d'un indicateur. Le rattrapage calcule les
#      indicateurs UNE fois sur l'historique complet puis score plusieurs
#      dates passées : légitime tant que la valeur d'une ligne ne dépend
#      d'aucune ligne postérieure. Si un indicateur futur violait ça, le
#      rattrapage produirait des scores faux sans lever la moindre erreur.
#   3. L'écriture non demandée. `dry_run` est vrai par défaut ; un régression
#      sur ce point réécrirait l'historique sur une simple simulation.
#
# Aucune DB, aucun réseau : les dates sont fabriquées à la main et les accès
# base sont monkeypatchés.
# ============================================================

from datetime import date

import numpy as np
import pandas as pd
import pytest

import ranking
from ranking import (backfill_ranking, dates_a_backfiller, date_de_scoring,
                     valider_plage_backfill)


# Le trou réel : job de ranking figé du 24 au 31/08/2026 (roadmap #25).
# 23/08 et 30/08 sont des dimanches, 01/09 a bien été classé.
TROU_DEBUT = date(2026, 8, 23)
TROU_FIN = date(2026, 9, 1)
TROU_ATTENDU = [date(2026, 8, j) for j in (24, 25, 26, 27, 28, 29, 31)]


# ============================================================
# dates_a_backfiller — quelles dates recalculer
# ============================================================

def test_reproduit_exactement_le_trou_du_24_au_31_aout():
    """Le cas réel, bornes comprises : 7 dates, dimanches exclus."""
    cibles = dates_a_backfiller(TROU_DEBUT, TROU_FIN,
                                dates_existantes=[date(2026, 9, 1)])
    assert cibles == TROU_ATTENDU


def test_les_dimanches_ne_sont_jamais_produits():
    """
    Le scheduler tourne lun-sam. Un rattrapage qui inventerait des dimanches
    rendrait le calendrier de ranking_hebdo incohérent avec lui-même.
    """
    cibles = dates_a_backfiller(date(2026, 8, 23), date(2026, 8, 30))
    assert all(d.weekday() != 6 for d in cibles)
    assert date(2026, 8, 23) not in cibles
    assert date(2026, 8, 30) not in cibles


def test_une_date_deja_classee_est_ignoree_par_defaut():
    deja = [date(2026, 8, 25), date(2026, 8, 26)]
    cibles = dates_a_backfiller(date(2026, 8, 24), date(2026, 8, 27), deja)
    assert cibles == [date(2026, 8, 24), date(2026, 8, 27)]


def test_overwrite_reprend_les_dates_deja_classees():
    """Mode calibration : on recalcule du connu pour le comparer au stocké."""
    deja = [date(2026, 8, 25), date(2026, 8, 26)]
    cibles = dates_a_backfiller(date(2026, 8, 24), date(2026, 8, 27), deja,
                                overwrite=True)
    assert cibles == [date(2026, 8, j) for j in (24, 25, 26, 27)]


def test_plage_inversee_ou_vide_ne_produit_rien():
    assert dates_a_backfiller(date(2026, 8, 27), date(2026, 8, 24)) == []
    assert dates_a_backfiller(None, date(2026, 8, 24)) == []
    assert dates_a_backfiller(date(2026, 8, 23), date(2026, 8, 23)) == []  # dimanche seul


# ============================================================
# date_de_scoring — sur quelle séance on classe
# ============================================================

SEANCES = [date(2026, 8, 20), date(2026, 8, 21), date(2026, 8, 24),
           date(2026, 8, 25), date(2026, 8, 26)]


def test_sans_cible_prend_la_derniere_seance():
    """Chemin quotidien : littéralement l'ancien `max(all_dates)`."""
    assert date_de_scoring(SEANCES) == date(2026, 8, 26)


def test_pas_de_look_ahead_par_defaut():
    """
    LE test de ce lot. Un ranking daté du 25/08 est une décision prise au
    matin du 25/08 : la clôture du 25 n'existe pas encore. L'inclure
    fabriquerait une ligne qui sait ce qui va se passer, et biaiserait
    précisément les analyses de rendement postérieur au classement.
    """
    assert date_de_scoring(SEANCES, date(2026, 8, 25)) == date(2026, 8, 24)


def test_seance_du_jour_incluse_sur_demande_explicite():
    """L'ancienne convention (< sync.py v1.7) reste atteignable, jamais par défaut."""
    assert date_de_scoring(SEANCES, date(2026, 8, 25),
                           inclure_seance_du_jour=True) == date(2026, 8, 25)


def test_samedi_score_sur_la_seance_de_vendredi():
    """
    Convention observée en prod : date_calcul samedi 22/08 → data_date 21/08,
    samedi 15/08 → 14/08, samedi 05/09 → 04/09.
    """
    assert date_de_scoring(SEANCES, date(2026, 8, 22)) == date(2026, 8, 21)


def test_aucune_seance_avant_la_cible():
    assert date_de_scoring(SEANCES, date(2026, 8, 20)) is None
    assert date_de_scoring([], date(2026, 8, 25)) is None


def test_les_dates_nulles_sont_ignorees():
    assert date_de_scoring([None, date(2026, 8, 21), None],
                           date(2026, 8, 25)) == date(2026, 8, 21)


# ============================================================
# Causalité des indicateurs — l'hypothèse qui fonde le rattrapage
# ============================================================

def _serie_synthetique(n=300, graine=42):
    """Prix/volume déterministes, assez longs pour que mom_r2 (252) existe."""
    rng = np.random.default_rng(graine)
    idx = pd.bdate_range("2025-01-01", periods=n)
    prix = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n))), index=idx)
    return pd.DataFrame({
        "prix_ajuste": prix,
        "prix_cloture": prix,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        "prix_haut": prix * 1.01,
        "prix_bas": prix * 0.99,
    }, index=idx)


def test_les_indicateurs_sont_causaux():
    """
    Calculés sur l'historique complet ou sur l'historique tronqué à la date
    de scoring, les indicateurs de cette date doivent être identiques.

    C'est ce qui autorise `_charger_contexte` à ne calculer qu'une fois pour
    toute une plage de dates. Si un indicateur non causal apparaît un jour
    (une normalisation sur la série entière, un `bfill`, un centrage), ce
    test tombe — avant que le rattrapage n'écrive des lignes fausses.
    """
    from backtest_ranking import compute_all_indicators

    df = _serie_synthetique()
    coupe = df.index[280]

    complet = compute_all_indicators(df.copy())
    tronque = compute_all_indicators(df.loc[:coupe].copy())

    colonnes = ["sma_200", "sma_150", "mom_r2", "rvol", "obv_slope", "atr_14"]
    for col in colonnes:
        attendu = tronque.loc[coupe, col]
        obtenu = complet.loc[coupe, col]
        assert obtenu == pytest.approx(attendu, rel=1e-9, nan_ok=True), (
            f"{col} dépend de données postérieures à la date de scoring"
        )


# ============================================================
# valider_plage_backfill — la règle partagée endpoint / fonction
# ============================================================

AUJ = date(2026, 9, 8)


def test_plage_correcte_ne_renvoie_aucun_message():
    assert valider_plage_backfill(date(2026, 8, 24), date(2026, 8, 31), AUJ) is None


def test_message_pour_plage_inversee():
    assert "précéder" in valider_plage_backfill(date(2026, 8, 31),
                                                date(2026, 8, 24), AUJ)


def test_message_pour_plage_trop_large():
    assert "maximum" in valider_plage_backfill(date(2026, 1, 1),
                                               date(2026, 8, 31), AUJ)


def test_message_pour_fin_non_passee():
    """Classer aujourd'hui est le travail du job quotidien."""
    assert "antérieure" in valider_plage_backfill(date(2026, 9, 1), AUJ, AUJ)
    assert "antérieure" in valider_plage_backfill(date(2026, 9, 1),
                                                  date(2026, 9, 9), AUJ)


# ============================================================
# Endpoint — délégation, pas de duplication
# ============================================================

def _appeler_endpoint(**params):
    """
    Appelle la fonction de route directement (ni serveur, ni client HTTP) et
    retourne (réponse, tâches de fond programmées).
    """
    import asyncio
    from backfill_api import creer_router_backfill

    taches = []

    class FausseBackgroundTasks:
        def add_task(self, func, *args, **kwargs):
            taches.append((func, args, kwargs))

    router = creer_router_backfill(engine=object(), run_job=lambda *a, **k: None)
    route = next(r for r in router.routes if r.path == "/backfill-ranking")
    reponse = asyncio.run(route.endpoint(background_tasks=FausseBackgroundTasks(),
                                         **params))
    return reponse, taches


def test_endpoint_refuse_une_date_non_iso_sans_rien_lancer():
    reponse, taches = _appeler_endpoint(debut="24/08/2026", fin="2026-08-31")
    assert reponse["status"] == "error"
    assert taches == []


def test_endpoint_et_fonction_appliquent_la_meme_regle():
    """
    Anti-#30 : l'endpoint et `backfill_ranking` doivent refuser les mêmes
    plages avec le même message. S'ils divergent un jour, c'est ici que ça
    se voit — pas en production.
    """
    debut, fin = date(2025, 1, 1), date(2025, 12, 31)   # trop large
    reponse, taches = _appeler_endpoint(debut=str(debut), fin=str(fin))
    direct = backfill_ranking(SENTINELLE, debut, fin)

    assert reponse["status"] == "error" and direct["status"] == "error"
    assert reponse["message"] == direct["message"]
    assert taches == []


def test_endpoint_simule_par_defaut():
    """dry_run doit rester vrai sans paramètre : on ne réécrit pas par défaut."""
    reponse, taches = _appeler_endpoint(debut="2026-08-24", fin="2026-08-31")

    assert reponse["status"] == "processing"
    assert reponse["mode"] == "dry_run"
    assert len(taches) == 1
    assert taches[0][2]["dry_run"] is True


def test_endpoint_transmet_l_ecriture_demandee():
    reponse, taches = _appeler_endpoint(debut="2026-08-24", fin="2026-08-31",
                                        dry_run=False, overwrite=True)
    assert reponse["mode"] == "ecriture"
    assert taches[0][2]["dry_run"] is False
    assert taches[0][2]["overwrite"] is True


# ============================================================
# backfill_ranking — garde-fous d'entrée
# ============================================================

SENTINELLE = object()  # les garde-fous rendent la main avant tout accès DB


def test_refuse_une_plage_inversee():
    r = backfill_ranking(SENTINELLE, date(2026, 8, 31), date(2026, 8, 24))
    assert r["status"] == "error"


def test_refuse_une_plage_trop_large():
    """Une plage de plusieurs mois n'est pas un rattrapage d'incident."""
    r = backfill_ranking(SENTINELLE, date(2025, 1, 1), date(2025, 12, 31))
    assert r["status"] == "error"
    assert "maximum" in r["message"]


def test_refuse_de_classer_aujourd_hui_ou_demain():
    """Le classement du jour est le travail du job quotidien."""
    r = backfill_ranking(SENTINELLE, date.today(), date.today())
    assert r["status"] == "error"


def test_refuse_sans_engine():
    r = backfill_ranking(None, date(2026, 8, 24), date(2026, 8, 31))
    assert r["status"] == "error"


# ============================================================
# backfill_ranking — écriture, simulation, calibration
# ============================================================

@pytest.fixture
def backfill_sans_db(monkeypatch):
    """
    Neutralise les trois accès base de `backfill_ranking` et enregistre les
    écritures. Permet de tester la logique de rattrapage sans DB.
    """
    ecritures = []
    appels = []

    def faux_contexte(engine):
        return {"ticker_data": {}, "secteur_mapping": {}, "force_data": {},
                "macro_data": {}, "seances": SEANCES}

    def faux_records(contexte, date_calcul, scoring_date, top_n, politique,
                     alerting=True):
        appels.append({"date_calcul": date_calcul, "scoring_date": scoring_date,
                       "alerting": alerting})
        records = [{"ticker": f"T{i}", "zone": "US"} for i in range(3)]
        meta = {"action": "classer", "scoring_date": scoring_date,
                "nb_eligible": 42, "nb_total": 400, "resume": "ok",
                "diagnostic": {"tickers_exclus": []},
                "zones": ["US", "US", "US"]}
        return records, meta

    def fausse_ecriture(engine, date_calcul, records):
        ecritures.append(date_calcul)

    monkeypatch.setattr(ranking, "_charger_contexte", faux_contexte)
    monkeypatch.setattr(ranking, "_construire_records", faux_records)
    monkeypatch.setattr(ranking, "_persister_records", fausse_ecriture)
    monkeypatch.setattr(ranking, "_ranking_existant",
                        lambda engine, debut, fin: {date(2026, 8, 25): ["T0", "X1", "T2"]})

    return {"ecritures": ecritures, "appels": appels}


def test_dry_run_n_ecrit_rien(backfill_sans_db):
    """Défaut de la fonction : simuler. Une régression ici réécrit la prod."""
    r = backfill_ranking(SENTINELLE, date(2026, 8, 24), date(2026, 8, 26))

    assert r["status"] == "ok"
    assert r["mode"] == "dry_run"
    assert backfill_sans_db["ecritures"] == []
    assert {l["statut"] for l in r["dates"]} == {"simulee"}


def test_ecriture_explicite_persiste_chaque_date(backfill_sans_db):
    r = backfill_ranking(SENTINELLE, date(2026, 8, 24), date(2026, 8, 26),
                         dry_run=False)

    # Le 25/08 est déjà classé (cf. fixture) → ignoré sans overwrite.
    assert backfill_sans_db["ecritures"] == [date(2026, 8, 24), date(2026, 8, 26)]
    assert r["nb_ranked"] == 2
    assert r["mode"] == "ecriture"


def test_calibration_compare_au_stocke_sans_ecrire(backfill_sans_db):
    """
    overwrite + dry_run : on recalcule une date connue pour mesurer l'écart
    au stocké. C'est ce qui permet de juger un rattrapage AVANT de l'écrire.
    """
    r = backfill_ranking(SENTINELLE, date(2026, 8, 24), date(2026, 8, 26),
                         overwrite=True, dry_run=True)

    assert backfill_sans_db["ecritures"] == []
    ligne = next(l for l in r["dates"] if l["date_calcul"] == "2026-08-25")
    comp = ligne["comparaison"]
    assert comp["nb_stocke"] == 3
    assert comp["nb_communs"] == 2          # T0 et T2 communs, X1 non
    assert comp["rangs_identiques"] == 2    # T0 au rang 1, T2 au rang 3


def test_le_rattrapage_ne_pousse_aucune_alerte(backfill_sans_db):
    """
    Sept jours rattrapés ne doivent pas déclencher sept alertes Telegram
    décrivant l'état du marché d'il y a deux semaines.
    """
    backfill_ranking(SENTINELLE, date(2026, 8, 24), date(2026, 8, 26),
                     dry_run=False)
    assert all(a["alerting"] is False for a in backfill_sans_db["appels"])


def test_chaque_date_est_scoree_sur_la_seance_precedente(backfill_sans_db):
    """Pas de look-ahead, sur le chemin réel de la fonction cette fois."""
    backfill_ranking(SENTINELLE, date(2026, 8, 24), date(2026, 8, 26),
                     overwrite=True)
    scoring = {a["date_calcul"]: a["scoring_date"] for a in backfill_sans_db["appels"]}
    assert scoring[date(2026, 8, 25)] == date(2026, 8, 24)
    assert scoring[date(2026, 8, 26)] == date(2026, 8, 25)


def test_aucune_date_a_traiter_est_un_succes_pas_une_erreur(backfill_sans_db):
    """Rejouer un rattrapage déjà fait ne doit pas ressembler à une panne."""
    r = backfill_ranking(SENTINELLE, date(2026, 8, 25), date(2026, 8, 25))
    assert r["status"] == "ok"
    assert r["nb_dates"] == 0
    assert backfill_sans_db["ecritures"] == []
