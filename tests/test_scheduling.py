# ============================================================
# tests/test_scheduling.py — Trading Brain
# ============================================================
# Verrouille les règles d'ordonnancement du pipeline nocturne.
# Régression de l'incident du 2026-09-07 : l'analyse tournait pendant que
# sync_prix écrivait encore, et se terminait en "ok" sur des données
# incomplètes (28 tickers en fin de liste sans indicateurs).
#
# Sans DB, sans réseau.
# ============================================================

from datetime import date

from scheduling import (analysis_should_follow_sync, analysis_should_skip,
                        masque_barres_closes)


# ------------------------------------------------------------
# analysis_should_skip — garde-fou anti-chevauchement
# ------------------------------------------------------------

def test_skip_quand_sync_prix_tourne():
    """LE cas de l'incident : la sync écrit encore, l'analyse doit renoncer."""
    skip, motif = analysis_should_skip({"sync_prix": {"status": "running"}})
    assert skip is True
    assert "sync_prix" in motif


def test_pas_de_skip_quand_sync_prix_terminee():
    skip, motif = analysis_should_skip({"sync_prix": {"status": "ok"}})
    assert skip is False
    assert motif == ""


def test_pas_de_skip_quand_sync_prix_en_erreur():
    """Une sync en échec ne bloque pas une analyse manuelle : c'est
    `analysis_should_follow_sync` qui décide pour le chaînage automatique."""
    skip, _ = analysis_should_skip({"sync_prix": {"status": "error"}})
    assert skip is False


def test_pas_de_skip_au_premier_demarrage():
    """job_status vide = conteneur fraîchement redémarré, aucune raison de bloquer."""
    assert analysis_should_skip({})[0] is False
    assert analysis_should_skip(None)[0] is False


def test_pas_de_skip_si_un_autre_job_tourne():
    """Seul sync_prix écrit dans actions_prix_historique."""
    etat = {"sync_metadata": {"status": "running"}, "sync_prix": {"status": "ok"}}
    assert analysis_should_skip(etat)[0] is False


def test_job_status_malforme_ne_bloque_pas():
    """Ne doit jamais lever ni bloquer sur une structure inattendue."""
    assert analysis_should_skip({"sync_prix": "running"})[0] is False
    assert analysis_should_skip({"sync_prix": None})[0] is False
    assert analysis_should_skip("pas un dict")[0] is False


# ------------------------------------------------------------
# analysis_should_follow_sync — chaînage
# ------------------------------------------------------------

def test_enchaine_apres_une_sync_reussie():
    suivre, motif = analysis_should_follow_sync({"sync_prix": {"status": "ok"}})
    assert suivre is True
    assert motif == ""


def test_n_enchaine_pas_apres_une_sync_en_erreur():
    """Mieux vaut pas d'indicateurs que des indicateurs sur prix partiels."""
    suivre, motif = analysis_should_follow_sync({"sync_prix": {"status": "error"}})
    assert suivre is False
    assert "error" in motif


def test_n_enchaine_pas_si_sync_encore_running():
    """Cas théoriquement impossible (appel séquentiel), verrouillé quand même."""
    suivre, _ = analysis_should_follow_sync({"sync_prix": {"status": "running"}})
    assert suivre is False


def test_n_enchaine_pas_sans_statut():
    assert analysis_should_follow_sync({})[0] is False
    assert analysis_should_follow_sync(None)[0] is False
    assert analysis_should_follow_sync({"sync_prix": {}})[0] is False


# ------------------------------------------------------------
# masque_barres_closes — rejet des séances en cours
# ------------------------------------------------------------

AUJOURDHUI = date(2026, 9, 7)


def test_barre_du_jour_rejetee():
    """LE cas KRX : à 01h00 UTC le marché coréen est ouvert depuis 1h."""
    assert masque_barres_closes([AUJOURDHUI], AUJOURDHUI) == [False]


def test_barres_passees_conservees():
    veille = date(2026, 9, 4)
    assert masque_barres_closes([veille], AUJOURDHUI) == [True]


def test_serie_mixte():
    """Cas réel : yfinance renvoie l'historique + la barre du jour en cours."""
    dates = [date(2026, 9, 2), date(2026, 9, 3), date(2026, 9, 4), AUJOURDHUI]
    assert masque_barres_closes(dates, AUJOURDHUI) == [True, True, True, False]


def test_barre_future_rejetee():
    """Ne devrait pas arriver, mais un fuseau mal géré côté feed le pourrait."""
    demain = date(2026, 9, 8)
    assert masque_barres_closes([demain], AUJOURDHUI) == [False]


def test_serie_vide():
    assert masque_barres_closes([], AUJOURDHUI) == []
