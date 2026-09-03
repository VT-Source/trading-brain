# ============================================================
# tests/test_alerting.py — Trading Brain
# ============================================================
# Verrouille le module d'alerting (alerting.py, roadmap #19).
#
# Deux exigences non négociables :
#   1. L'alerting ne doit JAMAIS casser l'appelant. Sans configuration
#      Telegram, sans réseau, avec une entrée malformée : on retourne un
#      dict, on ne lève pas.
#   2. Les règles de détection sont pures et vérifiables sans réseau.
#
# Aucun test n'envoie de message : les tests tournent sans TELEGRAM_*
# en environnement, donc send_alert court-circuite sur "not_configured".
# ============================================================

import time

import alerting
from alerting import (
    check_ranking_composition,
    check_stale_tickers,
    send_alert,
    alert_job_failure,
    alert_tickers_stale,
    alert_batch_avis_bloque,
    alert_ranking_composition,
    alerting_status,
    _should_send,
)


# ============================================================
# COMPOSITION DU RANKING (incident du 16/05)
# ============================================================

def test_ranking_equilibre_ne_declenche_rien():
    zones = ["US"] * 12 + ["EU"] * 6 + ["KR"] * 2
    assert check_ranking_composition(zones, nb_eligible=180) == []


def test_ranking_100_pct_us_est_signale():
    """Le scénario exact du 16/05 : 59 tickers EU non synchronisés."""
    anomalies = check_ranking_composition(["US"] * 20, nb_eligible=120)
    assert len(anomalies) == 1
    assert "Concentration zone US" in anomalies[0]


def test_ranking_vide():
    anomalies = check_ranking_composition([], nb_eligible=0)
    assert anomalies == ["Ranking vide : 0 ticker classé."]


def test_ranking_trop_court():
    anomalies = check_ranking_composition(["US"] * 3 + ["EU"] * 2, nb_eligible=90)
    assert any("Seulement 5 ticker(s) classé(s)" in a for a in anomalies)


def test_peu_de_tickers_eligibles():
    zones = ["US"] * 10 + ["EU"] * 8
    anomalies = check_ranking_composition(zones, nb_eligible=12)
    assert any("éligible(s)" in a for a in anomalies)


def test_seuil_de_concentration_est_strict():
    """90 % pile ne déclenche pas — c'est au-delà que ça alerte."""
    zones = ["US"] * 9 + ["EU"]          # 90.0 %
    assert check_ranking_composition(zones, nb_eligible=120) == []
    zones = ["US"] * 19 + ["EU"]         # 95 %
    assert check_ranking_composition(zones, nb_eligible=120) != []


def test_zone_none_ne_crashe_pas():
    zones = ["US"] * 10 + [None] * 8
    check_ranking_composition(zones, nb_eligible=100)   # ne doit pas lever


def test_nb_eligible_none_tolere():
    """L'appelant peut ne pas connaître nb_eligible — pas de crash."""
    assert check_ranking_composition(["US"] * 10 + ["EU"] * 8, nb_eligible=None) == []


# ============================================================
# TICKERS EN RETARD
# ============================================================

def test_stale_sous_le_seuil():
    assert check_stale_tickers([("ROG.SW", "2026-05-10")]) is False


def test_stale_au_dessus_du_seuil():
    stale = [(f"T{i}.PA", "2026-05-10") for i in range(59)]
    assert check_stale_tickers(stale) is True


def test_stale_liste_vide():
    assert check_stale_tickers([]) is False
    assert check_stale_tickers(None) is False


# ============================================================
# ANTI-SPAM
# ============================================================

def test_cooldown_bloque_le_second_envoi():
    cle = "test:cooldown"
    alerting._LAST_SENT.pop(cle, None)
    assert _should_send(cle, cooldown_min=60) is True

    alerting._LAST_SENT[cle] = time.time()
    assert _should_send(cle, cooldown_min=60) is False

    # une fois la fenêtre écoulée, on ré-alerte
    alerting._LAST_SENT[cle] = time.time() - 61 * 60
    assert _should_send(cle, cooldown_min=60) is True
    alerting._LAST_SENT.pop(cle, None)


def test_sans_cle_pas_de_deduplication():
    assert _should_send(None, cooldown_min=600) is True


# ============================================================
# ROBUSTESSE — ne jamais casser l'appelant
# ============================================================

def test_send_alert_non_configure_retourne_un_dict():
    res = send_alert("Titre", "corps")
    assert isinstance(res, dict)
    assert res["sent"] is False


def test_helpers_ne_levent_jamais():
    """Chaque point d'appel doit être sûr, y compris sur entrée dégradée."""
    assert alert_job_failure("sync_prix", "boom", 12.3)["sent"] is False
    assert alert_job_failure("sync_prix", None)["sent"] is False
    assert alert_tickers_stale([], "2026-06-12")["sent"] is False
    assert alert_batch_avis_bloque({})["sent"] is False
    assert alert_batch_avis_bloque(None)["sent"] is False
    assert alert_ranking_composition(["US"] * 18, 120)["sent"] is False


def test_batch_sain_ne_declenche_pas():
    sain = {"status": "ok", "pending": 1, "stale": 0, "alerte": False,
            "seuil_h": 26, "jobs": [{"batch_id": "msgbatch_x", "stale": False}]}
    assert alert_batch_avis_bloque(sain)["reason"] == "pas_d_alerte"


def test_batch_bloque_tente_l_envoi():
    bloque = {"status": "ok", "pending": 1, "stale": 1, "alerte": True,
              "seuil_h": 26,
              "jobs": [{"batch_id": "msgbatch_01RnbNS4A2Vu8FoTUKjbrgBy",
                        "semaine": "2026-08-31", "age_h": 31.4, "stale": True}]}
    res = alert_batch_avis_bloque(bloque)
    # non configuré en test → l'envoi est court-circuité, pas ignoré
    assert res["reason"] == "not_configured"


def test_alerting_status_ne_fuit_pas_le_token(monkeypatch):
    """
    /health-jobs est un endpoint public (l'auth API est le point R6, pas
    encore fait) : le statut ne doit exposer que des booléens.
    """
    monkeypatch.setattr(alerting, "TELEGRAM_BOT_TOKEN", "123456:SECRET-TOKEN")
    monkeypatch.setattr(alerting, "TELEGRAM_CHAT_ID", "987654321")

    st = alerting_status()
    rendu = str(st)
    assert "SECRET-TOKEN" not in rendu
    assert "987654321" not in rendu
    assert st["token_set"] is True and st["chat_id_set"] is True
    assert isinstance(st["configured"], bool)
    assert st["version"] == alerting.ALERTING_VERSION
