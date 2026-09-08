# ============================================================
# tests/test_ranking_politique.py — Trading Brain
# ============================================================
# Verrouille `ranking.appliquer_politique_fraicheur` — la traduction d'un
# diagnostic de fraîcheur en décision de classement.
#
# Pourquoi ces tests existent : jusqu'au 08/09 ce branchement vivait à
# l'intérieur de `compute_and_store_ranking`, une fonction de 170 lignes qui
# touche la DB. Il décidait pourtant du sort d'un ranking entier — donc
# d'entrées en position — sans qu'une seule ligne de test puisse l'atteindre.
# L'extraction (roadmap #5) rend les 4 politiques vérifiables AVANT que
# FRAICHEUR_POLITIQUE ne quitte le mode `observer` en prod.
#
# Aucune DB, aucun réseau : les diagnostics sont fabriqués à la main, au
# format exact de freshness.diagnostic_fraicheur.
# ============================================================

from datetime import date

from ranking import appliquer_politique_fraicheur

J04 = date(2026, 9, 4)   # vendredi — dernière clôture US (Labor Day le 07)
J07 = date(2026, 9, 7)   # lundi    — clôture KR / EU partielle


# ============================================================
# HELPERS
# ============================================================

def diag_ok(derniere_us=J07, derniere_kr=J07):
    """Diagnostic sain : toutes les places exploitables."""
    return {
        "date_reference": J07,
        "places": {
            "NYSE": {"verdict": "complete", "derniere_date": derniere_us,
                     "tickers": ["AAPL", "MSFT"]},
            "KRX":  {"verdict": "complete", "derniere_date": derniere_kr,
                     "tickers": ["005930.KS"]},
        },
        "places_saines": ["KRX", "NYSE"],
        "places_douteuses": [],
        "tickers_exclus": [],
        "zones_touchees": [],
        "ok": True,
        "resume": "2 places, toutes exploitables au 2026-09-07",
    }


def diag_douteux():
    """Le cas réel du 07/09 : BME 1/5 et SIX 3/5, couverture partielle."""
    return {
        "date_reference": J07,
        "places": {
            "NYSE": {"verdict": "complete", "derniere_date": J07,
                     "tickers": ["AAPL", "MSFT"]},
            "BME":  {"verdict": "partielle", "derniere_date": J04,
                     "tickers": ["ITX.MC", "SAN.MC"]},
        },
        "places_saines": ["NYSE"],
        "places_douteuses": ["BME"],
        "tickers_exclus": ["ITX.MC", "SAN.MC"],
        "zones_touchees": ["EU"],
        "ok": False,
        "resume": "1 place(s) non exploitable(s) au 2026-09-07 : BME 1/5 (partielle)",
    }


# ============================================================
# OBSERVER — diagnostique et alerte, mais ne change RIEN
# ============================================================

def test_observer_alerte_mais_ne_change_rien():
    d = appliquer_politique_fraicheur(diag_douteux(), "observer", J07)
    assert d["action"] == "classer"
    assert d["alerter"] is True          # déclencheur (e)
    assert d["ecarter"] is False
    assert d["tickers_exclus"] == []     # le ranking est INCHANGÉ
    assert d["date_classement"] == J07
    assert d["aligne"] is False


def test_observer_diagnostic_sain_nalerte_pas():
    d = appliquer_politique_fraicheur(diag_ok(), "observer", J07)
    assert d["action"] == "classer"
    assert d["alerter"] is False
    assert d["tickers_exclus"] == []


# ============================================================
# REFUSER — pas de ranking du tout
# ============================================================

def test_refuser_bloque_le_ranking():
    d = appliquer_politique_fraicheur(diag_douteux(), "refuser", J07)
    assert d["action"] == "refuser"
    assert d["alerter"] is True
    assert d["date_classement"] == J07


def test_refuser_laisse_passer_un_diagnostic_sain():
    """`refuser` ne bloque que sur diagnostic douteux — jamais par principe."""
    d = appliquer_politique_fraicheur(diag_ok(), "refuser", J07)
    assert d["action"] == "classer"
    assert d["alerter"] is False


# ============================================================
# EXCLURE — écarte les places douteuses, ne recule pas la date
# ============================================================

def test_exclure_ecarte_les_tickers_de_la_place_douteuse():
    d = appliquer_politique_fraicheur(diag_douteux(), "exclure", J07)
    assert d["action"] == "classer"
    assert d["ecarter"] is True
    assert d["tickers_exclus"] == ["ITX.MC", "SAN.MC"]
    assert d["date_classement"] == J07     # exclure ne recule JAMAIS la date
    assert d["aligne"] is False


def test_exclure_nexclut_rien_si_le_diagnostic_est_sain():
    d = appliquer_politique_fraicheur(diag_ok(), "exclure", J07)
    assert d["ecarter"] is False
    assert d["tickers_exclus"] == []


# ============================================================
# ALIGNER — écarte PUIS recule à la dernière séance commune
# ============================================================

def test_aligner_ecarte_et_recule():
    d = appliquer_politique_fraicheur(diag_douteux(), "aligner", J07)
    assert d["ecarter"] is True
    assert d["tickers_exclus"] == ["ITX.MC", "SAN.MC"]
    # BME est écartée, donc ignorée par derniere_seance_commune : seule NYSE
    # reste saine et elle est au 07/09 → aucun recul nécessaire.
    assert d["date_classement"] == J07
    assert d["aligne"] is False


def test_aligner_recule_meme_sur_diagnostic_sain():
    """
    Le cœur du mode `aligner`, et la raison pour laquelle `exclure` ne suffit
    pas : le 07/09 les US sont fermés (Labor Day) — diagnostic parfaitement
    sain — mais classer au 07/09 compare du 04/09 américain à du 07/09 coréen.
    """
    d = appliquer_politique_fraicheur(diag_ok(derniere_us=J04), "aligner", J07)
    assert d["alerter"] is False           # rien d'anormal à signaler
    assert d["ecarter"] is False
    assert d["aligne"] is True
    assert d["date_classement"] == J04     # recul à la séance commune


def test_aligner_ne_recule_pas_quand_tout_est_a_la_meme_date():
    d = appliquer_politique_fraicheur(diag_ok(), "aligner", J07)
    assert d["aligne"] is False
    assert d["date_classement"] == J07


def test_aligner_ne_avance_jamais_la_date():
    """Garde-fou : une séance commune postérieure ne doit pas tirer en avant."""
    d = appliquer_politique_fraicheur(diag_ok(), "aligner", J04)
    assert d["date_classement"] == J04
    assert d["aligne"] is False


# ============================================================
# CONTRAT — les autres politiques ne touchent jamais à la date
# ============================================================

def test_seul_aligner_touche_la_date():
    for politique in ("observer", "exclure", "refuser"):
        d = appliquer_politique_fraicheur(diag_ok(derniere_us=J04), politique, J07)
        assert d["date_classement"] == J07, politique
        assert d["aligne"] is False, politique


def test_diagnostic_vide_ne_leve_pas():
    """freshness retourne ok=True et des listes vides quand il n'a rien à évaluer."""
    vide = {"places": {}, "places_saines": [], "places_douteuses": [],
            "tickers_exclus": [], "ok": True, "resume": "aucune donnée à évaluer"}
    for politique in ("observer", "exclure", "aligner", "refuser"):
        d = appliquer_politique_fraicheur(vide, politique, J07)
        assert d["action"] == "classer", politique
        assert d["date_classement"] == J07, politique


def test_la_decision_ne_mute_pas_le_diagnostic():
    """La fonction est pure : l'appelant réutilise `diagnostic` après coup
    (resume, places_douteuses) pour son alerte et sa valeur de retour."""
    diag = diag_douteux()
    avant = dict(diag)
    appliquer_politique_fraicheur(diag, "aligner", J07)
    assert diag == avant
    assert diag["tickers_exclus"] == ["ITX.MC", "SAN.MC"]
