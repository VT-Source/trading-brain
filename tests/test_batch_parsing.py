# ============================================================
# tests/test_batch_parsing.py — Trading Brain
# ============================================================
# Verrouille le parsing des réponses batch avis IA (ai_opinion.py).
#
# Chaque bug vécu = un test de régression. Couverture :
#   R3  (17/07) — regex CLASSEMENT ancrée : une mention en prose dans
#                 LECTURE DU LOT ne doit plus être capturée à la place
#                 de la vraie ligne CLASSEMENT.
#   11/07       — matching des entêtes KRX en 3 passes (le modèle
#                 abandonne les zéros de tête et le suffixe .KS).
#   R2  (17/07) — distinction 0 avis parsé (→ ERREUR + re-soumission)
#                 vs parsing partiel (→ TERMINÉ + note « partiel x/y »).
#   Fallback    — sans ligne CLASSEMENT exploitable, l'ordre des
#                 sections fait foi.
#
# Ces fonctions sont PURES (pas de DB, pas d'appel Anthropic) : c'est
# précisément pour ça qu'on commence le filet de sécurité ici.
# ============================================================

from ai_opinion import (
    _extract_batch_opinions,
    _extract_conviction,
    _extract_risque_evenementiel,
    _extract_resume,
)


# ============================================================
# FIXTURES — réponses batch réalistes
# ============================================================

# Piège R3 : le mot « classement » apparaît en prose AVANT la vraie ligne,
# et il est suivi d'un ordre DIFFÉRENT (AMD, NVDA, MU au lieu de MU, NVDA,
# AMD). Avant le fix, re.search sans ancre capturait cette ligne de prose
# → classement_ia pollué, et le fallback ne se déclenchait jamais puisque
# le dict n'était pas vide.
REPONSE_PIEGE_PROSE = """LECTURE DU LOT : trois semi-conducteurs sur le lot.
Le classement penche vers AMD, NVDA, MU si l'on ne regarde que le momentum
brut, mais la qualité de tendance rebat les cartes.

CLASSEMENT : MU, NVDA, AMD

=== TICKER : MU ===
CONVICTION : FORT
RISQUE ÉVÉNEMENTIEL : NON
RÉSUMÉ : Cycle HBM intact, tendance propre.

=== TICKER : NVDA ===
CONVICTION : MODÉRÉ
RISQUE ÉVÉNEMENTIEL : OUI
RÉSUMÉ : Publication de résultats dans 8 jours.

=== TICKER : AMD ===
CONVICTION : FAIBLE
RISQUE ÉVÉNEMENTIEL : EARNINGS NON TROUVÉ
RÉSUMÉ : Tendance fragile sous la SMA 200.
"""

# Variantes d'entêtes KRX réellement renvoyées par le modèle (incident
# du 06/07 : 105560.KS et 086790.KS perdus).
REPONSE_KRX = """LECTURE DU LOT : trois valeurs coréennes.

CLASSEMENT : 105560.KS, 086790.KS, 005930.KS

=== TICKER : 105560.KS ===
CONVICTION : FORT
RISQUE ÉVÉNEMENTIEL : NON
RÉSUMÉ : KB Financial bien orienté.

=== TICKER : 86790 Hana Financial Group ===
CONVICTION : MODÉRÉ
RISQUE ÉVÉNEMENTIEL : NON
RÉSUMÉ : Hana suit le secteur.

=== TICKER : Samsung Electronics 005930.KS ===
CONVICTION : FORT
RISQUE ÉVÉNEMENTIEL : OUI
RÉSUMÉ : Résultats trimestriels imminents.
"""


# ============================================================
# R3 — ANCRAGE DE LA REGEX CLASSEMENT
# ============================================================

def test_r3_prose_ne_pollue_pas_le_classement():
    """La ligne CLASSEMENT réelle prime sur toute mention en prose."""
    out = _extract_batch_opinions(REPONSE_PIEGE_PROSE, ["MU", "NVDA", "AMD"])

    assert out["tickers"]["MU"]["classement_ia"] == 1
    assert out["tickers"]["NVDA"]["classement_ia"] == 2
    assert out["tickers"]["AMD"]["classement_ia"] == 3

    # Avant le fix : la prose donnait AMD=1 — c'est exactement ce qu'on interdit.
    assert out["tickers"]["AMD"]["classement_ia"] != 1


def test_r3_classement_brut_est_la_vraie_ligne():
    out = _extract_batch_opinions(REPONSE_PIEGE_PROSE, ["MU", "NVDA", "AMD"])
    assert out["classement_brut"].startswith("MU")
    assert "penche" not in out["classement_brut"]


def test_lecture_du_lot_capturee():
    out = _extract_batch_opinions(REPONSE_PIEGE_PROSE, ["MU", "NVDA", "AMD"])
    assert out["lecture_lot"] is not None
    assert "semi-conducteurs" in out["lecture_lot"]


def test_classement_ignore_les_tickers_hors_lot():
    """Un ticker cité dans la ligne CLASSEMENT mais absent du lot est ignoré."""
    reponse = REPONSE_PIEGE_PROSE.replace(
        "CLASSEMENT : MU, NVDA, AMD",
        "CLASSEMENT : MU, INTC, NVDA, AMD",
    )
    out = _extract_batch_opinions(reponse, ["MU", "NVDA", "AMD"])
    assert out["tickers"]["MU"]["classement_ia"] == 1
    assert out["tickers"]["NVDA"]["classement_ia"] == 2   # et non 3
    assert "INTC" not in out["tickers"]


def test_classement_variante_markdown():
    """Entête markdown : **CLASSEMENT** : ... reste reconnu."""
    reponse = REPONSE_PIEGE_PROSE.replace(
        "CLASSEMENT : MU, NVDA, AMD",
        "**CLASSEMENT** : MU, NVDA, AMD",
    )
    out = _extract_batch_opinions(reponse, ["MU", "NVDA", "AMD"])
    assert out["tickers"]["MU"]["classement_ia"] == 1
    assert out["tickers"]["AMD"]["classement_ia"] == 3


# ============================================================
# FALLBACK — pas de ligne CLASSEMENT exploitable
# ============================================================

def test_fallback_ordre_des_sections():
    """Sans ligne CLASSEMENT, l'ordre d'apparition des sections fait foi."""
    reponse = """LECTURE DU LOT : rien de particulier.

=== TICKER : NVDA ===
CONVICTION : FORT
RÉSUMÉ : ok.

=== TICKER : MU ===
CONVICTION : MODÉRÉ
RÉSUMÉ : ok.
"""
    out = _extract_batch_opinions(reponse, ["NVDA", "MU"])
    assert out["tickers"]["NVDA"]["classement_ia"] == 1
    assert out["tickers"]["MU"]["classement_ia"] == 2


# ============================================================
# MATCHING KRX — 3 PASSES (incident du 06/07)
# ============================================================

def test_krx_trois_passes():
    attendus = ["105560.KS", "086790.KS", "005930.KS"]
    out = _extract_batch_opinions(REPONSE_KRX, attendus)

    # Aucun ticker perdu : c'est le cœur de l'incident du 06/07.
    assert set(out["tickers"].keys()) == set(attendus)

    # Passe 1 — token exact
    assert out["tickers"]["105560.KS"]["conviction"] == "FORT"
    # Passe 3 — zéro de tête et suffixe .KS abandonnés par le modèle
    assert out["tickers"]["086790.KS"]["conviction"] == "MODÉRÉ"
    # Passe 2 — le ticker attendu apparaît tel quel dans un entête bavard
    assert out["tickers"]["005930.KS"]["conviction"] == "FORT"


def test_krx_classement_conserve():
    out = _extract_batch_opinions(REPONSE_KRX, ["105560.KS", "086790.KS", "005930.KS"])
    assert out["tickers"]["105560.KS"]["classement_ia"] == 1
    assert out["tickers"]["086790.KS"]["classement_ia"] == 2
    assert out["tickers"]["005930.KS"]["classement_ia"] == 3


def test_entete_non_reconnue_est_ignoree_sans_crash():
    reponse = """CLASSEMENT : MU

=== TICKER : Société Inconnue SA ===
CONVICTION : FORT
RÉSUMÉ : hors lot.

=== TICKER : MU ===
CONVICTION : FORT
RÉSUMÉ : ok.
"""
    out = _extract_batch_opinions(reponse, ["MU"])
    assert list(out["tickers"].keys()) == ["MU"]


# ============================================================
# R2 — 0 PARSÉ vs PARTIEL
# ============================================================

def test_r2_zero_parse_declenche_la_branche_erreur():
    """
    Réponse hors format : aucune section exploitable.
    poll_batch_opinions doit alors marquer ERREUR + re-soumettre.
    """
    reponse = "Je n'ai pas pu accéder aux données de marché cette semaine."
    out = _extract_batch_opinions(reponse, ["MU", "NVDA"])
    assert out["tickers"] == {}


def test_r2_parsing_partiel_conserve_les_avis_valides():
    """
    3 sections sur 5 : les 3 obtenus doivent être persistables, et les
    2 manquants identifiables (note « partiel x/y » côté appelant).
    """
    attendus = ["MU", "NVDA", "AMD", "AVGO", "KLAC"]
    out = _extract_batch_opinions(REPONSE_PIEGE_PROSE, attendus)

    assert set(out["tickers"].keys()) == {"MU", "NVDA", "AMD"}
    manquants = [t for t in attendus if t not in out["tickers"]]
    assert manquants == ["AVGO", "KLAC"]


# ============================================================
# EXTRACTIONS UNITAIRES
# ============================================================

def test_conviction_variantes_ranking():
    assert _extract_conviction("CONVICTION : FORT") == "FORT"
    assert _extract_conviction("**CONVICTION : MODÉRÉ**") == "MODÉRÉ"
    assert _extract_conviction("## CONVICTION\n\nFAIBLE") == "FAIBLE"
    assert _extract_conviction("CONVICTION 🟡 MODERE") == "MODÉRÉ"   # sans accent


def test_conviction_variantes_position():
    assert _extract_conviction("CONVICTION : VENDRE") == "VENDRE"
    assert _extract_conviction("**CONVICTION** — GARDER") == "GARDER"
    assert _extract_conviction("CONVICTION : RENFORCER") == "RENFORCER"


def test_conviction_fallback():
    """Aucun pattern → MODÉRÉ (jamais d'exception, jamais de None)."""
    assert _extract_conviction("Texte sans le mot attendu.") == "MODÉRÉ"


def test_risque_evenementiel():
    assert _extract_risque_evenementiel("RISQUE ÉVÉNEMENTIEL : OUI") is True
    assert _extract_risque_evenementiel("RISQUE ÉVÉNEMENTIEL : NON") is False
    assert _extract_risque_evenementiel(
        "RISQUE ÉVÉNEMENTIEL : EARNINGS NON TROUVÉ") is None
    assert _extract_risque_evenementiel("Aucune mention.") is None


def test_resume_extrait_et_borne():
    body = "CONVICTION : FORT\nRÉSUMÉ : Momentum intact.\nANALYSE : bla bla."
    assert _extract_resume(body) == "Momentum intact."
    assert _extract_resume("Pas de marqueur de résumé.") == ""


def test_risque_evenementiel_par_section():
    out = _extract_batch_opinions(REPONSE_PIEGE_PROSE, ["MU", "NVDA", "AMD"])
    assert out["tickers"]["MU"]["risque_evenementiel"] is False
    assert out["tickers"]["NVDA"]["risque_evenementiel"] is True
    assert out["tickers"]["AMD"]["risque_evenementiel"] is None
