# ============================================================
# tests/test_zones_macro.py — Trading Brain
# ============================================================
# Verrouille le mapping ticker → zone et la lecture du régime macro.
#
# Bug de référence (15/05) : un ticker sans suffixe Yahoo était classé
# selon le PAYS retourné par Yahoo Finance et non selon sa place de
# cotation. Les 13 ADR US (NXPI, ACN, APTV, LIN…) partaient donc en zone
# EU — mauvais ETF sectoriel, mauvais indice macro, mauvaise décision.
#
# ⚠️ PRÉREQUIS : ces tests portent sur `zone_priority_for(ticker, pays)`,
# extraite de `load_secteur_mapping()` pour être testable sans base de
# données. Voir la modification proposée dans backtest_ranking.py.
# ============================================================

import pandas as pd

from backtest_ranking import (
    zone_priority_for,
    get_ticker_zone,
    get_macro_regime,
    PAYS_EU,
)

JOUR = pd.Timestamp("2026-06-15")


# ============================================================
# RÈGLE DE COTATION — le suffixe Yahoo prime sur le pays du siège
# ============================================================

def test_us_listed_sans_suffixe():
    assert zone_priority_for("AAPL", "United States") == ["US", "EU"]


def test_adr_us_malgre_siege_europeen():
    """Régression 15/05 — les ADR restent en zone US."""
    for ticker, pays in [("NXPI", "Netherlands"), ("ACN", "Ireland"),
                         ("APTV", "Switzerland"), ("LIN", "United Kingdom"),
                         ("ASML", "United States")]:
        assert zone_priority_for(ticker, pays) == ["US", "EU"], ticker


def test_actions_europeennes_avec_suffixe():
    assert zone_priority_for("ASML.AS", "Netherlands") == ["EU", "US"]
    assert zone_priority_for("AI.PA", "France") == ["EU", "US"]
    assert zone_priority_for("ABBN.SW", "Switzerland") == ["EU", "US"]


def test_belgique_rangee_en_eu():
    """La zone BE est dormante : les actions belges sont en zone EU."""
    assert zone_priority_for("UCB.BR", "Belgium") == ["EU", "US"]


def test_coree_sans_repli():
    """
    KRX → ['KR'] strict. Pas de fallback US/EU : un titre coréen jugé
    sur ^GSPC produirait un signal macro faux.
    """
    assert zone_priority_for("000660.KS", "South Korea") == ["KR"]
    assert zone_priority_for("005930.KS", "South Korea") == ["KR"]
    assert zone_priority_for("123456.KQ", "South Korea") == ["KR"]


def test_pays_vide_ou_none_ne_crashe_pas():
    assert zone_priority_for("AAPL", None) == ["US", "EU"]
    assert zone_priority_for("ASML.AS", None) == ["US", "EU"]   # pas de pays → défaut
    assert zone_priority_for("ASML.AS", "") == ["US", "EU"]


def test_hors_univers_couvert_bascule_en_us():
    """
    Comportement ACTUEL documenté, pas un idéal : un suffixe hors PAYS_EU
    et hors KRX (Tokyo, Toronto…) part en ['US', 'EU']. À revoir le jour
    où une zone JP/CA sera ajoutée.
    """
    assert zone_priority_for("7203.T", "Japan") == ["US", "EU"]


def test_pays_eu_couvre_les_places_de_l_univers():
    for pays in ["Belgium", "France", "Germany", "Netherlands",
                 "Switzerland", "Sweden", "Denmark", "Italy", "Spain"]:
        assert pays in PAYS_EU


# ============================================================
# get_ticker_zone
# ============================================================

def test_get_ticker_zone_prend_la_priorite_haute():
    mapping = {"ASML.AS": {"secteur": "Technology", "zone_priority": ["EU", "US"]}}
    assert get_ticker_zone("ASML.AS", mapping) == "EU"


def test_get_ticker_zone_ticker_inconnu():
    """Ticker absent du mapping (secteur NULL) → US par défaut."""
    assert get_ticker_zone("INCONNU", {}) == "US"


def test_coherence_zone_priority_et_get_ticker_zone():
    tickers = {"AAPL": "United States", "ASML.AS": "Netherlands",
               "000660.KS": "South Korea"}
    mapping = {t: {"secteur": "Technology", "zone_priority": zone_priority_for(t, p)}
               for t, p in tickers.items()}
    assert get_ticker_zone("AAPL", mapping) == "US"
    assert get_ticker_zone("ASML.AS", mapping) == "EU"
    assert get_ticker_zone("000660.KS", mapping) == "KR"


# ============================================================
# RÉGIME MACRO
# ============================================================

def test_macro_regime_lit_la_derniere_valeur_connue():
    idx = pd.to_datetime(["2026-06-10", "2026-06-12", "2026-06-20"])
    macro = {"US": pd.DataFrame({"macro_bull": [True, False, True]}, index=idx)}
    # au 15/06 la dernière valeur connue est celle du 12/06
    assert get_macro_regime(macro, JOUR) == {"US": False}


def test_macro_regime_sans_donnee_anterieure_laisse_passer():
    idx = pd.to_datetime(["2026-07-01"])
    macro = {"EU": pd.DataFrame({"macro_bull": [False]}, index=idx)}
    assert get_macro_regime(macro, JOUR) == {"EU": True}


def test_macro_regime_trois_zones():
    idx = pd.to_datetime(["2026-06-10"])
    macro = {
        "US": pd.DataFrame({"macro_bull": [True]}, index=idx),
        "EU": pd.DataFrame({"macro_bull": [False]}, index=idx),
        "KR": pd.DataFrame({"macro_bull": [True]}, index=idx),
    }
    assert get_macro_regime(macro, JOUR) == {"US": True, "EU": False, "KR": True}


def test_macro_regime_retourne_des_bool_python():
    """np.bool_ dans un json.dumps() casse — on veut de vrais bool."""
    idx = pd.to_datetime(["2026-06-10"])
    macro = {"US": pd.DataFrame({"macro_bull": [True]}, index=idx)}
    assert type(get_macro_regime(macro, JOUR)["US"]) is bool
