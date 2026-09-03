# ============================================================
# tests/test_exit_conditions.py — Trading Brain
# ============================================================
# Verrouille les 5 conditions de sortie absolue v4.1 et le k adaptatif
# (backtest_ranking.py). C'est le cœur du moteur : une régression ici
# se traduit directement par une position gardée ou vendue à tort.
#
# Conditions (n'importe laquelle suffit), dans l'ordre de priorité :
#   1. Trailing stop ATR touché
#   2. Prix < SMA 200
#   3. Momentum R² < 0
#   4. Secteur hors force relative
#   5. Macro bearish (indice de zone < SMA 200)
#
# Aucune DB, aucun réseau : les DataFrames sont fabriqués à la main.
# ============================================================

import pandas as pd
import pytest

from backtest_ranking import (
    check_absolute_exit,
    compute_adaptive_k,
    K_MIN,
    K_MAX,
)

JOUR = pd.Timestamp("2026-06-15")


# ============================================================
# HELPERS
# ============================================================

def make_df(prix=100.0, atr=2.0, sma_200=90.0, mom_r2=0.15, day=JOUR):
    """DataFrame minimal d'un ticker à une date."""
    return pd.DataFrame(
        {"prix_ajuste": [prix], "atr_14": [atr],
         "sma_200": [sma_200], "mom_r2": [mom_r2]},
        index=[day],
    )


def make_pos(entry=100.0, max_price=100.0, stop=94.0, atr_entry=2.0, k=3.0):
    return {"entry_price": entry, "max_price": max_price, "stop": stop,
            "atr_entry": atr_entry, "k": k, "entry_date": pd.Timestamp("2026-05-01")}


def mapping(zone="US", secteur="Technology"):
    return {"AAA": {"secteur": secteur, "zone_priority": [zone]}}


def force(en_force=True, secteur="Technology", zone="US", day=JOUR):
    return {(secteur, zone): pd.DataFrame({"en_force_relative": [en_force]}, index=[day])}


def macro(bull=True, zone="US", day=JOUR):
    return {zone: pd.DataFrame({"macro_bull": [bull]}, index=[day])}


# ============================================================
# POSITION SAINE
# ============================================================

def test_position_saine_ne_sort_pas():
    assert check_absolute_exit("AAA", make_pos(), make_df(), JOUR,
                               mapping(), force(), macro()) is None


def test_jour_absent_du_dataframe_ne_sort_pas():
    """Jour férié / donnée manquante : on ne prend pas de décision."""
    df = make_df(day=pd.Timestamp("2026-06-12"))
    assert check_absolute_exit("AAA", make_pos(), df, JOUR,
                               mapping(), force(), macro()) is None


# ============================================================
# 1. TRAILING STOP
# ============================================================

def test_trailing_stop_touche():
    # max_price 120, k=3, atr=2 → stop = 120 - 6 = 114 ; prix 110 < 114
    pos = make_pos(max_price=120.0, stop=114.0)
    assert check_absolute_exit("AAA", pos, make_df(prix=110.0), JOUR,
                               mapping(), force(), macro()) == "TRAILING_STOP"


def test_trailing_stop_remonte_avec_le_plus_haut():
    """Le stop est un cliquet : il monte, il ne redescend jamais."""
    pos = make_pos(max_price=100.0, stop=94.0)
    check_absolute_exit("AAA", pos, make_df(prix=130.0, atr=2.0), JOUR,
                        mapping(), force(), macro())
    assert pos["max_price"] == 130.0
    assert pos["stop"] == pytest.approx(124.0)   # 130 - 3×2

    # ATR qui explose ensuite : le stop calculé baisserait, il doit tenir.
    check_absolute_exit("AAA", pos, make_df(prix=128.0, atr=20.0), JOUR,
                        mapping(), force(), macro())
    assert pos["stop"] == pytest.approx(124.0)


def test_trailing_stop_prioritaire_sur_les_autres_conditions():
    """Stop touché ET tendance cassée → la raison retenue est le stop."""
    pos = make_pos(max_price=120.0, stop=114.0)
    df = make_df(prix=110.0, sma_200=115.0, mom_r2=-0.2)
    assert check_absolute_exit("AAA", pos, df, JOUR, mapping(),
                               force(en_force=False), macro(bull=False)) == "TRAILING_STOP"


def test_atr_nan_retombe_sur_atr_entree():
    pos = make_pos(max_price=100.0, stop=90.0, atr_entry=2.0, k=3.0)
    df = make_df(prix=99.0, atr=float("nan"))
    # stop recalculé = 100 - 3×2 = 94 ; prix 99 > 94 → pas de sortie
    assert check_absolute_exit("AAA", pos, df, JOUR,
                               mapping(), force(), macro()) is None
    assert pos["stop"] == pytest.approx(94.0)


# ============================================================
# 2. PRIX < SMA 200
# ============================================================

def test_tendance_cassee():
    df = make_df(prix=95.0, sma_200=100.0)
    assert check_absolute_exit("AAA", make_pos(stop=50.0), df, JOUR,
                               mapping(), force(), macro()) == "TREND_BROKEN"


def test_prix_egal_sma_ne_sort_pas():
    """Le test est strict (<) : à égalité on reste investi."""
    df = make_df(prix=100.0, sma_200=100.0)
    assert check_absolute_exit("AAA", make_pos(stop=50.0), df, JOUR,
                               mapping(), force(), macro()) is None


def test_sma_nan_ne_declenche_pas_de_sortie():
    """SMA non calculable (historique court) : on ne sort pas sur du vide."""
    df = make_df(prix=95.0, sma_200=float("nan"))
    assert check_absolute_exit("AAA", make_pos(stop=50.0), df, JOUR,
                               mapping(), force(), macro()) is None


# ============================================================
# 3. MOMENTUM R² < 0
# ============================================================

def test_momentum_perdu():
    df = make_df(mom_r2=-0.05)
    assert check_absolute_exit("AAA", make_pos(stop=50.0), df, JOUR,
                               mapping(), force(), macro()) == "MOMENTUM_LOST"


def test_momentum_nul_ne_sort_pas():
    """Sortie sur mom_r2 < 0 strict — 0 exactement reste investi."""
    df = make_df(mom_r2=0.0)
    assert check_absolute_exit("AAA", make_pos(stop=50.0), df, JOUR,
                               mapping(), force(), macro()) is None


# ============================================================
# 4. SECTEUR HORS FORCE RELATIVE
# ============================================================

def test_secteur_faible():
    assert check_absolute_exit("AAA", make_pos(stop=50.0), make_df(), JOUR,
                               mapping(), force(en_force=False),
                               macro()) == "SECTOR_WEAK"


def test_secteur_sans_donnee_laisse_passer():
    """Pas d'ETF pour ce secteur/zone → on ne filtre pas (fallback documenté)."""
    assert check_absolute_exit("AAA", make_pos(stop=50.0), make_df(), JOUR,
                               mapping(), {}, macro()) is None


# ============================================================
# 5. MACRO BEARISH
# ============================================================

def test_macro_bearish():
    assert check_absolute_exit("AAA", make_pos(stop=50.0), make_df(), JOUR,
                               mapping(), force(), macro(bull=False)) == "MACRO_BEARISH"


def test_macro_kr_ne_retombe_pas_sur_us():
    """
    Régression zone KR (08/06) : un ticker coréen doit être jugé sur ^KS11,
    jamais sur ^GSPC. US bullish + KR bearish → la position KR sort.
    """
    m = {"US": pd.DataFrame({"macro_bull": [True]}, index=[JOUR]),
         "KR": pd.DataFrame({"macro_bull": [False]}, index=[JOUR])}
    map_kr = {"AAA": {"secteur": "Technology", "zone_priority": ["KR"]}}
    assert check_absolute_exit("AAA", make_pos(stop=50.0), make_df(), JOUR,
                               map_kr, force(zone="KR"), m) == "MACRO_BEARISH"


def test_macro_absente_laisse_passer():
    assert check_absolute_exit("AAA", make_pos(stop=50.0), make_df(), JOUR,
                               mapping(), force(), {}) is None


# ============================================================
# K ADAPTATIF — k = 2.0 + atr_pct × 0.5, clampé [2.0, 4.0]
# ============================================================

def test_k_adaptatif_valeur_nominale():
    # ATR 4 sur prix 250 → 1.6 % → k = 2.0 + 0.8 = 2.8
    assert compute_adaptive_k(4.0, 250.0) == pytest.approx(2.8)


def test_k_adaptatif_clamp_haut():
    # ATR 8 sur prix 130 → 6.15 % → 5.08 → clampé à 4.0
    assert compute_adaptive_k(8.0, 130.0) == K_MAX


def test_k_adaptatif_clamp_bas():
    assert compute_adaptive_k(0.0, 100.0) == K_MIN


def test_k_adaptatif_prix_invalide():
    """Prix <= 0 (donnée corrompue) → fallback 3.0, jamais de division par zéro."""
    assert compute_adaptive_k(4.0, 0.0) == 3.0
    assert compute_adaptive_k(4.0, -10.0) == 3.0


def test_k_adaptatif_toujours_dans_les_bornes():
    for atr, prix in [(0.1, 500.0), (50.0, 100.0), (2.0, 170.0), (25.0, 900.0)]:
        assert K_MIN <= compute_adaptive_k(atr, prix) <= K_MAX
