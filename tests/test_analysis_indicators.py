# ============================================================
# tests/test_analysis_indicators.py — Trading Brain
# ============================================================
# Verrouille `analysis.compute_analysis_indicators` — le calcul des
# indicateurs techniques écrits dans actions_prix_historique.
#
# Pourquoi ces tests existent : cette fonction était IMBRIQUÉE dans
# `run_analysis_logic` jusqu'au 08/09. Pure (DataFrame → DataFrame), mais
# inatteignable depuis un test — et c'est pourtant là qu'est né R4 :
# `min_periods=1` sur la SMA 200 produisait une moyenne biaisée vers le prix
# courant, silencieusement. Le premier test ci-dessous est la régression R4.
#
# ⚠️ PÉRIMÈTRE (vérifié le 2026-09-07) : ces colonnes n'alimentent AUCUN des
#    trois chemins de décision, qui rechargent tous l'historique complet via
#    backtest_ranking. Un test vert ici ne dit rien du moteur de trade.
#
# Aucune DB, aucun réseau : les séries sont fabriquées à la main.
# ============================================================

import numpy as np
import pandas as pd
import pytest

from analysis import compute_analysis_indicators


# ============================================================
# HELPERS
# ============================================================

def serie(n, prix=None, volume=1000.0, avec_hl=True, depart="2024-01-01"):
    """DataFrame d'un ticker, n barres quotidiennes consécutives."""
    idx = pd.date_range(depart, periods=n, freq="D")
    if prix is None:
        prix = [100.0 + i * 0.1 for i in range(n)]
    df = pd.DataFrame({
        "ticker":      ["AAA"] * n,
        "date":        idx,
        "prix_ajuste": pd.Series(prix, dtype="float64"),
        "volume":      pd.Series([volume] * n, dtype="float64"),
    })
    if avec_hl:
        df["prix_haut"] = df["prix_ajuste"] * 1.01
        df["prix_bas"]  = df["prix_ajuste"] * 0.99
    return df


# ============================================================
# RÉGRESSION R4 (2026-09-04) — SMA 200 stricte
# ============================================================

def test_r4_sma200_reste_nan_sous_200_barres():
    """
    Le bug R4 : avec min_periods=1, la SMA 200 valait ~le prix courant tant
    que la fenêtre n'avait pas 200 barres — une SMA~150 déguisée en SMA 200,
    invisible parce que non nulle. Une valeur absente est le comportement
    voulu ; une valeur fausse ne l'est jamais.
    """
    out = compute_analysis_indicators(serie(199))
    assert out["sma_200"].isna().all()


def test_r4_sma200_apparait_a_la_200e_barre():
    out = compute_analysis_indicators(serie(200))
    assert out["sma_200"].iloc[:199].isna().all()
    assert not pd.isna(out["sma_200"].iloc[-1])


def test_sma200_est_bien_la_moyenne_des_200_dernieres():
    df = serie(250)
    out = compute_analysis_indicators(df)
    attendu = df["prix_ajuste"].iloc[-200:].mean()
    assert out["sma_200"].iloc[-1] == pytest.approx(attendu, abs=1e-9)


# ============================================================
# GARDE-FOU D'ENTRÉE
# ============================================================

def test_moins_de_50_barres_retourne_le_groupe_intact():
    """Sous 50 barres, aucune colonne n'est ajoutée — pas de valeur inventée."""
    df = serie(49)
    out = compute_analysis_indicators(df)
    assert list(out.columns) == list(df.columns)
    assert "rsi_14" not in out.columns


def test_50_barres_declenche_le_calcul():
    out = compute_analysis_indicators(serie(50))
    for col in ("rsi_14", "sma_50", "bb_lower", "bb_position",
                "vol_avg_20", "atr_14", "regime_marche"):
        assert col in out.columns


# ============================================================
# RSI 14
# ============================================================

def test_rsi_sature_en_haut_sur_serie_strictement_croissante():
    out = compute_analysis_indicators(serie(80))
    assert out["rsi_14"].iloc[-1] > 99


def test_rsi_sature_en_bas_sur_serie_strictement_decroissante():
    prix = [200.0 - i * 0.5 for i in range(80)]
    out = compute_analysis_indicators(serie(80, prix=prix))
    assert out["rsi_14"].iloc[-1] < 1


# ============================================================
# ATR 14 — réel (H-L) si disponible, approché (C-C) sinon
# ============================================================

def test_atr_utilise_le_true_range_quand_haut_et_bas_existent():
    out = compute_analysis_indicators(serie(80, avec_hl=True))
    # TR ≈ 2 % du prix (haut/bas à ±1 %), très au-dessus du pas de 0,10
    assert out["atr_14"].iloc[-1] > 1.0


def test_atr_retombe_sur_le_close_a_close_sans_haut_ni_bas():
    out = compute_analysis_indicators(serie(80, avec_hl=False))
    # Sans H/L, TR = |ΔC| = 0,10 par barre
    assert out["atr_14"].iloc[-1] < 0.2


def test_atr_ignore_des_colonnes_haut_bas_entierement_nulles():
    """Colonnes présentes mais vides (tickers non couverts par /fill-high-low)
    → repli sur le close-to-close, pas un ATR NaN."""
    df = serie(80, avec_hl=True)
    df["prix_haut"] = np.nan
    df["prix_bas"]  = np.nan
    out = compute_analysis_indicators(df)
    assert not out["atr_14"].isna().all()
    assert out["atr_14"].iloc[-1] < 0.2


# ============================================================
# RÉGIME DE MARCHÉ — seuils ±2 % autour de la SMA 50
# ============================================================

def test_regime_bull_en_tendance_haussiere_marquee():
    prix = [100.0 * (1.01 ** i) for i in range(80)]
    out = compute_analysis_indicators(serie(80, prix=prix))
    assert out["regime_marche"].iloc[-1] == "BULL"


def test_regime_bear_en_tendance_baissiere_marquee():
    prix = [100.0 * (0.99 ** i) for i in range(80)]
    out = compute_analysis_indicators(serie(80, prix=prix))
    assert out["regime_marche"].iloc[-1] == "BEAR"


def test_regime_neutre_sur_prix_plat():
    out = compute_analysis_indicators(serie(80, prix=[100.0] * 80))
    assert out["regime_marche"].iloc[-1] == "NEUTRE"


# ============================================================
# BANDES DE BOLLINGER — pas de division par zéro
# ============================================================

def test_bollinger_ne_divise_pas_par_zero_sur_prix_plat():
    """Prix constant → largeur de bande nulle. La position doit être NaN,
    jamais un inf qui se propagerait dans les features ML."""
    out = compute_analysis_indicators(serie(80, prix=[100.0] * 80))
    assert not np.isinf(out["bb_position"].dropna()).any()


# ============================================================
# FEATURES ML DÉRIVÉES
# ============================================================

def test_vol_ratio_vaut_environ_1_a_volume_constant():
    out = compute_analysis_indicators(serie(80, volume=5000.0))
    assert abs(out["vol_ratio"].iloc[-1] - 1.0) < 0.01


def test_dist_sma200_positive_quand_le_prix_est_au_dessus():
    df = serie(250)
    out = compute_analysis_indicators(df)
    assert out["dist_sma200"].iloc[-1] > 0


# ============================================================
# PURETÉ — la fonction ne doit rien lire hors de son DataFrame
# ============================================================

def test_deux_appels_donnent_le_meme_resultat():
    df = serie(120)
    a = compute_analysis_indicators(df.copy())
    b = compute_analysis_indicators(df.copy())
    pd.testing.assert_frame_equal(a, b)


def test_signal_achat_nest_plus_produit():
    """v3.3 supprimée : la colonne ne doit plus jamais réapparaître."""
    out = compute_analysis_indicators(serie(120))
    assert "signal_achat" not in out.columns
