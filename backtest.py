# ============================================================
# backtest.py — Backtest Signal v3.5 | Trading Brain
# VT-Source | Phase 1 Étape 2
# ============================================================
# Appel : GET /run-backtest?tickers=NVDA,AAPL,MSFT,AMZN,ASML
#
# Ce que ce module teste :
#   1. Signal v3.5 (momentum R² + breakout 20j + RVOL>2 + OBV)
#   2. Sortie : Trailing Stop ATR (Chandelier Exit) — k = 2.0/2.5/3.0/3.5
#   3. Comparaison : signaux en secteur fort vs hors secteur fort
#   4. Critère de validation : Sharpe Ratio > 1.0 sur 3+ tickers
#
# Données : PostgreSQL (actions_prix_historique + secteurs_etf_prix)
# ============================================================

import os
import numpy as np
import pandas as pd
import vectorbt as vbt
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True) if DATABASE_URL else None

# ============================================================
# PARAMÈTRES
# ============================================================
TICKERS_DEFAULT    = ["NVDA", "AAPL", "MSFT", "AMZN", "ASML"]
K_VALUES           = [2.0, 2.5, 3.0, 3.5]   # multiplicateurs ATR à tester
MOMENTUM_WINDOW    = 252                      # 12 mois de trading
ROC_WINDOW         = 252                      # rendement annuel
BREAKOUT_WINDOW    = 20                       # plus haut 20j
RVOL_THRESHOLD     = 2.0                      # conviction institutionnelle
VOL_AVG_WINDOW     = 20                       # base RVOL
OBV_SLOPE_WINDOW   = 10                       # pente OBV
ATR_WINDOW         = 14                       # ATR standard
CAPITAL_INITIAL    = 10_000                   # € par ticker pour le backtest


# ============================================================
# CHARGEMENT DES DONNÉES DEPUIS POSTGRESQL
# ============================================================

def load_ticker_data(ticker: str) -> pd.DataFrame:
    """
    Charge l'historique complet d'un ticker depuis actions_prix_historique.
    Retourne un DataFrame indexé par date avec toutes les colonnes OHLCV.
    """
    if engine is None:
        raise RuntimeError("Engine PostgreSQL non connecté.")

    query = text("""
        SELECT date, prix_cloture, prix_ajuste, volume,
               prix_haut, prix_bas,
               sma_200, vol_avg_20, rsi_14
        FROM actions_prix_historique
        WHERE ticker = :ticker
          AND prix_ajuste IS NOT NULL
          AND prix_ajuste > 0
        ORDER BY date ASC
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"ticker": ticker})

    if df.empty:
        raise ValueError(f"Aucune donnée pour {ticker} en base.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df["prix_ajuste"] = df["prix_ajuste"].fillna(df["prix_cloture"])

    return df


def load_secteur_force(ticker: str) -> pd.DataFrame:
    """
    Charge l'historique de force relative sectorielle pour le secteur du ticker.
    Retourne un DataFrame indexé par date avec en_force_relative (bool).
    Joint via tickers_info.secteur_yahoo → secteurs_etf_prix.
    Priorité zone US si disponible, sinon EU.
    """
    if engine is None:
        return pd.DataFrame()

    query = text("""
        SELECT sep.date, sep.en_force_relative
        FROM secteurs_etf_prix sep
        JOIN secteurs_etf se ON se.ticker_etf = sep.ticker_etf
        JOIN tickers_info ti ON ti.secteur = se.secteur_yahoo
        WHERE ti.ticker = :ticker
          AND se.actif = TRUE
        ORDER BY
            CASE se.zone WHEN 'US' THEN 1 WHEN 'EU' THEN 2 ELSE 3 END,
            sep.date ASC
    """)

    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"ticker": ticker})
        if df.empty:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"])
        df = df.drop_duplicates(subset="date").set_index("date").sort_index()
        return df
    except Exception:
        return pd.DataFrame()


# ============================================================
# CALCUL DES INDICATEURS v3.5
# ============================================================

def momentum_r2_score(prices: pd.Series, window: int = MOMENTUM_WINDOW) -> pd.Series:
    """
    Momentum ajusté R² = ROC_12m × R² de la régression log-linéaire.
    Pénalise les tendances erratiques, favorise les tendances propres.
    Décision 2026-03-17 : remplace le ROC brut de v3.3.
    """
    scores = pd.Series(index=prices.index, dtype=float)

    for i in range(window, len(prices)):
        slice_prices = prices.iloc[i - window:i]
        if slice_prices.isnull().any() or (slice_prices <= 0).any():
            continue

        # ROC 12 mois
        roc = (slice_prices.iloc[-1] - slice_prices.iloc[0]) / slice_prices.iloc[0]

        # R² de la tendance log-linéaire
        x = np.arange(window).reshape(-1, 1)
        y = np.log(slice_prices.values)
        try:
            r2 = LinearRegression().fit(x, y).score(x, y)
        except Exception:
            r2 = 0.0

        scores.iloc[i] = roc * r2

    return scores


def compute_atr(df: pd.DataFrame, window: int = ATR_WINDOW) -> pd.Series:
    """
    ATR (Average True Range) sur 14 jours.
    - Mode réel    : utilise prix_haut et prix_bas (True Range complet)
                     activé automatiquement après /fill-high-low
    - Mode approché : close-to-close si high/low absents (null en base)
    """
    close = df["prix_ajuste"]

    has_hl = (
        "prix_haut" in df.columns and
        "prix_bas"  in df.columns and
        df["prix_haut"].notna().any() and
        df["prix_bas"].notna().any()
    )

    if has_hl:
        prev_close = close.shift(1)
        tr = pd.concat([
            df["prix_haut"] - df["prix_bas"],
            (df["prix_haut"] - prev_close).abs(),
            (df["prix_bas"]  - prev_close).abs()
        ], axis=1).max(axis=1)
        print(f"      ATR mode : réel (High-Low) ✅")
    else:
        tr = close.diff().abs()
        print(f"      ATR mode : approché (Close-Close) ⚠️ — lancer /fill-high-low")

    return tr.rolling(window, min_periods=1).mean()


def compute_signals_v35(df: pd.DataFrame, df_secteur: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule tous les indicateurs v3.5 et le signal d'achat composite.

    Niveaux implémentés :
        Niveau 1 : prix > SMA_200
        Niveau 2 : secteur en force relative (si données dispo)
        Niveau 3 : momentum_r2 > 0 + breakout_20j + RVOL > 2.0 + OBV accumulation
    """
    price = df["prix_ajuste"]
    vol   = df["volume"].fillna(0)

    # --- Niveau 1 : Filtre tendance ---
    sma_200    = price.rolling(200, min_periods=1).mean()
    tendance_ok = price > sma_200

    # --- Niveau 2 : Force relative sectorielle ---
    if not df_secteur.empty:
        # Aligner sur l'index du ticker (forward fill pour les jours sans ETF)
        force_rel = df_secteur["en_force_relative"].reindex(
            df.index, method="ffill"
        ).fillna(False)
    else:
        # Pas de données secteur → on laisse passer (pas de filtre dur)
        force_rel = pd.Series(True, index=df.index)

    # --- Niveau 3 : Stock picking ---

    # Momentum ajusté R²
    mom_r2    = momentum_r2_score(price, MOMENTUM_WINDOW)
    mom_ok    = mom_r2 > 0

    # Breakout plus haut 20j
    breakout_20j = price >= price.rolling(BREAKOUT_WINDOW, min_periods=1).max()

    # RVOL > 2.0
    vol_avg_20 = vol.rolling(VOL_AVG_WINDOW, min_periods=1).mean()
    rvol       = vol / (vol_avg_20 + 1e-9)
    rvol_fort  = rvol > RVOL_THRESHOLD

    # OBV accumulation
    obv            = (np.sign(price.diff()) * vol).fillna(0).cumsum()
    obv_slope      = obv.diff(OBV_SLOPE_WINDOW)
    obv_accumulation = obv_slope > 0

    # --- Signal final v3.5 ---
    signal_achat = (
        tendance_ok &
        force_rel &
        mom_ok &
        breakout_20j &
        rvol_fort &
        obv_accumulation
    )

    # --- ATR pour le trailing stop ---
    atr_14 = compute_atr(df, ATR_WINDOW)

    result = df[["prix_ajuste", "volume"]].copy()
    result["sma_200"]         = sma_200
    result["tendance_ok"]     = tendance_ok
    result["force_rel"]       = force_rel
    result["mom_r2"]          = mom_r2
    result["breakout_20j"]    = breakout_20j
    result["rvol"]            = rvol
    result["rvol_fort"]       = rvol_fort
    result["obv_accumulation"]= obv_accumulation
    result["signal_achat"]    = signal_achat
    result["atr_14"]          = atr_14

    return result


# ============================================================
# TRAILING STOP ATR (CHANDELIER EXIT)
# ============================================================

def build_exit_signals(entries: pd.Series, price: pd.Series,
                        atr: pd.Series, k: float) -> pd.Series:
    """
    Construit les signaux de sortie via Chandelier Exit.

    Logique :
        - À l'entrée : stop_initial = prix_entree - k × ATR_entree
        - En cours : stop = max_depuis_entree - k × ATR (ne peut que monter)
        - Sortie si prix <= stop

    Retourne une Series booléenne (True = sortie ce jour).
    """
    exits      = pd.Series(False, index=price.index)
    in_trade   = False
    max_price  = 0.0
    stop       = 0.0

    for i in range(len(price)):
        if not in_trade:
            if entries.iloc[i]:
                in_trade  = True
                max_price = price.iloc[i]
                stop      = price.iloc[i] - k * atr.iloc[i]
        else:
            # Mise à jour du plus haut
            if price.iloc[i] > max_price:
                max_price = price.iloc[i]

            # Stop ne peut que monter
            new_stop = max_price - k * atr.iloc[i]
            stop     = max(stop, new_stop)

            # Condition de sortie
            if price.iloc[i] <= stop:
                exits.iloc[i] = True
                in_trade      = False
                max_price     = 0.0
                stop          = 0.0

    return exits


# ============================================================
# BACKTEST PRINCIPAL PAR TICKER
# ============================================================

def run_backtest_ticker(ticker: str) -> dict:
    """
    Lance le backtest complet pour un ticker.
    Teste les 4 valeurs de k ATR et retourne les métriques pour chaque.
    """
    print(f"   📊 Backtest {ticker}...")

    try:
        df         = load_ticker_data(ticker)
        df_secteur = load_secteur_force(ticker)
    except Exception as e:
        return {"ticker": ticker, "erreur": str(e)}

    if len(df) < MOMENTUM_WINDOW + 50:
        return {"ticker": ticker, "erreur": f"Historique insuffisant ({len(df)} jours)"}

    # Calcul des indicateurs v3.5
    df_signals = compute_signals_v35(df, df_secteur)

    price   = df_signals["prix_ajuste"]
    entries = df_signals["signal_achat"]
    atr     = df_signals["atr_14"]

    nb_signaux = entries.sum()
    print(f"      Signaux générés : {nb_signaux}")

    if nb_signaux == 0:
        return {
            "ticker"    : ticker,
            "nb_signaux": 0,
            "message"   : "Aucun signal v3.5 sur la période — conditions trop strictes ou données insuffisantes.",
            "resultats_k": {}
        }

    resultats_k = {}

    for k in K_VALUES:
        try:
            exits = build_exit_signals(entries, price, atr, k)

            # VectorBT portfolio
            portfolio = vbt.Portfolio.from_signals(
                close      = price,
                entries    = entries,
                exits      = exits,
                init_cash  = CAPITAL_INITIAL,
                freq       = "D",
                upon_opposite_entry="ignore"  # ignorer signal si déjà en position
            )

            stats = portfolio.stats()

            # Métriques clés
            sharpe      = round(float(stats.get("Sharpe Ratio", 0) or 0), 3)
            total_ret   = round(float(stats.get("Total Return [%]", 0) or 0), 2)
            max_dd      = round(float(stats.get("Max Drawdown [%]", 0) or 0), 2)
            nb_trades   = int(stats.get("Total Trades", 0) or 0)
            win_rate    = round(float(stats.get("Win Rate [%]", 0) or 0), 2)
            avg_trade   = round(float(stats.get("Avg Winning Trade [%]", 0) or 0), 2)

            resultats_k[str(k)] = {
                "sharpe"       : sharpe,
                "total_return" : total_ret,
                "max_drawdown" : max_dd,
                "nb_trades"    : nb_trades,
                "win_rate"     : win_rate,
                "avg_win_trade": avg_trade,
                "valide"       : sharpe >= 1.0
            }

            print(f"      k={k} → Sharpe={sharpe} | Return={total_ret}% | MaxDD={max_dd}% | Trades={nb_trades} | WinRate={win_rate}%")

        except Exception as e:
            resultats_k[str(k)] = {"erreur": str(e)}

    # Meilleur k (Sharpe le plus élevé)
    valid_k = {k: v for k, v in resultats_k.items() if "sharpe" in v}
    best_k  = max(valid_k, key=lambda k: valid_k[k]["sharpe"]) if valid_k else None

    return {
        "ticker"     : ticker,
        "nb_signaux" : int(nb_signaux),
        "nb_jours"   : len(df),
        "resultats_k": resultats_k,
        "best_k"     : best_k,
        "best_sharpe": valid_k[best_k]["sharpe"] if best_k else None
    }


# ============================================================
# ANALYSE SECTORIELLE A POSTERIORI
# ============================================================

def analyse_secteur_a_posteriori(ticker: str, df_signals: pd.DataFrame,
                                   horizon: int = 30) -> dict:
    """
    Compare les performances des signaux en secteur fort vs hors secteur fort.
    Mesure le rendement moyen à J+horizon pour chaque groupe.
    """
    price    = df_signals["prix_ajuste"]
    signaux  = df_signals[df_signals["signal_achat"]].copy()

    if signaux.empty:
        return {}

    # Rendement à J+horizon
    future_returns = []
    for date_signal in signaux.index:
        loc = price.index.get_loc(date_signal)
        if loc + horizon < len(price):
            ret = (price.iloc[loc + horizon] - price.iloc[loc]) / price.iloc[loc] * 100
            future_returns.append({
                "date"        : date_signal,
                "force_rel"   : bool(df_signals.loc[date_signal, "force_rel"]),
                "rendement_pct": round(float(ret), 2)
            })

    if not future_returns:
        return {}

    df_ret     = pd.DataFrame(future_returns)
    en_force   = df_ret[df_ret["force_rel"] == True]["rendement_pct"]
    hors_force = df_ret[df_ret["force_rel"] == False]["rendement_pct"]

    return {
        "horizon_jours"          : horizon,
        "en_force_relative"      : {
            "nb_signaux"    : len(en_force),
            "rendement_moyen": round(float(en_force.mean()), 2) if len(en_force) > 0 else None,
            "taux_succes_pct": round(float((en_force > 0).mean() * 100), 1) if len(en_force) > 0 else None
        },
        "hors_force_relative"    : {
            "nb_signaux"    : len(hors_force),
            "rendement_moyen": round(float(hors_force.mean()), 2) if len(hors_force) > 0 else None,
            "taux_succes_pct": round(float((hors_force > 0).mean() * 100), 1) if len(hors_force) > 0 else None
        }
    }


# ============================================================
# SYNTHÈSE GLOBALE
# ============================================================

def synthesize_results(all_results: list) -> dict:
    """
    Agrège les résultats de tous les tickers pour identifier :
    - Le k optimal global (Sharpe moyen le plus élevé)
    - Le nombre de tickers validant Sharpe > 1.0
    - La recommandation finale
    """
    valid_results = [r for r in all_results if "resultats_k" in r and r["resultats_k"]]

    if not valid_results:
        return {"message": "Aucun résultat valide — vérifier les données en base."}

    # Sharpe moyen par valeur de k
    sharpe_par_k = {}
    for k in K_VALUES:
        k_str   = str(k)
        sharpes = [
            r["resultats_k"][k_str]["sharpe"]
            for r in valid_results
            if k_str in r["resultats_k"] and "sharpe" in r["resultats_k"][k_str]
        ]
        if sharpes:
            sharpe_par_k[k_str] = {
                "sharpe_moyen"   : round(np.mean(sharpes), 3),
                "nb_tickers_ok"  : sum(1 for s in sharpes if s >= 1.0),
                "nb_tickers_total": len(sharpes)
            }

    # Meilleur k global
    best_k_global = max(
        sharpe_par_k,
        key=lambda k: sharpe_par_k[k]["sharpe_moyen"]
    ) if sharpe_par_k else None

    # Critère de validation : Sharpe > 1.0 sur 3+ tickers
    nb_valides = sharpe_par_k.get(best_k_global, {}).get("nb_tickers_ok", 0) if best_k_global else 0
    signal_valide = nb_valides >= 3

    return {
        "sharpe_par_k"   : sharpe_par_k,
        "best_k_global"  : best_k_global,
        "nb_tickers_valides": nb_valides,
        "signal_v35_valide" : signal_valide,
        "recommandation" : (
            f"✅ Signal v3.5 VALIDÉ — k={best_k_global} recommandé ({nb_valides} tickers Sharpe>1.0)"
            if signal_valide else
            f"⚠️ Signal v3.5 NON VALIDÉ — seulement {nb_valides}/3 tickers Sharpe>1.0 avec k={best_k_global}"
        )
    }


# ============================================================
# FONCTION PRINCIPALE — appelée par l'endpoint FastAPI
# ============================================================

def run_backtest_logic(tickers: list[str] = None, horizon: int = 30) -> dict:
    """
    Point d'entrée principal du backtest.
    Appelée en arrière-plan par l'endpoint /run-backtest.

    Paramètres
    ----------
    tickers : list[str]
        Liste des tickers à tester. Défaut : NVDA, AAPL, MSFT, AMZN, ASML.
    horizon : int
        Horizon en jours pour l'analyse sectorielle a posteriori (défaut : 30).
    """
    if tickers is None:
        tickers = TICKERS_DEFAULT

    print(f"🚀 Backtest v3.5 démarré — {len(tickers)} tickers, horizon={horizon}j")
    print(f"   Tickers : {', '.join(tickers)}")
    print(f"   K ATR testés : {K_VALUES}")

    all_results      = []
    analyse_secteur  = {}

    for ticker in tickers:
        result = run_backtest_ticker(ticker)
        all_results.append(result)

        # Analyse sectorielle a posteriori si des signaux existent
        if result.get("nb_signaux", 0) > 0:
            try:
                df         = load_ticker_data(ticker)
                df_secteur = load_secteur_force(ticker)
                df_signals = compute_signals_v35(df, df_secteur)
                analyse_secteur[ticker] = analyse_secteur_a_posteriori(
                    ticker, df_signals, horizon
                )
            except Exception as e:
                analyse_secteur[ticker] = {"erreur": str(e)}

    synthese = synthesize_results(all_results)

    print(f"\n{'='*60}")
    print(f"📋 SYNTHÈSE BACKTEST v3.5")
    print(f"{'='*60}")
    print(f"   {synthese.get('recommandation', '')}")
    if synthese.get("sharpe_par_k"):
        for k, stats in synthese["sharpe_par_k"].items():
            print(f"   k={k} → Sharpe moyen={stats['sharpe_moyen']} | Tickers OK={stats['nb_tickers_ok']}/{stats['nb_tickers_total']}")

    return {
        "parametres": {
            "tickers" : tickers,
            "k_values": K_VALUES,
            "horizon" : horizon,
            "capital" : CAPITAL_INITIAL
        },
        "resultats_par_ticker": all_results,
        "analyse_sectorielle" : analyse_secteur,
        "synthese"            : synthese
    }
