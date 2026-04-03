# ============================================================
# backtest_ranking.py — Backtest Momentum Ranking v4.0
# Trading Brain | VT-Source
# ============================================================
# Appel : GET /run-backtest?mode=ranking&top_n=5
#
# Architecture fondamentalement différente de v3.5/v3.5b :
#   - Ancien : filtre binaire par ticker (signal oui/non)
#   - Nouveau : ranking cross-ticker (top N par score composite)
#
# Logique :
#   1. Chaque lundi, scorer TOUS les tickers éligibles
#   2. Classer par score composite décroissant
#   3. Acheter le top N, vendre ceux qui sortent du top N
#   4. Trailing stop ATR actif en continu (protection intra-semaine)
#
# Score composite = w1 × mom_r2_norm + w2 × rvol_norm + w3 × obv_norm
# Pondérations : mom_r2 (50%) + rvol (25%) + obv_slope (25%)
# ============================================================

import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from datetime import timedelta

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True) if DATABASE_URL else None

# ============================================================
# PARAMÈTRES
# ============================================================
TOP_N              = 5            # positions simultanées max
K_VALUES           = [2.5, 3.0, 3.5]   # trailing stop ATR (fixes)
K_MIN              = 2.0          # k adaptatif — plancher
K_MAX              = 4.0          # k adaptatif — plafond
K_ADAPTIVE_COEFF   = 0.5          # k = K_MIN + atr_pct × coeff, clampé [K_MIN, K_MAX]
MOMENTUM_WINDOW    = 252          # 12 mois
VOL_AVG_WINDOW     = 20           # base RVOL
OBV_SLOPE_WINDOW   = 10           # pente OBV
ATR_WINDOW         = 14           # ATR standard
CAPITAL_INITIAL    = 50_000       # capital total (réparti sur N positions)
MIN_HISTORY        = 260          # jours min pour calculer les indicateurs

# Pondérations du score composite
W_MOM_R2           = 0.50         # momentum ajusté R² (qualité de tendance)
W_RVOL             = 0.25         # volume relatif (conviction)
W_OBV              = 0.25         # accumulation OBV (smart money)

# Pays européens pour le mapping zone ETF
PAYS_EU = {
    'Belgium', 'France', 'Germany', 'Netherlands', 'Italy', 'Spain',
    'Switzerland', 'Sweden', 'Denmark', 'Finland', 'Norway', 'Austria',
    'Portugal', 'Ireland', 'Luxembourg', 'United Kingdom'
}


# ============================================================
# CHARGEMENT DONNÉES
# ============================================================

def load_all_tickers() -> list[str]:
    """Retourne la liste de tous les tickers en base avec assez d'historique."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT ticker, COUNT(*) as nb
            FROM actions_prix_historique
            WHERE prix_ajuste IS NOT NULL AND prix_ajuste > 0
            GROUP BY ticker
            HAVING COUNT(*) >= :min_hist
            ORDER BY ticker
        """), {"min_hist": MIN_HISTORY}).fetchall()
    return [r[0] for r in rows]


def load_all_price_data(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """
    Charge l'historique OHLCV de tous les tickers en un seul query.
    Retourne un dict {ticker: DataFrame indexé par date}.
    """
    if not tickers:
        return {}

    query = text("""
        SELECT ticker, date, prix_cloture, prix_ajuste, volume,
               prix_haut, prix_bas
        FROM actions_prix_historique
        WHERE ticker IN :tickers
          AND prix_ajuste IS NOT NULL
          AND prix_ajuste > 0
        ORDER BY ticker, date ASC
    """)

    with engine.connect() as conn:
        df_all = pd.read_sql(query, conn, params={"tickers": tuple(tickers)})

    df_all["date"] = pd.to_datetime(df_all["date"])
    df_all["prix_ajuste"] = df_all["prix_ajuste"].fillna(df_all["prix_cloture"])

    result = {}
    for ticker, group in df_all.groupby("ticker"):
        df_t = group.set_index("date").sort_index()
        if len(df_t) >= MIN_HISTORY:
            result[ticker] = df_t

    return result


def load_secteur_mapping() -> dict[str, dict]:
    """
    Retourne un dict {ticker: {"secteur": ..., "zone_priority": [...]}}
    basé sur tickers_info.secteur et tickers_info.pays.
    """
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT ticker, secteur, pays
            FROM tickers_info
            WHERE secteur IS NOT NULL
        """)).fetchall()

    mapping = {}
    for ticker, secteur, pays in rows:
        pays = pays or ''
        if pays == 'Belgium' or pays in PAYS_EU:
            zone_priority = ['EU', 'US']
        else:
            zone_priority = ['US', 'EU']
        mapping[ticker] = {"secteur": secteur, "zone_priority": zone_priority}

    return mapping


def load_all_secteur_force() -> dict[str, pd.DataFrame]:
    """
    Charge toutes les données de force relative sectorielle.
    Retourne {(secteur, zone): DataFrame indexé par date avec en_force_relative}.
    """
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT se.secteur_yahoo, se.zone, sep.date, sep.en_force_relative
            FROM secteurs_etf_prix sep
            JOIN secteurs_etf se ON se.ticker_etf = sep.ticker_etf
            WHERE se.actif = TRUE
            ORDER BY se.secteur_yahoo, se.zone, sep.date ASC
        """), conn)

    if df.empty:
        return {}

    df["date"] = pd.to_datetime(df["date"])
    result = {}
    for (secteur, zone), group in df.groupby(["secteur_yahoo", "zone"]):
        df_sz = group[["date", "en_force_relative"]].drop_duplicates(subset="date")
        df_sz = df_sz.set_index("date").sort_index()
        result[(secteur, zone)] = df_sz

    return result


def load_macro_data() -> dict[str, pd.DataFrame]:
    """
    Charge les prix des indices de référence macro et calcule la SMA 200.
    Retourne {zone: DataFrame indexé par date avec prix_ajuste et sma_200_idx}.

    Mapping zone → indice :
        US → ^GSPC (S&P 500)
        EU → ^STOXX (STOXX Europe 600)
    """
    ZONE_INDEX = {"US": "^GSPC", "EU": "^STOXX"}

    result = {}
    for zone, idx_ticker in ZONE_INDEX.items():
        with engine.connect() as conn:
            df = pd.read_sql(text("""
                SELECT date, prix_ajuste
                FROM indices_prix
                WHERE ticker_indice = :ticker
                ORDER BY date ASC
            """), conn, params={"ticker": idx_ticker})

        if df.empty:
            continue

        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df["sma_200_idx"] = df["prix_ajuste"].rolling(200, min_periods=200).mean()
        df["macro_bull"] = df["prix_ajuste"] > df["sma_200_idx"]
        result[zone] = df

    return result


def get_macro_regime(
    macro_data: dict[str, pd.DataFrame],
    as_of: pd.Timestamp
) -> dict[str, bool]:
    """
    Retourne l'état macro par zone à la date donnée.
    {"US": True/False, "EU": True/False}
    True = bullish (indice > SMA200), False = bearish.
    """
    regime = {}
    for zone, df in macro_data.items():
        valid = df[df.index <= as_of]
        if not valid.empty:
            regime[zone] = bool(valid["macro_bull"].iloc[-1])
        else:
            regime[zone] = True  # pas de données → on laisse passer
    return regime


def get_ticker_zone(ticker: str, secteur_mapping: dict) -> str:
    """Retourne la zone principale d'un ticker (US ou EU)."""
    info = secteur_mapping.get(ticker)
    if not info:
        return "US"
    return info["zone_priority"][0]


def get_secteur_force_for_ticker(
    ticker: str,
    secteur_mapping: dict,
    force_data: dict,
    as_of: pd.Timestamp
) -> bool:
    """
    Retourne True si le secteur du ticker est en force relative à la date donnée.
    Utilise le mapping pays → zone pour trouver le bon ETF.
    """
    info = secteur_mapping.get(ticker)
    if not info:
        return True  # pas d'info → on laisse passer

    secteur = info["secteur"]
    for zone in info["zone_priority"]:
        key = (secteur, zone)
        if key in force_data:
            df_force = force_data[key]
            # Trouver la dernière valeur connue <= as_of
            valid = df_force[df_force.index <= as_of]
            if not valid.empty:
                return bool(valid["en_force_relative"].iloc[-1])

    return True  # pas de données → on laisse passer


# ============================================================
# INDICATEURS
# ============================================================

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule tous les indicateurs nécessaires au scoring.
    Retourne le DataFrame enrichi.
    """
    price = df["prix_ajuste"]
    vol   = df["volume"].fillna(0)

    # SMA 200
    df["sma_200"] = price.rolling(200, min_periods=200).mean()

    # Momentum R² (vectorisé par fenêtre glissante)
    scores = pd.Series(index=df.index, dtype=float)
    log_prices = np.log(price.values)
    x = np.arange(MOMENTUM_WINDOW).reshape(-1, 1)

    for i in range(MOMENTUM_WINDOW, len(price)):
        slice_log = log_prices[i - MOMENTUM_WINDOW:i]
        if np.any(np.isnan(slice_log)) or np.any(np.isinf(slice_log)):
            continue

        roc = (price.iloc[i] - price.iloc[i - MOMENTUM_WINDOW]) / price.iloc[i - MOMENTUM_WINDOW]
        try:
            r2 = LinearRegression().fit(x, slice_log).score(x, slice_log)
        except Exception:
            r2 = 0.0

        scores.iloc[i] = roc * r2

    df["mom_r2"] = scores

    # RVOL
    vol_avg = vol.rolling(VOL_AVG_WINDOW, min_periods=1).mean()
    df["rvol"] = vol / (vol_avg + 1e-9)

    # OBV slope
    obv = (np.sign(price.diff()) * vol).fillna(0).cumsum()
    df["obv_slope"] = obv.diff(OBV_SLOPE_WINDOW)

    # ATR
    has_hl = (
        "prix_haut" in df.columns and
        "prix_bas" in df.columns and
        df["prix_haut"].notna().any() and
        df["prix_bas"].notna().any()
    )

    if has_hl:
        prev_close = price.shift(1)
        tr = pd.concat([
            df["prix_haut"] - df["prix_bas"],
            (df["prix_haut"] - prev_close).abs(),
            (df["prix_bas"] - prev_close).abs()
        ], axis=1).max(axis=1)
    else:
        tr = price.diff().abs()

    df["atr_14"] = tr.rolling(ATR_WINDOW, min_periods=1).mean()

    return df


# ============================================================
# SCORING & RANKING
# ============================================================

def compute_composite_score(
    ticker_data: dict[str, pd.DataFrame],
    date: pd.Timestamp,
    secteur_mapping: dict,
    force_data: dict
) -> list[dict]:
    """
    Score tous les tickers éligibles à une date donnée.
    Retourne une liste triée par score décroissant.

    Éligibilité :
        1. Prix > SMA 200 (tendance haussière)
        2. Secteur en force relative
        3. Mom R² > 0 (tendance positive)
    """
    candidates = []

    for ticker, df in ticker_data.items():
        # Vérifier qu'on a des données à cette date
        if date not in df.index:
            # Prendre le dernier jour disponible avant cette date
            valid = df[df.index <= date]
            if valid.empty:
                continue
            row = valid.iloc[-1]
            row_date = valid.index[-1]
            # Ne pas utiliser si plus de 5 jours d'écart (données stale)
            if (date - row_date).days > 5:
                continue
        else:
            row = df.loc[date]

        # Filtre 1 : tendance haussière
        sma_200 = row.get("sma_200")
        prix = row.get("prix_ajuste")
        if pd.isna(sma_200) or pd.isna(prix) or prix <= sma_200:
            continue

        # Filtre 2 : secteur en force
        if not get_secteur_force_for_ticker(ticker, secteur_mapping, force_data, date):
            continue

        # Filtre 3 : momentum positif
        mom_r2 = row.get("mom_r2")
        if pd.isna(mom_r2) or mom_r2 <= 0:
            continue

        rvol      = row.get("rvol", 0)
        obv_slope = row.get("obv_slope", 0)
        atr_14    = row.get("atr_14", 0)

        if pd.isna(rvol):
            rvol = 0
        if pd.isna(obv_slope):
            obv_slope = 0

        candidates.append({
            "ticker"   : ticker,
            "prix"     : float(prix),
            "mom_r2"   : float(mom_r2),
            "rvol"     : float(rvol),
            "obv_slope": float(obv_slope),
            "atr_14"   : float(atr_14) if not pd.isna(atr_14) else 0,
            "sma_200"  : float(sma_200),
        })

    if not candidates:
        return []

    # Normalisation min-max pour chaque métrique
    df_cand = pd.DataFrame(candidates)

    for col in ["mom_r2", "rvol", "obv_slope"]:
        col_min = df_cand[col].min()
        col_max = df_cand[col].max()
        col_range = col_max - col_min
        if col_range > 0:
            df_cand[f"{col}_norm"] = (df_cand[col] - col_min) / col_range
        else:
            df_cand[f"{col}_norm"] = 0.5

    # Score composite
    df_cand["score"] = (
        W_MOM_R2 * df_cand["mom_r2_norm"] +
        W_RVOL   * df_cand["rvol_norm"] +
        W_OBV    * df_cand["obv_slope_norm"]
    )

    # Tri décroissant
    df_cand = df_cand.sort_values("score", ascending=False).reset_index(drop=True)
    df_cand["rank"] = range(1, len(df_cand) + 1)

    return df_cand.to_dict("records")


# ============================================================
# K ADAPTATIF — ATR% based
# ============================================================

def compute_adaptive_k(atr_14: float, prix: float) -> float:
    """
    Calcule le k du trailing stop adapté à la volatilité du ticker.

    Formule : k = K_MIN + (atr_pct × K_ADAPTIVE_COEFF), clampé entre K_MIN et K_MAX.

    Exemples :
        AAPL  : ATR=4$, prix=250$ → atr_pct=1.6% → k = 2.0 + 1.6×0.5 = 2.8
        NVDA  : ATR=8$, prix=130$ → atr_pct=6.2% → k = 2.0 + 6.2×0.5 = 5.1 → clamp → 4.0
        PG    : ATR=2$, prix=170$ → atr_pct=1.2% → k = 2.0 + 1.2×0.5 = 2.6
        ASML  : ATR=25€, prix=900€ → atr_pct=2.8% → k = 2.0 + 2.8×0.5 = 3.4
    """
    if prix <= 0:
        return 3.0  # fallback
    atr_pct = (atr_14 / prix) * 100
    k = K_MIN + atr_pct * K_ADAPTIVE_COEFF
    return max(K_MIN, min(K_MAX, round(k, 2)))


# ============================================================
# SIMULATION PORTFOLIO
# ============================================================

def run_ranking_backtest(
    top_n: int = TOP_N,
    k = 3.0,
    macro_mode: str = "off",
    start_date: str = None,
    end_date: str = None,
) -> dict:
    """
    Backtest principal avec ranking hebdomadaire.

    Chaque lundi :
        1. Score et classe tous les tickers éligibles
        2. Vend les positions qui ont quitté le top N
        3. Achète les nouveaux entrants du top N
    En continu :
        4. Trailing stop ATR sur chaque position

    Paramètre k :
        float (ex. 3.0)    → k fixe pour toutes les positions
        "adaptive"         → k calculé par ticker via ATR% à l'entrée

    Paramètre macro_mode :
        "off"    → pas de filtre macro (comportement actuel)
        "cut"    → 0 position si la zone du ticker est bearish
        "reduce" → max 2 positions par zone bearish (au lieu de 5)

    Retourne les métriques détaillées du portefeuille.
    """
    adaptive_mode = (k == "adaptive")
    k_label = "adaptive (ATR%)" if adaptive_mode else str(k)
    print(f"🚀 Backtest Ranking v4.0 — top {top_n}, k={k_label}, macro={macro_mode}")
    print(f"   Pondérations : Mom R² {W_MOM_R2:.0%} | RVOL {W_RVOL:.0%} | OBV {W_OBV:.0%}")
    if adaptive_mode:
        print(f"   K adaptatif : k = {K_MIN} + atr_pct × {K_ADAPTIVE_COEFF}, clampé [{K_MIN}, {K_MAX}]")

    # 1. Charger tout
    print("   📥 Chargement des tickers...")
    all_tickers = load_all_tickers()
    print(f"   {len(all_tickers)} tickers avec historique suffisant.")

    print("   📥 Chargement des prix...")
    ticker_data = load_all_price_data(all_tickers)
    print(f"   {len(ticker_data)} tickers chargés.")

    print("   📥 Chargement mapping sectoriel...")
    secteur_mapping = load_secteur_mapping()
    force_data      = load_all_secteur_force()
    print(f"   {len(secteur_mapping)} tickers mappés, {len(force_data)} secteurs-zones chargés.")

    # Chargement macro (indices + SMA 200)
    macro_data = {}
    if macro_mode != "off":
        print("   📥 Chargement données macro (indices SMA200)...")
        macro_data = load_macro_data()
        print(f"   {len(macro_data)} zones macro chargées : {list(macro_data.keys())}")

    # 2. Calculer les indicateurs pour chaque ticker
    print("   📊 Calcul des indicateurs...")
    for ticker in ticker_data:
        ticker_data[ticker] = compute_all_indicators(ticker_data[ticker])
    print("   Indicateurs calculés.")

    # 3. Construire la timeline des lundis
    all_dates = set()
    for df in ticker_data.values():
        all_dates.update(df.index)
    all_dates = sorted(all_dates)

    if start_date:
        all_dates = [d for d in all_dates if d >= pd.Timestamp(start_date)]
    if end_date:
        all_dates = [d for d in all_dates if d <= pd.Timestamp(end_date)]

    # On commence après MOMENTUM_WINDOW jours pour avoir les indicateurs
    if len(all_dates) > MOMENTUM_WINDOW + 50:
        all_dates = all_dates[MOMENTUM_WINDOW + 50:]

    mondays = [d for d in all_dates if d.weekday() == 0]

    if not mondays:
        return {"erreur": "Pas assez de données pour le backtest."}

    print(f"   📅 {len(mondays)} lundis de {mondays[0].date()} à {mondays[-1].date()}")

    # 4. Simulation
    capital        = CAPITAL_INITIAL
    cash           = CAPITAL_INITIAL
    positions      = {}   # {ticker: {entry_price, entry_date, shares, max_price, stop, atr_entry}}
    all_trades     = []
    portfolio_history = []
    weekly_rankings   = []

    position_size = CAPITAL_INITIAL / top_n  # taille par position

    for i, monday in enumerate(mondays):
        # --- A. Vérifier les trailing stops sur toutes les positions ---
        # On vérifie chaque jour de la semaine précédente
        if i > 0:
            prev_monday = mondays[i - 1]
            week_days = [d for d in all_dates if prev_monday < d <= monday]

            for day in week_days:
                tickers_to_close = []
                for ticker, pos in positions.items():
                    if ticker not in ticker_data:
                        continue
                    df_t = ticker_data[ticker]
                    if day not in df_t.index:
                        continue

                    current_price = float(df_t.loc[day, "prix_ajuste"])
                    current_atr   = float(df_t.loc[day, "atr_14"]) if not pd.isna(df_t.loc[day, "atr_14"]) else pos["atr_entry"]

                    # Mise à jour du plus haut
                    if current_price > pos["max_price"]:
                        pos["max_price"] = current_price

                    # Nouveau stop (ne peut que monter)
                    pos_k = pos["k"]
                    new_stop = pos["max_price"] - pos_k * current_atr
                    pos["stop"] = max(pos["stop"], new_stop)

                    # Vérification sortie
                    if current_price <= pos["stop"]:
                        ret_pct = (current_price - pos["entry_price"]) / pos["entry_price"] * 100
                        cash += pos["shares"] * current_price
                        all_trades.append({
                            "ticker"      : ticker,
                            "entry_date"  : str(pos["entry_date"].date()),
                            "exit_date"   : str(day.date()),
                            "exit_reason" : "TRAILING_STOP",
                            "entry_price" : round(pos["entry_price"], 2),
                            "exit_price"  : round(current_price, 2),
                            "shares"      : round(pos["shares"], 4),
                            "pnl_eur"     : round(pos["shares"] * (current_price - pos["entry_price"]), 2),
                            "return_pct"  : round(ret_pct, 2),
                            "duration_days": (day - pos["entry_date"]).days,
                            "k_used"      : pos["k"],
                        })
                        tickers_to_close.append(ticker)

                for t in tickers_to_close:
                    del positions[t]

        # --- B. Ranking hebdomadaire ---
        ranking = compute_composite_score(ticker_data, monday, secteur_mapping, force_data)

        # --- B2. Filtre macro — exclure ou réduire les zones bearish ---
        macro_regime = get_macro_regime(macro_data, monday) if macro_mode != "off" else {}

        if ranking and macro_mode == "cut":
            # Exclure les tickers dont la zone est bearish
            ranking = [r for r in ranking if macro_regime.get(get_ticker_zone(r["ticker"], secteur_mapping), True)]

        if ranking:
            effective_top_n = top_n
            if macro_mode == "reduce":
                # Compter combien de zones sont bearish
                nb_bearish_zones = sum(1 for z, bull in macro_regime.items() if not bull)
                if nb_bearish_zones > 0:
                    effective_top_n = max(2, top_n - nb_bearish_zones)  # réduire de 1 par zone bearish, min 2

            top_tickers = [r["ticker"] for r in ranking[:effective_top_n]]
            weekly_rankings.append({
                "date"          : str(monday.date()),
                "nb_eligible"   : len(ranking),
                "macro_regime"  : macro_regime if macro_mode != "off" else None,
                "effective_top_n": effective_top_n if macro_mode == "reduce" else top_n,
                "top"           : [{"ticker": r["ticker"], "score": round(r["score"], 4),
                                    "mom_r2": round(r["mom_r2"], 4), "rvol": round(r["rvol"], 2),
                                    "rank": r["rank"]} for r in ranking[:effective_top_n]],
            })
        else:
            top_tickers = []
            weekly_rankings.append({
                "date"        : str(monday.date()),
                "nb_eligible" : 0,
                "macro_regime": macro_regime if macro_mode != "off" else None,
                "top"         : [],
            })

        # --- C. Vendre les positions qui ne sont plus dans le top N ---
        tickers_to_sell = [t for t in positions if t not in top_tickers]

        # En mode "cut", vendre aussi les positions dans les zones bearish
        if macro_mode == "cut":
            for t in list(positions.keys()):
                zone = get_ticker_zone(t, secteur_mapping)
                if not macro_regime.get(zone, True) and t not in tickers_to_sell:
                    tickers_to_sell.append(t)
        for ticker in tickers_to_sell:
            pos = positions[ticker]
            df_t = ticker_data.get(ticker)
            if df_t is not None and monday in df_t.index:
                exit_price = float(df_t.loc[monday, "prix_ajuste"])
            else:
                # Prendre le dernier prix connu
                valid = df_t[df_t.index <= monday] if df_t is not None else pd.DataFrame()
                exit_price = float(valid["prix_ajuste"].iloc[-1]) if not valid.empty else pos["entry_price"]

            ret_pct = (exit_price - pos["entry_price"]) / pos["entry_price"] * 100
            cash += pos["shares"] * exit_price
            # Déterminer la raison de sortie
            ticker_zone = get_ticker_zone(ticker, secteur_mapping)
            if macro_mode == "cut" and not macro_regime.get(ticker_zone, True):
                exit_reason = "MACRO_BEARISH"
            else:
                exit_reason = "RANKING_OUT"

            all_trades.append({
                "ticker"      : ticker,
                "entry_date"  : str(pos["entry_date"].date()),
                "exit_date"   : str(monday.date()),
                "exit_reason" : exit_reason,
                "entry_price" : round(pos["entry_price"], 2),
                "exit_price"  : round(exit_price, 2),
                "shares"      : round(pos["shares"], 4),
                "pnl_eur"     : round(pos["shares"] * (exit_price - pos["entry_price"]), 2),
                "return_pct"  : round(ret_pct, 2),
                "duration_days": (monday - pos["entry_date"]).days,
                "k_used"      : pos["k"],
            })
            del positions[ticker]

        # --- D. Acheter les nouveaux entrants du top N ---
        for ticker in top_tickers:
            if ticker in positions:
                continue  # déjà en portefeuille
            if len(positions) >= top_n:
                break  # portefeuille plein

            df_t = ticker_data.get(ticker)
            if df_t is None or monday not in df_t.index:
                continue

            entry_price = float(df_t.loc[monday, "prix_ajuste"])
            atr_val     = float(df_t.loc[monday, "atr_14"]) if not pd.isna(df_t.loc[monday, "atr_14"]) else entry_price * 0.02

            # K adaptatif ou fixe selon le mode
            if adaptive_mode:
                pos_k = compute_adaptive_k(atr_val, entry_price)
            else:
                pos_k = k

            # Investir une fraction du capital
            invest = min(position_size, cash)
            if invest < 100:  # minimum pour ouvrir
                continue

            shares = invest / entry_price
            cash  -= invest

            positions[ticker] = {
                "entry_price" : entry_price,
                "entry_date"  : monday,
                "shares"      : shares,
                "max_price"   : entry_price,
                "stop"        : entry_price - pos_k * atr_val,
                "atr_entry"   : atr_val,
                "k"           : pos_k,
            }

        # --- E. Valeur du portefeuille ---
        portfolio_value = cash
        for ticker, pos in positions.items():
            df_t = ticker_data.get(ticker)
            if df_t is not None:
                valid = df_t[df_t.index <= monday]
                if not valid.empty:
                    portfolio_value += pos["shares"] * float(valid["prix_ajuste"].iloc[-1])

        portfolio_history.append({
            "date"          : str(monday.date()),
            "portfolio_value": round(portfolio_value, 2),
            "cash"          : round(cash, 2),
            "nb_positions"  : len(positions),
            "positions"     : list(positions.keys()),
        })

        # Log mensuel
        if i % 4 == 0:
            pct = round((portfolio_value / CAPITAL_INITIAL - 1) * 100, 1)
            print(f"   📅 {monday.date()} | Portefeuille: {portfolio_value:,.0f}€ ({pct:+.1f}%) | Positions: {len(positions)} | Cash: {cash:,.0f}€")

    # --- F. Fermer les positions restantes ---
    last_date = all_dates[-1] if all_dates else mondays[-1]
    for ticker, pos in list(positions.items()):
        df_t = ticker_data.get(ticker)
        if df_t is not None:
            valid = df_t[df_t.index <= last_date]
            exit_price = float(valid["prix_ajuste"].iloc[-1]) if not valid.empty else pos["entry_price"]
        else:
            exit_price = pos["entry_price"]

        ret_pct = (exit_price - pos["entry_price"]) / pos["entry_price"] * 100
        all_trades.append({
            "ticker"      : ticker,
            "entry_date"  : str(pos["entry_date"].date()),
            "exit_date"   : str(last_date.date()) + " (FINAL)",
            "exit_reason" : "END_OF_BACKTEST",
            "entry_price" : round(pos["entry_price"], 2),
            "exit_price"  : round(exit_price, 2),
            "shares"      : round(pos["shares"], 4),
            "pnl_eur"     : round(pos["shares"] * (exit_price - pos["entry_price"]), 2),
            "return_pct"  : round(ret_pct, 2),
            "duration_days": (last_date - pos["entry_date"]).days,
            "k_used"      : pos["k"],
        })

    # --- G. Métriques globales ---
    final_value   = portfolio_history[-1]["portfolio_value"] if portfolio_history else CAPITAL_INITIAL
    total_return  = round((final_value / CAPITAL_INITIAL - 1) * 100, 2)
    nb_trades     = len(all_trades)

    wins    = [t for t in all_trades if t["return_pct"] > 0]
    losses  = [t for t in all_trades if t["return_pct"] <= 0]
    win_rate = round(len(wins) / nb_trades * 100, 1) if nb_trades > 0 else 0

    avg_win  = round(np.mean([t["return_pct"] for t in wins]), 2) if wins else 0
    avg_loss = round(np.mean([t["return_pct"] for t in losses]), 2) if losses else 0

    # Drawdown
    values = [h["portfolio_value"] for h in portfolio_history]
    peak       = values[0]
    max_dd     = 0
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100
        if dd > max_dd:
            max_dd = dd
    max_dd = round(max_dd, 2)

    # Sharpe (annualisé, sur rendements hebdomadaires)
    if len(values) > 2:
        weekly_returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
        mean_r = np.mean(weekly_returns)
        std_r  = np.std(weekly_returns)
        sharpe = round(mean_r / (std_r + 1e-9) * np.sqrt(52), 3)  # annualisé
    else:
        sharpe = 0

    # Trades par raison de sortie
    exit_reasons = {}
    for t in all_trades:
        reason = t["exit_reason"]
        if reason not in exit_reasons:
            exit_reasons[reason] = {"count": 0, "avg_return": []}
        exit_reasons[reason]["count"] += 1
        exit_reasons[reason]["avg_return"].append(t["return_pct"])

    for reason in exit_reasons:
        rets = exit_reasons[reason]["avg_return"]
        exit_reasons[reason]["avg_return"] = round(np.mean(rets), 2)
        exit_reasons[reason]["win_rate"] = round(sum(1 for r in rets if r > 0) / len(rets) * 100, 1)

    # Tickers les plus fréquents dans le top N
    ticker_frequency = {}
    for wr in weekly_rankings:
        for entry in wr["top"]:
            t = entry["ticker"]
            ticker_frequency[t] = ticker_frequency.get(t, 0) + 1

    top_tickers_freq = sorted(ticker_frequency.items(), key=lambda x: -x[1])[:20]

    result = {
        "version"    : "v4.0-ranking",
        "parametres" : {
            "top_n"          : top_n,
            "k_atr"          : k_label,
            "k_adaptive"     : adaptive_mode,
            "k_range"        : f"[{K_MIN}, {K_MAX}]" if adaptive_mode else None,
            "capital_initial" : CAPITAL_INITIAL,
            "weights"        : {"mom_r2": W_MOM_R2, "rvol": W_RVOL, "obv": W_OBV},
            "period"         : f"{mondays[0].date()} → {mondays[-1].date()}",
            "nb_weeks"       : len(mondays),
        },
        "metriques"  : {
            "total_return_pct"  : total_return,
            "final_value"       : final_value,
            "sharpe_ratio"      : sharpe,
            "max_drawdown_pct"  : max_dd,
            "nb_trades"         : nb_trades,
            "win_rate_pct"      : win_rate,
            "avg_win_pct"       : avg_win,
            "avg_loss_pct"      : avg_loss,
            "profit_factor"     : round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else 0,
        },
        "exit_reasons"       : exit_reasons,
        "top_tickers_freq"   : [{"ticker": t, "weeks_in_top": w} for t, w in top_tickers_freq],
        "trades"             : sorted(all_trades, key=lambda t: t["entry_date"]),
        "portfolio_history"  : portfolio_history,
        "sample_rankings"    : weekly_rankings[:5] + weekly_rankings[-5:],  # premiers et derniers pour diagnostic
    }

    print(f"\n{'='*60}")
    print(f"📋 SYNTHÈSE BACKTEST RANKING v4.0")
    print(f"{'='*60}")
    print(f"   Total Return  : {total_return:+.1f}%")
    print(f"   Sharpe Ratio  : {sharpe}")
    print(f"   Max Drawdown  : {max_dd:.1f}%")
    print(f"   Trades        : {nb_trades} (Win rate: {win_rate}%)")
    print(f"   Avg Win/Loss  : {avg_win:+.1f}% / {avg_loss:.1f}%")
    print(f"   Profit Factor : {result['metriques']['profit_factor']}")

    return result


# ============================================================
# MULTI-K TEST
# ============================================================

def run_ranking_multi_k(top_n: int = TOP_N) -> dict:
    """
    Teste le backtest ranking avec plusieurs valeurs de k fixes + le mode adaptatif.
    Retourne les métriques comparées.
    """
    results = {}

    # Tests k fixes
    for k_val in K_VALUES:
        print(f"\n{'='*60}")
        print(f"🔄 Test k={k_val}")
        print(f"{'='*60}")
        results[str(k_val)] = run_ranking_backtest(top_n=top_n, k=k_val)

    # Test k adaptatif
    print(f"\n{'='*60}")
    print(f"🔄 Test k=adaptive (ATR%)")
    print(f"{'='*60}")
    results["adaptive"] = run_ranking_backtest(top_n=top_n, k="adaptive")

    # Résumé comparatif
    summary = {}
    for k_str, res in results.items():
        m = res["metriques"]
        summary[k_str] = {
            "sharpe"       : m["sharpe_ratio"],
            "total_return" : m["total_return_pct"],
            "max_drawdown" : m["max_drawdown_pct"],
            "nb_trades"    : m["nb_trades"],
            "win_rate"     : m["win_rate_pct"],
            "profit_factor": m["profit_factor"],
        }

    best_k = max(summary, key=lambda k: summary[k]["sharpe"])

    print(f"\n{'='*60}")
    print(f"📋 COMPARATIF K VALUES")
    print(f"{'='*60}")
    for k_str, s in summary.items():
        marker = " ← BEST" if k_str == best_k else ""
        print(f"   k={k_str} → Sharpe={s['sharpe']} | Return={s['total_return']}% | MaxDD={s['max_drawdown']}% | Trades={s['nb_trades']}{marker}")

    return {
        "summary": summary,
        "best_k" : best_k,
        "details": results,
    }


# ============================================================
# POINT D'ENTRÉE — appelé par l'endpoint FastAPI
# ============================================================

def run_backtest_ranking_logic(top_n: int = TOP_N, k = None, macro: str = "off") -> dict:
    """
    Point d'entrée pour le endpoint /run-backtest-ranking.
    - k=None         → teste k fixes (2.5, 3.0, 3.5) + adaptatif
    - k=3.0          → teste uniquement k=3.0
    - k="adaptive"   → teste uniquement le mode adaptatif ATR%
    - macro="off"    → pas de filtre macro
    - macro="cut"    → exclure les zones bearish du ranking
    - macro="reduce" → réduire le top N pour les zones bearish
    """
    if k is not None:
        if k == "adaptive":
            return run_ranking_backtest(top_n=top_n, k="adaptive", macro_mode=macro)
        return run_ranking_backtest(top_n=top_n, k=float(k), macro_mode=macro)
    else:
        return run_ranking_multi_k(top_n=top_n)
