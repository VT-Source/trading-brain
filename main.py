import io
import os
import time
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
from fastapi import FastAPI, BackgroundTasks
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

try:
    from train_model import train_brain
except ImportError:
    def train_brain():
        print("⚠️ Fonction train_brain non trouvée dans train_model.py")

try:
    from backtest import run_backtest_logic
except ImportError:
    def run_backtest_logic(**kwargs):
        return {"erreur": "backtest.py non trouvé"}
try:
    from backtest_ranking import run_backtest_ranking_logic
except ImportError:
    def run_backtest_ranking_logic(**kwargs):
         return {"erreur": "backtest_ranking.py non trouvé"}

load_dotenv()

app = FastAPI()

# --- CONFIGURATION DATABASE ---
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True) if DATABASE_URL else None

# ============================================================
# Contexte nécessaire pour chaque indicateur glissant :
#   SMA_200  → 200 jours
#   RSI_14   → 14 jours
#   BB_lower → 20 jours
#   vol_avg  → 20 jours
#   ATR_14   → 14 jours (réel si prix_haut/prix_bas disponibles)
# On charge 220 jours pour avoir la SMA_200 correcte + marge.
# On ne sauvegarde que les 5 derniers jours (nouveaux / modifiés).
# ============================================================
INCREMENTAL_LOOKBACK_DAYS = 220
INCREMENTAL_SAVE_DAYS     = 5


# ============================================================
# ENDPOINTS API
# ============================================================

@app.get("/")
def home():
    return {
        "status"          : "Service Trading IA Actif",
        "version"         : "5.9.0",
        "engine_connected": engine is not None
    }

@app.get("/check-model")
def check_model():
    """Diagnostic — vérifie si le modèle est présent en base."""
    if engine is None:
        return {"error": "engine non connecté"}
    try:
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT model_name, accuracy, updated_at,
                       LENGTH(model_data)   AS model_size_bytes,
                       LENGTH(columns_data) AS cols_size_bytes
                FROM models_store
                WHERE model_name = 'trading_forest'
            """)).fetchone()
        if row:
            return {
                "model_found"      : True,
                "model_name"       : row[0],
                "accuracy"         : f"{round(row[1] * 100, 2)}%",
                "updated_at"       : str(row[2]),
                "model_size_bytes" : row[3],
                "cols_size_bytes"  : row[4]
            }
        return {"model_found": False, "message": "Aucun modèle en base — appeler /train-model"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/train-model")
async def trigger_training(background_tasks: BackgroundTasks):
    background_tasks.add_task(train_brain)
    return {"status": "processing", "message": "Entraînement lancé en arrière-plan."}

@app.get("/sync-metadata")
async def trigger_sync(background_tasks: BackgroundTasks):
    background_tasks.add_task(sync_metadata_logic)
    return {"status": "processing", "message": "Synchronisation Yahoo lancée."}

@app.get("/sync-etf-sectoriels")
async def trigger_sync_etf(background_tasks: BackgroundTasks, full: bool = False):
    """
    Synchronise les prix des ETF sectoriels et indices de référence.
    - full=False (défaut) : 30 derniers jours — appel quotidien scheduler 06h15
    - full=True           : 5 ans d'historique — initialisation manuelle uniquement
    """
    background_tasks.add_task(sync_secteurs_etf_logic, full=full)
    mode = "COMPLÈTE (5 ans)" if full else "incrémentale (30j)"
    return {
        "status" : "processing",
        "message": f"Sync ETF sectoriels lancée en arrière-plan — mode {mode}."
    }

@app.get("/secteurs-actifs")
def get_secteurs_actifs_endpoint():
    """
    Retourne les secteurs actuellement en force relative.
    Utilisé par le dashboard Streamlit (Onglet 1 & 3).
    """
    secteurs = get_secteurs_en_force()
    return {
        "date"              : str(date.today()),
        "nb_secteurs_actifs": len(secteurs),
        "secteurs"          : secteurs
    }

@app.get("/fill-high-low")
async def trigger_fill_high_low(background_tasks: BackgroundTasks):
    """
    Remplit prix_haut et prix_bas pour tout l'historique depuis yfinance.
    À appeler UNE SEULE FOIS après déploiement de cette version.
    Durée estimée : 20-40 minutes selon le nombre de tickers.
    ⚠️ Action manuelle uniquement — NE PAS planifier dans le scheduler.
    """
    background_tasks.add_task(fill_high_low_logic)
    return {
        "status" : "processing",
        "message": "⚠️ Remplissage prix_haut/prix_bas lancé sur tout l'historique (action manuelle unique)."
    }

@app.get("/run-analysis")
async def trigger_analysis(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_analysis_logic, full=False)
    return {
        "status" : "processing",
        "message": f"Analyse incrémentale lancée (contexte {INCREMENTAL_LOOKBACK_DAYS}j, sauvegarde {INCREMENTAL_SAVE_DAYS}j)."
    }

@app.get("/run-analysis-full")
async def trigger_full_analysis(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_analysis_logic, full=True)
    return {
        "status" : "processing",
        "message": "⚠️ Recalcul COMPLET lancé sur tout l'historique (action manuelle)."
    }

@app.get("/backtest-detail")
def backtest_detail(ticker: str = "MSFT", k: float = 2.0, horizon: int = 30):
    """
    Retourne le détail de chaque trade pour un ticker donné.
    Synchrone — résultat immédiat dans le navigateur.
    """
    from backtest import (load_ticker_data, load_secteur_force,
                          compute_signals_v35, build_exit_signals)
    try:
        df         = load_ticker_data(ticker)
        df_secteur = load_secteur_force(ticker)
        df_signals = compute_signals_v35(df, df_secteur)

        price      = df_signals["prix_ajuste"]
        entries    = df_signals["signal_achat"]
        atr        = df_signals["atr_14"]
        exits      = build_exit_signals(entries, price, atr, k)
        atr_type   = "réel (H-L)" if df["prix_haut"].notna().any() else "approché (C-C)"

        trades      = []
        in_trade    = False
        entry_date  = None
        entry_price = None

        for i in range(len(price)):
            if not in_trade and entries.iloc[i]:
                in_trade    = True
                entry_date  = price.index[i]
                entry_price = float(price.iloc[i])
            elif in_trade and exits.iloc[i]:
                exit_date  = price.index[i]
                exit_price = float(price.iloc[i])
                ret_pct    = round((exit_price - entry_price) / entry_price * 100, 2)
                loc        = price.index.get_loc(entry_date)
                prix_jh    = float(price.iloc[loc + horizon]) if loc + horizon < len(price) else None
                row        = df_signals.loc[entry_date]
                trades.append({
                    "entry_date"       : str(entry_date.date()),
                    "exit_date"        : str(exit_date.date()),
                    "duree_jours"      : (exit_date - entry_date).days,
                    "prix_entree"      : entry_price,
                    "prix_sortie"      : exit_price,
                    "rendement_pct"    : ret_pct,
                    "resultat"         : "✅ WIN" if ret_pct > 0 else "❌ LOSS",
                    f"prix_j{horizon}" : round(prix_jh, 2) if prix_jh else None,
                    "contexte_signal"  : {
                        "tendance_ok"     : bool(row["tendance_ok"]),
                        "force_rel"       : bool(row["force_rel"]),
                        "mom_r2"          : round(float(row["mom_r2"]), 4),
                        "breakout_20j"    : bool(row["breakout_20j"]),
                        "rvol"            : round(float(row["rvol"]), 2),
                        "obv_accumulation": bool(row["obv_accumulation"]),
                        "atr_14"          : round(float(row["atr_14"]), 3),
                        "atr_type"        : atr_type,
                        "sma_200"         : round(float(row["sma_200"]), 2),
                    }
                })
                in_trade = False

        # Signaux sans sortie (position encore ouverte)
        for i in range(len(price)):
            if entries.iloc[i]:
                date_signal = price.index[i]
                if not any(t["entry_date"] == str(date_signal.date()) for t in trades):
                    row = df_signals.loc[date_signal]
                    trades.append({
                        "entry_date"   : str(date_signal.date()),
                        "exit_date"    : "OUVERT",
                        "prix_entree"  : float(price.iloc[i]),
                        "rendement_pct": round((float(price.iloc[-1]) - float(price.iloc[i])) / float(price.iloc[i]) * 100, 2),
                        "resultat"     : "🔄 OUVERT",
                        "contexte_signal": {
                            "tendance_ok"     : bool(row["tendance_ok"]),
                            "force_rel"       : bool(row["force_rel"]),
                            "mom_r2"          : round(float(row["mom_r2"]), 4),
                            "breakout_20j"    : bool(row["breakout_20j"]),
                            "rvol"            : round(float(row["rvol"]), 2),
                            "obv_accumulation": bool(row["obv_accumulation"]),
                        }
                    })

        return {
            "ticker"    : ticker,
            "k"         : k,
            "atr_type"  : atr_type,
            "nb_signaux": int(entries.sum()),
            "nb_trades" : len(trades),
            "win_rate"  : round(sum(1 for t in trades if "WIN" in t.get("resultat", "")) / len(trades) * 100, 1) if trades else 0,
            "trades"    : sorted(trades, key=lambda t: t["entry_date"])
        }

    except Exception as e:
        return {"erreur": str(e)}

@app.get("/schema-diagnostic")
def schema_diagnostic():
    """
    Retourne le schéma de chaque table + pourcentage de valeurs NULL par colonne.
    À partager dans le projet Claude pour contextualiser le modèle de données.
    """
    if engine is None:
        return {"error": "engine non connecté"}

    tables = [
        "actions_prix_historique",
        "tickers_info",
        "positions",
        "signaux_log",
        "models_store",
        "secteurs_etf",
        "secteurs_etf_prix",
        "indices_prix"
    ]

    result = {}

    with engine.connect() as conn:
        for table in tables:
            # 1. Schéma
            schema_rows = conn.execute(text("""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_name = :table
                ORDER BY ordinal_position
            """), {"table": table}).fetchall()

            if not schema_rows:
                result[table] = {"erreur": "table absente"}
                continue

            columns = [r[0] for r in schema_rows]
            schema  = [
                {
                    "colonne" : r[0],
                    "type"    : r[1],
                    "nullable": r[2],
                    "defaut"  : r[3]
                }
                for r in schema_rows
            ]

            # 2. Nombre total de lignes
            try:
                total = conn.execute(text(
                    f"SELECT COUNT(*) FROM {table}"
                )).scalar()
            except Exception:
                total = None

            # 3. Pourcentage NULL par colonne
            null_stats = {}
            if total and total > 0:
                for col in columns:
                    try:
                        null_count = conn.execute(text(
                            f"SELECT COUNT(*) FROM {table} WHERE {col} IS NULL"
                        )).scalar()
                        null_stats[col] = {
                            "null_count": null_count,
                            "null_pct"  : round(100.0 * null_count / total, 1)
                        }
                    except Exception:
                        null_stats[col] = {"erreur": "calcul impossible"}

            result[table] = {
                "nb_lignes" : total,
                "schema"    : schema,
                "null_stats": null_stats
            }

    return result

@app.get("/run-backtest")
async def trigger_backtest(
    background_tasks: BackgroundTasks,
    tickers: str = "NVDA,AAPL,MSFT,AMZN,ASML",
    horizon: int = 30
):
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    background_tasks.add_task(run_backtest_logic, tickers=ticker_list, horizon=horizon)
    return {
        "status" : "processing",
        "message": f"Backtest v3.5 lancé en arrière-plan — {len(ticker_list)} tickers, horizon={horizon}j.",
        "tickers": ticker_list
    }

@app.get("/run-backtest-ranking")
async def trigger_backtest_ranking(
    background_tasks: BackgroundTasks,
    top_n: int = 5,
    k: str = None,
    macro: str = "off"
):
    """
    Backtest momentum ranking — top N positions, réévaluation hebdo.
    - top_n : nombre de positions simultanées (défaut 5)
    - k     : "adaptive" pour ATR%, ou un float (ex. 3.0), ou rien pour multi-k
    - macro : "off" (défaut), "cut" (exclure zones bearish), "reduce" (réduire top N)
    Appels :
      GET /run-backtest-ranking                              → multi-k + adaptive, sans macro
      GET /run-backtest-ranking?k=adaptive&macro=cut         → k adaptatif + filtre macro hard
      GET /run-backtest-ranking?k=3.0&macro=cut              → k=3.0 + filtre macro hard
      GET /run-backtest-ranking?k=adaptive&macro=reduce      → k adaptatif + macro soft
    """
    if k is not None and k != "adaptive":
        try:
            k = float(k)
        except ValueError:
            return {"error": f"k doit être un nombre ou 'adaptive', reçu: {k}"}
    if macro not in ("off", "cut", "reduce"):
        return {"error": f"macro doit être 'off', 'cut' ou 'reduce', reçu: {macro}"}
    background_tasks.add_task(run_backtest_ranking_logic, top_n=top_n, k=k, macro=macro)
    k_label = "adaptive (ATR%)" if k == "adaptive" else f"k={k}" if k else "multi-k + adaptive"
    return {
        "status" : "processing",
        "message": f"Backtest ranking v4.0 lancé — top {top_n}, {k_label}, macro={macro}.",
    }
 
        
# ============================================================
# CHARGEMENT DU MODÈLE DEPUIS POSTGRESQL
# ============================================================

def load_model_from_db():
    """
    Charge le modèle ML depuis models_store (PostgreSQL).
    Résiste aux redéploiements Railway (filesystem éphémère).
    Retourne (model, model_cols) ou (None, None) si absent.
    """
    if engine is None:
        return None, None
    try:
        with engine.connect() as conn:
                        row = conn.execute(text("""
                SELECT model_data, columns_data, accuracy, updated_at
                FROM models_store
                WHERE model_name = 'trading_forest'
            """)).fetchone()

        if row:
            model      = joblib.load(io.BytesIO(bytes(row[0])))
            model_cols = joblib.load(io.BytesIO(bytes(row[1])))
            print(f"✅ Modèle ML chargé depuis DB (précision : {round(float(row[2]) * 100, 1)}%, entraîné le {row[3].date()})")
            return model, model_cols

        print("⚠️ Aucun modèle trouvé en base — score_ia sera 0.0. Appeler /train-model.")
        return None, None

    except Exception as e:
        print(f"❌ Erreur chargement modèle depuis DB : {e}")
        return None, None


# ============================================================
# LOGIQUE SYNC METADATA
# ============================================================

def sync_metadata_logic():
    if engine is None: return
    print("🔄 Sync Yahoo Finance (Metadata)...")
    try:
        with engine.connect() as conn:
            result  = conn.execute(text("SELECT DISTINCT ticker FROM actions_prix_historique LIMIT 500;"))
            tickers = [row[0] for row in result]

        for ticker_symbol in tickers:
            try:
                stock = yf.Ticker(ticker_symbol)
                data  = stock.info
                if "symbol" in data:
                    metadata = {
                        "ticker"    : ticker_symbol,
                        "name"      : data.get("longName"),
                        "secteur"   : data.get("sector"),
                        "industrie" : data.get("industry"),
                        "pays"      : data.get("country"),
                        "monnaie"   : data.get("currency"),
                        "market_cap": data.get("marketCap"),
                        "pe_ratio"  : data.get("trailingPE")
                    }
                    with engine.begin() as conn:
                        conn.execute(text("""
                            INSERT INTO tickers_info
                                (ticker, name, secteur, industrie, pays, monnaie,
                                 market_cap, pe_ratio, derniere_maj)
                            VALUES
                                (:ticker, :name, :secteur, :industrie, :pays, :monnaie,
                                 :market_cap, :pe_ratio, CURRENT_DATE)
                            ON CONFLICT (ticker) DO UPDATE SET
                                pe_ratio     = EXCLUDED.pe_ratio,
                                market_cap   = EXCLUDED.market_cap,
                                derniere_maj = CURRENT_DATE;
                        """), metadata)
                time.sleep(1)
            except Exception:
                continue
    except Exception as e:
        print(f"❌ Erreur Sync: {e}")


# ============================================================
# LOGIQUE FILL HIGH/LOW — ACTION MANUELLE UNIQUE
# ============================================================

def fill_high_low_logic():
    """
    Remplit prix_haut, prix_bas et prix_ouverture pour tout l'historique depuis yfinance.
    Correction : Utilise la même transaction (conn) pour to_sql et l'UPDATE.
    """
    if engine is None:
        print("❌ fill_high_low_logic : engine non connecté.")
        return

    print("📥 Remplissage prix_haut / prix_bas / prix_ouverture — historique complet...")

    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT DISTINCT ticker FROM actions_prix_historique ORDER BY ticker"))
            all_tickers = [row[0] for row in result]

        print(f"   {len(all_tickers)} tickers à traiter.")

        chunk_size = 20
        total_chunks = (len(all_tickers) - 1) // chunk_size + 1
        total_updated = 0

        for i in range(0, len(all_tickers), chunk_size):
            chunk = all_tickers[i:i + chunk_size]
            chunk_updated = 0

            for ticker_symbol in chunk:
                try:
                    df_yf = yf.download(
                        ticker_symbol,
                        period="5y",
                        interval="1d",
                        auto_adjust=True,
                        progress=False
                    )

                    if df_yf.empty:
                        continue

                    df_yf = df_yf.reset_index()
                    df_yf.columns = [c[0] if isinstance(c, tuple) else c for c in df_yf.columns]
                    df_yf = df_yf.rename(columns={
                        "Date": "date",
                        "High": "prix_haut",
                        "Low": "prix_bas",
                        "Open": "prix_ouverture",
                    })

                    df_yf["date"] = pd.to_datetime(df_yf["date"]).dt.date
                    df_yf["ticker"] = ticker_symbol

                    cols_needed = ["ticker", "date", "prix_haut", "prix_bas", "prix_ouverture"]
                    df_to_save = df_yf[[c for c in cols_needed if c in df_yf.columns]].dropna(subset=["prix_haut", "prix_bas"])

                    if df_to_save.empty:
                        continue

                    # On nettoie le nom de la table (minuscules et sans caractères spéciaux)
                    clean_ticker = ticker_symbol.lower().replace('.', '_').replace('-', '_').replace('^', '')
                    tmp_table_name = f"_tmp_hl_{clean_ticker}"

                    # --- CORE FIX: TOUTE L'OPÉRATION DANS LA MÊME TRANSACTION ---
                    with engine.begin() as conn:
                        # 1. On écrit les données dans la table temporaire en utilisant la connexion active
                        df_to_save.to_sql(tmp_table_name, conn, if_exists="replace", index=False)

                        # 2. On fait l'UPDATE
                        conn.execute(text(f"""
                            UPDATE actions_prix_historique a SET
                                prix_haut      = t.prix_haut,
                                prix_bas       = t.prix_bas,
                                prix_ouverture = t.prix_ouverture
                            FROM {tmp_table_name} t
                            WHERE a.ticker = t.ticker
                              AND a.date   = t.date::date
                        """))
                        
                        # 3. On supprime la table
                        conn.execute(text(f"DROP TABLE IF EXISTS {tmp_table_name}"))

                    chunk_updated += len(df_to_save)
                    time.sleep(0.5)

                except Exception as e:
                    print(f"   ❌ {ticker_symbol} : {e}")
                    continue

            total_updated += chunk_updated
            print(f"   🟢 Chunk {i // chunk_size + 1}/{total_chunks} terminé.")

        print(f"🏁 Fill high/low terminé — {total_updated} lignes mises à jour.")

    except Exception as e:
        print(f"❌ Erreur fill_high_low_logic : {e}")


# ============================================================
# LOGIQUE SYNC ETF SECTORIELS
# ============================================================

def sync_secteurs_etf_logic(full: bool = False):
    """
    Télécharge et persiste les prix des ETF sectoriels + indices de référence.
    Calcule le ratio de force relative et le ratio vs MM50 pour chaque ETF.

    Paramètres
    ----------
    full : bool
        False → 30 derniers jours (mode quotidien, scheduler)
        True  → 5 ans d'historique (initialisation manuelle)
    """
    if engine is None:
        print("❌ sync_secteurs_etf_logic : engine non connecté.")
        return

    mode   = "COMPLET (5 ans)" if full else "INCRÉMENTAL (30j)"
    period = "5y" if full else "1mo"
    print(f"🔄 Sync ETF sectoriels — mode {mode}...")

    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT ticker_etf, indice_reference
                FROM secteurs_etf
                WHERE actif = TRUE
            """)).fetchall()

        if not rows:
            print("⚠️ Aucun ETF trouvé dans secteurs_etf — exécuter secteurs_etf.sql d'abord.")
            return

        etf_list   = [r[0] for r in rows]
        etf_to_idx = {r[0]: r[1] for r in rows}
        indices    = list(set(r[1] for r in rows))

        print(f"   {len(etf_list)} ETF sectoriels + {len(indices)} indices à synchroniser.")

        print("   📥 Téléchargement indices de référence...")
        for idx_ticker in indices:
            try:
                df_idx = yf.download(
                    idx_ticker, period=period, interval="1d",
                    auto_adjust=True, progress=False
                )
                if df_idx.empty:
                    print(f"   ⚠️ Aucune donnée pour l'indice {idx_ticker}")
                    continue

                df_idx = df_idx.reset_index()
                df_idx.columns = [c[0] if isinstance(c, tuple) else c for c in df_idx.columns]
                df_idx = df_idx.rename(columns={"Date": "date", "Close": "prix_cloture", "Adj Close": "prix_ajuste"})
                if "prix_ajuste" not in df_idx.columns:
                    df_idx["prix_ajuste"] = df_idx["prix_cloture"]
                df_idx["date"]          = pd.to_datetime(df_idx["date"]).dt.date
                df_idx["ticker_indice"] = idx_ticker
                records = df_idx[["ticker_indice", "date", "prix_cloture", "prix_ajuste"]].to_dict("records")

                with engine.begin() as conn:
                    conn.execute(text("""
                        INSERT INTO indices_prix (ticker_indice, date, prix_cloture, prix_ajuste)
                        VALUES (:ticker_indice, :date, :prix_cloture, :prix_ajuste)
                        ON CONFLICT (ticker_indice, date) DO UPDATE SET
                            prix_cloture = EXCLUDED.prix_cloture,
                            prix_ajuste  = EXCLUDED.prix_ajuste
                    """), records)

                print(f"   ✅ Indice {idx_ticker} — {len(records)} jours enregistrés.")
                time.sleep(1)

            except Exception as e:
                print(f"   ❌ Erreur indice {idx_ticker} : {e}")
                continue

        print("   📥 Téléchargement ETF sectoriels...")
        for etf_ticker in etf_list:
            try:
                df_etf = yf.download(
                    etf_ticker, period=period, interval="1d",
                    auto_adjust=True, progress=False
                )
                if df_etf.empty:
                    print(f"   ⚠️ Aucune donnée pour {etf_ticker}")
                    continue

                df_etf = df_etf.reset_index()
                df_etf.columns = [c[0] if isinstance(c, tuple) else c for c in df_etf.columns]
                df_etf = df_etf.rename(columns={
                    "Date": "date", "Close": "prix_cloture",
                    "Adj Close": "prix_ajuste", "Volume": "volume"
                })
                if "prix_ajuste" not in df_etf.columns:
                    df_etf["prix_ajuste"] = df_etf["prix_cloture"]
                if "volume" not in df_etf.columns:
                    df_etf["volume"] = 0
                df_etf["date"]       = pd.to_datetime(df_etf["date"]).dt.date
                df_etf["ticker_etf"] = etf_ticker

                idx_ticker = etf_to_idx[etf_ticker]
                with engine.connect() as conn:
                    df_indice = pd.read_sql(text("""
                        SELECT date, prix_ajuste AS prix_indice
                        FROM indices_prix WHERE ticker_indice = :idx ORDER BY date ASC
                    """), conn, params={"idx": idx_ticker})

                if df_indice.empty:
                    print(f"   ⚠️ Pas de données indice {idx_ticker} pour {etf_ticker} — skip.")
                    continue

                df_indice["date"] = pd.to_datetime(df_indice["date"]).dt.date
                df_merged         = df_etf.merge(df_indice, on="date", how="inner")

                first_etf = df_merged["prix_ajuste"].iloc[0]
                first_idx = df_merged["prix_indice"].iloc[0]
                df_merged["ratio_force_relative"] = (
                    (df_merged["prix_ajuste"] / first_etf) /
                    (df_merged["prix_indice"] / first_idx)
                )
                df_merged["mm50_ratio"]        = df_merged["ratio_force_relative"].rolling(50, min_periods=1).mean()
                df_merged["ratio_vs_mm50"]     = df_merged["ratio_force_relative"] / df_merged["mm50_ratio"]
                df_merged["en_force_relative"] = df_merged["ratio_vs_mm50"] > 1.0

                records = df_merged[[
                    "ticker_etf", "date", "prix_cloture", "prix_ajuste", "volume",
                    "prix_indice", "ratio_force_relative", "ratio_vs_mm50", "en_force_relative"
                ]].to_dict("records")

                for r in records:
                    r["en_force_relative"] = bool(r["en_force_relative"])
                    r["volume"]            = int(r["volume"]) if r["volume"] is not None else 0

                with engine.begin() as conn:
                    conn.execute(text("""
                        INSERT INTO secteurs_etf_prix
                            (ticker_etf, date, prix_cloture, prix_ajuste, volume,
                             prix_indice, ratio_force_relative, ratio_vs_mm50, en_force_relative)
                        VALUES
                            (:ticker_etf, :date, :prix_cloture, :prix_ajuste, :volume,
                             :prix_indice, :ratio_force_relative, :ratio_vs_mm50, :en_force_relative)
                        ON CONFLICT (ticker_etf, date) DO UPDATE SET
                            prix_cloture         = EXCLUDED.prix_cloture,
                            prix_ajuste          = EXCLUDED.prix_ajuste,
                            volume               = EXCLUDED.volume,
                            prix_indice          = EXCLUDED.prix_indice,
                            ratio_force_relative = EXCLUDED.ratio_force_relative,
                            ratio_vs_mm50        = EXCLUDED.ratio_vs_mm50,
                            en_force_relative    = EXCLUDED.en_force_relative,
                            updated_at           = CURRENT_TIMESTAMP
                    """), records)

                statut_fr = "OUI ✅" if df_merged["en_force_relative"].iloc[-1] else "NON ❌"
                ratio_val = df_merged["ratio_vs_mm50"].iloc[-1]
                print(f"   ✅ {etf_ticker} — {len(records)} jours | Force relative : {statut_fr} | ratio_vs_mm50 = {ratio_val:.3f}")
                time.sleep(1)

            except Exception as e:
                print(f"   ❌ Erreur ETF {etf_ticker} : {e}")
                continue

        print("🏁 Sync ETF sectoriels terminée.")

    except Exception as e:
        print(f"❌ Erreur sync_secteurs_etf_logic : {e}")


# ============================================================
# HELPER : Secteurs en force relative
# ============================================================

def get_secteurs_en_force() -> list[dict]:
    """
    Retourne la liste des secteurs Yahoo actuellement en force relative.
    Interroge la vue v_secteurs_en_force (dernière date disponible).
    Si la table est vide ou absente → retourne [] sans planter run_analysis_logic.
    """
    if engine is None:
        return []
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT secteur_yahoo, zone, ticker_etf, indice_reference,
                       date, ratio_force_relative, ratio_vs_mm50
                FROM v_secteurs_en_force
                ORDER BY zone, ratio_vs_mm50 DESC
            """)).fetchall()

        return [
            {
                "secteur_yahoo"       : r[0],
                "zone"                : r[1],
                "ticker_etf"          : r[2],
                "indice_reference"    : r[3],
                "date"                : str(r[4]),
                "ratio_force_relative": float(r[5]) if r[5] is not None else None,
                "ratio_vs_mm50"       : float(r[6]) if r[6] is not None else None,
            }
            for r in rows
        ]
    except Exception as e:
        print(f"⚠️ get_secteurs_en_force : {e} — retour liste vide.")
        return []


# ============================================================
# LOGIQUE ANALYSE — INCRÉMENTALE ET COMPLÈTE
# ============================================================

def run_analysis_logic(full: bool = False):
    if engine is None: return

    mode = "COMPLET" if full else "INCRÉMENTAL"
    print(f"🚀 Démarrage Analyse v5.9.0 — mode {mode}...")

    try:
        # 1. Liste des tickers
        with engine.connect() as conn:
            result      = conn.execute(text("SELECT DISTINCT ticker FROM actions_prix_historique"))
            all_tickers = [row[0] for row in result]

        if not all_tickers: return
        print(f"   {len(all_tickers)} tickers trouvés en base.")

        # 2. Chargement du modèle ML depuis PostgreSQL
        model, model_cols = load_model_from_db()

        # 3. Définition de la fenêtre de chargement
        if full:
            date_filter = ""
            date_params = {}
        else:
            date_filter = "AND a.date >= :date_from"
            date_params = {
                "date_from": (pd.Timestamp.today() - pd.Timedelta(days=INCREMENTAL_LOOKBACK_DAYS)).date()
            }

        # 4. Traitement par chunks
        chunk_size   = 50
        total_chunks = (len(all_tickers) - 1) // chunk_size + 1

        for i in range(0, len(all_tickers), chunk_size):
            tickers_chunk = all_tickers[i:i + chunk_size]

            # prix_haut et prix_bas inclus pour ATR réel (remplis par /fill-high-low)
            query = text(f"""
                SELECT a.id, a.ticker, a.date,
                       a.prix_cloture, a.prix_ajuste, a.volume,
                       a.prix_haut, a.prix_bas,
                       t.secteur, t.market_cap, t.pe_ratio
                FROM actions_prix_historique a
                LEFT JOIN tickers_info t ON a.ticker = t.ticker
                WHERE a.ticker IN :tickers
                {date_filter}
                ORDER BY a.ticker, a.date ASC
            """)

            params = {"tickers": tuple(tickers_chunk), **date_params}
            df     = pd.read_sql(query, engine, params=params)
            if df.empty: continue

            df['prix_ajuste'] = df['prix_ajuste'].fillna(df['prix_cloture'])
            df = df.dropna(subset=['prix_ajuste']).sort_values(['ticker', 'date'])

            def compute_indicators(group):
                if len(group) < 50: return group
                price = group['prix_ajuste']
                vol   = group['volume']

                # RSI 14
                delta = price.diff()
                gain  = delta.where(delta > 0, 0).rolling(14).mean()
                loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
                group['rsi_14'] = 100 - (100 / (1 + gain / (loss + 1e-9)))

                # Moyennes mobiles
                group['sma_200'] = price.rolling(200, min_periods=1).mean()
                group['sma_50']  = price.rolling(50,  min_periods=1).mean()

                # Bandes de Bollinger
                sma_20               = price.rolling(20).mean()
                std_20               = price.rolling(20).std()
                bb_upper             = sma_20 + (std_20 * 2)
                group['bb_lower']    = sma_20 - (std_20 * 2)
                bb_range             = (bb_upper - group['bb_lower']).replace(0, np.nan)
                group['bb_position'] = (price - group['bb_lower']) / bb_range

                # Volume moyen 20j
                group['vol_avg_20'] = vol.rolling(20).mean()

                # Features ML
                group['rsi_slope']   = group['rsi_14'].diff(3)
                group['vol_ratio']   = vol / (group['vol_avg_20'] + 1e-9)
                group['dist_sma200'] = (price - group['sma_200']) / (group['sma_200'] + 1e-9)

                # ATR 14 — réel (H-L) si disponible, approché (C-C) sinon
                has_hl = (
                    'prix_haut' in group.columns and
                    'prix_bas'  in group.columns and
                    group['prix_haut'].notna().any() and
                    group['prix_bas'].notna().any()
                )
                if has_hl:
                    prev_close = price.shift(1)
                    tr = pd.concat([
                        group['prix_haut'] - group['prix_bas'],
                        (group['prix_haut'] - prev_close).abs(),
                        (group['prix_bas']  - prev_close).abs()
                    ], axis=1).max(axis=1)
                else:
                    tr = price.diff().abs()

                group['atr_14'] = tr.rolling(14, min_periods=1).mean()

                # Régime de marché
                group['regime_marche'] = np.where(
                    price > group['sma_50'] * 1.02, 'BULL',
                    np.where(price < group['sma_50'] * 0.98, 'BEAR', 'NEUTRE')
                )

                # ============================================================
                # SIGNAL ACHAT — v3.3 (obsolète, conservé en production)
                # ⚠️ À remplacer par signal v3.5 après validation backtest
                # ============================================================
                group['signal_achat'] = (
                    (group['rsi_14'] < 35) &
                    (price > group['sma_200']) &
                    (vol > group['vol_avg_20']) &
                    (price < group['bb_lower'])
                )
                return group

            df = df.groupby('ticker', group_keys=False).apply(compute_indicators)

            # 5. Score ML
            if model is not None:
                feat_df = pd.get_dummies(df, columns=['secteur', 'regime_marche'])
                for col in model_cols:
                    if col not in feat_df.columns:
                        feat_df[col] = 0
                X_input            = feat_df[model_cols].fillna(0).replace([np.inf, -np.inf], 0)
                df['confiance_ml'] = model.predict_proba(X_input)[:, 1]
            else:
                df['confiance_ml'] = 0.0

            # 6. Filtrage des lignes à sauvegarder
            if not full:
                save_from  = pd.Timestamp.today() - pd.Timedelta(days=INCREMENTAL_SAVE_DAYS)
                df_to_save = df[pd.to_datetime(df['date']) >= save_from].copy()
            else:
                df_to_save = df.copy()

            if df_to_save.empty:
                continue

            # 7. Sauvegarde — table temporaire unique par chunk
            tmp_table    = f"_tmp_update_{i}"
            cols_to_save = [
                'id', 'rsi_14', 'sma_200', 'bb_lower', 'bb_position',
                'vol_avg_20', 'regime_marche', 'signal_achat', 'confiance_ml',
                'rsi_slope', 'vol_ratio', 'dist_sma200', 'atr_14'
            ]
            df_update = df_to_save[[c for c in cols_to_save if c in df_to_save.columns]].copy()
            df_update.to_sql(tmp_table, engine, if_exists='replace', index=False)

            with engine.begin() as conn:
                conn.execute(text(f"""
                    UPDATE actions_prix_historique a SET
                        rsi_14        = t.rsi_14,
                        sma_200       = t.sma_200,
                        bb_lower      = t.bb_lower,
                        bb_position   = t.bb_position,
                        vol_avg_20    = t.vol_avg_20,
                        regime_marche = t.regime_marche,
                        signal_achat  = t.signal_achat,
                        score_ia      = t.confiance_ml,
                        rsi_slope     = t.rsi_slope,
                        vol_ratio     = t.vol_ratio,
                        dist_sma200   = t.dist_sma200,
                        atr_14        = t.atr_14
                    FROM {tmp_table} t
                    WHERE a.id = t.id
                """))
                conn.execute(text(f"DROP TABLE IF EXISTS {tmp_table}"))

            rows_saved = len(df_update)
            print(f"🟢 Chunk {i // chunk_size + 1}/{total_chunks} — {rows_saved} lignes sauvegardées.")

        print(f"🏁 Analyse {mode} terminée.")

    except Exception as e:
        print(f"❌ Erreur run_analysis_logic (full={full}) : {e}")
