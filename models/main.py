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
        "version"         : "5.7.0",
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
                SELECT model_name, precision, created_at,
                       LENGTH(model_data) AS model_size_bytes,
                       LENGTH(cols_data)  AS cols_size_bytes
                FROM models_store
                WHERE model_name = 'trading_forest'
            """)).fetchone()
        if row:
            return {
                "model_found"      : True,
                "model_name"       : row[0],
                "precision"        : f"{round(row[1] * 100, 2)}%",
                "trained_at"       : str(row[2]),
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

# --- Endpoint ETF sectoriels ---
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

# --- Endpoint quotidien (scheduler 06h00) ---
# Charge 220 jours de contexte, ne sauvegarde que les 5 derniers jours.
@app.get("/run-analysis")
async def trigger_analysis(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_analysis_logic, full=False)
    return {
        "status" : "processing",
        "message": f"Analyse incrémentale lancée (contexte {INCREMENTAL_LOOKBACK_DAYS}j, sauvegarde {INCREMENTAL_SAVE_DAYS}j)."
    }

# --- Endpoint recalcul complet (action manuelle uniquement) ---
# À appeler après un changement de logique AT, ajout d'indicateur,
# ou pour initialiser un nouveau ticker.
# NE PAS planifier dans le scheduler.
@app.get("/run-analysis-full")
async def trigger_full_analysis(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_analysis_logic, full=True)
    return {
        "status" : "processing",
        "message": "⚠️ Recalcul COMPLET lancé sur tout l'historique (action manuelle)."
    }


# --- Endpoint backtest (action manuelle uniquement) ---
# Tickers par défaut : NVDA, AAPL, MSFT, AMZN, ASML
# Paramètres optionnels : tickers=NVDA,AAPL&horizon=30
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
                SELECT model_data, cols_data, precision, created_at
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
        # 1. Récupérer tous les ETF actifs + leurs indices de référence
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

        # --------------------------------------------------------
        # 2. Télécharger les indices de référence
        # --------------------------------------------------------
        print("   📥 Téléchargement indices de référence...")
        for idx_ticker in indices:
            try:
                df_idx = yf.download(
                    idx_ticker,
                    period=period,
                    interval="1d",
                    auto_adjust=True,
                    progress=False
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

        # --------------------------------------------------------
        # 3. Télécharger et traiter les ETF sectoriels
        # --------------------------------------------------------
        print("   📥 Téléchargement ETF sectoriels...")
        for etf_ticker in etf_list:
            try:
                df_etf = yf.download(
                    etf_ticker,
                    period=period,
                    interval="1d",
                    auto_adjust=True,
                    progress=False
                )
                if df_etf.empty:
                    print(f"   ⚠️ Aucune donnée pour {etf_ticker}")
                    continue

                df_etf = df_etf.reset_index()
                df_etf.columns = [c[0] if isinstance(c, tuple) else c for c in df_etf.columns]
                df_etf = df_etf.rename(columns={
                    "Date"      : "date",
                    "Close"     : "prix_cloture",
                    "Adj Close" : "prix_ajuste",
                    "Volume"    : "volume"
                })

                if "prix_ajuste" not in df_etf.columns:
                    df_etf["prix_ajuste"] = df_etf["prix_cloture"]
                if "volume" not in df_etf.columns:
                    df_etf["volume"] = 0

                df_etf["date"]       = pd.to_datetime(df_etf["date"]).dt.date
                df_etf["ticker_etf"] = etf_ticker

                # ------------------------------------------------
                # 4. Récupérer le prix de l'indice de référence
                # ------------------------------------------------
                idx_ticker = etf_to_idx[etf_ticker]

                with engine.connect() as conn:
                    df_indice = pd.read_sql(text("""
                        SELECT date, prix_ajuste AS prix_indice
                        FROM indices_prix
                        WHERE ticker_indice = :idx
                        ORDER BY date ASC
                    """), conn, params={"idx": idx_ticker})

                if df_indice.empty:
                    print(f"   ⚠️ Pas de données indice {idx_ticker} pour {etf_ticker} — skip.")
                    continue

                df_indice["date"] = pd.to_datetime(df_indice["date"]).dt.date

                # ------------------------------------------------
                # 5. Calcul ratio de force relative
                # ------------------------------------------------
                df_merged = df_etf.merge(df_indice, on="date", how="inner")

                # Normalisation à la première date disponible
                first_etf_price = df_merged["prix_ajuste"].iloc[0]
                first_idx_price = df_merged["prix_indice"].iloc[0]

                df_merged["ratio_force_relative"] = (
                    (df_merged["prix_ajuste"] / first_etf_price) /
                    (df_merged["prix_indice"] / first_idx_price)
                )

                # MM50 du ratio (décision 2026-03-17 — plus réactif aux rotations sectorielles)
                # min_periods=1 : évite les NaN en mode incrémental (< 50 jours chargés)
                df_merged["mm50_ratio"]         = df_merged["ratio_force_relative"].rolling(50, min_periods=1).mean()
                df_merged["ratio_vs_mm50"]      = df_merged["ratio_force_relative"] / df_merged["mm50_ratio"]
                df_merged["en_force_relative"]  = df_merged["ratio_vs_mm50"] > 1.0

                # ------------------------------------------------
                # 6. Upsert dans secteurs_etf_prix
                # ------------------------------------------------
                cols_to_save = [
                    "ticker_etf", "date", "prix_cloture", "prix_ajuste",
                    "volume", "prix_indice", "ratio_force_relative",
                    "ratio_vs_mm50", "en_force_relative"
                ]
                records = df_merged[cols_to_save].to_dict("records")

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
# Utilisée par run_analysis_logic (Phase 1 Étape 3) et /secteurs-actifs
# ============================================================

def get_secteurs_en_force() -> list[dict]:
    """
    Retourne la liste des secteurs Yahoo actuellement en force relative.
    Interroge la vue v_secteurs_en_force (dernière date disponible).

    Retourne une liste de dicts :
      [{"secteur_yahoo": "Technology", "zone": "US", "ticker_etf": "XLK", ...}, ...]

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
    print(f"🚀 Démarrage Analyse v5.6.0 — mode {mode}...")

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

            query = text(f"""
                SELECT a.id, a.ticker, a.date, a.prix_cloture, a.prix_ajuste, a.volume,
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
                sma_20            = price.rolling(20).mean()
                std_20            = price.rolling(20).std()
                bb_upper          = sma_20 + (std_20 * 2)
                group['bb_lower'] = sma_20 - (std_20 * 2)
                bb_range          = (bb_upper - group['bb_lower']).replace(0, np.nan)
                group['bb_position'] = (price - group['bb_lower']) / bb_range

                # Volume moyen 20j
                group['vol_avg_20'] = vol.rolling(20).mean()

                # Features ML
                group['rsi_slope']   = group['rsi_14'].diff(3)
                group['vol_ratio']   = vol / (group['vol_avg_20'] + 1e-9)
                group['dist_sma200'] = (price - group['sma_200']) / (group['sma_200'] + 1e-9)

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
                X_input          = feat_df[model_cols].fillna(0).replace([np.inf, -np.inf], 0)
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
                'rsi_slope', 'vol_ratio', 'dist_sma200'
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
                        dist_sma200   = t.dist_sma200
                    FROM {tmp_table} t
                    WHERE a.id = t.id
                """))
                conn.execute(text(f"DROP TABLE IF EXISTS {tmp_table}"))

            rows_saved = len(df_update)
            print(f"🟢 Chunk {i // chunk_size + 1}/{total_chunks} — {rows_saved} lignes sauvegardées.")

        print(f"🏁 Analyse {mode} terminée.")

    except Exception as e:
        print(f"❌ Erreur run_analysis_logic (full={full}) : {e}")
