import io
import os
import time
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from fastapi import FastAPI, BackgroundTasks
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

try:
    from train_model import train_brain
except ImportError:
    def train_brain():
        print("⚠️ Fonction train_brain non trouvée dans train_model.py")

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
        "version"         : "5.5.1",
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
# LOGIQUE ANALYSE — INCRÉMENTALE ET COMPLÈTE
# ============================================================

def run_analysis_logic(full: bool = False):
    if engine is None: return

    mode = "COMPLET" if full else "INCRÉMENTAL"
    print(f"🚀 Démarrage Analyse v5.5.1 — mode {mode}...")

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
                # SIGNAL ACHAT — 4 conditions AT (PROJECT_STATUS v3.1)
                # 1. RSI < 35            : survente
                # 2. Prix > SMA_200      : tendance haussière long terme
                # 3. Volume > vol_avg_20 : confirmation d'intérêt marché
                # 4. Prix < BB_lower     : opportunité de rebond Bollinger
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
