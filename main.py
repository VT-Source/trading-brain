import os
import time
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from fastapi import FastAPI, BackgroundTasks
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Import de la fonction d'entraînement depuis train_model.py
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

MODEL_PATH = "models/trading_forest.joblib"
COLS_PATH  = "models/trading_forest_cols.joblib"

# ============================================================
# ENDPOINTS API
# ============================================================

@app.get("/")
def home():
    return {
        "status": "Service Trading IA Actif",
        "version": "5.3.1",
        "engine_connected": engine is not None
    }

@app.get("/train-model")
async def trigger_training(background_tasks: BackgroundTasks):
    background_tasks.add_task(train_brain)
    return {"status": "processing", "message": "Entraînement v3.3 lancé."}

@app.get("/sync-metadata")
async def trigger_sync(background_tasks: BackgroundTasks):
    background_tasks.add_task(sync_metadata_logic)
    return {"status": "processing", "message": "Synchronisation Yahoo lancée."}

@app.get("/run-analysis")
async def trigger_analysis(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_analysis_logic)
    return {"status": "processing", "message": "Analyse massive lancée..."}

# ============================================================
# LOGIQUE SYNC METADATA
# ============================================================

def sync_metadata_logic():
    if engine is None: return
    print("🔄 Sync Yahoo Finance (Metadata)...")
    query = "SELECT DISTINCT ticker FROM actions_prix_historique LIMIT 500;"
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))
            tickers = [row[0] for row in result]

        for ticker_symbol in tickers:
            try:
                stock = yf.Ticker(ticker_symbol)
                data  = stock.info
                if "symbol" in data:
                    metadata = {
                        "ticker"   : ticker_symbol,
                        "name"     : data.get("longName"),
                        "secteur"  : data.get("sector"),
                        "industrie": data.get("industry"),
                        "pays"     : data.get("country"),
                        "monnaie"  : data.get("currency"),
                        "market_cap": data.get("marketCap"),
                        "pe_ratio" : data.get("trailingPE")
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
# LOGIQUE ANALYSE MASSIVE
# ============================================================

def run_analysis_logic():
    if engine is None: return
    print("🚀 Démarrage Analyse v5.3.1...")

    try:
        # 1. Liste des tickers
        with engine.connect() as conn:
            result = conn.execute(text("SELECT DISTINCT ticker FROM actions_prix_historique"))
            all_tickers = [row[0] for row in result]

        if not all_tickers: return

        # 2. Chargement du modèle
        model, model_cols = None, None
        if os.path.exists(MODEL_PATH) and os.path.exists(COLS_PATH):
            model = joblib.load(MODEL_PATH)
            model_cols = joblib.load(COLS_PATH)
            print("✅ Modèle ML chargé.")

        # 3. Traitement par Chunks
        chunk_size = 50
        for i in range(0, len(all_tickers), chunk_size):
            tickers_chunk = all_tickers[i:i + chunk_size]

            query = text("""
                SELECT a.id, a.ticker, a.date, a.prix_cloture, a.prix_ajuste, a.volume,
                       t.secteur, t.market_cap, t.pe_ratio
                FROM actions_prix_historique a
                LEFT JOIN tickers_info t ON a.ticker = t.ticker
                WHERE a.ticker IN :tickers
                ORDER BY a.ticker, a.date ASC
            """)

            df = pd.read_sql(query, engine, params={"tickers": tuple(tickers_chunk)})
            if df.empty: continue

            df['prix_ajuste'] = df['prix_ajuste'].fillna(df['prix_cloture'])
            df = df.dropna(subset=['prix_ajuste']).sort_values(['ticker', 'date'])

            def compute_indicators(group):
                if len(group) < 50: return group
                price = group['prix_ajuste']
                vol = group['volume']

                # Indicateurs techniques
                delta = price.diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                group['rsi_14'] = 100 - (100 / (1 + gain / (loss + 1e-9)))
                group['sma_200'] = price.rolling(200, min_periods=1).mean()
                group['sma_50'] = price.rolling(50, min_periods=1).mean()
                
                std_20 = price.rolling(20).std()
                sma_20 = price.rolling(20).mean()
                group['bb_lower'] = sma_20 - (std_20 * 2)

                # Features ML
                group['rsi_slope'] = group['rsi_14'].diff(3)
                group['vol_ratio'] = vol / (vol.rolling(20).mean() + 1e-9)
                group['dist_sma200'] = (price - group['sma_200']) / (group['sma_200'] + 1e-9)

                # Régime de Marché
                group['regime_marche'] = np.where(
                    price > group['sma_50'] * 1.02, 'BULL',
                    np.where(price < group['sma_50'] * 0.98, 'BEAR', 'NEUTRE')
                )

                # Signal AT Durci
                group['signal_achat'] = (
                    (group['rsi_14'] < 30) & 
                    (price > group['sma_200']) & 
                    (group['regime_marche'] != 'BEAR')
                )
                return group

            df = df.groupby('ticker', group_keys=False).apply(compute_indicators)

            # 4. Score ML
            if model is not None:
                feat_df = pd.get_dummies(df, columns=['secteur', 'regime_marche'])
                for col in model_cols:
                    if col not in feat_df.columns: feat_df[col] = 0
                
                X_input = feat_df[model_cols].fillna(0).replace([np.inf, -np.inf], 0)
                df['confiance_ml'] = model.predict_proba(X_input)[:, 1]
            else:
                df['confiance_ml'] = 0.0

            # 5. Sauvegarde
            df_update = df[['id', 'rsi_14', 'sma_200', 'bb_lower', 'regime_marche', 'signal_achat', 'confiance_ml']].copy()
            df_update.to_sql("_tmp_update", engine, if_exists='replace', index=False)
            
            with engine.begin() as conn:
                conn.execute(text("""
                    UPDATE actions_prix_historique a SET
                        rsi_14 = t.rsi_14, sma_200 = t.sma_200, bb_lower = t.bb_lower,
                        regime_marche = t.regime_marche, signal_achat = t.signal_achat,
                        score_ia = t.confiance_ml
                    FROM _tmp_update t WHERE a.id = t.id
                """))
            print(f"🟢 Chunk {i//chunk_size + 1} traité.")

        print("🏁 Analyse massive terminée.")
    except Exception as e:
        print(f"❌ Erreur : {e}")
