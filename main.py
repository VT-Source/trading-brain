import os
import time
import pickle
import pandas as pd
import numpy as np
import yfinance as yf
from fastapi import FastAPI, BackgroundTasks
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Import de ta fonction d'entraînement sécurisée
try:
    from train_model import train_brain
except ImportError:
    def train_brain(): 
        print("⚠️ Fonction train_brain non trouvée.")

load_dotenv()

app = FastAPI()

# --- Configuration DB ---
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True) if DATABASE_URL else None

@app.get("/")
def home():
    return {
        "status": "Service Trading IA Actif", 
        "version": "4.9.9", 
        "features": ["Price-Safety-Net", "Strict-SQL-Casting", "Memory-Optimized"]
    }

# --- ENDPOINTS ---
@app.get("/train-model")
async def trigger_training(background_tasks: BackgroundTasks):
    background_tasks.add_task(train_brain)
    return {"status": "processing", "message": "Entraînement lancé."}

@app.get("/sync-metadata")
async def trigger_sync(background_tasks: BackgroundTasks):
    background_tasks.add_task(sync_metadata_logic)
    return {"status": "processing", "message": "Sync Yahoo Finance lancée."}

@app.get("/run-analysis")
async def trigger_analysis(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_analysis_logic)
    return {"status": "processing", "message": "Analyse technique massive lancée..."}

# --- LOGIQUE SYNC METADATA ---
def sync_metadata_logic():
    if engine is None: return
    print("🔄 Sync Yahoo Finance (Metadata)...")
    query = """
        SELECT DISTINCT ticker FROM actions_prix_historique
        WHERE ticker NOT IN (SELECT ticker FROM tickers_info)
        OR ticker IN (SELECT ticker FROM tickers_info WHERE derniere_maj < CURRENT_DATE - INTERVAL '30 days')
        LIMIT 500;
    """
    try:
        with engine.connect() as conn:
            tickers = [row[0] for row in conn.execute(text(query))]

        for ticker_symbol in tickers:
            try:
                stock = yf.Ticker(ticker_symbol)
                data = stock.info
                if "symbol" in data:
                    metadata = {
                        "ticker": ticker_symbol,
                        "name": data.get("longName"),
                        "secteur": data.get("sector"),
                        "industrie": data.get("industry"),
                        "pays": data.get("country"),
                        "monnaie": data.get("currency"),
                        "market_cap": data.get("marketCap"),
                        "pe_ratio": data.get("trailingPE")
                    }
                    with engine.begin() as conn:
                        conn.execute(text("""
                            INSERT INTO tickers_info (ticker, name, secteur, industrie, pays, monnaie, market_cap, pe_ratio, derniere_maj)
                            VALUES (:ticker, :name, :secteur, :industrie, :pays, :monnaie, :market_cap, :pe_ratio, CURRENT_DATE)
                            ON CONFLICT (ticker) DO UPDATE SET 
                                pe_ratio = EXCLUDED.pe_ratio, market_cap = EXCLUDED.market_cap, derniere_maj = CURRENT_DATE;
                        """), metadata)
                time.sleep(1) # Courtoisie API
            except Exception:
                continue
    except Exception as e:
        print(f"❌ Erreur Sync: {e}")

# --- LOGIQUE ANALYSE MASSIVE (V4.9.9) ---
def run_analysis_logic():
    if engine is None: return
    print("🚀 Démarrage de l'analyse V4.9.9 (Sécurisée)...")
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT DISTINCT ticker FROM actions_prix_historique"))
            all_tickers = [row[0] for row in result]
        
        if not all_tickers: return
        
        # Chargement modèle IA
        model, model_cols = None, None
        with engine.connect() as conn:
            res_model = conn.execute(text("SELECT model_data, columns_data FROM models_store WHERE model_name='trading_forest'")).fetchone()
            if res_model:
                model = pickle.loads(res_model[0])
                model_cols = pickle.loads(res_model[1])

        chunk_size = 20
        total_rows = 0

        for i in range(0, len(all_tickers), chunk_size):
            tickers_chunk = all_tickers[i:i + chunk_size]
            
            # Récupération avec fallback prix_cloture si prix_ajuste est NULL
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

            # Sécurité Prix : On remplace le NULL ajusté par le brut
            df['prix_ajuste'] = df['prix_ajuste'].fillna(df['prix_cloture'])
            df['prix_ajuste'] = pd.to_numeric(df['prix_ajuste'], errors='coerce')
            df = df.dropna(subset=['prix_ajuste']).sort_values(['ticker', 'date'])

            # Calculs techniques
            def compute_group(group):
                if len(group) < 14: return group
                
                # RSI 14
                delta = group['prix_ajuste'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=7).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=7).mean()
                rs = gain / (loss + 1e-9)
                group['rsi_14'] = 100 - (100 / (1 + rs))
                
                # SMA & Vol
                group['sma_200'] = group['prix_ajuste'].rolling(window=200, min_periods=1).mean()
                group['vol_avg_20'] = group['volume'].rolling(window=20, min_periods=1).mean()
                
                # Bollinger
                sma_20 = group['prix_ajuste'].rolling(window=20, min_periods=1).mean()
                std_20 = group['prix_ajuste'].rolling(window=20, min_periods=1).std()
                group['bb_lower'] = sma_20 - (std_20 * 2)
                
                # Target
                group['prix_futur'] = group['prix_ajuste'].shift(-7)
                group['target_ml'] = ((group['prix_futur'] - group['prix_ajuste']) / group['prix_ajuste'] >= 0.05).astype(float)
                return group

            df = df.groupby('ticker', group_keys=False).apply(compute_group)

            # Scoring IA
            if model:
                # Préparation features (match avec colonnes d'entraînement)
                feat_df = pd.get_dummies(df[['rsi_14', 'vol_avg_20', 'sma_200', 'bb_lower', 'secteur', 'market_cap', 'pe_ratio']], columns=['secteur'])
                for col in model_cols:
                    if col not in feat_df.columns: feat_df[col] = 0
                df['score_ia'] = model.predict_proba(feat_df[model_cols].fillna(0))[:, 1]
            else:
                df['score_ia'] = 0.0

            df['signal_achat'] = ((df['rsi_14'].fillna(100) < 35) & (df['score_ia'] >= 0.5))

            # Sauvegarde sécurisée (Casting numeric pour Postgres)
            cols_to_save = ['id', 'rsi_14', 'vol_avg_20', 'sma_200', 'bb_lower', 'signal_achat', 'score_ia', 'target_ml']
            df_save = df[cols_to_save].copy()
            df_save.to_sql("_tmp_chunk", engine, if_exists='replace', index=False)
            
            with engine.begin() as conn:
                conn.execute(text("""
                    UPDATE actions_prix_historique a
                    SET rsi_14 = CAST(t.rsi_14 AS numeric), 
                        vol_avg_20 = CAST(t.vol_avg_20 AS numeric), 
                        sma_200 = CAST(t.sma_200 AS numeric), 
                        bb_lower = CAST(t.bb_lower AS numeric), 
                        signal_achat = t.signal_achat, 
                        score_ia = CAST(t.score_ia AS numeric), 
                        target_ml = CAST(t.target_ml AS numeric)
                    FROM _tmp_chunk t WHERE a.id = t.id;
                """))
            
            total_rows += len(df)
            print(f"🟢 Paquet {i//chunk_size + 1} traité ({total_rows} lignes).")

        print("🏁 ANALYSE TERMINÉE.")
    except Exception as e:
        print(f"❌ Erreur critique : {e}")
