import os
import time
import pickle
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, BackgroundTasks
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Import de ta fonction d'entraînement
try:
    from train_model import train_brain
except ImportError:
    def train_brain(): 
        print("⚠️ Fonction train_brain non trouvée.")

load_dotenv()

app = FastAPI()

# --- Configuration DB ---
DATABASE_URL = os.getenv("DATABASE_URL")
engine = None

if DATABASE_URL:
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    if DATABASE_URL.endswith("/") and "?" not in DATABASE_URL:
        DATABASE_URL = DATABASE_URL[:-1]
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

@app.get("/")
def home():
    return {
        "status": "Service Trading IA Actif", 
        "version": "4.9.4", 
        "features": ["Safe-Columns-Check", "Chunk-Processing", "Memory-Optimized"]
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
                time.sleep(1)
            except Exception:
                continue
    except Exception as e:
        print(f"❌ Erreur Sync: {e}")

# --- LOGIQUE ANALYSE MASSIVE (CHUNK MODE + SAFE INDEX) ---
def run_analysis_logic():
    if engine is None: return
    print("🚀 Démarrage de l'analyse V4.9.4 (Optimisation & Sécurité)...")
    
    try:
        # 1. Récupérer TOUS les tickers uniques
        with engine.connect() as conn:
            result = conn.execute(text("SELECT DISTINCT ticker FROM actions_prix_historique"))
            all_tickers = [row[0] for row in result]
        
        if not all_tickers: return
        print(f"📈 {len(all_tickers)} tickers à traiter.")

        # 2. Charger le modèle IA une seule fois
        model, model_cols = None, None
        with engine.connect() as conn:
            res_model = conn.execute(text("SELECT model_data, columns_data FROM models_store WHERE model_name='trading_forest'")).fetchone()
            if res_model:
                model = pickle.loads(res_model[0])
                model_cols = pickle.loads(res_model[1])

        chunk_size = 20
        total_processed_rows = 0
        # Liste des colonnes techniques attendues
        tech_cols = ['rsi_14', 'vol_avg_20', 'sma_200', 'bb_lower', 'target_ml']

        for i in range(0, len(all_tickers), chunk_size):
            current_chunk = all_tickers[i:i + chunk_size]
            print(f"📦 Paquet {i//chunk_size + 1}/{len(all_tickers)//chunk_size + 1} ({current_chunk[0]}...)")

            query = text("""
                SELECT a.id, a.ticker, a.date, a.prix_ajuste, a.volume, 
                       t.secteur, t.market_cap, t.pe_ratio 
                FROM actions_prix_historique a 
                LEFT JOIN tickers_info t ON a.ticker = t.ticker
                WHERE a.ticker IN :tickers
                ORDER BY a.ticker, a.date ASC
            """)
            
            df = pd.read_sql(query, engine, params={"tickers": tuple(current_chunk)})
            if df.empty: continue

            df['date'] = pd.to_datetime(df['date']).dt.date
            df['prix_ajuste'] = pd.to_numeric(df['prix_ajuste'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            df = df.sort_values(['ticker', 'date']).dropna(subset=['prix_ajuste'])

            # --- Calculs techniques sécurisés ---
            def compute_group(group):
                if len(group) < 2: return group 
                
                delta = group['prix_ajuste'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
                rs = gain / (loss + 1e-9)
                group['rsi_14'] = 100 - (100 / (1 + rs))
                
                group['sma_200'] = group['prix_ajuste'].rolling(window=200, min_periods=1).mean()
                group['vol_avg_20'] = group['volume'].rolling(window=20, min_periods=1).mean()
                
                sma_20 = group['prix_ajuste'].rolling(window=20, min_periods=1).mean()
                std_20 = group['prix_ajuste'].rolling(window=20, min_periods=1).std()
                group['bb_lower'] = sma_20 - (std_20 * 2)
                
                group['prix_futur'] = group['prix_ajuste'].shift(-7)
                group['target_ml'] = ((group['prix_futur'] - group['prix_ajuste']) / group['prix_ajuste'] >= 0.05).astype(float)
                return group

            df = df.groupby('ticker', group_keys=False).apply(compute_group)

            # --- SÉCURITÉ : Forcer la présence des colonnes même si calcul impossible ---
            for col in tech_cols:
                if col not in df.columns:
                    df[col] = None

            # --- Scoring IA ---
            if model and not df.empty:
                # Préparation des features
                feat_list = ['rsi_14', 'vol_avg_20', 'sma_200', 'bb_lower', 'secteur', 'market_cap', 'pe_ratio']
                features_df = df[feat_list].copy()
                features_df = pd.get_dummies(features_df, columns=['secteur'])
                
                for col in model_cols:
                    if col not in features_df.columns:
                        features_df[col] = 0
                
                # Predict
                df['score_ia'] = model.predict_proba(features_df[model_cols].fillna(0))[:, 1]
            else:
                df['score_ia'] = 0

            # Signal final (RSI fillna à 100 pour éviter les faux signaux sur données vides)
            df['signal_achat'] = ((df['rsi_14'].fillna(100) < 35) & (df['score_ia'] >= 0.5))

            # --- Sauvegarde par ID ---
            df_save = df[['id', 'rsi_14', 'vol_avg_20', 'sma_200', 'bb_lower', 'signal_achat', 'score_ia', 'target_ml']].copy()
            df_save.to_sql("_tmp_chunk", engine, if_exists='replace', index=False)
            
            with engine.begin() as conn:
                conn.execute(text("""
                    UPDATE actions_prix_historique a
                    SET rsi_14 = t.rsi_14, vol_avg_20 = t.vol_avg_20, sma_200 = t.sma_200, 
                        bb_lower = t.bb_lower, signal_achat = t.signal_achat, 
                        score_ia = t.score_ia, target_ml = t.target_ml
                    FROM _tmp_chunk t WHERE a.id = t.id;
                """))
            
            total_processed_rows += len(df)
            print(f"🟢 {total_processed_rows} lignes traitées avec succès.")

        print(f"🏁 ANALYSE COMPLÈTE TERMINÉE.")

    except Exception as e:
        print(f"❌ Erreur critique : {e}")
