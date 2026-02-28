import os
import time
import requests
import pickle
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from fastapi import FastAPI, BackgroundTasks
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Import de ta fonction d'entraînement
from train_model import train_brain

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
        "version": "4.6", 
        "features": ["Native-Pandas-Math", "Robust-RSI", "Anti-Deadlock"]
    }

# --- ENDPOINT 1 : ENTRAÎNEMENT ---
@app.get("/train-model")
async def trigger_training(background_tasks: BackgroundTasks):
    background_tasks.add_task(train_brain)
    return {"status": "processing", "message": "Entraînement du cerveau lancé en tâche de fond..."}

# --- ENDPOINT 2 : SYNC METADATA ---
@app.get("/sync-metadata")
async def trigger_sync(background_tasks: BackgroundTasks):
    background_tasks.add_task(sync_metadata_logic)
    return {"status": "processing", "message": "Synchronisation Yahoo Finance lancée..."}

# --- ENDPOINT 3 : ANALYSE + PRÉDICTION IA ---
@app.get("/run-analysis")
async def trigger_analysis(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_analysis_logic)
    return {"status": "processing", "message": "Analyse technique et Prédictions IA lancées..."}

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

                if "symbol" in data or "longName" in data:
                    m_cap_raw = data.get("marketCap")
                    pe_ratio_raw = data.get("trailingPE")

                    metadata = {
                        "ticker": ticker_symbol,
                        "name": data.get("longName"),
                        "secteur": data.get("sector"),
                        "industrie": data.get("industry"),
                        "pays": data.get("country"),
                        "monnaie": data.get("currency"),
                        "market_cap": int(m_cap_raw) if m_cap_raw else None,
                        "pe_ratio": float(pe_ratio_raw) if pe_ratio_raw else None
                    }
                    
                    with engine.begin() as conn:
                        conn.execute(text("""
                            INSERT INTO tickers_info (ticker, name, secteur, industrie, pays, monnaie, market_cap, pe_ratio, derniere_maj)
                            VALUES (:ticker, :name, :secteur, :industrie, :pays, :monnaie, :market_cap, :pe_ratio, CURRENT_DATE)
                            ON CONFLICT (ticker) DO UPDATE SET 
                                pe_ratio = EXCLUDED.pe_ratio, 
                                market_cap = EXCLUDED.market_cap, 
                                derniere_maj = CURRENT_DATE;
                        """), metadata)
                    print(f"✅ {ticker_symbol} synchronisé.")
                
                time.sleep(1)
            except Exception as e:
                print(f"⚠️ Erreur ticker {ticker_symbol}: {e}")
                
    except Exception as e:
        print(f"❌ Erreur Sync: {e}")

def run_analysis_logic():
    if engine is None: return
    print("🚀 Démarrage de l'analyse (Version 4.6 - Native Pandas Math)...")
    
    try:
        # 1. Charger les données
        query = "SELECT a.*, t.secteur, t.market_cap, t.pe_ratio FROM actions_prix_historique a LEFT JOIN tickers_info t ON a.ticker = t.ticker"
        df = pd.read_sql(query, engine)
        if df.empty: 
            print("⚠️ Base vide.")
            return

        # --- NETTOYAGE ET TRI ---
        df['date'] = pd.to_datetime(df['date']).dt.date
        df = df.sort_values(['ticker', 'date'])
        
        # SÉCURITÉ : Forcer le numérique et supprimer les prix nuls ou NaN
        df['prix_ajuste'] = pd.to_numeric(df['prix_ajuste'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        df = df.dropna(subset=['prix_ajuste'])

        print(f"📊 Analyse de {len(df)} lignes en cours...")

        # 2. CALCULS TECHNIQUES NATIFS (Sans bibliothèques externes pour éviter les bugs)
        
        # --- RSI NATIF ---
        def pandas_rsi(series, period=14):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        df['rsi_14'] = df.groupby('ticker')['prix_ajuste'].transform(lambda x: pandas_rsi(x))
        
        # --- SMA & VOLUME ---
        df['sma_200'] = df.groupby('ticker')['prix_ajuste'].transform(lambda x: x.rolling(window=200, min_periods=1).mean())
        df['vol_avg_20'] = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(window=20, min_periods=1).mean())
        
        # --- BOLLINGER NATIF ---
        df['sma_20'] = df.groupby('ticker')['prix_ajuste'].transform(lambda x: x.rolling(window=20).mean())
        df['std_20'] = df.groupby('ticker')['prix_ajuste'].transform(lambda x: x.rolling(window=20).std())
        df['bb_lower'] = df['sma_20'] - (df['std_20'] * 2)

        # 3. PRÉDICTION IA
        with engine.connect() as conn:
            res = conn.execute(text("SELECT model_data, columns_data FROM models_store WHERE model_name='trading_forest'")).fetchone()
        
        if res:
            print("🧠 Modèle trouvé ! Scoring...")
            model = pickle.loads(res[0])
            model_cols = pickle.loads(res[1])
            
            # Préparation des features
            features = df[['rsi_14', 'vol_avg_20', 'sma_200', 'bb_lower', 'secteur', 'market_cap', 'pe_ratio']].copy()
            features = pd.get_dummies(features, columns=['secteur'])
            for col in model_cols:
                if col not in features.columns: features[col] = 0
            
            df['score_ia'] = model.predict_proba(features[model_cols].fillna(0))[:, 1]
        else:
            df['score_ia'] = 0

        # 4. SIGNAL ET TARGET
        # Un signal d'achat se déclenche si RSI bas ET prix au-dessus de la SMA 200 ET score IA favorable
        df['signal_achat'] = (
            (df['rsi_14'] < 35) & 
            (df['prix_ajuste'] > df['sma_200']) & 
            (df['score_ia'] >= 0.7) # Seuil abaissé à 0.5 pour tester la réactivité
        ).fillna(False)

        print("🎯 Calcul du Target ML...")
        df['prix_futur'] = df.groupby('ticker')['prix_ajuste'].shift(-7)
        df['target_ml'] = ((df['prix_futur'] - df['prix_ajuste']) / df['prix_ajuste'] >= 0.05).astype(int)

        # 5. SAUVEGARDE
        print("💾 Sauvegarde finale...")
        cols_to_save = ['ticker', 'date', 'rsi_14', 'vol_avg_20', 'sma_200', 'bb_lower', 'signal_achat', 'score_ia', 'target_ml']
        df_save = df[cols_to_save].copy()
        df_save['date'] = df_save['date'].astype(str)
        
        df_save.to_sql("_tmp_analysis", engine, if_exists='replace', index=False)
        
        with engine.begin() as conn:
            conn.execute(text("LOCK TABLE actions_prix_historique IN EXCLUSIVE MODE;"))
            conn.execute(text("""
                UPDATE actions_prix_historique a
                SET rsi_14 = t.rsi_14, 
                    vol_avg_20 = t.vol_avg_20, 
                    sma_200 = t.sma_200, 
                    bb_lower = t.bb_lower, 
                    signal_achat = t.signal_achat, 
                    score_ia = t.score_ia, 
                    target_ml = t.target_ml
                FROM _tmp_analysis t 
                WHERE a.ticker = t.ticker AND a.date::DATE = t.date::DATE;
            """))
        print("✅ Analyse terminée avec succès (Méthode Native).")

    except Exception as e:
        print(f"❌ Erreur critique : {e}")
