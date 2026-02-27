import os
import time
import requests
import pandas as pd
import pandas_ta as ta
from fastapi import FastAPI, BackgroundTasks
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# --- Configuration DB ---
DATABASE_URL = os.getenv("DATABASE_URL")
ALPHA_VANTAGE_KEY = "6LSG483HD1NHNK2R" # Ta clé API
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
        "status": "Service Trading Actif", 
        "version": "3.2", 
        "features": ["ML-Ready", "AlphaVantage-Sync", "Fix-Numpy-Types"]
    }

# --- ENDPOINT 1 : ANALYSE TECHNIQUE (Quotidien) ---
@app.get("/run-analysis")
async def trigger_analysis(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_analysis_logic)
    return {"status": "processing", "message": "Calculs techniques et Target ML lancés..."}

# --- ENDPOINT 2 : SYNC METADATA (Hebdomadaire/Cron) ---
@app.get("/sync-metadata")
async def trigger_sync(background_tasks: BackgroundTasks):
    background_tasks.add_task(sync_metadata_logic)
    return {"status": "processing", "message": "Synchronisation des secteurs et P/E Ratios lancée..."}

def sync_metadata_logic():
    if engine is None: return
    print("🔄 Démarrage de la synchronisation Alpha Vantage...")
    
    # On cherche les tickers inconnus OU mis à jour il y a plus de 30 jours
    query = """
        SELECT DISTINCT ticker FROM actions_prix_historique
        WHERE ticker NOT IN (SELECT ticker FROM tickers_info)
        OR ticker IN (SELECT ticker FROM tickers_info WHERE derniere_maj < CURRENT_DATE - INTERVAL '30 days')
        LIMIT 20;
    """
    
    try:
        with engine.connect() as conn:
            tickers = [row[0] for row in conn.execute(text(query))]

        for ticker in tickers:
            print(f"🚀 Tentative pour {ticker}...")
            url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_VANTAGE_KEY}'
            r = requests.get(url)
            data = r.json()

            if "Symbol" in data:
                # CORRECTION : Conversion explicite pour éviter l'erreur numpy.int64
                m_cap_raw = pd.to_numeric(data.get("MarketCapitalization"), errors='coerce')
                pe_ratio_raw = pd.to_numeric(data.get("PERatio"), errors='coerce')

                metadata = {
                    "ticker": data.get("Symbol"),
                    "name": data.get("Name"),
                    "secteur": data.get("Sector"),
                    "industrie": data.get("Industry"),
                    "pays": data.get("Country"),
                    "monnaie": data.get("Currency"),
                    # On force le type Python standard int/float ou None
                    "market_cap": int(m_cap_raw) if pd.notnull(m_cap_raw) else None,
                    "pe_ratio": float(pe_ratio_raw) if pd.notnull(pe_ratio_raw) else None
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
                print(f"✅ {ticker} synchronisé avec succès.")
            else:
                print(f"⚠️ Pas de données Symbol pour {ticker}. Réponse API : {data}")
            
            time.sleep(15) # Pause de sécurité pour la limite Alpha Vantage
            
    except Exception as e:
        print(f"❌ Erreur Sync Metadata: {e}")

def run_analysis_logic():
    if engine is None: return
    print("🚀 Démarrage de l'analyse technique...")
    
    try:
        query = 'SELECT * FROM public."actions_prix_historique" ORDER BY ticker, date'
        df = pd.read_sql(query, engine)
        if df.empty: return

        # Préparation
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
        df = df.sort_values(['ticker', 'date'])
        df['prix_ajuste'] = pd.to_numeric(df['prix_ajuste'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

        # Indicateurs
        df['rsi_14'] = df.groupby('ticker')['prix_ajuste'].transform(lambda x: ta.rsi(x, length=14))
        df['vol_avg_20'] = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(window=20).mean())
        df['sma_200'] = df.groupby('ticker')['prix_ajuste'].transform(lambda x: x.rolling(window=200).mean())
        
        def get_bb_lower(x):
            if len(x) < 20: return pd.Series([None] * len(x))
            return ta.bbands(x, length=20, std=2).iloc[:, 0]
        df['bb_lower'] = df.groupby('ticker')['prix_ajuste'].transform(get_bb_lower)

        df['prix_veille'] = df.groupby('ticker')['prix_ajuste'].shift(1)

        # Signal d'Achat
        df['signal_achat'] = (
            (df['rsi_14'] < 35) & 
            (df['volume'] > df['vol_avg_20']) & 
            (df['prix_ajuste'] > df['sma_200']) & 
            (df['prix_ajuste'] <= df['bb_lower']) &
            (df['prix_ajuste'] > df['prix_veille'])
        ).fillna(False)

        # Target ML (+6% à 30 jours)
        df['max_30j'] = df.groupby('ticker')['prix_ajuste'].transform(lambda x: x.shift(-30).rolling(window=30, min_periods=1).max())
        df['target_ml'] = ((df['max_30j'] - df['prix_ajuste']) / df['prix_ajuste'] >= 0.06).astype(int)

        # Sauvegarde via table temporaire pour performance
        calc_cols = ['ticker', 'date', 'rsi_14', 'vol_avg_20', 'sma_200', 'bb_lower', 'signal_achat', 'target_ml']
        df[calc_cols].to_sql("_calc_tmp_v3", engine, if_exists='replace', index=False)

        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE public."actions_prix_historique" a
                SET rsi_14 = t.rsi_14, vol_avg_20 = t.vol_avg_20, sma_200 = t.sma_200, 
                    bb_lower = t.bb_lower, signal_achat = t.signal_achat, target_ml = t.target_ml
                FROM _calc_tmp_v3 t
                WHERE a.ticker = t.ticker AND a.date = t.date;
            """))
            conn.execute(text('DROP TABLE IF EXISTS _calc_tmp_v3;'))

        print("✅ Analyse technique et Target ML terminées.")

    except Exception as e:
        print(f"❌ Erreur Analyse: {e}")
