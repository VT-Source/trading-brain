import os
import pandas as pd
import pandas_ta as ta
from fastapi import FastAPI, BackgroundTasks
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# --- Configuration DB (Railway & n8n) ---
DATABASE_URL = os.getenv("DATABASE_URL")

engine = None
if DATABASE_URL:
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

    if DATABASE_URL.endswith("/") and "?" not in DATABASE_URL:
        DATABASE_URL = DATABASE_URL[:-1]

    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        connect_args={"options": "-csearch_path=public"}
    )

@app.get("/")
def home():
    return {"status": "Service Trading Actif", "version": "2.0", "features": ["RSI", "SMA200", "Bollinger"]}

@app.get("/debug/db-info")
def db_info():
    if engine is None: return {"error": "Engine not initialized"}
    with engine.connect() as conn:
        info = conn.execute(text("SELECT current_database(), current_setting('search_path');")).mappings().first()
        tables = conn.execute(text("SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = 'public';")).mappings().all()
    return {"info": dict(info), "tables": [dict(t) for t in tables]}

@app.get("/run-analysis")
async def trigger_analysis(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_analysis)
    return {"status": "processing", "message": "Calculs techniques avancés lancés..."}

def run_analysis():
    if engine is None:
        print("❌ Engine non initialisé")
        return

    print("🚀 Démarrage de l'analyse technique (V2 - Trend Following)...")
    table_name = "actions_prix_historique"
    schema_name = "public"

    try:
        query = f'SELECT * FROM {schema_name}."{table_name}" ORDER BY ticker, date'
        df = pd.read_sql(query, engine)

        if df.empty:
            print("⚠️ Table vide.")
            return

        # --- Préparation ---
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
        df = df.sort_values(['ticker', 'date'])
        df['prix_ajuste'] = pd.to_numeric(df['prix_ajuste'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

        # --- Calculs Techniques ---
        
        # 1. RSI 14
        df['rsi_14'] = df.groupby('ticker')['prix_ajuste'].transform(lambda x: ta.rsi(x, length=14))

        # 2. Moyenne Mobile Volume 20
        df['vol_avg_20'] = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(window=20).mean())

        # 3. SMA 200 (Filtre de Tendance)
        df['sma_200'] = df.groupby('ticker')['prix_ajuste'].transform(lambda x: x.rolling(window=200).mean())

        # 4. Bandes de Bollinger (Bande Inférieure)
        def get_bb_lower(x):
            if len(x) < 20: return pd.Series([None] * len(x))
            # bb_lower est la première colonne retournée par bbands (BBL_20_2.0)
            return ta.bbands(x, length=20, std=2).iloc[:, 0]

        df['bb_lower'] = df.groupby('ticker')['prix_ajuste'].transform(get_bb_lower)

        # --- Signal d'Achat Amélioré ---
        # Conditions : RSI Bas + Volume Haut + Tendance Haussière (Prix > SMA200) + Extension Basse (Prix <= BB_Lower)
        df['signal_achat'] = (
            (df['rsi_14'] < 35) & 
            (df['volume'] > df['vol_avg_20']) & 
            (df['prix_ajuste'] > df['sma_200']) & 
            (df['prix_ajuste'] <= df['bb_lower'])
        ).fillna(False)

        # --- Sauvegarde ---
        calc_cols = ['ticker', 'date', 'rsi_14', 'vol_avg_20', 'sma_200', 'bb_lower', 'signal_achat']
        tmp_table = "_calc_tmp_v2"
        
        df[calc_cols].to_sql(tmp_table, engine, if_exists='replace', index=False, schema=schema_name)

        with engine.begin() as conn:
            conn.execute(text(f"""
                UPDATE {schema_name}."{table_name}" a
                SET rsi_14 = t.rsi_14,
                    vol_avg_20 = t.vol_avg_20,
                    sma_200 = t.sma_200,
                    bb_lower = t.bb_lower,
                    signal_achat = t.signal_achat
                FROM {schema_name}."{tmp_table}" t
                WHERE a.ticker = t.ticker AND a.date = t.date;
            """))
            conn.execute(text(f'DROP TABLE IF EXISTS {schema_name}."{tmp_table}";'))

        print(f"✅ Analyse V2 terminée. SMA200 et Bollinger mis à jour.")

    except Exception as e:
        print(f"❌ Erreur : {e}")
