import os
import pandas as pd
import pandas_ta as ta
from fastapi import FastAPI, BackgroundTasks
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

# --- CRUCIAL : Railway cherche cet objet 'app' ---
app = FastAPI()

# --- Configuration DB ---
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
    return {"status": "Service Trading Actif", "version": "3.0", "features": ["ML-Ready", "SMA200", "Bollinger"]}

@app.get("/run-analysis")
async def trigger_analysis(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_analysis)
    return {"status": "processing", "message": "Calculs V3 (Préparation ML) lancés..."}

def run_analysis():
    if engine is None:
        print("❌ Engine non initialisé")
        return

    print("🚀 Démarrage de l'analyse (V3 - Machine Learning Preparation)...")
    table_name = "actions_prix_historique"
    schema_name = "public"

    try:
        # 1. Lecture
        query = f'SELECT * FROM {schema_name}."{table_name}" ORDER BY ticker, date'
        df = pd.read_sql(query, engine)
        if df.empty:
            print("⚠️ Table vide.")
            return

        # 2. Préparation
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
        df = df.sort_values(['ticker', 'date'])
        df['prix_ajuste'] = pd.to_numeric(df['prix_ajuste'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

        # 3. Calculs Techniques
        df['rsi_14'] = df.groupby('ticker')['prix_ajuste'].transform(lambda x: ta.rsi(x, length=14))
        df['vol_avg_20'] = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(window=20).mean())
        df['sma_200'] = df.groupby('ticker')['prix_ajuste'].transform(lambda x: x.rolling(window=200).mean())
        
        def get_bb_lower(x):
            if len(x) < 20: return pd.Series([None] * len(x))
            return ta.bbands(x, length=20, std=2).iloc[:, 0]
        df['bb_lower'] = df.groupby('ticker')['prix_ajuste'].transform(get_bb_lower)

        # Sécurité de rebond (Prix > Prix Veille)
        df['prix_veille'] = df.groupby('ticker')['prix_ajuste'].shift(1)

        # 4. Signal d'Achat (Tes critères validés + Rebond)
        df['signal_achat'] = (
            (df['rsi_14'] < 35) & 
            (df['volume'] > df['vol_avg_20']) & 
            (df['prix_ajuste'] > df['sma_200']) & 
            (df['prix_ajuste'] <= df['bb_lower']) &
            (df['prix_ajuste'] > df['prix_veille'])
        ).fillna(False)

        # 5. Calcul de la Target pour le ML (+6% à 30 jours)
        # On regarde le futur : quel est le prix max sur les 30 prochains jours ?
        df['max_30j'] = (
            df.groupby('ticker')['prix_ajuste']
            .transform(lambda x: x.shift(-30).rolling(window=30, min_periods=1).max())
        )
        # Target = 1 si le gain max futur a atteint +6%
        df['target_ml'] = ((df['max_30j'] - df['prix_ajuste']) / df['prix_ajuste'] >= 0.06).astype(int)

        # 6. Sauvegarde via table temporaire
        calc_cols = ['ticker', 'date', 'rsi_14', 'vol_avg_20', 'sma_200', 'bb_lower', 'signal_achat', 'target_ml']
        tmp_table = "_calc_tmp_ml"
        
        df[calc_cols].to_sql(tmp_table, engine, if_exists='replace', index=False, schema=schema_name)

        with engine.begin() as conn:
            conn.execute(text(f"""
                UPDATE {schema_name}."{table_name}" a
                SET rsi_14 = t.rsi_14, 
                    vol_avg_20 = t.vol_avg_20, 
                    sma_200 = t.sma_200, 
                    bb_lower = t.bb_lower, 
                    signal_achat = t.signal_achat, 
                    target_ml = t.target_ml
                FROM {schema_name}."{tmp_table}" t
                WHERE a.ticker = t.ticker AND a.date = t.date;
            """))
            conn.execute(text(f'DROP TABLE IF EXISTS {schema_name}."{tmp_table}";'))

        print(f"✅ Analyse terminée. Données prêtes pour le ML.")

    except Exception as e:
        print(f"❌ Erreur : {e}")

