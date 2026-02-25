import os
import pandas as pd
import pandas_ta as ta
from fastapi import FastAPI, BackgroundTasks
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# 1. Configuration de la DB
DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL) if DATABASE_URL else None

@app.get("/")
def home():
    return {"status": "Service Trading Actif", "version": "1.5", "mode": "Upsert Optimized"}

@app.get("/run-analysis")
async def trigger_analysis(background_tasks: BackgroundTasks):
    """Réponse immédiate à n8n, calcul en arrière-plan"""
    if not engine:
        return {"status": "error", "message": "DB non configurée"}
    
    background_tasks.add_task(run_analysis)
    return {"status": "processing", "message": "Analyse lancée en tâche de fond"}

def run_analysis():
    print("🚀 Début de l'analyse (Mode Upsert)...")
    
    try:
        # 1. Lecture
        df = pd.read_sql("SELECT * FROM actions_prix_historique ORDER BY ticker, date", engine)
        if df.empty: return

        # 2. Calculs techniques (Vectorisés)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['ticker', 'date'])
        
        df['rsi_14'] = df.groupby('ticker')['prix_ajuste'].transform(lambda x: ta.rsi(x, length=14))
        df['vol_avg_20'] = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(window=20).mean())
        df['signal_achat'] = (df['rsi_14'] < 35) & (df['volume'] > df['vol_avg_20'])

        # Nettoyage pour SQL (remplacer NaN par None pour Postgres)
        df = df.where(pd.notnull(df), None)

        # 3. Logique UPSERT via Table Temporaire
        with engine.begin() as conn:
            # A. Créer une table temporaire vide identique à la structure
            conn.execute(text("CREATE TEMP TABLE temp_analysis AS SELECT * FROM actions_prix_historique WITH NO DATA"))
            
            # B. Charger les nouveaux calculs dans la table temporaire
            df.to_sql('temp_analysis', conn, if_exists='append', index=False)
            
            # C. L'UPSERT : On met à jour la table principale depuis la tempo
            # On suppose que 'ticker' + 'date' forment l'identité unique
            upsert_query = text("""
                UPDATE actions_prix_historique h
                SET rsi_14 = t.rsi_14,
                    vol_avg_20 = t.vol_avg_20,
                    signal_achat = t.signal_achat
                FROM temp_analysis t
                WHERE h.ticker = t.ticker AND h.date = t.date;
            """)
            conn.execute(upsert_query)
            
        print("✅ Analyse terminée et Table mise à jour (Upsert).")

    except Exception as e:
        print(f"❌ Erreur lors de l'analyse : {e}")

if __name__ == "__main__":
    run_analysis()
