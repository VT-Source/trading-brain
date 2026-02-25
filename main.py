import os
import pandas as pd
import pandas_ta as ta
from fastapi import FastAPI, BackgroundTasks
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# Configuration alignée sur Railway & n8n
DATABASE_URL = os.getenv('DATABASE_URL')

if DATABASE_URL:
    # 1. Correction du protocole pour SQLAlchemy
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    
    # 2. Correction de la base : On retire tout ajout manuel de "n8n"
    # Railway injecte normalement l'URL complète avec /railway à la fin.
    # Si l'URL finit par un slash, on l'enlève pour laisser SQLAlchemy gérer.
    if DATABASE_URL.endswith("/"):
        DATABASE_URL = DATABASE_URL[:-1]

    # Note : Si Railway ne donne pas de nom, SQLAlchemy utilisera 'postgres'
    engine = create_engine(DATABASE_URL)
else:
    engine = None

@app.get("/")
def home():
    return {"status": "Service Trading Actif", "version": "1.6"}

@app.get("/run-analysis")
async def trigger_analysis(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_analysis)
    return {"status": "processing", "message": "Calculs lancés sur la base partagée"}

def run_analysis():
    if engine is None: return
    print("🚀 Démarrage de l'analyse technique...")
    
    try:
        # On utilise le nom de table EXACT vu dans ton JSON n8n
        table_name = "actions_prix_historique"
        
        # Lecture
        query = f'SELECT * FROM "{table_name}" ORDER BY ticker, date'
        df = pd.read_sql(query, engine)
        
        if df.empty:
            print("⚠️ Table trouvée mais vide.")
            return

        # Calculs techniques (RSI et Volume Moyen)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['ticker', 'date'])
        
        # On s'assure que les colonnes sont numériques
        df['prix_ajuste'] = pd.to_numeric(df['prix_ajuste'])
        df['volume'] = pd.to_numeric(df['volume'])
        
        df['rsi_14'] = df.groupby('ticker')['prix_ajuste'].transform(lambda x: ta.rsi(x, length=14))
        df['vol_avg_20'] = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(window=20).mean())
        
        # Signal d'achat
        df['signal_achat'] = (df['rsi_14'] < 35) & (df['volume'] > df['vol_avg_20'])

        # Remplacement des NaN par None (pour SQL) et sauvegarde
        df = df.where(pd.notnull(df), None)
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        
        print(f"✅ Analyse terminée. {len(df)} lignes mises à jour.")

    except Exception as e:
        print(f"❌ Erreur lors de l'exécution : {e}")
