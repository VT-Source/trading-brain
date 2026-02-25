import os
import pandas as pd
import pandas_ta as ta
from fastapi import FastAPI
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# 1. Configuration de la base de données Railway
DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL:
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    engine = create_engine(DATABASE_URL)
else:
    engine = None
    print("ATTENTION : DATABASE_URL non trouvée.")

# 2. Points d'entrée de l'API (FastAPI)
@app.get("/")
def home():
    return {"status": "Service Trading Actif", "version": "1.4"}

@app.get("/run-analysis")
async def trigger_analysis():
    """Route appelée par n8n pour déclencher les calculs"""
    try:
        run_analysis()
        return {"status": "success", "message": "Analyse technique complétée"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# 3. Logique de calcul
def run_analysis():
    if engine is None:
        raise Exception("Connexion à la base de données impossible")

    print("Début de l'analyse technique...")
    
    # Lecture des données
    query = "SELECT * FROM actions_prix_historique ORDER BY ticker, date"
    df = pd.read_sql(query, engine)
    
    if df.empty:
        print("Erreur : La table est vide.")
        return

    # Calculs techniques par ticker
    df['rsi_14'] = df.groupby('ticker')['prix_ajuste'].transform(lambda x: ta.rsi(x, length=14))
    df['vol_avg_20'] = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(window=20).mean())

    # Signal d'Achat
    df['signal_achat'] = (df['rsi_14'] < 35) & (df['volume'] > df['vol_avg_20'])

    # Mise à jour de la base
    df.to_sql('actions_prix_historique', engine, if_exists='replace', index=False)
    print("Analyse terminée. Base mise à jour.")

if __name__ == "__main__":
    run_analysis()
