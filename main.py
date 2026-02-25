import os
import pandas as pd
import pandas_ta as ta
from fastapi import FastAPI, BackgroundTasks
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# 1. Configuration de la base de données
DATABASE_URL = os.getenv('DATABASE_URL')

if DATABASE_URL:
    # Nettoyage classique (postgres -> postgresql)
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    
    # FORCE LE NOM DE LA BASE : 
    # Si l'URL s'arrête au port (5432/), on ajoute 'n8n' (nom de ta DB sur la capture)
    if DATABASE_URL.endswith("/"):
        DATABASE_URL += "n8n"
    elif not DATABASE_URL.split('/')[-1]:
        DATABASE_URL += "n8n"

    print(f"🔗 Connexion ciblée sur la base : {DATABASE_URL.split('/')[-1]}")
    engine = create_engine(DATABASE_URL)
else:
    engine = None

@app.get("/")
def home():
    return {"status": "Service Trading Actif", "database": "Ciblée sur n8n"}

@app.get("/run-analysis")
async def trigger_analysis(background_tasks: BackgroundTasks):
    """Réponse immédiate, calcul en tâche de fond"""
    background_tasks.add_task(run_analysis)
    return {"status": "processing", "message": "Analyse lancée sur la base n8n"}

def run_analysis():
    if engine is None: return
    print("🚀 Début de l'analyse technique...")
    
    try:
        # On lit la table (en forçant le nom entre guillemets pour éviter les erreurs de casse)
        query = 'SELECT * FROM "actions_prix_historique" ORDER BY ticker, date'
        df = pd.read_sql(query, engine)
        
        if df.empty:
            print("⚠️ La table est vide.")
            return

        # Calculs techniques
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['ticker', 'date'])
        df['rsi_14'] = df.groupby('ticker')['prix_ajuste'].transform(lambda x: ta.rsi(x, length=14))
        df['vol_avg_20'] = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(window=20).mean())
        df['signal_achat'] = (df['rsi_14'] < 35) & (df['volume'] > df['vol_avg_20'])

        # Sauvegarde
        df = df.fillna(0)
        df.to_sql('actions_prix_historique', engine, if_exists='replace', index=False)
        print("✅ Analyse terminée et table mise à jour !")

    except Exception as e:
        print(f"❌ Erreur lors de l'exécution : {e}")
