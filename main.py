import os
import pandas as pd
import pandas_ta as ta
from fastapi import FastAPI
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# Configuration de la base de données Railway
DATABASE_URL = os.getenv('DATABASE_URL')

if DATABASE_URL:
    # Nettoyage de l'URL pour SQLAlchemy (Postgres -> Postgressql)
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    engine = create_engine(DATABASE_URL)
else:
    engine = None
    print("ATTENTION : DATABASE_URL non trouvée. Le mode analyse sera indisponible.")

@app.get("/")
def home():
   @app.get("/run-analysis")
async def trigger_analysis():
    try:
        run_analysis() # C'est la fonction de calcul qu'on a créée
        return {"status": "success", "message": "Analyse technique complétée"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def run_analysis():
    """Fonction principale de calcul appelée par n8n via 'Execute Command'"""
    print("Début de l'analyse technique via la base de données...")
    
    # 1. Chargement des données depuis Postgres
    query = "SELECT * FROM actions_prix_historique ORDER BY ticker, date"
    df = pd.read_sql(query, engine)
    
    if df.empty:
        print("Erreur : La table est vide.")
        return

    # 2. Calculs techniques (par groupe de ticker pour ne pas mélanger les actions)
    # On utilise prix_ajuste pour le RSI !
    df['rsi_14'] = df.groupby('ticker')['prix_ajuste'].transform(lambda x: ta.rsi(x, length=14))
    
    # Moyenne de volume sur 20 jours
    df['vol_avg_20'] = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(window=20).mean())

    # 3. Logique du Signal d'Achat (Calculé pour chaque ligne)
    # Signal si RSI < 35 et Volume actuel > Moyenne 20 jours
    df['signal_achat'] = (df['rsi_14'] < 35) & (df['volume'] > df['vol_avg_20'])

    # 4. Sauvegarde/Mise à jour de la table
    # On utilise 'replace' pour mettre à jour toutes les colonnes d'un coup
    df.to_sql('actions_prix_historique', engine, if_exists='replace', index=False)
    print("Analyse terminée. Base de données mise à jour avec succès.")

if __name__ == "__main__":
    # Permet de lancer le script manuellement ou via n8n avec : python main.py
    run_analysis()
