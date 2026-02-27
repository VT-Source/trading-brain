import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
from dotenv import load_dotenv

load_dotenv()

# Connexion DB (Même logique que main.py)
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)

def train_brain():
    print("🧠 Récupération des données pour l'entraînement...")
    
    # Jointure entre tes prix (indicateurs) et tes infos d'entreprise
    query = """
        SELECT 
            a.rsi_14, a.vol_avg_20, a.sma_200, a.bb_lower,
            t.secteur, t.market_cap, t.pe_ratio,
            a.target_ml
        FROM actions_prix_historique a
        JOIN tickers_info t ON a.ticker = t.ticker
        WHERE a.rsi_14 IS NOT NULL 
          AND a.target_ml IS NOT NULL 
          AND t.secteur IS NOT NULL
    """
    
    df = pd.read_sql(query, engine)
    
    if len(df) < 50:
        print(f"⚠️ Pas assez de données ({len(df)} lignes). Continue à synchroniser tes tickers !")
        return

    print(f"📊 Données chargées : {len(df)} lignes.")

    # 1. Préparation : Transformer le texte (Secteur) en nombres (One-Hot Encoding)
    # L'IA ne comprend pas "Technology", elle comprend 0 ou 1.
    df = pd.get_dummies(df, columns=['secteur'])
    
    # 2. Séparation Features (X) et Cible (y)
    X = df.drop('target_ml', axis=1)
    y = df['target_ml']

    # Sauvegarder la liste des colonnes pour que main.py sache dans quel ordre envoyer les données
    model_columns = list(X.columns)
    joblib.dump(model_columns, 'model_columns.joblib')

    # 3. Entraînement
    print("🏗️ Apprentissage du modèle (Random Forest)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # 4. Score
    score = model.score(X_test, y_test)
    print(f"✅ Modèle prêt ! Précision : {round(score * 100, 2)}%")

    # 5. Sauvegarde locale (sur le disque Railway temporaire ou ton PC)
    joblib.dump(model, 'trading_model.joblib')
    print("💾 Fichiers sauvegardés : trading_model.joblib et model_columns.joblib")

if __name__ == "__main__":
    train_brain()
