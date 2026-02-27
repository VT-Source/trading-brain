import pandas as pd
import io
import pickle
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL").replace("postgres://", "postgresql://", 1)
engine = create_engine(DATABASE_URL)

def train_brain():
    print("🧠 Cloud Training : Chargement des données...")
    
    # On récupère les colonnes nécessaires
    query = """
        SELECT a.rsi_14, a.vol_avg_20, a.sma_200, a.bb_lower,
               t.secteur, t.market_cap, t.pe_ratio, a.target_ml
        FROM actions_prix_historique a
        JOIN tickers_info t ON a.ticker = t.ticker
        WHERE a.target_ml IS NOT NULL 
          AND t.secteur IS NOT NULL
    """
    
    df = pd.read_sql(query, engine)
    
    if len(df) < 50:
        print("❌ Données insuffisantes pour l'entraînement.")
        return

    # --- ÉTAPE DE NETTOYAGE (Crucial pour Yahoo Finance) ---
    # 1. Remplissage des indicateurs techniques par la moyenne (si un calcul a échoué)
    df['rsi_14'] = df['rsi_14'].fillna(50) # Neutre
    df['pe_ratio'] = df['pe_ratio'].fillna(df['pe_ratio'].median()) # Médiane par secteur serait mieux, mais globale suffit
    df['market_cap'] = df['market_cap'].fillna(df['market_cap'].median())
    
    # 2. Pour les moyennes mobiles et bandes de bollinger, on remplit par 0 ou médiane
    df = df.fillna(0) 

    # --- PRÉPARATION ---
    df = pd.get_dummies(df, columns=['secteur'])
    X = df.drop('target_ml', axis=1)
    y = df['target_ml']
    cols = list(X.columns)

    # --- VÉRIFICATION FINALE ---
    if X.isnull().values.any():
        print("⚠️ Attention: Des valeurs NaN persistent. Nettoyage forcé...")
        X = X.fillna(0)

    # Entraînement
    print(f"📊 Entraînement sur {len(df)} lignes et {len(cols)} features...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # On garde max_depth pour éviter l'overfitting (tes 98% de réussite)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    # --- SAUVEGARDE DANS LA DB (Format Binaire) ---
    print(f"💾 Sauvegarde du modèle (Précision test: {round(score*100, 2)}%)...")
    
    model_binary = pickle.dumps(model)
    cols_binary = pickle.dumps(cols)

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO models_store (model_name, model_data, columns_data, accuracy, updated_at)
            VALUES ('trading_forest', :m_data, :c_data, :acc, CURRENT_TIMESTAMP)
            ON CONFLICT (model_name) DO UPDATE SET
                model_data = EXCLUDED.model_data,
                columns_data = EXCLUDED.columns_data,
                accuracy = EXCLUDED.accuracy,
                updated_at = CURRENT_TIMESTAMP;
        """), {"m_data": model_binary, "c_data": cols_binary, "acc": score})

    print("✅ Cerveau sauvegardé dans la table models_store.")

if __name__ == "__main__":
    train_brain()
