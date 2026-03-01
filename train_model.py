import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
from dotenv import load_dotenv

load_dotenv()

# Sécurité URL Database
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

def train_brain():
    print("🧠 Cloud Training V2 : Chargement des données massives...")
    
    # On récupère uniquement ce qui est nécessaire pour limiter la RAM
    query = """
        SELECT a.rsi_14, a.vol_avg_20, a.sma_200, a.bb_lower,
               t.secteur, t.market_cap, t.pe_ratio, a.target_ml
        FROM actions_prix_historique a
        JOIN tickers_info t ON a.ticker = t.ticker
        WHERE a.target_ml IS NOT NULL 
          AND a.rsi_14 IS NOT NULL
    """
    
    try:
        df = pd.read_sql(query, engine)
    except Exception as e:
        print(f"❌ Erreur lors de la lecture DB : {e}")
        return

    if len(df) < 100:
        print(f"❌ Données insuffisantes ({len(df)} lignes). Attendez que l'analyse technique soit finie.")
        return

    # --- NETTOYAGE SÉCURISÉ ---
    # Remplacer les INFINIS (souvent générés par les ratios PE ou RSI) par NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Nettoyage intelligent : Médiane pour les chiffres, 'Inconnu' pour les secteurs
    df['rsi_14'] = df['rsi_14'].fillna(50)
    df['pe_ratio'] = df['pe_ratio'].fillna(df['pe_ratio'].median())
    df['market_cap'] = df['market_cap'].fillna(df['market_cap'].median())
    df['secteur'] = df['secteur'].fillna('Unknown')
    
    # Sécurité supplémentaire pour SMA et BB (si NULL, on prend 0 ou la valeur actuelle n'aiderait pas)
    df = df.fillna(0) 

    # --- PRÉPARATION DES FEATURES ---
    # Conversion du secteur en colonnes binaires (One-Hot Encoding)
    df = pd.get_dummies(df, columns=['secteur'])
    
    X = df.drop('target_ml', axis=1)
    y = df['target_ml'].astype(int) # S'assurer que les cibles sont des entiers (0 ou 1)
    cols = list(X.columns)

    # --- ENTRAÎNEMENT OPTIMISÉ ---
    print(f"📊 Entraînement sur {len(df)} lignes et {len(cols)} indicateurs...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # n_jobs=-1 utilise tous les processeurs de ton serveur
    # class_weight='balanced' corrige ton problème de taux de réussite faible (16%)
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=12, 
        random_state=42, 
        n_jobs=-1, 
        class_weight='balanced_subsample' 
    )
    
    model.fit(X_train, y_train)
    
    # Calcul de la précision
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    # --- SAUVEGARDE ---
    print(f"💾 Sauvegarde du modèle (Précision Entraînement: {round(train_score*100, 2)}% | Test: {round(test_score*100, 2)}%)")
    
    model_binary = pickle.dumps(model)
    cols_binary = pickle.dumps(cols)

    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO models_store (model_name, model_data, columns_data, accuracy, updated_at)
                VALUES ('trading_forest', :m_data, :c_data, :acc, CURRENT_TIMESTAMP)
                ON CONFLICT (model_name) DO UPDATE SET
                    model_data = EXCLUDED.model_data,
                    columns_data = EXCLUDED.columns_data,
                    accuracy = EXCLUDED.accuracy,
                    updated_at = CURRENT_TIMESTAMP;
            """), {"m_data": model_binary, "c_data": cols_binary, "acc": test_score})
        print("✅ Modèle 'trading_forest' mis à jour avec les données de 399 tickers.")
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde du modèle : {e}")

if __name__ == "__main__":
    train_brain()
