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
    
    query = """
        SELECT a.rsi_14, a.vol_avg_20, a.sma_200, a.bb_lower,
               t.secteur, t.market_cap, t.pe_ratio, a.target_ml
        FROM actions_prix_historique a
        JOIN tickers_info t ON a.ticker = t.ticker
        WHERE a.rsi_14 IS NOT NULL AND a.target_ml IS NOT NULL AND t.secteur IS NOT NULL
    """
    
    df = pd.read_sql(query, engine)
    
    if len(df) < 50:
        print("❌ Données insuffisantes.")
        return

    # Préparation
    df = pd.get_dummies(df, columns=['secteur'])
    X = df.drop('target_ml', axis=1)
    y = df['target_ml']
    cols = list(X.columns)

    # Entraînement
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    # --- SAUVEGARDE DANS LA DB (Format Binaire) ---
    print(f"💾 Sauvegarde du modèle (Précision: {round(score*100, 2)}%)...")
    
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

if __name__ == "__main__":
    train_brain()
