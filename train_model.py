import os, joblib
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

# ... (garder ton code d'init engine/env) ...

def train_brain():
    # ÉTAPE 1 : On charge TOUT le contexte, pas seulement les signaux
    query = """
        SELECT a.rsi_14, a.vol_ratio, a.dist_sma200, a.bb_position,
               a.regime_marche, a.signal_achat, t.secteur, a.target_ml, a.date
        FROM actions_prix_historique a
        JOIN tickers_info t ON a.ticker = t.ticker
        WHERE a.target_ml IS NOT NULL AND a.rsi_14 IS NOT NULL
        ORDER BY a.date ASC
    """
    df = pd.read_sql(query, engine)

    # ÉTAPE 2 : Nettoyage des Warnings (pd.set_option pour éviter tes logs d'erreur)
    pd.set_option('future.no_silent_downcasting', True)
    df = df.fillna(0).replace([np.inf, -np.inf], 0)
    
    # ÉTAPE 3 : Pondération Temporelle (Donner plus de poids au présent)
    # Les données de 2024-2026 sont plus importantes que celles de 2020
    sample_weights = np.exp(np.linspace(0, 1, len(df)))
    
    # ÉTAPE 4 : Features & Encodage
    df = pd.get_dummies(df, columns=['secteur', 'regime_marche'])
    X = df.drop(['target_ml', 'date'], axis=1)
    y = df['target_ml'].astype(int)

    # ÉTAPE 5 : Modèle avec contraintes pour éviter l'Overfitting
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,        # Profondeur limitée pour forcer la généralisation
        min_samples_leaf=10, # Évite d'apprendre des cas isolés
        class_weight='balanced',
        random_state=42
    )
    
    model.fit(X, y, sample_weight=sample_weights)
    
    # Sauvegarde
    joblib.dump(model, "models/trading_forest.joblib")
    joblib.dump(list(X.columns), "models/trading_forest_cols.joblib")
    print("✅ Modèle v3.3 optimisé et sauvegardé.")
