import os
import joblib
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, accuracy_score
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

MODEL_PATH = "models/trading_forest.joblib"
COLS_PATH  = "models/trading_forest_cols.joblib"

def train_brain():
    print("🧠 Entraînement v3.3 : Apprentissage Global (Contexte vs Signal)...")

    # --------------------------------------------------------
    # ÉTAPE 1 : Chargement Global (Inclusion des échecs et du hors-signal)
    # --------------------------------------------------------
    query = """
        SELECT a.rsi_14, a.vol_avg_20, a.sma_200, a.bb_lower,
               a.rsi_slope, a.vol_ratio, a.dist_sma200, a.bb_position,
               a.regime_marche, a.signal_achat,
               t.secteur, t.market_cap, t.pe_ratio,
               a.target_ml,
               a.date
        FROM actions_prix_historique a
        JOIN tickers_info t ON a.ticker = t.ticker
        WHERE a.target_ml IS NOT NULL 
          AND a.rsi_14 IS NOT NULL
        ORDER BY a.date ASC
    """

    try:
        df = pd.read_sql(query, engine)
    except Exception as e:
        print(f"❌ Erreur lecture DB : {e}")
        return

    if len(df) < 500:
        print(f"❌ Données insuffisantes ({len(df)} lignes). Besoin d'un historique plus large pour le ML global.")
        return

    # Nettoyage immédiat pour éviter les erreurs de calcul
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Distribution pour debug
    regime_dist = df['regime_marche'].value_counts().to_dict()
    print(f"📊 Dataset : {len(df)} lignes | Signaux AT : {df['signal_achat'].sum()}")
    print(f"📊 Distribution régimes : {regime_dist}")

    # --------------------------------------------------------
    # ÉTAPE 2 : Preprocessing & Encodage
    # --------------------------------------------------------
    # Remplissage intelligent
    df['rsi_slope']   = df['rsi_slope'].fillna(0)
    df['vol_ratio']   = df['vol_ratio'].fillna(1)
    df['dist_sma200'] = df['dist_sma200'].fillna(0)
    df['bb_position'] = df['bb_position'].fillna(0)
    df['regime_marche'] = df['regime_marche'].fillna('NEUTRE')
    df['pe_ratio']    = df['pe_ratio'].fillna(df['pe_ratio'].median())
    
    # Encodage (One-Hot)
    df = pd.get_dummies(df, columns=['secteur', 'regime_marche'])
    
    # Conversion booléen en int pour le modèle
    df['signal_achat'] = df['signal_achat'].astype(int)

    # --------------------------------------------------------
    # ÉTAPE 3 : Pondération Hybride (Temps + Importance Signal)
    # --------------------------------------------------------
    # 1. Poids temporel (plus c'est récent, plus ça compte)
    time_weights = np.exp(np.linspace(0, 1, len(df)))
    
    # 2. Sur-pondération des signaux AT (x5) pour forcer le modèle à bien les classer
    df['final_weight'] = time_weights
    df.loc[df['signal_achat'] == 1, 'final_weight'] *= 5

    X = df.drop(['target_ml', 'date', 'final_weight'], axis=1)
    y = df['target_ml'].astype(int)
    weights = df['final_weight'].values
    cols = list(X.columns)

    # --------------------------------------------------------
    # ÉTAPE 4 : Validation Croisée Temporelle
    # --------------------------------------------------------
    tscv = TimeSeriesSplit(n_splits=5)
    metrics = {'acc': [], 'prec': [], 'rec': []}

    print("⏳ Validation croisée en cours...")

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        w_train = weights[train_idx]

        clf = RandomForestClassifier(
            n_estimators=150, # Augmenté pour la complexité globale
            max_depth=7,      # Contrôle de l'overfitting
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train, sample_weight=w_train)
        
        preds = clf.predict(X_test)
        metrics['acc'].append(accuracy_score(y_test, preds))
        metrics['prec'].append(precision_score(y_test, preds, zero_division=0))
        metrics['rec'].append(recall_score(y_test, preds, zero_division=0))

    print(f"\n📈 Résultats Moyens :")
    print(f"   Précision : {round(np.mean(metrics['prec'])*100, 2)}% (Cible > 52%)")
    print(f"   Recall    : {round(np.mean(metrics['rec'])*100, 2)}%")

    # --------------------------------------------------------
    # ÉTAPE 5 : Entraînement Final & Sauvegarde
    # --------------------------------------------------------
    print("\n💪 Entraînement final sur la totalité des données...")
    final_model = RandomForestClassifier(
        n_estimators=150, max_depth=7, 
        class_weight='balanced_subsample', random_state=42, n_jobs=-1
    )
    final_model.fit(X, y, sample_weight=weights)

    os.makedirs("models", exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(cols, COLS_PATH)
    
    # Update DB métriques
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO models_store (model_name, accuracy, updated_at)
                VALUES ('trading_forest_v3', :acc, CURRENT_TIMESTAMP)
                ON CONFLICT (model_name) DO UPDATE SET accuracy = EXCLUDED.accuracy;
            """), {"acc": float(np.mean(metrics['prec']))})
    except Exception as e:
        print(f"⚠️ Erreur logs DB: {e}")

    print(f"✅ Modèle sauvegardé. Prêt pour /run-analysis.")

if __name__ == "__main__":
    train_brain()
