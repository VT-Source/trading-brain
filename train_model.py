import os
import joblib
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score
from dotenv import load_dotenv
 
load_dotenv()
 
# --- CONFIGURATION DATABASE ---
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
 
engine = create_engine(DATABASE_URL, pool_pre_ping=True) if DATABASE_URL else None
 
MODEL_PATH = "models/trading_forest.joblib"
COLS_PATH  = "models/trading_forest_cols.joblib"
 
 
def train_brain():
    if engine is None:
        print("❌ Erreur : DATABASE_URL non configurée.")
        return
 
    print("🧠 Entraînement v3.4 : Apprentissage Global (Contexte vs Signal)...")
 
    # 1. Chargement depuis la base
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
 
    if len(df) < 200:
        print(f"❌ Données insuffisantes ({len(df)} lignes).")
        return
 
    # 2. Nettoyage & Preprocessing
    pd.set_option('future.no_silent_downcasting', True)
    df = df.replace([np.inf, -np.inf], np.nan)
 
    df['rsi_slope']   = df['rsi_slope'].fillna(0)
    df['vol_ratio']   = df['vol_ratio'].fillna(1)
    df['dist_sma200'] = df['dist_sma200'].fillna(0)
    df['bb_position'] = df['bb_position'].fillna(0)
    df['pe_ratio']    = df['pe_ratio'].fillna(df['pe_ratio'].median())
 
    # FIX : Normaliser regime_marche — remplacer les anciennes valeurs résiduelles
    # main.py écrit BULL/BEAR/NEUTRE ; des lignes historiques peuvent contenir HAUSSIER/BAISSIER
    df['regime_marche'] = df['regime_marche'].replace({
        'HAUSSIER': 'BULL',
        'BAISSIER': 'BEAR'
    }).fillna('NEUTRE')
 
    # FIX : Forcer les 3 catégories connues pour garantir des colonnes stables après get_dummies
    # Sans ça, si BEAR est absent du batch, la colonne regime_marche_BEAR n'est pas créée
    # et le modèle est entraîné sur des colonnes différentes de celles de l'inférence
    df['regime_marche'] = pd.Categorical(
        df['regime_marche'],
        categories=['BULL', 'BEAR', 'NEUTRE']
    )
 
    # FIX : Même chose pour secteur — forcer les catégories connues
    known_secteurs = list(df['secteur'].dropna().unique())
    df['secteur'] = pd.Categorical(df['secteur'].fillna('INCONNU'), categories=known_secteurs + ['INCONNU'])
 
    # Encodage one-hot avec colonnes stables
    df = pd.get_dummies(df, columns=['secteur', 'regime_marche'])
    df['signal_achat'] = df['signal_achat'].astype(int)
 
    # 3. Pondération Hybride (Temps + Signal)
    time_weights = np.exp(np.linspace(0, 1, len(df)))
    df['final_weight'] = time_weights
    df.loc[df['signal_achat'] == 1, 'final_weight'] *= 5
 
    X       = df.drop(['target_ml', 'date', 'final_weight'], axis=1)
    y       = df['target_ml'].astype(int)
    weights = df['final_weight'].values
    cols    = list(X.columns)
 
    # 4. Validation Croisée Temporelle
    tscv       = TimeSeriesSplit(n_splits=5)
    precisions = []
 
    print("⏳ Validation croisée en cours...")
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        w_train         = weights[train_idx]
 
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=6,
            class_weight='balanced', random_state=42
        )
        clf.fit(X_train, y_train, sample_weight=w_train)
 
        preds = clf.predict(X_test)
        precisions.append(precision_score(y_test, preds, zero_division=0))
 
    mean_prec = np.mean(precisions)
 
    # FIX : Alerte précision < 52% (mentionnée dans PROJECT_STATUS mais absente du code)
    if mean_prec < 0.52:
        print(f"⚠️ ALERTE : Précision {round(mean_prec * 100, 2)}% < 52% — modèle peu fiable.")
        print("   → Vérifier signal_achat (4 conditions AT) et target_ml avant de déployer.")
    else:
        print(f"✅ Précision moyenne : {round(mean_prec * 100, 2)}% — modèle acceptable.")
 
    # 5. Entraînement Final & Sauvegarde
    print("💪 Entraînement final sur l'ensemble du dataset...")
    final_model = RandomForestClassifier(
        n_estimators=100, max_depth=6,
        class_weight='balanced', random_state=42
    )
    final_model.fit(X, y, sample_weight=weights)
 
    os.makedirs("models", exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(cols, COLS_PATH)
 
    print(f"✅ Modèle sauvegardé : {MODEL_PATH}")
    print(f"✅ Colonnes sauvegardées : {COLS_PATH} ({len(cols)} features)")
 
 
if __name__ == "__main__":
    train_brain()
