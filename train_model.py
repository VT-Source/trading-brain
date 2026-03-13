import os
import joblib
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score
from dotenv import load_dotenv
 
load_dotenv()
 
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
 
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
 
MODEL_PATH = "models/trading_forest.joblib"
COLS_PATH  = "models/trading_forest_cols.joblib"
 
def train_brain():
    print("🧠 Entraînement v3.2 : AT signal → ML confirm (régime dynamique)...")
 
    # --------------------------------------------------------
    # ÉTAPE 1 : Chargement — UNIQUEMENT les signaux AT
    # --------------------------------------------------------
    query = """
        SELECT a.rsi_14, a.vol_avg_20, a.sma_200, a.bb_lower,
               a.rsi_slope, a.vol_ratio, a.dist_sma200, a.bb_position,
               a.regime_marche,
               t.secteur, t.market_cap, t.pe_ratio,
               a.target_ml,
               a.date
        FROM actions_prix_historique a
        JOIN tickers_info t ON a.ticker = t.ticker
        WHERE a.signal_achat = TRUE
          AND a.target_ml    IS NOT NULL
          AND a.rsi_14       IS NOT NULL
        ORDER BY a.date ASC
    """
 
    try:
        df = pd.read_sql(query, engine)
    except Exception as e:
        print(f"❌ Erreur lecture DB : {e}")
        return
 
    if len(df) < 100:
        print(
            f"❌ Données insuffisantes ({len(df)} signaux AT avec target_ml).\n"
            f"   → Lancez /run-analysis puis target_ml.sql avant de réentraîner"
        )
        return
 
    print(f"📊 Dataset : {len(df)} signaux AT | "
          f"Taux de succès brut : {round(df['target_ml'].mean() * 100, 1)}%")
 
    # Afficher la distribution des régimes dans le dataset
    if 'regime_marche' in df.columns:
        regime_dist = df['regime_marche'].value_counts()
        print(f"📊 Distribution régimes : {regime_dist.to_dict()}")
 
    # --------------------------------------------------------
    # ÉTAPE 2 : Nettoyage
    # --------------------------------------------------------
    df = df.replace([np.inf, -np.inf], np.nan)
    df['rsi_14']      = df['rsi_14'].fillna(50)
    df['rsi_slope']   = df['rsi_slope'].fillna(0)
    df['vol_ratio']   = df['vol_ratio'].fillna(1)
    df['dist_sma200'] = df['dist_sma200'].fillna(0)
    df['bb_position'] = df['bb_position'].fillna(0)
    df['regime_marche'] = df['regime_marche'].fillna('NEUTRE')
    df['pe_ratio']    = df['pe_ratio'].fillna(df['pe_ratio'].median())
    df['market_cap']  = df['market_cap'].fillna(df['market_cap'].median())
    df['secteur']     = df['secteur'].fillna('Unknown')
    df                = df.fillna(0)
 
    # --------------------------------------------------------
    # ÉTAPE 3 : Préparation features + pondération temporelle
    # --------------------------------------------------------
    df = pd.get_dummies(df, columns=['secteur', 'regime_marche'])
 
    # Tri chronologique garanti avant pondération
    df = df.sort_values('date').reset_index(drop=True)
 
    X    = df.drop(['target_ml', 'date'], axis=1)
    y    = df['target_ml'].astype(int)
    cols = list(X.columns)
 
    # --------------------------------------------------------
    # APPROCHE 1 — Pondération temporelle exponentielle
    # --------------------------------------------------------
    # Les signaux récents comptent exponentiellement plus
    # que les signaux anciens (facteur e^1 ≈ 2.7x)
    # Exemple : un signal de 2025 pèse ~2.7x plus qu'un signal de 2021
    sample_weights = np.exp(np.linspace(0, 1, len(X)))
 
    # --------------------------------------------------------
    # ÉTAPE 4 : Validation temporelle (TimeSeriesSplit)
    # --------------------------------------------------------
    tscv          = TimeSeriesSplit(n_splits=5)
    scores_test   = []
    scores_prec   = []
    scores_recall = []
 
    print("⏳ Validation croisée temporelle (5 splits)...")
 
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        w_train         = sample_weights[train_idx]  # ← poids temporels
 
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced_subsample'
        )
        # Entraînement avec pondération temporelle
        clf.fit(X_train, y_train, sample_weight=w_train)
 
        acc  = clf.score(X_test, y_test)
        pred = clf.predict(X_test)
        prec = precision_score(y_test, pred, zero_division=0)
        rec  = recall_score(y_test, pred, zero_division=0)
 
        scores_test.append(acc)
        scores_prec.append(prec)
        scores_recall.append(rec)
 
        print(f"   Fold {fold + 1} → Accuracy: {round(acc*100,1)}% | "
              f"Précision: {round(prec*100,1)}% | "
              f"Recall: {round(rec*100,1)}%")
 
    mean_acc  = round(np.mean(scores_test)  * 100, 2)
    mean_prec = round(np.mean(scores_prec)  * 100, 2)
    mean_rec  = round(np.mean(scores_recall)* 100, 2)
 
    print(f"\n📈 Résultats moyens sur 5 folds :")
    print(f"   Accuracy  : {mean_acc}%")
    print(f"   Précision : {mean_prec}%  ← % de vrais positifs parmi les signaux ML")
    print(f"   Recall    : {mean_rec}%   ← % de succès AT détectés par le ML")
 
    if mean_prec < 52:
        print(f"⚠️  Précision < 52% — modèle encore en amélioration.")
    else:
        print(f"✅ Précision acceptable (> 52%)")
 
    # --------------------------------------------------------
    # ÉTAPE 5 : Entraînement final avec pondération temporelle
    # --------------------------------------------------------
    print("\n💪 Entraînement final sur l'ensemble du dataset...")
    final_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced_subsample'
    )
    final_model.fit(X, y, sample_weight=sample_weights)
    final_score = final_model.score(X, y)
 
    # --------------------------------------------------------
    # ÉTAPE 6 : Sauvegarde joblib
    # --------------------------------------------------------
    os.makedirs("models", exist_ok=True)
 
    try:
        joblib.dump(final_model, MODEL_PATH)
        joblib.dump(cols,        COLS_PATH)
        print(f"💾 Modèle sauvegardé → {MODEL_PATH}")
        print(f"💾 Colonnes sauvegardées → {COLS_PATH}")
    except Exception as e:
        print(f"❌ Erreur sauvegarde joblib : {e}")
        return
 
    # Sauvegarde métriques en base
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO models_store
                    (model_name, model_data, columns_data, accuracy, updated_at)
                VALUES
                    ('trading_forest_meta', :placeholder, :placeholder2,
                     :acc, CURRENT_TIMESTAMP)
                ON CONFLICT (model_name) DO UPDATE SET
                    accuracy   = EXCLUDED.accuracy,
                    updated_at = CURRENT_TIMESTAMP;
            """), {
                "placeholder" : b"joblib_file",
                "placeholder2": b"joblib_file",
                "acc"         : float(mean_prec / 100)
            })
        print(f"📊 Métriques sauvegardées en base (précision CV : {mean_prec}%)")
    except Exception as e:
        print(f"⚠️ Sauvegarde métriques en base : {e}")
 
    print(f"\n✅ Modèle 'trading_forest' prêt.")
    print(f"   Dataset     : {len(df)} signaux AT uniquement")
    print(f"   Features    : {len(cols)} colonnes")
    print(f"   Précision CV: {mean_prec}% | Score final : {round(final_score*100,2)}%")
    print(f"   ⚠️  Si score final >> précision CV → overfitting résiduel normal")
 
if __name__ == "__main__":
    train_brain()
