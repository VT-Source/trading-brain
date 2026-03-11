
Copier

import os
import time
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from fastapi import FastAPI, BackgroundTasks
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Import de la fonction d'entraînement
try:
    from train_model import train_brain
except ImportError:
    def train_brain():
        print("⚠️ Fonction train_brain non trouvée.")

load_dotenv()

app = FastAPI()

# --- Configuration DB ---
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True) if DATABASE_URL else None

# --- Chemin du modèle ML ---
MODEL_PATH = "models/trading_forest.joblib"
COLS_PATH  = "models/trading_forest_cols.joblib"

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
def home():
    return {
        "status": "Service Trading IA Actif",
        "version": "5.0.0",
        "architecture": "AT-signal → ML-confirm → NLP-veto"
    }

@app.get("/train-model")
async def trigger_training(background_tasks: BackgroundTasks):
    background_tasks.add_task(train_brain)
    return {"status": "processing", "message": "Entraînement lancé."}

@app.get("/sync-metadata")
async def trigger_sync(background_tasks: BackgroundTasks):
    background_tasks.add_task(sync_metadata_logic)
    return {"status": "processing", "message": "Sync Yahoo Finance lancée."}

@app.get("/run-analysis")
async def trigger_analysis(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_analysis_logic)
    return {"status": "processing", "message": "Analyse technique massive lancée..."}

# ============================================================
# LOGIQUE SYNC METADATA
# ============================================================

def sync_metadata_logic():
    if engine is None: return
    print("🔄 Sync Yahoo Finance (Metadata)...")
    query = """
        SELECT DISTINCT ticker FROM actions_prix_historique
        WHERE ticker NOT IN (SELECT ticker FROM tickers_info)
        OR ticker IN (
            SELECT ticker FROM tickers_info
            WHERE derniere_maj < CURRENT_DATE - INTERVAL '30 days'
        )
        LIMIT 500;
    """
    try:
        with engine.connect() as conn:
            tickers = [row[0] for row in conn.execute(text(query))]

        for ticker_symbol in tickers:
            try:
                stock = yf.Ticker(ticker_symbol)
                data  = stock.info
                if "symbol" in data:
                    metadata = {
                        "ticker"  : ticker_symbol,
                        "name"    : data.get("longName"),
                        "secteur" : data.get("sector"),
                        "industrie": data.get("industry"),
                        "pays"    : data.get("country"),
                        "monnaie" : data.get("currency"),
                        "market_cap": data.get("marketCap"),
                        "pe_ratio": data.get("trailingPE")
                    }
                    with engine.begin() as conn:
                        conn.execute(text("""
                            INSERT INTO tickers_info
                                (ticker, name, secteur, industrie, pays, monnaie,
                                 market_cap, pe_ratio, derniere_maj)
                            VALUES
                                (:ticker, :name, :secteur, :industrie, :pays, :monnaie,
                                 :market_cap, :pe_ratio, CURRENT_DATE)
                            ON CONFLICT (ticker) DO UPDATE SET
                                pe_ratio    = EXCLUDED.pe_ratio,
                                market_cap  = EXCLUDED.market_cap,
                                derniere_maj = CURRENT_DATE;
                        """), metadata)
                time.sleep(1)
            except Exception:
                continue
    except Exception as e:
        print(f"❌ Erreur Sync: {e}")

# ============================================================
# LOGIQUE ANALYSE MASSIVE v5.0.0
# Architecture : AT signal → ML confirm → log signaux
# ============================================================

def run_analysis_logic():
    if engine is None: return
    print("🚀 Démarrage de l'analyse v5.0.0 (AT → ML → Log)...")

    try:
        # --- Récupération de tous les tickers ---
        with engine.connect() as conn:
            result      = conn.execute(text(
                "SELECT DISTINCT ticker FROM actions_prix_historique"
            ))
            all_tickers = [row[0] for row in result]

        if not all_tickers: return

        # --- Chargement modèle ML (joblib) ---
        model, model_cols = None, None
        if os.path.exists(MODEL_PATH) and os.path.exists(COLS_PATH):
            try:
                model      = joblib.load(MODEL_PATH)
                model_cols = joblib.load(COLS_PATH)
                print("✅ Modèle ML chargé depuis joblib.")
            except Exception as e:
                print(f"⚠️ Impossible de charger le modèle ML : {e}")

        chunk_size = 20
        total_rows = 0

        for i in range(0, len(all_tickers), chunk_size):
            tickers_chunk = all_tickers[i:i + chunk_size]

            query = text("""
                SELECT a.id, a.ticker, a.date,
                       a.prix_cloture, a.prix_ajuste, a.volume,
                       t.secteur, t.market_cap, t.pe_ratio
                FROM actions_prix_historique a
                LEFT JOIN tickers_info t ON a.ticker = t.ticker
                WHERE a.ticker IN :tickers
                ORDER BY a.ticker, a.date ASC
            """)

            df = pd.read_sql(query, engine, params={"tickers": tuple(tickers_chunk)})
            if df.empty: continue

            # Sécurité prix
            df['prix_ajuste'] = df['prix_ajuste'].fillna(df['prix_cloture'])
            df['prix_ajuste'] = pd.to_numeric(df['prix_ajuste'], errors='coerce')
            df = df.dropna(subset=['prix_ajuste']).sort_values(['ticker', 'date'])

            # --- CALCULS TECHNIQUES PAR TICKER ---
            def compute_group(group):
                if len(group) < 14: return group

                price = group['prix_ajuste']
                vol   = group['volume']

                # RSI 14
                delta = price.diff()
                gain  = delta.where(delta > 0, 0).rolling(14, min_periods=7).mean()
                loss  = (-delta.where(delta < 0, 0)).rolling(14, min_periods=7).mean()
                group['rsi_14'] = 100 - (100 / (1 + gain / (loss + 1e-9)))

                # SMA 200
                group['sma_200'] = price.rolling(200, min_periods=1).mean()

                # Volume moyen 20j
                group['vol_avg_20'] = vol.rolling(20, min_periods=1).mean()

                # Bollinger Band inférieure
                sma_20           = price.rolling(20, min_periods=1).mean()
                std_20           = price.rolling(20, min_periods=1).std()
                group['bb_lower'] = sma_20 - (std_20 * 2)

                # --------------------------------------------------------
                # COUCHE 1 — SIGNAL D'ACHAT AT (4 conditions cumulatives)
                # --------------------------------------------------------
                group['signal_achat'] = (
                    (group['rsi_14']    < 35)               # Survente
                    & (price            > group['sma_200'])  # Tendance haussière
                    & (vol              > group['vol_avg_20'])# Confirmation volume
                    & (price            < group['bb_lower'])  # Rebond Bollinger
                )

                # Target ML (pour réentraînement futur)
                group['prix_futur'] = price.shift(-10)
                group['target_ml']  = (
                    (group['signal_achat']) &
                    ((group['prix_futur'] - price) / price >= 0.05)
                ).astype(float)

                return group

            df = df.groupby('ticker', group_keys=False).apply(compute_group)

            # --------------------------------------------------------
            # COUCHE 2 — SCORE ML (confirmation du signal AT)
            # --------------------------------------------------------
            if model is not None:
                feat_df = pd.get_dummies(
                    df[['rsi_14', 'vol_avg_20', 'sma_200', 'bb_lower',
                        'secteur', 'market_cap', 'pe_ratio']],
                    columns=['secteur']
                )
                # Aligner les colonnes avec celles de l'entraînement
                for col in model_cols:
                    if col not in feat_df.columns:
                        feat_df[col] = 0
                feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0)

                df['confiance_ml'] = model.predict_proba(
                    feat_df[model_cols]
                )[:, 1]
            else:
                df['confiance_ml'] = 0.0

            # --------------------------------------------------------
            # LOGGING DANS signaux_log (traçabilité)
            # --------------------------------------------------------
            today        = pd.Timestamp.today().date()
            df_today     = df[df['date'] == str(today)].copy()

            if not df_today.empty:
                log_rows = []
                for _, row in df_today.iterrows():
                    signal  = bool(row.get('signal_achat', False))
                    conf_ml = float(row.get('confiance_ml', 0.0))

                    if not signal:
                        decision = 'IGNORÉ'
                    elif conf_ml <= 0.6:
                        decision = 'REJETÉ_ML'
                    else:
                        decision = 'SUGGESTION'

                    log_rows.append({
                        "ticker"         : row['ticker'],
                        "date"           : today,
                        "rsi_14"         : row.get('rsi_14'),
                        "sma_200"        : row.get('sma_200'),
                        "vol_avg_20"     : row.get('vol_avg_20'),
                        "bb_lower"       : row.get('bb_lower'),
                        "signal_achat"   : signal,
                        "confiance_ml"   : conf_ml if signal else None,
                        "sentiment_label": None,        # Phase 3
                        "decision_finale": decision,
                        "prix_j0"        : row.get('prix_ajuste'),
                    })

                if log_rows:
                    df_log = pd.DataFrame(log_rows)
                    with engine.begin() as conn:
                        for _, lr in df_log.iterrows():
                            conn.execute(text("""
                                INSERT INTO signaux_log
                                    (ticker, date, rsi_14, sma_200, vol_avg_20,
                                     bb_lower, signal_achat, confiance_ml,
                                     sentiment_label, decision_finale, prix_j0)
                                VALUES
                                    (:ticker, :date, :rsi_14, :sma_200, :vol_avg_20,
                                     :bb_lower, :signal_achat, :confiance_ml,
                                     :sentiment_label, :decision_finale, :prix_j0)
                                ON CONFLICT DO NOTHING;
                            """), lr.to_dict())

            # --------------------------------------------------------
            # SAUVEGARDE EN BASE (casting strict pour PostgreSQL)
            # --------------------------------------------------------
            cols_to_save = [
                'id', 'rsi_14', 'vol_avg_20', 'sma_200', 'bb_lower',
                'signal_achat', 'confiance_ml', 'target_ml'
            ]
            df_save = df[cols_to_save].copy()
            df_save.to_sql("_tmp_chunk", engine, if_exists='replace', index=False)

            with engine.begin() as conn:
                conn.execute(text("""
                    UPDATE actions_prix_historique a
                    SET
                        rsi_14       = CAST(t.rsi_14       AS numeric),
                        vol_avg_20   = CAST(t.vol_avg_20   AS numeric),
                        sma_200      = CAST(t.sma_200      AS numeric),
                        bb_lower     = CAST(t.bb_lower     AS numeric),
                        signal_achat = t.signal_achat,
                        score_ia     = CAST(t.confiance_ml AS numeric),
                        target_ml    = CAST(t.target_ml    AS numeric)
                    FROM _tmp_chunk t
                    WHERE a.id = t.id;
                """))

            total_rows += len(df)
            print(f"🟢 Paquet {i // chunk_size + 1} traité ({total_rows} lignes).")

        # Mise à jour des résultats J+10 dans signaux_log
        try:
            with engine.begin() as conn:
                updated = conn.execute(
                    text("SELECT update_signaux_log_resultats()")
                ).scalar()
                if updated:
                    print(f"📊 {updated} résultats J+10 mis à jour dans signaux_log.")
        except Exception as e:
            print(f"⚠️ Mise à jour signaux_log J+10 : {e}")

        print("🏁 ANALYSE TERMINÉE.")

    except Exception as e:
        print(f"❌ Erreur critique : {e}")
