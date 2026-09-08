# ============================================================
# analysis.py — Trading Brain — v1.0
# ============================================================
# Analyse incrémentale et complète : calcul des indicateurs techniques
# et du score ML, puis persistance dans actions_prix_historique.
#
# Extrait de main.py (roadmap #5, 2026-09-08). Le lot est à
# ISO-COMPORTEMENT : aucune règle n'est modifiée, seul le découpage change.
#
# ⚠️ RAPPEL DE PÉRIMÈTRE (vérifié le 2026-09-07) : les colonnes écrites ici
#    n'alimentent AUCUN des trois chemins de décision. Le ranking, les 5
#    conditions de sortie et le snapshot avis IA rechargent tous l'historique
#    complet via backtest_ranking.load_all_price_data + compute_all_indicators.
#    Ce module alimente les features ML (train_model.py, gelé),
#    /backtest-detail et /schema-diagnostic. Une valeur fausse ici est une
#    dette de données réelle, pas une décision de trade faussée.
#
# `compute_analysis_indicators` était une fonction IMBRIQUÉE dans
# `run_analysis_logic` : pure (DataFrame → DataFrame), mais inatteignable
# depuis un test. C'est pourtant là que sont nés R4 (min_periods) et une
# partie de #26. Elle est désormais au niveau module et verrouillée par tests.
#
# `engine` et `job_status` sont INJECTÉS et non importés : un
# `from main import engine` créerait un import circulaire. Précédent dans le
# code : `_poll_ai_opinions_with_alert` et `_pipeline_prix_puis_analyse`
# prennent déjà `engine` en paramètre.
#
# ⚠️ DÉPLOIEMENT COUPLÉ : analysis.py + ranking.py + main.py, un seul commit.
#    Import NON gardé côté main.py, comme scheduling.py et freshness.py.
# ============================================================

import io

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text

# Import NON gardé, volontairement : un garde-fou anti-chevauchement qui
# disparaît en silence est pire que son absence (cf. main.py).
from scheduling import analysis_should_skip

ANALYSIS_VERSION = "1.0"

# ============================================================
# Contexte nécessaire pour chaque indicateur glissant :
#   SMA_200  → 200 jours
#   RSI_14   → 14 jours
#   BB_lower → 20 jours
#   vol_avg  → 20 jours
#   ATR_14   → 14 jours (réel si prix_haut/prix_bas disponibles)
# R4 (2026-09-04) : 220 jours CALENDAIRES ≈ 150 barres de bourse
# (jours ouvrés - fériés), donc insuffisant pour une SMA_200 réelle —
# celle écrite en incrémental était en pratique une SMA~150. Or ce
# seuil sert à la fois de filtre d'entrée et de condition de sortie
# (prix < SMA 200) : le moteur décidait sur un indicateur faux.
# 320 jours calendaires ≈ 228 barres, marge suffisante au-dessus de 200
# même les périodes chargées en jours fériés (Noël/nouvel an).
# On ne sauvegarde que les 5 derniers jours (nouveaux / modifiés).
# ============================================================
INCREMENTAL_LOOKBACK_DAYS = 320
INCREMENTAL_SAVE_DAYS     = 5

# ============================================================
# CHARGEMENT DU MODÈLE DEPUIS POSTGRESQL
# ============================================================

def load_model_from_db(engine):
    """
    Charge le modèle ML depuis models_store (PostgreSQL).
    Résiste aux redéploiements Railway (filesystem éphémère).
    Retourne (model, model_cols) ou (None, None) si absent.
    """
    if engine is None:
        return None, None
    try:
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT model_data, columns_data, accuracy, updated_at
                FROM models_store
                WHERE model_name = 'trading_forest'
            """)).fetchone()

        if row:
            model      = joblib.load(io.BytesIO(bytes(row[0])))
            model_cols = joblib.load(io.BytesIO(bytes(row[1])))
            print(f"✅ Modèle ML chargé depuis DB (précision : {round(float(row[2]) * 100, 1)}%, entraîné le {row[3].date()})")
            return model, model_cols

        print("⚠️ Aucun modèle trouvé en base — score_ia sera 0.0. Appeler /train-model.")
        return None, None

    except Exception as e:
        print(f"❌ Erreur chargement modèle depuis DB : {e}")
        return None, None


# ============================================================
# RÈGLE PURE — indicateurs techniques d'un ticker
# ============================================================
# DataFrame → DataFrame, sans DB ni réseau. Était imbriquée dans
# run_analysis_logic jusqu'au 08/09 : intestable alors qu'elle porte
# le correctif R4 (min_periods=200 strict).

def compute_analysis_indicators(group):
    if len(group) < 50: return group
    price = group['prix_ajuste']
    vol   = group['volume']

    # RSI 14
    delta = price.diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    group['rsi_14'] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    # Moyennes mobiles
    # R4 (2026-09-04) : min_periods=200 strict — avec min_periods=1,
    # la moyenne était biaisée vers le prix courant tant que la fenêtre
    # n'avait pas 200 barres (cf. note "Calculs rolling", CLAUDE.md).
    # Lookback 320j (ci-dessus) garantit désormais assez de barres.
    group['sma_200'] = price.rolling(200, min_periods=200).mean()
    group['sma_50']  = price.rolling(50,  min_periods=1).mean()

    # Bandes de Bollinger
    sma_20               = price.rolling(20).mean()
    std_20               = price.rolling(20).std()
    bb_upper             = sma_20 + (std_20 * 2)
    group['bb_lower']    = sma_20 - (std_20 * 2)
    bb_range             = (bb_upper - group['bb_lower']).replace(0, np.nan)
    group['bb_position'] = (price - group['bb_lower']) / bb_range

    # Volume moyen 20j
    group['vol_avg_20'] = vol.rolling(20).mean()

    # Features ML
    group['rsi_slope']   = group['rsi_14'].diff(3)
    group['vol_ratio']   = vol / (group['vol_avg_20'] + 1e-9)
    group['dist_sma200'] = (price - group['sma_200']) / (group['sma_200'] + 1e-9)

    # ATR 14 — réel (H-L) si disponible, approché (C-C) sinon
    has_hl = (
        'prix_haut' in group.columns and
        'prix_bas'  in group.columns and
        group['prix_haut'].notna().any() and
        group['prix_bas'].notna().any()
    )
    if has_hl:
        prev_close = price.shift(1)
        tr = pd.concat([
            group['prix_haut'] - group['prix_bas'],
            (group['prix_haut'] - prev_close).abs(),
            (group['prix_bas']  - prev_close).abs()
        ], axis=1).max(axis=1)
    else:
        tr = price.diff().abs()

    group['atr_14'] = tr.rolling(14, min_periods=1).mean()

    # Régime de marché
    group['regime_marche'] = np.where(
        price > group['sma_50'] * 1.02, 'BULL',
        np.where(price < group['sma_50'] * 0.98, 'BEAR', 'NEUTRE')
    )

    # SIGNAL ACHAT v3.3 supprimé (mean-reversion empiriquement invalidée).
    # L'entrée v4.1 se fait via le ranking momentum top 5
    # (compute_and_store_ranking → ranking_hebdo).
    # La colonne actions_prix_historique.signal_achat n'est plus mise à jour.
    return group

# ============================================================
# LOGIQUE ANALYSE — INCRÉMENTALE ET COMPLÈTE
# ============================================================

def run_analysis_logic(engine, job_status: dict, full: bool = False,
                       app_version: str = ANALYSIS_VERSION):
    """
    Args:
        engine      : SQLAlchemy engine, injecté par main.py (pas d'import
                      circulaire — cf. entête du module)
        job_status  : dict de suivi des jobs scheduler, LU seul (jamais muté
                      ici) par le garde-fou anti-chevauchement
        full        : True = recalcul complet, False = fenêtre incrémentale
        app_version : reportée telle quelle dans le log de démarrage, pour
                      que la ligne continue d'afficher la version de l'API
    """
    if engine is None: return

    # --- Garde-fou anti-chevauchement (incident du 2026-09-07) ---
    # Le chaînage (_pipeline_prix_puis_analyse) supprime la course côté
    # scheduler. Ce garde-fou couvre ce qu'il ne couvre pas : un appel
    # manuel à /run-analysis ou /run-analysis-full pendant que la sync
    # nocturne écrit encore. Règle pure et testée → scheduling.py.
    skip, motif = analysis_should_skip(job_status)
    if skip:
        print(f"⏭️  Analyse annulée — {motif}")
        return {"status": "skipped", "message": motif}

    mode = "COMPLET" if full else "INCRÉMENTAL"
    print(f"🚀 Démarrage Analyse v{app_version} — mode {mode}...")

    try:
        # 1. Liste des tickers
        with engine.connect() as conn:
            result      = conn.execute(text("SELECT DISTINCT ticker FROM actions_prix_historique"))
            all_tickers = [row[0] for row in result]

        if not all_tickers: return
        print(f"   {len(all_tickers)} tickers trouvés en base.")

        # 2. Chargement du modèle ML depuis PostgreSQL
        model, model_cols = load_model_from_db(engine)

        # 3. Définition de la fenêtre de chargement
        if full:
            date_filter = ""
            date_params = {}
        else:
            date_filter = "AND a.date >= :date_from"
            date_params = {
                "date_from": (pd.Timestamp.today() - pd.Timedelta(days=INCREMENTAL_LOOKBACK_DAYS)).date()
            }

        # 4. Traitement par chunks
        chunk_size   = 50
        total_chunks = (len(all_tickers) - 1) // chunk_size + 1

        for i in range(0, len(all_tickers), chunk_size):
            tickers_chunk = all_tickers[i:i + chunk_size]

            # prix_haut et prix_bas inclus pour ATR réel (remplis par /fill-high-low)
            query = text(f"""
                SELECT a.id, a.ticker, a.date,
                       a.prix_cloture, a.prix_ajuste, a.volume,
                       a.prix_haut, a.prix_bas,
                       t.secteur, t.market_cap, t.pe_ratio
                FROM actions_prix_historique a
                LEFT JOIN tickers_info t ON a.ticker = t.ticker
                WHERE a.ticker IN :tickers
                {date_filter}
                ORDER BY a.ticker, a.date ASC
            """)

            params = {"tickers": tuple(tickers_chunk), **date_params}
            df     = pd.read_sql(query, engine, params=params)
            if df.empty: continue

            df['prix_ajuste'] = df['prix_ajuste'].fillna(df['prix_cloture'])
            df = df.dropna(subset=['prix_ajuste']).sort_values(['ticker', 'date'])

            # Règle pure, testée → compute_analysis_indicators (haut du module)
            df = df.groupby('ticker', group_keys=False).apply(compute_analysis_indicators)

            # 5. Score ML
            if model is not None:
                feat_df = pd.get_dummies(df, columns=['secteur', 'regime_marche'])
                for col in model_cols:
                    if col not in feat_df.columns:
                        feat_df[col] = 0
                X_input            = feat_df[model_cols].fillna(0).replace([np.inf, -np.inf], 0)
                df['confiance_ml'] = model.predict_proba(X_input)[:, 1]
            else:
                df['confiance_ml'] = 0.0

            # 6. Filtrage des lignes à sauvegarder
            if not full:
                # .normalize() : sans lui, pd.Timestamp.today() porte l'heure
                # courante et la borne, comparée à des dates à minuit, excluait
                # le jour le plus ancien — 4 jours sauvegardés au lieu de 5
                # (roadmap #26).
                save_from  = (pd.Timestamp.today().normalize()
                              - pd.Timedelta(days=INCREMENTAL_SAVE_DAYS))
                df_to_save = df[pd.to_datetime(df['date']) >= save_from].copy()
            else:
                df_to_save = df.copy()

            if df_to_save.empty:
                continue

            # 7. Sauvegarde — table temporaire unique par chunk
            tmp_table    = f"_tmp_update_{i}"
            cols_to_save = [
                'id', 'rsi_14', 'sma_200', 'bb_lower', 'bb_position',
                'vol_avg_20', 'regime_marche', 'confiance_ml',
                'rsi_slope', 'vol_ratio', 'dist_sma200', 'atr_14'
            ]
            df_update = df_to_save[[c for c in cols_to_save if c in df_to_save.columns]].copy()
            df_update.to_sql(tmp_table, engine, if_exists='replace', index=False)

            with engine.begin() as conn:
                conn.execute(text(f"""
                    UPDATE actions_prix_historique a SET
                        rsi_14        = t.rsi_14,
                        sma_200       = t.sma_200,
                        bb_lower      = t.bb_lower,
                        bb_position   = t.bb_position,
                        vol_avg_20    = t.vol_avg_20,
                        regime_marche = t.regime_marche,
                        score_ia      = t.confiance_ml,
                        rsi_slope     = t.rsi_slope,
                        vol_ratio     = t.vol_ratio,
                        dist_sma200   = t.dist_sma200,
                        atr_14        = t.atr_14
                    FROM {tmp_table} t
                    WHERE a.id = t.id
                """))
                conn.execute(text(f"DROP TABLE IF EXISTS {tmp_table}"))

            rows_saved = len(df_update)
            print(f"🟢 Chunk {i // chunk_size + 1}/{total_chunks} — {rows_saved} lignes sauvegardées.")

        print(f"🏁 Analyse {mode} terminée.")

    except Exception as e:
        print(f"❌ Erreur run_analysis_logic (full={full}) : {e}")
