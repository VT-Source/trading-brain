import io
import os
import time
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

try:
    from train_model import train_brain
except ImportError:
    def train_brain():
        print("⚠️ Fonction train_brain non trouvée dans train_model.py")

try:
    from backtest import run_backtest_logic
except ImportError:
    def run_backtest_logic(**kwargs):
        return {"erreur": "backtest.py non trouvé"}
try:
    from backtest_ranking import (
        run_backtest_ranking_logic,
        load_all_tickers,
        load_all_price_data,
        compute_all_indicators,
        compute_composite_score,
        compute_adaptive_k,
        load_secteur_mapping,
        load_all_secteur_force,
        load_macro_data,
        get_macro_regime,
        get_ticker_zone,
        get_ticker_zone,
        get_secteur_force_for_ticker,
    )
except ImportError:
    def run_backtest_ranking_logic(**kwargs):
        return {"erreur": "backtest_ranking.py non trouvé"}
    # Stubs pour les fonctions individuelles
    load_all_tickers = None
    load_all_price_data = None
    compute_all_indicators = None
    compute_composite_score = None
    compute_adaptive_k = None
    load_secteur_mapping = None
    load_all_secteur_force = None
    load_macro_data = None
    get_macro_regime = None
    get_ticker_zone = None
    get_secteur_force_for_ticker = None

load_dotenv()

app = FastAPI()

# --- CONFIGURATION DATABASE ---
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True) if DATABASE_URL else None

# ============================================================
# Contexte nécessaire pour chaque indicateur glissant :
#   SMA_200  → 200 jours
#   RSI_14   → 14 jours
#   BB_lower → 20 jours
#   vol_avg  → 20 jours
#   ATR_14   → 14 jours (réel si prix_haut/prix_bas disponibles)
# On charge 220 jours pour avoir la SMA_200 correcte + marge.
# On ne sauvegarde que les 5 derniers jours (nouveaux / modifiés).
# ============================================================
INCREMENTAL_LOOKBACK_DAYS = 220
INCREMENTAL_SAVE_DAYS     = 5


# ============================================================
# ENDPOINTS API
# ============================================================

@app.get("/")
def home():
    return {
        "status"          : "Service Trading IA Actif",
        "version"         : "6.3.0",
        "engine_connected": engine is not None
    }

@app.get("/check-model")
def check_model():
    """Diagnostic — vérifie si le modèle est présent en base."""
    if engine is None:
        return {"error": "engine non connecté"}
    try:
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT model_name, accuracy, updated_at,
                       LENGTH(model_data)   AS model_size_bytes,
                       LENGTH(columns_data) AS cols_size_bytes
                FROM models_store
                WHERE model_name = 'trading_forest'
            """)).fetchone()
        if row:
            return {
                "model_found"      : True,
                "model_name"       : row[0],
                "accuracy"         : f"{round(row[1] * 100, 2)}%",
                "updated_at"       : str(row[2]),
                "model_size_bytes" : row[3],
                "cols_size_bytes"  : row[4]
            }
        return {"model_found": False, "message": "Aucun modèle en base — appeler /train-model"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/train-model")
async def trigger_training(background_tasks: BackgroundTasks):
    background_tasks.add_task(train_brain)
    return {"status": "processing", "message": "Entraînement lancé en arrière-plan."}

# --- Endpoint sync prix quotidien (remplace n8n) ---
@app.get("/sync-prix")
async def trigger_sync_prix(background_tasks: BackgroundTasks, full: bool = False, period: str = None):
    """
    Synchronise les prix OHLCV depuis yfinance → actions_prix_historique.
    - full=False (défaut) : 30 derniers jours — appel quotidien scheduler
    - full=True           : 5 ans d'historique — rattrapage manuel
    """
    background_tasks.add_task(sync_prix_logic, full=full, period_override=period)
    mode = f"custom ({period})" if period else ("COMPLÈTE (5 ans)" if full else "incrémentale (30j)")
    return {
        "status" : "processing",
        "message": f"Sync prix lancée en arrière-plan — mode {mode}."
    }

@app.get("/sync-metadata")
async def trigger_sync(background_tasks: BackgroundTasks):
    background_tasks.add_task(sync_metadata_logic)
    return {"status": "processing", "message": "Synchronisation Yahoo lancée."}

@app.get("/sync-etf-sectoriels")
async def trigger_sync_etf(background_tasks: BackgroundTasks, full: bool = False):
    """
    Synchronise les prix des ETF sectoriels et indices de référence.
    - full=False (défaut) : 30 derniers jours — appel quotidien scheduler 06h15
    - full=True           : 5 ans d'historique — initialisation manuelle uniquement
    """
    background_tasks.add_task(sync_secteurs_etf_logic, full=full)
    mode = "COMPLÈTE (5 ans)" if full else "incrémentale (30j)"
    return {
        "status" : "processing",
        "message": f"Sync ETF sectoriels lancée en arrière-plan — mode {mode}."
    }

@app.get("/secteurs-actifs")
def get_secteurs_actifs_endpoint():
    """
    Retourne les secteurs actuellement en force relative.
    Utilisé par le dashboard Streamlit (Onglet 1 & 3).
    """
    secteurs = get_secteurs_en_force()
    return {
        "date"              : str(date.today()),
        "nb_secteurs_actifs": len(secteurs),
        "secteurs"          : secteurs
    }

@app.get("/compute-ranking")
async def trigger_compute_ranking(background_tasks: BackgroundTasks, top_n: int = 20):
    """
    Lance le calcul du ranking en arrière-plan.
    Résultat stocké dans ranking_hebdo, lu par /ranking-live.
    """
    background_tasks.add_task(compute_and_store_ranking, top_n=top_n)
    return {
        "status": "processing",
        "message": f"Calcul ranking lancé en arrière-plan (top {top_n})."
    }

@app.get("/ranking-live")
def ranking_live(top_n: int = 20):
    """
    Retourne le dernier ranking pré-calculé depuis ranking_hebdo.
    Instantané (~50ms).
    """
    if engine is None:
        return {"error": "engine non connecté"}
 
    try:
        import json
 
        with engine.connect() as conn:
            last_date = conn.execute(text(
                "SELECT MAX(date_calcul) FROM ranking_hebdo"
            )).scalar()
 
            if last_date is None:
                return {"error": "Aucun ranking calculé. Appeler /compute-ranking d'abord."}
 
            rows = conn.execute(text("""
                SELECT rank, ticker, score, mom_r2, rvol, obv_slope,
                       prix, sma_200, atr_14, k_adaptatif, zone, secteur,
                       macro_regime, nb_eligible, nb_total, data_date
                FROM ranking_hebdo
                WHERE date_calcul = :d
                ORDER BY rank ASC
                LIMIT :n
            """), {"d": last_date, "n": top_n}).fetchall()
 
        if not rows:
            return {"error": "Ranking vide."}
 
        raw = rows[0][12]
        macro_regime = raw if isinstance(raw, dict) else json.loads(raw) if raw else {}
 
        ranking = []
        for r in rows:
            ranking.append({
                "rank":      r[0],
                "ticker":    r[1],
                "score":     float(r[2]) if r[2] else 0,
                "mom_r2":    float(r[3]) if r[3] else 0,
                "rvol":      float(r[4]) if r[4] else 0,
                "obv_slope": float(r[5]) if r[5] else 0,
                "prix":      float(r[6]) if r[6] else 0,
                "sma_200":   float(r[7]) if r[7] else 0,
                "atr_14":    float(r[8]) if r[8] else 0,
                "k":         float(r[9]) if r[9] else 3.0,
                "zone":      r[10],
                "secteur":   r[11],
            })
 
        return {
            "ranking": ranking,
            "macro_regime": macro_regime,
            "meta": {
                "data_date":        str(rows[0][15]),
                "nb_total_tickers": int(rows[0][14]) if rows[0][14] else 0,
                "nb_eligible":      int(rows[0][13]) if rows[0][13] else 0,
                "top_n":            top_n,
                "date_calcul":      str(last_date),
            },
        }
 
    except Exception as e:
        return {"error": str(e)}
        
@app.get("/macro-status")
def macro_status():
    """
    Retourne l'état macro complet : régime par zone + détails indices.
    Utilisé par le dashboard Streamlit (page Macro & Secteurs).
    """
    if engine is None:
        return {"error": "engine non connecté"}
 
    try:
        import pandas as pd
 
        # Charger les données d'indices
        with engine.connect() as conn:
            df_indices = pd.read_sql(text("""
                SELECT ticker_indice, date, prix_ajuste
                FROM indices_prix
                WHERE ticker_indice IN ('^GSPC', '^STOXX')
                ORDER BY ticker_indice, date ASC
            """), conn)
 
        if df_indices.empty:
            return {"error": "Aucune donnée d'indices en base"}
 
        df_indices["date"] = pd.to_datetime(df_indices["date"])
 
        zones = []
        zone_map = {
            "^GSPC":  {"zone": "US", "nom": "S&P 500"},
            "^STOXX": {"zone": "EU", "nom": "STOXX Europe 600"},
        }
 
        for ticker_indice, info in zone_map.items():
            df_z = df_indices[df_indices["ticker_indice"] == ticker_indice].set_index("date").sort_index()
            if df_z.empty:
                continue
 
            prix = float(df_z["prix_ajuste"].iloc[-1])
            sma_200 = float(df_z["prix_ajuste"].rolling(200, min_periods=200).mean().iloc[-1]) \
                if len(df_z) >= 200 else None
 
            is_bull = prix > sma_200 if sma_200 else None
 
            zones.append({
                "zone":         info["zone"],
                "indice":       info["nom"],
                "ticker":       ticker_indice,
                "prix_indice":  round(prix, 2),
                "sma_200":      round(sma_200, 2) if sma_200 else None,
                "bullish":      is_bull,
                "date":         str(df_z.index[-1].date()),
            })
 
        return {
            "zones": zones,
            "date":  str(date.today()),
        }
 
    except Exception as e:
        return {"error": str(e)}

# ============================================================
# ENDPOINTS — DÉCISIONS HUMAINES (v2)
# ============================================================

class DecisionPayload(BaseModel):
    semaine:     str            # format YYYY-MM-DD (lundi de la semaine)
    ticker:      str
    rang:        Optional[int]  = None
    decision:    str            # 'suivi' | 'ignore' | 'modifie'
    commentaire: Optional[str]  = None

class PositionOpenPayload(BaseModel):
    ticker:      str
    date_achat:  str              # format YYYY-MM-DD
    prix_achat:  float
    quantite:    float
    decision_id: Optional[int]  = None
    source:      Optional[str]  = "ranking"   # 'ranking' | 'manuel'
    commentaire: Optional[str]  = None
 
 
class PositionClosePayload(BaseModel):
    date_vente:   str            # format YYYY-MM-DD
    prix_vente:   float
    raison_vente: str            # TRAILING_STOP | TREND_BROKEN | MOMENTUM_LOST | SECTOR_WEAK | MACRO_BEARISH | MANUEL
 
 
class PositionEditPayload(BaseModel):
    prix_achat:  Optional[float] = None
    quantite:    Optional[float] = None
    date_achat:  Optional[str]   = None
    commentaire: Optional[str]   = None
    decision_id: Optional[int]   = None

@app.post("/decisions")
def upsert_decision(payload: DecisionPayload):
    """
    Crée ou met à jour la décision humaine pour un ticker/semaine.
    UPSERT : si (semaine, ticker) existe déjà, on écrase.
    """
    if engine is None:
        return {"error": "engine non connecté"}

    decisions_valides = {"suivi", "ignore", "modifie"}
    if payload.decision not in decisions_valides:
        return {"error": f"Décision invalide. Valeurs acceptées : {decisions_valides}"}

    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO decisions_humaines (semaine, ticker, rang, decision, commentaire, updated_at)
                VALUES (:semaine, :ticker, :rang, :decision, :commentaire, NOW())
                ON CONFLICT (semaine, ticker)
                DO UPDATE SET
                    decision    = EXCLUDED.decision,
                    commentaire = EXCLUDED.commentaire,
                    rang        = EXCLUDED.rang,
                    updated_at  = NOW()
            """), {
                "semaine":     payload.semaine,
                "ticker":      payload.ticker.upper(),
                "rang":        payload.rang,
                "decision":    payload.decision,
                "commentaire": payload.commentaire,
            })
        return {
            "status":  "ok",
            "semaine": payload.semaine,
            "ticker":  payload.ticker.upper(),
            "decision": payload.decision,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/decisions")
def get_decisions(semaine: Optional[str] = None):
    """
    Retourne les décisions humaines.
    - ?semaine=YYYY-MM-DD  → décisions de cette semaine uniquement
    - sans paramètre        → toutes les semaines disponibles + leurs décisions
    """
    if engine is None:
        return {"error": "engine non connecté"}

    try:
        with engine.connect() as conn:
            if semaine:
                rows = conn.execute(text("""
                    SELECT semaine, ticker, rang, decision, commentaire, updated_at
                    FROM decisions_humaines
                    WHERE semaine = :semaine
                    ORDER BY rang ASC NULLS LAST, ticker ASC
                """), {"semaine": semaine}).fetchall()
            else:
                rows = conn.execute(text("""
                    SELECT semaine, ticker, rang, decision, commentaire, updated_at
                    FROM decisions_humaines
                    ORDER BY semaine DESC, rang ASC NULLS LAST
                """)).fetchall()

        decisions = [
            {
                "semaine":     str(r[0]),
                "ticker":      r[1],
                "rang":        r[2],
                "decision":    r[3],
                "commentaire": r[4],
                "updated_at":  str(r[5]),
            }
            for r in rows
        ]

        # Statistiques rapides par semaine
        semaines = {}
        for d in decisions:
            s = d["semaine"]
            if s not in semaines:
                semaines[s] = {"suivi": 0, "ignore": 0, "modifie": 0, "total": 0}
            semaines[s][d["decision"]] += 1
            semaines[s]["total"] += 1

        return {
            "nb_total":  len(decisions),
            "semaines":  semaines,
            "decisions": decisions,
        }
    except Exception as e:
        return {"error": str(e)}

# ============================================================
# ENDPOINTS — POSITIONS (v4.1)
# ============================================================

@app.post("/positions")
def open_position(payload: PositionOpenPayload):
    """
    Ouvre une nouvelle position (saisie manuelle après exécution broker).
    """
    if engine is None:
        return {"error": "engine non connecté"}

    sources_valides = {"ranking", "manuel"}
    if payload.source not in sources_valides:
        return {"error": f"Source invalide. Valeurs acceptées : {sources_valides}"}

    try:
        with engine.begin() as conn:
            result = conn.execute(text("""
                INSERT INTO positions (ticker, date_achat, prix_achat, quantite,
                                       decision_id, source, commentaire)
                VALUES (:ticker, :date_achat, :prix_achat, :quantite,
                        :decision_id, :source, :commentaire)
                RETURNING id, montant_investi
            """), {
                "ticker":      payload.ticker.upper(),
                "date_achat":  payload.date_achat,
                "prix_achat":  payload.prix_achat,
                "quantite":    payload.quantite,
                "decision_id": payload.decision_id,
                "source":      payload.source,
                "commentaire": payload.commentaire,
            })
            row = result.fetchone()

        return {
            "status":          "ok",
            "id":              row[0],
            "ticker":          payload.ticker.upper(),
            "montant_investi": float(row[1]),
            "message":         f"Position ouverte : {payload.quantite} × {payload.ticker.upper()} à {payload.prix_achat}",
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/positions")
def list_positions(status: Optional[str] = None):
    """
    Liste les positions.
    - ?status=open   → positions ouvertes uniquement
    - ?status=closed → positions fermées uniquement
    - sans paramètre → toutes
    """
    if engine is None:
        return {"error": "engine non connecté"}

    try:
        where_clause = ""
        if status == "open":
            where_clause = "WHERE p.statut = 'OUVERT'"
        elif status == "closed":
            where_clause = "WHERE p.statut = 'FERMÉ'"

        with engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT p.id, p.ticker, p.date_achat, p.prix_achat, p.quantite,
                       p.montant_investi, p.statut,
                       p.date_vente, p.prix_vente, p.raison_vente,
                       p.resultat_eur, p.resultat_pct,
                       p.decision_id, p.source, p.commentaire,
                       p.created_at, p.updated_at
                FROM positions p
                {where_clause}
                ORDER BY p.date_achat DESC, p.id DESC
            """)).fetchall()

        positions = []
        for r in rows:
            positions.append({
                "id":              r[0],
                "ticker":          r[1],
                "date_achat":      str(r[2]),
                "prix_achat":      float(r[3]),
                "quantite":        float(r[4]),
                "montant_investi": float(r[5]) if r[5] else None,
                "statut":          r[6],
                "date_vente":      str(r[7]) if r[7] else None,
                "prix_vente":      float(r[8]) if r[8] else None,
                "raison_vente":    r[9],
                "resultat_eur":    float(r[10]) if r[10] else None,
                "resultat_pct":    float(r[11]) if r[11] else None,
                "decision_id":     r[12],
                "source":          r[13],
                "commentaire":     r[14],
                "created_at":      str(r[15]),
                "updated_at":      str(r[16]),
            })

        ouvertes = [p for p in positions if p["statut"] == "OUVERT"]
        fermees  = [p for p in positions if p["statut"] == "FERMÉ"]

        return {
            "nb_total":    len(positions),
            "nb_ouvertes": len(ouvertes),
            "nb_fermees":  len(fermees),
            "positions":   positions,
        }
    except Exception as e:
        return {"error": str(e)}


@app.patch("/positions/{position_id}")
def edit_position(position_id: int, payload: PositionEditPayload):
    """
    Modifie une position ouverte (correction prix, quantité, commentaire...).
    Seules les positions OUVERTES peuvent être éditées.
    """
    if engine is None:
        return {"error": "engine non connecté"}

    updates = []
    params = {"position_id": position_id}

    if payload.prix_achat is not None:
        updates.append("prix_achat = :prix_achat")
        params["prix_achat"] = payload.prix_achat
    if payload.quantite is not None:
        updates.append("quantite = :quantite")
        params["quantite"] = payload.quantite
    if payload.date_achat is not None:
        updates.append("date_achat = :date_achat")
        params["date_achat"] = payload.date_achat
    if payload.commentaire is not None:
        updates.append("commentaire = :commentaire")
        params["commentaire"] = payload.commentaire
    if payload.decision_id is not None:
        updates.append("decision_id = :decision_id")
        params["decision_id"] = payload.decision_id

    if not updates:
        return {"error": "Aucun champ à modifier"}

    updates.append("updated_at = NOW()")

    try:
        with engine.begin() as conn:
            check = conn.execute(text("""
                SELECT statut FROM positions WHERE id = :position_id
            """), {"position_id": position_id}).fetchone()

            if not check:
                return {"error": f"Position {position_id} introuvable"}
            if check[0] != "OUVERT":
                return {"error": f"Position {position_id} déjà fermée — modification impossible"}

            conn.execute(text(f"""
                UPDATE positions
                SET {', '.join(updates)}
                WHERE id = :position_id
            """), params)

        return {
            "status":  "ok",
            "id":      position_id,
            "updated": [u.split(" = ")[0] for u in updates if u != "updated_at = NOW()"],
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/positions/{position_id}/close")
def close_position(position_id: int, payload: PositionClosePayload):
    """
    Ferme une position ouverte (saisie manuelle après vente sur broker).
    """
    if engine is None:
        return {"error": "engine non connecté"}

    raisons_valides = {
        "TRAILING_STOP", "TREND_BROKEN", "MOMENTUM_LOST",
        "SECTOR_WEAK", "MACRO_BEARISH", "MANUEL"
    }
    if payload.raison_vente not in raisons_valides:
        return {"error": f"Raison invalide. Valeurs acceptées : {raisons_valides}"}

    try:
        with engine.begin() as conn:
            check = conn.execute(text("""
                SELECT statut, ticker, prix_achat, quantite
                FROM positions WHERE id = :position_id
            """), {"position_id": position_id}).fetchone()

            if not check:
                return {"error": f"Position {position_id} introuvable"}
            if check[0] != "OUVERT":
                return {"error": f"Position {position_id} déjà fermée"}

            conn.execute(text("""
                UPDATE positions
                SET statut       = 'FERMÉ',
                    date_vente   = :date_vente,
                    prix_vente   = :prix_vente,
                    raison_vente = :raison_vente,
                    updated_at   = NOW()
                WHERE id = :position_id
            """), {
                "position_id":  position_id,
                "date_vente":   payload.date_vente,
                "prix_vente":   payload.prix_vente,
                "raison_vente": payload.raison_vente,
            })

        pnl_pct = round(100.0 * (payload.prix_vente - float(check[2])) / float(check[2]), 2)
        pnl_eur = round((payload.prix_vente - float(check[2])) * float(check[3]), 2)

        return {
            "status":       "ok",
            "id":           position_id,
            "ticker":       check[1],
            "resultat_pct": pnl_pct,
            "resultat_eur": pnl_eur,
            "raison_vente": payload.raison_vente,
            "message":      f"Position {check[1]} fermée — {'+' if pnl_pct >= 0 else ''}{pnl_pct}% ({'+' if pnl_eur >= 0 else ''}{pnl_eur}€)",
        }
    except Exception as e:
        return {"error": str(e)}

# ============================================================
# ENDPOINT — ÉVALUATION POSITIONS OUVERTES (v4.1)
# ============================================================

@app.get("/positions-ouvertes-eval")
def evaluate_open_positions():
    """
    Retourne toutes les positions ouvertes avec évaluation temps réel
    des 5 conditions de sortie v4.1.

    Pour chaque position :
      - P&L latent (% et €)
      - Feu par condition : OK (🟢), ALERTE (🟡), VIOLÉE (🔴)
      - Trailing stop calculé depuis le high watermark réel

    Temps d'exécution : ~2-5s (max 5 tickers à charger).
    """
    if engine is None:
        return {"error": "engine non connecté"}

    # Vérifier que les fonctions backtest_ranking sont disponibles
    if load_all_price_data is None:
        return {"error": "backtest_ranking.py non disponible — fonctions non importées"}

    try:
        # 1. Charger les positions ouvertes
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT id, ticker, date_achat, prix_achat, quantite,
                       montant_investi, decision_id, source, commentaire
                FROM positions
                WHERE statut = 'OUVERT'
                ORDER BY date_achat ASC
            """)).fetchall()

        if not rows:
            return {"nb_positions": 0, "positions": [], "date_evaluation": str(date.today())}

        positions_db = []
        tickers_needed = set()
        for r in rows:
            positions_db.append({
                "id":              r[0],
                "ticker":          r[1],
                "date_achat":      r[2],
                "prix_achat":      float(r[3]),
                "quantite":        float(r[4]),
                "montant_investi": float(r[5]) if r[5] else None,
                "decision_id":     r[6],
                "source":          r[7],
                "commentaire":     r[8],
            })
            tickers_needed.add(r[1])

        # 2. Charger les données de prix pour ces tickers uniquement
        ticker_data = load_all_price_data(list(tickers_needed))

        # 3. Calculer les indicateurs pour chaque ticker
        for ticker in ticker_data:
            ticker_data[ticker] = compute_all_indicators(ticker_data[ticker])

        # 4. Charger contexte sectoriel et macro
        secteur_mapping = load_secteur_mapping()
        force_data      = load_all_secteur_force()
        macro_data      = load_macro_data()
        macro_regime    = {}
        if macro_data:
            # Trouver la dernière date disponible dans les données macro
            latest_macro_dates = [df.index.max() for df in macro_data.values() if not df.empty]
            if latest_macro_dates:
                macro_eval_date = min(latest_macro_dates)
                macro_regime = get_macro_regime(macro_data, macro_eval_date)

        # 5. Évaluer chaque position
        results = []
        for pos in positions_db:
            ticker = pos["ticker"]
            df_t = ticker_data.get(ticker)

            if df_t is None or df_t.empty:
                results.append({
                    **pos,
                    "date_achat":     str(pos["date_achat"]),
                    "error":          f"Pas de données prix pour {ticker}",
                    "conditions":     {},
                    "alerte_globale": "INCONNU",
                })
                continue

            # Dernière date avec des données
            latest_date = df_t.index.max()
            if latest_date not in df_t.index:
                results.append({
                    **pos,
                    "date_achat":     str(pos["date_achat"]),
                    "error":          f"Pas de données récentes pour {ticker}",
                    "conditions":     {},
                    "alerte_globale": "INCONNU",
                })
                continue

            current_price = float(df_t.loc[latest_date, "prix_ajuste"])
            current_atr   = float(df_t.loc[latest_date, "atr_14"]) if not pd.isna(df_t.loc[latest_date, "atr_14"]) else 0

            # --- P&L latent ---
            pnl_pct = round(100.0 * (current_price - pos["prix_achat"]) / pos["prix_achat"], 2)
            pnl_eur = round((current_price - pos["prix_achat"]) * pos["quantite"], 2)
            jours   = (latest_date.date() - pos["date_achat"]).days

            # --- Trailing stop : calculer depuis date_achat ---
            date_achat_ts = pd.Timestamp(pos["date_achat"])
            df_since_entry = df_t[df_t.index >= date_achat_ts]

            # K adaptatif basé sur l'ATR actuel
            k = compute_adaptive_k(current_atr, current_price) if current_atr > 0 else 3.0

            # High watermark = max(prix_achat, max close depuis entrée)
            if not df_since_entry.empty:
                max_price = max(pos["prix_achat"], float(df_since_entry["prix_ajuste"].max()))
            else:
                max_price = pos["prix_achat"]

            # Trailing stop = high watermark - k × ATR
            trailing_stop = max_price - k * current_atr if current_atr > 0 else 0
            stop_distance_pct = round(100.0 * (current_price - trailing_stop) / current_price, 2) if current_price > 0 else 0

            # --- Évaluation des 5 conditions ---
            conditions = {}

            # 1. Trailing stop
            if current_price <= trailing_stop:
                conditions["trailing_stop"] = {"status": "VIOLATED", "feu": "🔴"}
            elif stop_distance_pct < 2.0:
                conditions["trailing_stop"] = {"status": "WARNING", "feu": "🟡"}
            else:
                conditions["trailing_stop"] = {"status": "OK", "feu": "🟢"}
            conditions["trailing_stop"]["stop_level"]     = round(trailing_stop, 2)
            conditions["trailing_stop"]["distance_pct"]   = stop_distance_pct
            conditions["trailing_stop"]["k"]              = k
            conditions["trailing_stop"]["max_price"]      = round(max_price, 2)

            # 2. Prix < SMA 200
            sma_200 = df_t.loc[latest_date, "sma_200"] if "sma_200" in df_t.columns else None
            if sma_200 is not None and not pd.isna(sma_200):
                sma_200 = float(sma_200)
                sma_dist_pct = round(100.0 * (current_price - sma_200) / sma_200, 2)
                if current_price < sma_200:
                    conditions["trend_sma200"] = {"status": "VIOLATED", "feu": "🔴"}
                elif sma_dist_pct < 3.0:
                    conditions["trend_sma200"] = {"status": "WARNING", "feu": "🟡"}
                else:
                    conditions["trend_sma200"] = {"status": "OK", "feu": "🟢"}
                conditions["trend_sma200"]["sma_200"]      = round(sma_200, 2)
                conditions["trend_sma200"]["distance_pct"] = sma_dist_pct
            else:
                conditions["trend_sma200"] = {"status": "NO_DATA", "feu": "⚪"}

            # 3. Momentum R² < 0
            mom_r2 = df_t.loc[latest_date, "mom_r2"] if "mom_r2" in df_t.columns else None
            if mom_r2 is not None and not pd.isna(mom_r2):
                mom_r2 = float(mom_r2)
                if mom_r2 < 0:
                    conditions["momentum_r2"] = {"status": "VIOLATED", "feu": "🔴"}
                elif mom_r2 < 0.05:
                    conditions["momentum_r2"] = {"status": "WARNING", "feu": "🟡"}
                else:
                    conditions["momentum_r2"] = {"status": "OK", "feu": "🟢"}
                conditions["momentum_r2"]["value"] = round(mom_r2, 4)
            else:
                conditions["momentum_r2"] = {"status": "NO_DATA", "feu": "⚪"}

            # 4. Secteur en force relative
            sector_ok = get_secteur_force_for_ticker(ticker, secteur_mapping, force_data, latest_date)
            if not sector_ok:
                conditions["secteur"] = {"status": "VIOLATED", "feu": "🔴"}
            else:
                conditions["secteur"] = {"status": "OK", "feu": "🟢"}
            secteur_info = secteur_mapping.get(ticker, {})
            conditions["secteur"]["secteur_name"] = secteur_info.get("secteur", "—")

            # 5. Macro regime
            zone = get_ticker_zone(ticker, secteur_mapping)
            macro_bull = macro_regime.get(zone, True)
            if not macro_bull:
                conditions["macro"] = {"status": "VIOLATED", "feu": "🔴"}
            else:
                conditions["macro"] = {"status": "OK", "feu": "🟢"}
            conditions["macro"]["zone"] = zone

            # --- Alerte globale ---
            violated = [k for k, v in conditions.items() if v.get("status") == "VIOLATED"]
            warnings = [k for k, v in conditions.items() if v.get("status") == "WARNING"]

            if violated:
                alerte = "SORTIE_RECOMMANDÉE"
            elif warnings:
                alerte = "VIGILANCE"
            else:
                alerte = "SAIN"

            results.append({
                "id":              pos["id"],
                "ticker":          ticker,
                "date_achat":      str(pos["date_achat"]),
                "prix_achat":      pos["prix_achat"],
                "quantite":        pos["quantite"],
                "montant_investi": pos["montant_investi"],
                "prix_actuel":     round(current_price, 2),
                "pnl_pct":         pnl_pct,
                "pnl_eur":         pnl_eur,
                "jours_detention": jours,
                "conditions":      conditions,
                "nb_violated":     len(violated),
                "violated":        violated,
                "nb_warnings":     len(warnings),
                "warnings":        warnings,
                "alerte_globale":  alerte,
                "data_date":       str(latest_date.date()),
                "decision_id":     pos["decision_id"],
                "source":          pos["source"],
                "commentaire":     pos["commentaire"],
            })

        # Stats globales
        nb_violated = sum(1 for r in results if r.get("alerte_globale") == "SORTIE_RECOMMANDÉE")
        nb_warning  = sum(1 for r in results if r.get("alerte_globale") == "VIGILANCE")
        nb_sain     = sum(1 for r in results if r.get("alerte_globale") == "SAIN")

        return {
            "date_evaluation": str(date.today()),
            "nb_positions":    len(results),
            "nb_sortie_reco":  nb_violated,
            "nb_vigilance":    nb_warning,
            "nb_sain":         nb_sain,
            "positions":       results,
        }

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}

@app.get("/fill-high-low")
async def trigger_fill_high_low(background_tasks: BackgroundTasks):
    """
    Remplit prix_haut et prix_bas pour tout l'historique depuis yfinance.
    À appeler UNE SEULE FOIS après déploiement de cette version.
    Durée estimée : 20-40 minutes selon le nombre de tickers.
    ⚠️ Action manuelle uniquement — NE PAS planifier dans le scheduler.
    """
    background_tasks.add_task(fill_high_low_logic)
    return {
        "status" : "processing",
        "message": "⚠️ Remplissage prix_haut/prix_bas lancé sur tout l'historique (action manuelle unique)."
    }

@app.get("/run-analysis")
async def trigger_analysis(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_analysis_logic, full=False)
    return {
        "status" : "processing",
        "message": f"Analyse incrémentale lancée (contexte {INCREMENTAL_LOOKBACK_DAYS}j, sauvegarde {INCREMENTAL_SAVE_DAYS}j)."
    }

@app.get("/run-analysis-full")
async def trigger_full_analysis(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_analysis_logic, full=True)
    return {
        "status" : "processing",
        "message": "⚠️ Recalcul COMPLET lancé sur tout l'historique (action manuelle)."
    }

@app.get("/backtest-detail")
def backtest_detail(ticker: str = "MSFT", k: float = 2.0, horizon: int = 30):
    """
    Retourne le détail de chaque trade pour un ticker donné.
    Synchrone — résultat immédiat dans le navigateur.
    """
    from backtest import (load_ticker_data, load_secteur_force,
                          compute_signals_v35, build_exit_signals)
    try:
        df         = load_ticker_data(ticker)
        df_secteur = load_secteur_force(ticker)
        df_signals = compute_signals_v35(df, df_secteur)

        price      = df_signals["prix_ajuste"]
        entries    = df_signals["signal_achat"]
        atr        = df_signals["atr_14"]
        exits      = build_exit_signals(entries, price, atr, k)
        atr_type   = "réel (H-L)" if df["prix_haut"].notna().any() else "approché (C-C)"

        trades      = []
        in_trade    = False
        entry_date  = None
        entry_price = None

        for i in range(len(price)):
            if not in_trade and entries.iloc[i]:
                in_trade    = True
                entry_date  = price.index[i]
                entry_price = float(price.iloc[i])
            elif in_trade and exits.iloc[i]:
                exit_date  = price.index[i]
                exit_price = float(price.iloc[i])
                ret_pct    = round((exit_price - entry_price) / entry_price * 100, 2)
                loc        = price.index.get_loc(entry_date)
                prix_jh    = float(price.iloc[loc + horizon]) if loc + horizon < len(price) else None
                row        = df_signals.loc[entry_date]
                trades.append({
                    "entry_date"       : str(entry_date.date()),
                    "exit_date"        : str(exit_date.date()),
                    "duree_jours"      : (exit_date - entry_date).days,
                    "prix_entree"      : entry_price,
                    "prix_sortie"      : exit_price,
                    "rendement_pct"    : ret_pct,
                    "resultat"         : "✅ WIN" if ret_pct > 0 else "❌ LOSS",
                    f"prix_j{horizon}" : round(prix_jh, 2) if prix_jh else None,
                    "contexte_signal"  : {
                        "tendance_ok"     : bool(row["tendance_ok"]),
                        "force_rel"       : bool(row["force_rel"]),
                        "mom_r2"          : round(float(row["mom_r2"]), 4),
                        "breakout_20j"    : bool(row["breakout_20j"]),
                        "rvol"            : round(float(row["rvol"]), 2),
                        "obv_accumulation": bool(row["obv_accumulation"]),
                        "atr_14"          : round(float(row["atr_14"]), 3),
                        "atr_type"        : atr_type,
                        "sma_200"         : round(float(row["sma_200"]), 2),
                    }
                })
                in_trade = False

        # Signaux sans sortie (position encore ouverte)
        for i in range(len(price)):
            if entries.iloc[i]:
                date_signal = price.index[i]
                if not any(t["entry_date"] == str(date_signal.date()) for t in trades):
                    row = df_signals.loc[date_signal]
                    trades.append({
                        "entry_date"   : str(date_signal.date()),
                        "exit_date"    : "OUVERT",
                        "prix_entree"  : float(price.iloc[i]),
                        "rendement_pct": round((float(price.iloc[-1]) - float(price.iloc[i])) / float(price.iloc[i]) * 100, 2),
                        "resultat"     : "🔄 OUVERT",
                        "contexte_signal": {
                            "tendance_ok"     : bool(row["tendance_ok"]),
                            "force_rel"       : bool(row["force_rel"]),
                            "mom_r2"          : round(float(row["mom_r2"]), 4),
                            "breakout_20j"    : bool(row["breakout_20j"]),
                            "rvol"            : round(float(row["rvol"]), 2),
                            "obv_accumulation": bool(row["obv_accumulation"]),
                        }
                    })

        return {
            "ticker"    : ticker,
            "k"         : k,
            "atr_type"  : atr_type,
            "nb_signaux": int(entries.sum()),
            "nb_trades" : len(trades),
            "win_rate"  : round(sum(1 for t in trades if "WIN" in t.get("resultat", "")) / len(trades) * 100, 1) if trades else 0,
            "trades"    : sorted(trades, key=lambda t: t["entry_date"])
        }

    except Exception as e:
        return {"erreur": str(e)}

@app.get("/schema-diagnostic")
def schema_diagnostic():
    """
    Retourne le schéma de chaque table + pourcentage de valeurs NULL par colonne.
    À partager dans le projet Claude pour contextualiser le modèle de données.
    """
    if engine is None:
        return {"error": "engine non connecté"}

    tables = [
        "actions_prix_historique",
        "tickers_info",
        "positions",
        "signaux_log",
        "models_store",
        "secteurs_etf",
        "secteurs_etf_prix",
        "indices_prix"
    ]

    result = {}

    with engine.connect() as conn:
        for table in tables:
            # 1. Schéma
            schema_rows = conn.execute(text("""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_name = :table
                ORDER BY ordinal_position
            """), {"table": table}).fetchall()

            if not schema_rows:
                result[table] = {"erreur": "table absente"}
                continue

            columns = [r[0] for r in schema_rows]
            schema  = [
                {
                    "colonne" : r[0],
                    "type"    : r[1],
                    "nullable": r[2],
                    "defaut"  : r[3]
                }
                for r in schema_rows
            ]

            # 2. Nombre total de lignes
            try:
                total = conn.execute(text(
                    f"SELECT COUNT(*) FROM {table}"
                )).scalar()
            except Exception:
                total = None

            # 3. Pourcentage NULL par colonne
            null_stats = {}
            if total and total > 0:
                for col in columns:
                    try:
                        null_count = conn.execute(text(
                            f"SELECT COUNT(*) FROM {table} WHERE {col} IS NULL"
                        )).scalar()
                        null_stats[col] = {
                            "null_count": null_count,
                            "null_pct"  : round(100.0 * null_count / total, 1)
                        }
                    except Exception:
                        null_stats[col] = {"erreur": "calcul impossible"}

            result[table] = {
                "nb_lignes" : total,
                "schema"    : schema,
                "null_stats": null_stats
            }

    return result

@app.get("/run-backtest")
async def trigger_backtest(
    background_tasks: BackgroundTasks,
    tickers: str = "NVDA,AAPL,MSFT,AMZN,ASML",
    horizon: int = 30
):
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    background_tasks.add_task(run_backtest_logic, tickers=ticker_list, horizon=horizon)
    return {
        "status" : "processing",
        "message": f"Backtest v3.5 lancé en arrière-plan — {len(ticker_list)} tickers, horizon={horizon}j.",
        "tickers": ticker_list
    }

@app.get("/run-backtest-ranking")
async def trigger_backtest_ranking(
    background_tasks: BackgroundTasks,
    top_n: int = 5,
    k: str = None,
    macro: str = "off",
    sma: int = None,
    min_mom_r2: float = None
):
    """
    Backtest hybrid v4.1 — entrée ranking, sortie absolue.
    - top_n : positions max (défaut 5)
    - sma   : None = compare SMA200 vs SMA150, ou valeur spécifique (150, 200)
    Appels :
      GET /run-backtest-ranking              → compare SMA 200 vs SMA 150
      GET /run-backtest-ranking?sma=150      → SMA 150 uniquement
      GET /run-backtest-ranking?sma=200      → SMA 200 uniquement
    """
    background_tasks.add_task(run_backtest_ranking_logic, top_n=top_n, sma=sma, min_mom_r2=min_mom_r2)
    if min_mom_r2 == -1:
        label = "comparaison seuils mom_r2 (0 vs 0.01 vs 0.05)"
    elif min_mom_r2 is not None:
        label = f"min_mom_r2={min_mom_r2}"
    elif sma:
        label = f"SMA {sma}"
    else:
        label = "SMA 200 vs SMA 150 (compare)"
    return {
        "status" : "processing",
        "message": f"Backtest hybrid v4.1 lancé — top {top_n}, {label}.",
    }
 
        
# ============================================================
# CHARGEMENT DU MODÈLE DEPUIS POSTGRESQL
# ============================================================

def load_model_from_db():
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
# LOGIQUE SYNC PRIX QUOTIDIEN (remplace n8n)
# ============================================================

def sync_prix_logic(full: bool = False, period_override: str = None):
    """
    Télécharge les prix OHLCV depuis yfinance et upsert dans actions_prix_historique.
    Remplace le workflow n8n désactivé le 2026-04-11.

    Paramètres
    ----------
    full : bool
        False → 30 derniers jours (mode quotidien, scheduler)
        True  → 5 ans d'historique (initialisation / rattrapage)
    """
    if engine is None:
        print("❌ sync_prix_logic : engine non connecté.")
        return

    mode   = "COMPLET (5 ans)" if full else "INCRÉMENTAL (30j)"
    period = period_override or ("5y" if full else "1mo")
    print(f"🔄 Sync prix quotidien — mode {mode}...")

    try:
        # 1. Liste des tickers depuis la base
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT DISTINCT ticker FROM actions_prix_historique ORDER BY ticker"
            ))
            all_tickers = [row[0] for row in result]

        if not all_tickers:
            print("⚠️ Aucun ticker en base.")
            return

        print(f"   {len(all_tickers)} tickers à synchroniser.")

        chunk_size    = 20
        total_chunks  = (len(all_tickers) - 1) // chunk_size + 1
        total_upserted = 0
        total_errors   = 0

        for i in range(0, len(all_tickers), chunk_size):
            chunk = all_tickers[i:i + chunk_size]
            chunk_upserted = 0

            for ticker_symbol in chunk:
                try:
                    print(f"   ⏳ {ticker_symbol}...")
                    df_yf = yf.download(
                        ticker_symbol,
                        period=period,
                        interval="1d",
                        auto_adjust=True,
                        progress=False
                    )

                    if df_yf.empty:
                        continue

                    df_yf = df_yf.reset_index()
                    # yfinance peut retourner des MultiIndex columns
                    df_yf.columns = [c[0] if isinstance(c, tuple) else c for c in df_yf.columns]

                    df_yf = df_yf.rename(columns={
                        "Date" : "date",
                        "Open" : "prix_ouverture",
                        "High" : "prix_haut",
                        "Low"  : "prix_bas",
                        "Close": "prix_cloture",
                        "Volume": "volume",
                    })

                    # auto_adjust=True → Close est déjà ajusté
                    df_yf["prix_ajuste"] = df_yf["prix_cloture"]
                    df_yf["date"]   = pd.to_datetime(df_yf["date"]).dt.date
                    df_yf["ticker"] = ticker_symbol

                    cols = ["ticker", "date", "prix_ouverture", "prix_haut",
                            "prix_bas", "prix_cloture", "prix_ajuste", "volume"]
                    df_clean = df_yf[[c for c in cols if c in df_yf.columns]].dropna(
                        subset=["prix_cloture"]
                    )

                    if df_clean.empty:
                        continue

                    records = df_clean.to_dict("records")

                    with engine.begin() as conn:
                        conn.execute(text("""
                            INSERT INTO actions_prix_historique
                                (ticker, date, prix_ouverture, prix_haut,
                                 prix_bas, prix_cloture, prix_ajuste, volume)
                            VALUES
                                (:ticker, :date, :prix_ouverture, :prix_haut,
                                 :prix_bas, :prix_cloture, :prix_ajuste, :volume)
                            ON CONFLICT (ticker, date) DO UPDATE SET
                                prix_ouverture = EXCLUDED.prix_ouverture,
                                prix_haut      = EXCLUDED.prix_haut,
                                prix_bas       = EXCLUDED.prix_bas,
                                prix_cloture   = EXCLUDED.prix_cloture,
                                prix_ajuste    = EXCLUDED.prix_ajuste,
                                volume         = EXCLUDED.volume
                        """), records)

                    chunk_upserted += len(records)
                    time.sleep(0.5)

                except Exception as e:
                    print(f"   ❌ {ticker_symbol} : {e}")
                    total_errors += 1
                    continue

            total_upserted += chunk_upserted
            print(f"   🟢 Chunk {i // chunk_size + 1}/{total_chunks} — "
                  f"{chunk_upserted} lignes upsertées.")

        print(f"🏁 Sync prix terminée — {total_upserted} lignes upsertées, "
              f"{total_errors} erreurs.")

    except Exception as e:
        print(f"❌ Erreur sync_prix_logic : {e}")

# ============================================================
# LOGIQUE SYNC METADATA
# ============================================================

def sync_metadata_logic():
    if engine is None: return
    print("🔄 Sync Yahoo Finance (Metadata)...")
    try:
        with engine.connect() as conn:
            result  = conn.execute(text("SELECT DISTINCT ticker FROM actions_prix_historique LIMIT 500;"))
            tickers = [row[0] for row in result]

        for ticker_symbol in tickers:
            try:
                stock = yf.Ticker(ticker_symbol)
                data  = stock.info
                if "symbol" in data:
                    metadata = {
                        "ticker"    : ticker_symbol,
                        "name"      : data.get("longName"),
                        "secteur"   : data.get("sector"),
                        "industrie" : data.get("industry"),
                        "pays"      : data.get("country"),
                        "monnaie"   : data.get("currency"),
                        "market_cap": data.get("marketCap"),
                        "pe_ratio"  : data.get("trailingPE")
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
                                pe_ratio     = EXCLUDED.pe_ratio,
                                market_cap   = EXCLUDED.market_cap,
                                derniere_maj = CURRENT_DATE;
                        """), metadata)
                time.sleep(1)
            except Exception:
                continue
    except Exception as e:
        print(f"❌ Erreur Sync: {e}")


# ============================================================
# LOGIQUE FILL HIGH/LOW — ACTION MANUELLE UNIQUE
# ============================================================

def fill_high_low_logic():
    """
    Remplit prix_haut, prix_bas et prix_ouverture pour tout l'historique depuis yfinance.
    Correction : Utilise la même transaction (conn) pour to_sql et l'UPDATE.
    """
    if engine is None:
        print("❌ fill_high_low_logic : engine non connecté.")
        return

    print("📥 Remplissage prix_haut / prix_bas / prix_ouverture — historique complet...")

    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT DISTINCT ticker FROM actions_prix_historique ORDER BY ticker"))
            all_tickers = [row[0] for row in result]

        print(f"   {len(all_tickers)} tickers à traiter.")

        chunk_size = 20
        total_chunks = (len(all_tickers) - 1) // chunk_size + 1
        total_updated = 0

        for i in range(0, len(all_tickers), chunk_size):
            chunk = all_tickers[i:i + chunk_size]
            chunk_updated = 0

            for ticker_symbol in chunk:
                try:
                    df_yf = yf.download(
                        ticker_symbol,
                        period=period,
                        interval="1d",
                        auto_adjust=True,
                        progress=False,
                        timeout=30
                    )

                    if df_yf.empty:
                        continue

                    df_yf = df_yf.reset_index()
                    df_yf.columns = [c[0] if isinstance(c, tuple) else c for c in df_yf.columns]
                    df_yf = df_yf.rename(columns={
                        "Date": "date",
                        "High": "prix_haut",
                        "Low": "prix_bas",
                        "Open": "prix_ouverture",
                    })

                    df_yf["date"] = pd.to_datetime(df_yf["date"]).dt.date
                    df_yf["ticker"] = ticker_symbol

                    cols_needed = ["ticker", "date", "prix_haut", "prix_bas", "prix_ouverture"]
                    df_to_save = df_yf[[c for c in cols_needed if c in df_yf.columns]].dropna(subset=["prix_haut", "prix_bas"])

                    if df_to_save.empty:
                        continue

                    # On nettoie le nom de la table (minuscules et sans caractères spéciaux)
                    clean_ticker = ticker_symbol.lower().replace('.', '_').replace('-', '_').replace('^', '')
                    tmp_table_name = f"_tmp_hl_{clean_ticker}"

                    # --- CORE FIX: TOUTE L'OPÉRATION DANS LA MÊME TRANSACTION ---
                    with engine.begin() as conn:
                        # 1. On écrit les données dans la table temporaire en utilisant la connexion active
                        df_to_save.to_sql(tmp_table_name, conn, if_exists="replace", index=False)

                        # 2. On fait l'UPDATE
                        conn.execute(text(f"""
                            UPDATE actions_prix_historique a SET
                                prix_haut      = t.prix_haut,
                                prix_bas       = t.prix_bas,
                                prix_ouverture = t.prix_ouverture
                            FROM {tmp_table_name} t
                            WHERE a.ticker = t.ticker
                              AND a.date   = t.date::date
                        """))
                        
                        # 3. On supprime la table
                        conn.execute(text(f"DROP TABLE IF EXISTS {tmp_table_name}"))

                    chunk_updated += len(df_to_save)
                    time.sleep(0.5)

                except Exception as e:
                    print(f"   ❌ {ticker_symbol} : {e}")
                    continue

            total_updated += chunk_updated
            print(f"   🟢 Chunk {i // chunk_size + 1}/{total_chunks} terminé.")

        print(f"🏁 Fill high/low terminé — {total_updated} lignes mises à jour.")

    except Exception as e:
        print(f"❌ Erreur fill_high_low_logic : {e}")


# ============================================================
# LOGIQUE SYNC ETF SECTORIELS
# ============================================================

def sync_secteurs_etf_logic(full: bool = False):
    """
    Télécharge et persiste les prix des ETF sectoriels + indices de référence.
    Calcule le ratio de force relative et le ratio vs MM50 pour chaque ETF.

    Paramètres
    ----------
    full : bool
        False → 30 derniers jours (mode quotidien, scheduler)
        True  → 5 ans d'historique (initialisation manuelle)
    """
    if engine is None:
        print("❌ sync_secteurs_etf_logic : engine non connecté.")
        return

    mode   = "COMPLET (5 ans)" if full else "INCRÉMENTAL (30j)"
    period = "5y" if full else "1mo"
    print(f"🔄 Sync ETF sectoriels — mode {mode}...")

    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT ticker_etf, indice_reference
                FROM secteurs_etf
                WHERE actif = TRUE
            """)).fetchall()

        if not rows:
            print("⚠️ Aucun ETF trouvé dans secteurs_etf — exécuter secteurs_etf.sql d'abord.")
            return

        etf_list   = [r[0] for r in rows]
        etf_to_idx = {r[0]: r[1] for r in rows}
        indices    = list(set(r[1] for r in rows))

        print(f"   {len(etf_list)} ETF sectoriels + {len(indices)} indices à synchroniser.")

        print("   📥 Téléchargement indices de référence...")
        for idx_ticker in indices:
            try:
                df_idx = yf.download(
                    idx_ticker, period=period, interval="1d",
                    auto_adjust=True, progress=False
                )
                if df_idx.empty:
                    print(f"   ⚠️ Aucune donnée pour l'indice {idx_ticker}")
                    continue

                df_idx = df_idx.reset_index()
                df_idx.columns = [c[0] if isinstance(c, tuple) else c for c in df_idx.columns]
                df_idx = df_idx.rename(columns={"Date": "date", "Close": "prix_cloture", "Adj Close": "prix_ajuste"})
                if "prix_ajuste" not in df_idx.columns:
                    df_idx["prix_ajuste"] = df_idx["prix_cloture"]
                df_idx["date"]          = pd.to_datetime(df_idx["date"]).dt.date
                df_idx["ticker_indice"] = idx_ticker
                records = df_idx[["ticker_indice", "date", "prix_cloture", "prix_ajuste"]].to_dict("records")

                with engine.begin() as conn:
                    conn.execute(text("""
                        INSERT INTO indices_prix (ticker_indice, date, prix_cloture, prix_ajuste)
                        VALUES (:ticker_indice, :date, :prix_cloture, :prix_ajuste)
                        ON CONFLICT (ticker_indice, date) DO UPDATE SET
                            prix_cloture = EXCLUDED.prix_cloture,
                            prix_ajuste  = EXCLUDED.prix_ajuste
                    """), records)

                print(f"   ✅ Indice {idx_ticker} — {len(records)} jours enregistrés.")
                time.sleep(1)

            except Exception as e:
                print(f"   ❌ Erreur indice {idx_ticker} : {e}")
                continue

        print("   📥 Téléchargement ETF sectoriels...")
        for etf_ticker in etf_list:
            try:
                df_etf = yf.download(
                    etf_ticker, period=period, interval="1d",
                    auto_adjust=True, progress=False
                )
                if df_etf.empty:
                    print(f"   ⚠️ Aucune donnée pour {etf_ticker}")
                    continue

                df_etf = df_etf.reset_index()
                df_etf.columns = [c[0] if isinstance(c, tuple) else c for c in df_etf.columns]
                df_etf = df_etf.rename(columns={
                    "Date": "date", "Close": "prix_cloture",
                    "Adj Close": "prix_ajuste", "Volume": "volume"
                })
                if "prix_ajuste" not in df_etf.columns:
                    df_etf["prix_ajuste"] = df_etf["prix_cloture"]
                if "volume" not in df_etf.columns:
                    df_etf["volume"] = 0
                df_etf["date"]       = pd.to_datetime(df_etf["date"]).dt.date
                df_etf["ticker_etf"] = etf_ticker

                idx_ticker = etf_to_idx[etf_ticker]
                with engine.connect() as conn:
                    df_indice = pd.read_sql(text("""
                        SELECT date, prix_ajuste AS prix_indice
                        FROM indices_prix WHERE ticker_indice = :idx ORDER BY date ASC
                    """), conn, params={"idx": idx_ticker})

                if df_indice.empty:
                    print(f"   ⚠️ Pas de données indice {idx_ticker} pour {etf_ticker} — skip.")
                    continue

                df_indice["date"] = pd.to_datetime(df_indice["date"]).dt.date
                df_merged         = df_etf.merge(df_indice, on="date", how="inner")

                first_etf = df_merged["prix_ajuste"].iloc[0]
                first_idx = df_merged["prix_indice"].iloc[0]
                df_merged["ratio_force_relative"] = (
                    (df_merged["prix_ajuste"] / first_etf) /
                    (df_merged["prix_indice"] / first_idx)
                )
                df_merged["mm50_ratio"]        = df_merged["ratio_force_relative"].rolling(50, min_periods=1).mean()
                df_merged["ratio_vs_mm50"]     = df_merged["ratio_force_relative"] / df_merged["mm50_ratio"]
                df_merged["en_force_relative"] = df_merged["ratio_vs_mm50"] > 1.0

                records = df_merged[[
                    "ticker_etf", "date", "prix_cloture", "prix_ajuste", "volume",
                    "prix_indice", "ratio_force_relative", "ratio_vs_mm50", "en_force_relative"
                ]].to_dict("records")

                for r in records:
                    r["en_force_relative"] = bool(r["en_force_relative"])
                    r["volume"]            = int(r["volume"]) if r["volume"] is not None else 0

                with engine.begin() as conn:
                    conn.execute(text("""
                        INSERT INTO secteurs_etf_prix
                            (ticker_etf, date, prix_cloture, prix_ajuste, volume,
                             prix_indice, ratio_force_relative, ratio_vs_mm50, en_force_relative)
                        VALUES
                            (:ticker_etf, :date, :prix_cloture, :prix_ajuste, :volume,
                             :prix_indice, :ratio_force_relative, :ratio_vs_mm50, :en_force_relative)
                        ON CONFLICT (ticker_etf, date) DO UPDATE SET
                            prix_cloture         = EXCLUDED.prix_cloture,
                            prix_ajuste          = EXCLUDED.prix_ajuste,
                            volume               = EXCLUDED.volume,
                            prix_indice          = EXCLUDED.prix_indice,
                            ratio_force_relative = EXCLUDED.ratio_force_relative,
                            ratio_vs_mm50        = EXCLUDED.ratio_vs_mm50,
                            en_force_relative    = EXCLUDED.en_force_relative,
                            updated_at           = CURRENT_TIMESTAMP
                    """), records)

                statut_fr = "OUI ✅" if df_merged["en_force_relative"].iloc[-1] else "NON ❌"
                ratio_val = df_merged["ratio_vs_mm50"].iloc[-1]
                print(f"   ✅ {etf_ticker} — {len(records)} jours | Force relative : {statut_fr} | ratio_vs_mm50 = {ratio_val:.3f}")
                time.sleep(1)

            except Exception as e:
                print(f"   ❌ Erreur ETF {etf_ticker} : {e}")
                continue

        print("🏁 Sync ETF sectoriels terminée.")

    except Exception as e:
        print(f"❌ Erreur sync_secteurs_etf_logic : {e}")


# ============================================================
# HELPER : Secteurs en force relative
# ============================================================

def get_secteurs_en_force() -> list[dict]:
    """
    Retourne la liste des secteurs Yahoo actuellement en force relative.
    Interroge la vue v_secteurs_en_force (dernière date disponible).
    Si la table est vide ou absente → retourne [] sans planter run_analysis_logic.
    """
    if engine is None:
        return []
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT secteur_yahoo, zone, ticker_etf, indice_reference,
                       date, ratio_force_relative, ratio_vs_mm50
                FROM v_secteurs_en_force
                ORDER BY zone, ratio_vs_mm50 DESC
            """)).fetchall()

        return [
            {
                "secteur_yahoo"       : r[0],
                "zone"                : r[1],
                "ticker_etf"          : r[2],
                "indice_reference"    : r[3],
                "date"                : str(r[4]),
                "ratio_force_relative": float(r[5]) if r[5] is not None else None,
                "ratio_vs_mm50"       : float(r[6]) if r[6] is not None else None,
            }
            for r in rows
        ]
    except Exception as e:
        print(f"⚠️ get_secteurs_en_force : {e} — retour liste vide.")
        return []

def compute_and_store_ranking(top_n: int = 20):
    """
    Calcule le ranking momentum sur tous les tickers et le persiste
    dans ranking_hebdo. Appelé par le scheduler (lundi 06h45)
    et par l'endpoint /compute-ranking.
 
    Durée estimée : 3-4 min sur 400 tickers.
    """
    if compute_composite_score is None:
        print("❌ compute_and_store_ranking : backtest_ranking.py non disponible")
        return {"error": "backtest_ranking.py non disponible"}
    if engine is None:
        print("❌ compute_and_store_ranking : engine non connecté")
        return {"error": "engine non connecté"}
 
    try:
        print("📊 Calcul ranking hebdo...")
 
        # 1. Charger les tickers et données
        all_tickers = load_all_tickers()
        ticker_data = load_all_price_data(all_tickers)
 
        # 2. Calculer les indicateurs pour chaque ticker
        for ticker in list(ticker_data.keys()):
            ticker_data[ticker] = compute_all_indicators(ticker_data[ticker])
 
        # 3. Charger contexte sectoriel et macro
        secteur_mapping = load_secteur_mapping()
        force_data = load_all_secteur_force()
        macro_data = load_macro_data()
 
        # 4. Trouver le dernier jour de trading disponible
        all_dates = set()
        for df in ticker_data.values():
            all_dates.update(df.index)
        if not all_dates:
            return {"error": "Aucune donnée disponible"}
 
        latest_date = max(all_dates)
 
        # 5. Calculer le ranking
        ranking = compute_composite_score(
            ticker_data, latest_date, secteur_mapping, force_data, sma_period=200
        )
 
        # 6. Enrichir et préparer les records
        macro_regime = get_macro_regime(macro_data, latest_date)
        import json
        today = date.today()
 
        records = []
        for r in ranking[:top_n]:
            ticker = r["ticker"]
            zone = get_ticker_zone(ticker, secteur_mapping)
            secteur = secteur_mapping.get(ticker, {}).get("secteur", "—")
            k = compute_adaptive_k(r["atr_14"], r["prix"]) if r["atr_14"] > 0 else 3.0
 
            records.append({
                "date_calcul":  today,
                "rank":         r.get("rank", 0),
                "ticker":       ticker,
                "score":        round(r["score"], 4),
                "mom_r2":       round(r["mom_r2"], 4),
                "rvol":         round(r["rvol"], 2),
                "obv_slope":    round(r["obv_slope"], 2),
                "prix":         round(r["prix"], 2),
                "sma_200":      round(r["sma_200"], 2),
                "atr_14":       round(r["atr_14"], 2),
                "k_adaptatif":  k,
                "zone":         zone,
                "secteur":      secteur,
                "macro_regime": json.dumps(macro_regime),
                "nb_eligible":  len(ranking),
                "nb_total":     len(ticker_data),
                "data_date":    latest_date.date(),
            })
 
        # 7. Supprimer l'ancien ranking du jour et insérer
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM ranking_hebdo WHERE date_calcul = :d"), {"d": today})
            for rec in records:
                conn.execute(text("""
                    INSERT INTO ranking_hebdo
                        (date_calcul, rank, ticker, score, mom_r2, rvol, obv_slope,
                         prix, sma_200, atr_14, k_adaptatif, zone, secteur,
                         macro_regime, nb_eligible, nb_total, data_date)
                    VALUES
                        (:date_calcul, :rank, :ticker, :score, :mom_r2, :rvol, :obv_slope,
                         :prix, :sma_200, :atr_14, :k_adaptatif, :zone, :secteur,
                         CAST(:macro_regime AS jsonb), :nb_eligible, :nb_total, :data_date)
                """), rec)
 
        print(f"✅ Ranking hebdo sauvegardé : {len(records)} tickers, date données {latest_date.date()}")
        return {"status": "ok", "nb_ranked": len(records), "data_date": str(latest_date.date())}
 
    except Exception as e:
        print(f"❌ Erreur compute_and_store_ranking : {e}")
        return {"error": str(e)}

# ============================================================
# LOGIQUE ANALYSE — INCRÉMENTALE ET COMPLÈTE
# ============================================================

def run_analysis_logic(full: bool = False):
    if engine is None: return

    mode = "COMPLET" if full else "INCRÉMENTAL"
    print(f"🚀 Démarrage Analyse v5.9.0 — mode {mode}...")

    try:
        # 1. Liste des tickers
        with engine.connect() as conn:
            result      = conn.execute(text("SELECT DISTINCT ticker FROM actions_prix_historique"))
            all_tickers = [row[0] for row in result]

        if not all_tickers: return
        print(f"   {len(all_tickers)} tickers trouvés en base.")

        # 2. Chargement du modèle ML depuis PostgreSQL
        model, model_cols = load_model_from_db()

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

            def compute_all_indicators(group):
                if len(group) < 50: return group
                price = group['prix_ajuste']
                vol   = group['volume']

                # RSI 14
                delta = price.diff()
                gain  = delta.where(delta > 0, 0).rolling(14).mean()
                loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
                group['rsi_14'] = 100 - (100 / (1 + gain / (loss + 1e-9)))

                # Moyennes mobiles
                group['sma_200'] = price.rolling(200, min_periods=1).mean()
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

                # ============================================================
                # SIGNAL ACHAT — v3.3 (obsolète, conservé en production)
                # ⚠️ À remplacer par signal v3.5 après validation backtest
                # ============================================================
                group['signal_achat'] = (
                    (group['rsi_14'] < 35) &
                    (price > group['sma_200']) &
                    (vol > group['vol_avg_20']) &
                    (price < group['bb_lower'])
                )
                return group

            df = df.groupby('ticker', group_keys=False).apply(compute_all_indicators)

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
                save_from  = pd.Timestamp.today() - pd.Timedelta(days=INCREMENTAL_SAVE_DAYS)
                df_to_save = df[pd.to_datetime(df['date']) >= save_from].copy()
            else:
                df_to_save = df.copy()

            if df_to_save.empty:
                continue

            # 7. Sauvegarde — table temporaire unique par chunk
            tmp_table    = f"_tmp_update_{i}"
            cols_to_save = [
                'id', 'rsi_14', 'sma_200', 'bb_lower', 'bb_position',
                'vol_avg_20', 'regime_marche', 'signal_achat', 'confiance_ml',
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
                        signal_achat  = t.signal_achat,
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
