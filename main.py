import os
import time
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from fastapi import FastAPI, BackgroundTasks
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Import de l'entraînement
try:
    from train_model import train_brain
except ImportError:
    def train_brain():
        print("⚠️ Fonction train_brain non trouvée.")

load_dotenv()

app = FastAPI()

# --- CONFIGURATION DATABASE ---
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True) if DATABASE_URL else None

MODEL_PATH = "models/trading_forest.joblib"
COLS_PATH  = "models/trading_forest_cols.joblib"

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
def home():
    return {
        "status": "Service Trading IA Actif",
        "version": "5.3.0",
        "architecture": "AT-signal → ML-confirm (Régime) → NLP-veto"
    }

@app.get("/train-model")
async def trigger_training(background_tasks: BackgroundTasks):
    background_tasks.add_task(train_brain)
    return {"status": "processing", "message": "Entraînement v3.3 lancé."}

@app.get("/sync-metadata")
async def trigger_sync(background_tasks: BackgroundTasks):
    background_tasks.add_task(sync_metadata_logic)
    return {"status": "processing", "message": "Sync Yahoo Finance lancée."}

@app.get("/run-analysis")
async def trigger_analysis(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_analysis_logic)
    return {"status": "processing", "message": "Analyse massive v5.3.0 lancée..."}

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
        with engine
