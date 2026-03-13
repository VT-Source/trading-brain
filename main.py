import os
import time
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from fastapi import FastAPI, BackgroundTasks
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Import de la fonction d'entraînement depuis train_model.py
try:
    from train_model import train_brain
except ImportError:
    def train_brain():
        print("⚠️ Fonction train_brain non trouvée dans train_model.py")

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
# ENDPOINTS API
# ============================================================

@app.get("/")
def home():
    return {
        "status": "Service Trading IA Actif",
        "version": "5.3.1",
        "syntax_check": "Corrected"
    }

@app.get("/train-model")
async def trigger_training(background_tasks: BackgroundTasks):
    background_tasks.add_task(train_brain)
    return {"status": "processing", "message": "Entraînement v3.3 lancé en arrière-plan."}

@app.get("/sync-metadata")
async def trigger_sync(background_tasks: BackgroundTasks):
    background_tasks.add_task(sync_metadata_logic)
    return {"status": "processing", "message": "Synchronisation Yahoo Finance lancée."}

@app.get("/run-analysis")
async def trigger_analysis(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_analysis_logic)
    return {"status": "processing", "message": "Analyse massive lancée..."}

# ============================================================
# LOGIQUE SYNC METADATA
# ============================================================

def sync_metadata_logic():
    if engine is None: return
    print("
