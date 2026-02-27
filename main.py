import os
import time
import requests
import pandas as pd
import pandas_ta as ta
from fastapi import FastAPI, BackgroundTasks
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# --- Configuration DB ---
DATABASE_URL = os.getenv("DATABASE_URL")
ALPHA_VANTAGE_KEY = "6LSG483HD1NHNK2R" # Ta clé API
engine = None

if DATABASE_URL:
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    if DATABASE_URL.endswith("/") and "?" not in DATABASE_URL:
        DATABASE_URL = DATABASE_URL[:-1]
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

@app.get("/")
def home():
    return {
        "status": "Service Trading Actif", 
        "version": "3.2", 
        "features": ["ML-Ready", "AlphaVantage-Sync", "Fix-Numpy-Types"]
    }

# --- ENDPOINT 1 : ANALYSE TECHNIQUE (Quotidien) ---
@app.get("/run-analysis")
async def trigger_analysis(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_analysis_logic)
    return {"status": "processing", "message": "Calculs techniques et Target ML lancés..."}

# --- ENDPOINT 2 : SYNC METADATA (Hebdomadaire/Cron) ---
@app.get("/sync-metadata")
async def trigger_sync(background_tasks: BackgroundTasks):
    background_tasks.add_task(sync_metadata_logic)
    return {"status": "processing", "message": "Synchronisation des secteurs et P/E Ratios lancée..."}

def sync_metadata_logic():
    if engine is None: return
    print("🔄 Démarrage de la synchronisation Alpha Vantage...")
    
    # On cherche les tickers inconnus OU mis à jour il y a plus de 30 jours
    query = """
        SELECT DISTINCT ticker FROM actions_prix_historique
        WHERE ticker NOT IN (SELECT ticker FROM tickers_info)
        OR ticker IN (SELECT ticker FROM tickers_info WHERE derniere_maj < CURRENT_DATE - INTERVAL '30 days')
        LIMIT 20;
    """
    
    try:
        with engine.connect()
