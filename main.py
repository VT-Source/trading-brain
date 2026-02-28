import os
import time
import pickle
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, BackgroundTasks
from sqlalchemy import create_engine, text
from理论上,
from dotenv import load_dotenv

# On importe ta fonction d'entraînement (assure-toi que train_model.py est dans le même dossier)
try:
    from train_model import train_brain
except ImportError:
    def train_brain(): 
        print("⚠️ Fonction train_brain non trouvée.")

load_dotenv()

app = FastAPI()

# --- Configuration DB ---
DATABASE_URL = os.getenv("DATABASE_URL")
engine = None

if DATABASE_URL:
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    # Suppression d'un éventuel slash final pour éviter les erreurs de connexion
    if DATABASE_URL.endswith("/") and "?" not in DATABASE_URL:
        DATABASE_URL = DATABASE_URL[:-1]
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

@app.get("/")
def home():
    return {
        "status": "Service Trading IA Actif", 
        "version": "4.9.1", 
        "stats_target": "500k_rows_ready",
        "features": ["Force-Sort-Groups", "Update-By-ID", "Native-Pandas-Math"]
    }

# --- ENDPOINT 1 : ENTRAÎNEMENT ---
@app.get("/train-model")
async def trigger_training(background_tasks: BackgroundTasks):
    background_tasks.add_task(train_brain)
    return {"status": "processing", "message": "Entraînement du cerveau lancé en tâche de fond..."}

# --- ENDPOINT 2 : SYNC METADATA ---
@app.get("/sync-metadata")
async def trigger_sync(background_tasks: BackgroundTasks):
    background_tasks.add_task(sync_metadata_logic)
    return {"status": "processing", "message": "Synchronisation Yahoo Finance lancée..."}

# --- ENDPOINT 3 : ANALYSE + PRÉDICTION IA ---
@app.get("/run-analysis")
async def trigger_analysis(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_analysis_logic)
    return {"status": "processing", "message": "Analyse technique et Prédictions IA lancées..."}

# --- LOGIQUE SYNC METADATA ---
def sync_metadata_logic():
    if engine is None: return
    print("🔄 Sync Yahoo Finance (Metadata)...")
    query = """
        SELECT DISTINCT ticker FROM actions_prix_historique
        WHERE ticker NOT IN (SELECT ticker FROM tickers_info)
        OR ticker IN (SELECT ticker FROM tickers_info WHERE derniere_maj < CURRENT_DATE - INTERVAL '30 days')
        LIMIT 500;
    """
    try:
        with engine.connect() as conn:
            tickers = [row[0] for row in conn.execute(text(query))]

        for ticker_symbol in tickers:
            try:
                stock = yf.Ticker(ticker_symbol)
                data = stock.info
                if "symbol" in data:
                    metadata = {
                        "ticker": ticker_symbol,
                        "name": data.get("longName"),
                        "secteur": data.get("sector"),
                        "industrie": data.get("industry"),
                        "pays": data.get("country"),
                        "monnaie": data.get("currency"),
                        "market_cap": data.get("marketCap"),
                        "pe_ratio": data.get("trailingPE")
                    }
                    with engine.begin() as conn:
                        conn.execute(text("""
                            INSERT INTO tickers_info (ticker, name, secteur, industrie, pays, monnaie, market_cap, pe_ratio, derniere_maj)
                            VALUES (:ticker, :name, :secteur, :industrie, :pays, :monnaie, :market_cap, :pe_ratio, CURRENT_DATE)
                            ON CONFLICT (ticker) DO UPDATE SET 
                                pe_ratio = EXCLUDED.pe_ratio, market_cap = EXCLUDED.market_cap, derniere_maj = CURRENT_DATE;
                        """), metadata)
                time.sleep(1)
            except Exception as e:
                print(f"⚠️ Erreur ticker {ticker_symbol}: {e}")
    except Exception as e:
        print(f"❌ Erreur Sync: {e}")

# --- LOGIQUE ANALYSE (LE COEUR DU SYSTÈME) ---
def run_analysis_logic():
    if engine is None: return
    print("
