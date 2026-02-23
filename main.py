from fastapi import FastAPI, Body
from typing import List
import pandas as pd
import pandas_ta as ta

app = FastAPI()

@app.get("/")
def home():
    return {"status": "Service Trading Actif", "version": "1.2"}

@app.post("/analyze")
async def analyze_stock(data: List[dict] = Body(...)):
    # 1. Transformation des données reçues en tableau (Pandas)
    df = pd.DataFrame(data)
    
    # Vérification que les colonnes nécessaires existent
    if df.empty or 'prix_cloture' not in df.columns:
        return {"error": "Données incomplètes ou vides"}

    # 2. Calcul des indicateurs techniques
    # Calcul du RSI (période 14)
    df['rsi_14'] = ta.rsi(df['prix_cloture'], length=14)
    
    # Calcul de la moyenne du volume (20 derniers jours)
    df['vol_avg_20'] = df['volume'].rolling(window=20).mean()

    # 3. Logique du Signal d'Achat
    # Exemple : RSI < 30 (survendu) ET Volume > Moyenne
    last_row = df.iloc[-1].copy()
    
    rsi_value = float(last_row['rsi_14']) if not pd.isna(last_row['rsi_14']) else 50
    vol_value = float(last_row['volume'])
    vol_avg = float(last_row['vol_avg_20']) if not pd.isna(last_row['vol_avg_20']) else 0
    
    signal_achat = bool(rsi_value < 30 and vol_value > vol_avg)

    # 4. Retour des résultats à n8n
    return {
        "ticker": last_row.get('ticker', 'Inconnu'),
        "rsi_14": round(rsi_value, 2),
        "vol_avg_20": round(vol_avg, 2),
        "signal_achat": signal_achat,
        "derniere_cloture": float(last_row['prix_cloture'])
    }
