from fastapi import FastAPI, Body
import pandas as pd
import pandas_ta as ta
from typing import List

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "Service Trading Actif"}

@app.post("/analyze")
async def analyze_stock(data: List[dict] = Body(...)):
    df = pd.DataFrame(data)
    
    # On s'assure que les colonnes sont au bon format
    df['prix_cloture'] = pd.to_numeric(df['prix_cloture'])
    df['volume'] = pd.to_numeric(df['volume'])

    # Calcul RSI et Moyenne Volume
    df['rsi_14'] = ta.rsi(df['prix_cloture'], length=14)
    df['vol_avg_20'] = df['volume'].rolling(window=20).mean()
    
    # Extraction des dernières valeurs
    last_row = df.iloc[-1]
    
    # Logique du signal
    signal = False
    if last_row['volume'] > (last_row['vol_avg_20'] * 2.5) and last_row['rsi_14'] < 65:
        signal = True

    return {
        "rsi_14": float(last_row['rsi_14']) if not pd.isna(last_row['rsi_14']) else None,
        "vol_avg_20": float(last_row['vol_avg_20']) if not pd.isna(last_row['vol_avg_20']) else None,
        "signal_achat": bool(signal)
    }
