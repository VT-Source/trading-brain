import os
import pandas as pd
import pandas_ta as ta
from fastapi import FastAPI, BackgroundTasks
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# --- Configuration DB (Railway & n8n) ---
DATABASE_URL = os.getenv("DATABASE_URL")

engine = None
if DATABASE_URL:
    # Normalisation du protocole pour SQLAlchemy
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

    # On évite juste un slash final accidentel, sans toucher aux query params
    if DATABASE_URL.endswith("/") and "?" not in DATABASE_URL:
        DATABASE_URL = DATABASE_URL[:-1]

    # Force le search_path=public pour éviter toute ambiguïté de schéma
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        connect_args={"options": "-csearch_path=public"}
    )

# --- Health ---
@app.get("/")
def home():
    return {"status": "Service Trading Actif", "version": "1.8"}

# --- Debug DB ---
@app.get("/debug/db-info")
def db_info():
    if engine is None:
        return {"error": "Engine not initialized"}
    with engine.connect() as conn:
        info = conn.execute(text("""
            SELECT current_database() AS db,
                   current_user      AS usr,
                   inet_server_addr() AS host_ip,
                   inet_server_port() AS port,
                   current_setting('ssl') AS ssl_on,
                   current_setting('search_path') AS search_path;
        """)).mappings().first()

        tables = conn.execute(text("""
            SELECT schemaname, tablename
            FROM pg_catalog.pg_tables
            WHERE schemaname NOT IN ('pg_catalog','information_schema')
            ORDER BY schemaname, tablename;
        """)).mappings().all()

    return {"info": dict(info), "tables": [dict(t) for t in tables]}

# --- Endpoint d'exécution ---
@app.get("/run-analysis")
async def trigger_analysis(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_analysis)
    return {"status": "processing", "message": "Calculs lancés sur la base partagée"}

# --- Job principal ---
def run_analysis():
    if engine is None:
        print("❌ Engine non initialisé")
        return

    print("🚀 Démarrage de l'analyse technique...")

    table_name = "actions_prix_historique"
    schema_name = "public"

    try:
        # 1) Lecture (schéma qualifié)
        query = f'SELECT * FROM {schema_name}."{table_name}" ORDER BY ticker, date'
        df = pd.read_sql(query, engine)

        if df.empty:
            print("⚠️ Table trouvée mais vide. Rien à calculer.")
            return

        # 2) Prétraitement & types
        #    - Date en type date (pas timestamp) pour un JOIN propre sur DATE
        #    - Numérisation safe (coerce) pour éviter les plantages
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
        df = df.sort_values(['ticker', 'date'])

        df['prix_ajuste'] = pd.to_numeric(df.get('prix_ajuste'), errors='coerce')
        df['volume'] = pd.to_numeric(df.get('volume'), errors='coerce')

        # 3) Calculs techniques
        #    RSI 14 périodes sur prix_ajuste
        df['rsi_14'] = (
            df.groupby('ticker', dropna=False)['prix_ajuste']
              .transform(lambda x: ta.rsi(x, length=14))
        )

        #    Moyenne mobile de volume (20 jours) - on laisse min_periods=20 pour rester "classique"
        df['vol_avg_20'] = (
            df.groupby('ticker', dropna=False)['volume']
              .transform(lambda x: x.rolling(window=20, min_periods=20).mean())
        )

        #    Signal achat
        df['signal_achat'] = (df['rsi_14'] < 35) & (df['volume'] > df['vol_avg_20'])

        # 4) Ecriture SANS remplacer la table principale
        #    On passe par une table temporaire puis un UPDATE en masse
        calc_cols = ['ticker', 'date', 'rsi_14', 'vol_avg_20', 'signal_achat']
        tmp_table = "_calc_tmp"

        # Table temporaire (replace autorisé sur la TMP SEULEMENT)
        df_tmp = df[calc_cols].copy()
        df_tmp.to_sql(
            tmp_table,
            engine,
            if_exists='replace',
            index=False,
            schema=schema_name
        )

        # 5) Mise à jour en masse des champs calculés
        with engine.begin() as conn:
            conn.execute(text(f"""
                UPDATE {schema_name}."{table_name}" a
                SET rsi_14    = t.rsi_14,
                    vol_avg_20 = t.vol_avg_20,
                    signal_achat = t.signal_achat
                FROM {schema_name}."{tmp_table}" t
                WHERE a.ticker = t.ticker
                  AND a.date   = t.date;
            """))

            # Nettoyage de la table temporaire
            conn.execute(text(f'DROP TABLE IF EXISTS {schema_name}."{tmp_table}";'))

        print(f"✅ Analyse terminée. {len(df)} lignes traitées (updatées).")

    except Exception as e:
        # Aide au diagnostic typique : relation absente / base différente / schéma
        print(f"❌ Erreur lors de l'exécution : {e}")
        print("ℹ️ Vérifie que n8n et ce service pointent la MÊME base et que la table "
              f'{schema_name}."{table_name}" existe avec les colonnes attendues.')
