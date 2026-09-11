# ============================================================
# sonde.py — Sonde de publication Yahoo — Trading Brain v1.0
# VT-Source/trading-brain
# ============================================================
# Mesure TEMPORAIRE (roadmap #35, décision du 2026-09-11).
#
# Constat : au run de 01h00 UTC, Yahoo ne renvoie pas encore la séance de la
# veille pour la plupart des actions US et européennes (la Corée, elle, est
# à jour). La séance J entre donc en base à la DEUXIÈME nuit : depuis au
# moins juillet pour l'Europe, depuis le run du 2026-09-03 pour les US — sur
# le même déploiement, donc côté Yahoo et non côté yfinance. Le ranking et les
# 5 conditions de sortie tournent chaque nuit sur J-2 (US/EU) et J-1 (KR), et
# la règle de fraîcheur (#28) classe ces places « fermées ».
#
# Avant de décaler le pipeline, d'ajouter une resync ou d'accepter le retard,
# on mesure : toutes les heures, 15 instruments sont téléchargés avec les
# MÊMES paramètres que sync_prix_logic, et la dernière date reçue est stockée
# dans `sonde_publication`. `sql/analyse_sonde_publication.sql` en tire
# l'heure UTC de publication de la séance J, place par place.
#
# Ce que ce module ne fait PAS : il n'écrit dans aucune table lue par le
# moteur, ne passe pas par _run_job (aucune alerte, aucune clé dans
# /health-jobs, que le dashboard lit), et ne lève jamais d'exception.
#
# Inerte tant que la variable SONDE_PUBLICATION ne vaut pas "on".
#
# yfinance, sqlalchemy et apscheduler sont importés dans les fonctions qui
# s'en servent : les règles pures restent testables sans eux.
# ============================================================

import math
import os
import time
from datetime import datetime, timezone
from typing import NamedTuple

import pandas as pd

from scheduling import PIPELINE_NOCTURNE

VARIABLE_ACTIVATION = "SONDE_PUBLICATION"
JOB_ID = "sonde_publication"

# Minute de passage, chaque heure, 7j/7. Un passage dure ~30 s (15 appels,
# 1 s de pause) et au pire ~8 min si chaque appel atteint son timeout de
# 30 s : il ne peut pas empiéter sur sync_fx (00h55) ni sur le pipeline
# (01h00). Le passage de 01h20 tombe pendant le pipeline et se saute.
MINUTE_PASSAGE = 20
PAUSE_ENTRE_APPELS_S = 1.0

# Paramètres de yf.download, IDENTIQUES à sync_prix_logic en mode quotidien
# (sync.py, appel yf.download + `period = ... "1mo"`). La sonde doit voir ce
# que voit la sync, sinon elle mesure autre chose — la plage demandée peut
# changer la réponse de Yahoo. Verrouillé par tests/test_sonde.py.
PARAMETRES_TELECHARGEMENT = {
    "period": "1mo",
    "interval": "1d",
    "auto_adjust": True,
    "progress": False,
    "timeout": 30,
}


class InstrumentSonde(NamedTuple):
    ticker: str
    place: str
    type_instrument: str   # "indice" | "etf" | "action"


# Un indice, un ETF et des actions par zone : le 2026-09-11 à 01h30 UTC,
# ^GSPC avait déjà la séance du 10/09 alors que XLK ne l'avait pas. Côté
# Europe, une action par place, dont BME et SIX — les deux places
# « partielles » de la règle de fraîcheur.
# Pour les actions et ETF, `place` doit valoir freshness.place_de_cotation
# (vérifié par test, sans import : la sonde n'ajoute aucun couplage). Les
# indices n'ont pas de suffixe, leur place est déclarée à la main.
INSTRUMENTS_SONDE = (
    InstrumentSonde("^GSPC",     "NYSE/NASDAQ",        "indice"),
    InstrumentSonde("XLK",       "NYSE/NASDAQ",        "etf"),
    InstrumentSonde("AAPL",      "NYSE/NASDAQ",        "action"),
    InstrumentSonde("JPM",       "NYSE/NASDAQ",        "action"),
    InstrumentSonde("^STOXX",    "STOXX Europe 600",   "indice"),
    InstrumentSonde("EXV1.DE",   "Xetra",              "etf"),
    InstrumentSonde("SAP.DE",    "Xetra",              "action"),
    InstrumentSonde("MC.PA",     "Euronext Paris",     "action"),
    InstrumentSonde("ASML.AS",   "Euronext Amsterdam", "action"),
    InstrumentSonde("KBC.BR",    "Euronext Bruxelles", "action"),
    InstrumentSonde("IBE.MC",    "BME Madrid",         "action"),
    InstrumentSonde("NOVN.SW",   "SIX Zurich",         "action"),
    InstrumentSonde("^KS11",     "KRX",                "indice"),
    InstrumentSonde("091160.KS", "KRX",                "etf"),
    InstrumentSonde("005930.KS", "KRX",                "action"),
)

DDL_SONDE = """
    CREATE TABLE IF NOT EXISTS sonde_publication (
        id                    SERIAL PRIMARY KEY,
        sonde_at              TIMESTAMPTZ      NOT NULL,
        ticker                VARCHAR(20)      NOT NULL,
        place                 VARCHAR(40)      NOT NULL,
        type_instrument       VARCHAR(10)      NOT NULL,
        derniere_date_brute   DATE,
        derniere_date_valide  DATE,
        derniere_cloture      DOUBLE PRECISION,
        dernier_volume        DOUBLE PRECISION,
        nb_barres             INTEGER          NOT NULL DEFAULT 0,
        erreur                TEXT
    )
"""

INSERT_SONDE = """
    INSERT INTO sonde_publication
        (sonde_at, ticker, place, type_instrument, derniere_date_brute,
         derniere_date_valide, derniere_cloture, dernier_volume, nb_barres, erreur)
    VALUES
        (:sonde_at, :ticker, :place, :type_instrument, :derniere_date_brute,
         :derniere_date_valide, :derniere_cloture, :dernier_volume, :nb_barres, :erreur)
"""


# ============================================================
# Règles pures
# ============================================================

def sonde_peut_tourner(job_status) -> tuple[bool, str]:
    """
    La sonde peut-elle passer maintenant ?

    Non tant qu'une étape du pipeline nocturne est « running » : des appels
    yfinance concurrents pendant sync_prix exposeraient la sync aux réponses
    vides silencieuses du rate limiting Yahoo (incident du 2026-05-16). On
    lit l'état réel des étapes plutôt qu'un créneau horaire — le pipeline n'a
    plus d'heure de fin fixe depuis #33.

    Ne lève jamais : un job_status absent ou mal formé laisse passer.
    """
    if not isinstance(job_status, dict):
        return True, ""
    for etape in PIPELINE_NOCTURNE:
        etat = job_status.get(etape.id)
        if isinstance(etat, dict) and etat.get("status") == "running":
            return False, f"étape '{etape.id}' du pipeline en cours"
    return True, ""


def _en_float(valeur):
    """float Python, ou None pour NaN / valeur absente / non numérique."""
    try:
        f = float(valeur)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def resumer_reponse(df) -> dict:
    """
    Résume une réponse yf.download en une ligne de mesure.

    Applique la même normalisation que sync_prix_logic (colonnes MultiIndex
    aplaties, `Date` → datetime.date) mais PAS son `dropna` sur la clôture :
    on veut voir ce que Yahoo renvoie, y compris une dernière ligne sans
    clôture que la sync écarterait sans rien dire. D'où deux dates :

      - derniere_date_brute  : dernière ligne renvoyée, clôture ou non
      - derniere_date_valide : dernière ligne avec une clôture exploitable —
                               c'est celle qu'aurait retenue la sync

    `derniere_cloture` et `dernier_volume` sont ceux de la ligne brute : une
    barre de séance en cours (KRX à 10h KST) se reconnaît à son volume.
    """
    vide = {
        "derniere_date_brute": None,
        "derniere_date_valide": None,
        "derniere_cloture": None,
        "dernier_volume": None,
        "nb_barres": 0,
        "erreur": None,
    }
    if df is None or getattr(df, "empty", True):
        return {**vide, "erreur": "réponse vide"}

    df = df.reset_index()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    if "Date" not in df.columns or "Close" not in df.columns:
        return {**vide, "erreur": f"colonnes inattendues : {list(df.columns)[:8]}"}

    df = df.assign(_date=pd.to_datetime(df["Date"]).dt.date).sort_values("_date")
    brute = df.iloc[-1]
    valides = df[df["Close"].notna()]

    return {
        "derniere_date_brute": brute["_date"],
        "derniere_date_valide": valides["_date"].iloc[-1] if not valides.empty else None,
        "derniere_cloture": _en_float(brute["Close"]),
        "dernier_volume": _en_float(brute["Volume"]) if "Volume" in df.columns else None,
        "nb_barres": int(len(df)),
        "erreur": None,
    }


# ============================================================
# Exécution (DB + réseau)
# ============================================================

def executer_sonde(engine, job_status, telecharger=None, maintenant=None,
                   pause_s=PAUSE_ENTRE_APPELS_S) -> dict:
    """
    Un passage de la sonde : 15 téléchargements, une ligne par instrument.

    Ne lève jamais. Une erreur sur un instrument est stockée dans sa ligne ;
    une erreur globale (DB indisponible) est journalisée et renvoyée.
    `telecharger` et `maintenant` sont injectables pour les tests.
    """
    try:
        if engine is None:
            print("🛰️ Sonde de publication : engine absent — passage ignoré")
            return {"status": "skipped", "reason": "engine absent"}

        peut, motif = sonde_peut_tourner(job_status)
        if not peut:
            print(f"🛰️ Sonde de publication sautée : {motif}")
            return {"status": "skipped", "reason": motif}

        if telecharger is None:
            import yfinance as yf

            def telecharger(ticker):
                return yf.download(ticker, **PARAMETRES_TELECHARGEMENT)

        sonde_at = maintenant or datetime.now(timezone.utc)
        lignes = []
        for i, instrument in enumerate(INSTRUMENTS_SONDE):
            if i and pause_s:
                time.sleep(pause_s)
            try:
                mesure = resumer_reponse(telecharger(instrument.ticker))
            except Exception as e:
                mesure = {**resumer_reponse(None), "erreur": f"exception : {e}"[:500]}
            lignes.append({
                "sonde_at": sonde_at,
                "ticker": instrument.ticker,
                "place": instrument.place,
                "type_instrument": instrument.type_instrument,
                **mesure,
            })

        from sqlalchemy import text
        with engine.begin() as conn:
            conn.execute(text(DDL_SONDE))
        with engine.begin() as conn:
            conn.execute(text(INSERT_SONDE), lignes)

        resume = " | ".join(
            f"{l['ticker']}→{l['derniere_date_valide'] or 'ERR'}" for l in lignes
        )
        print(f"🛰️ Sonde de publication {sonde_at:%Y-%m-%d %H:%M} UTC — {resume}")
        return {
            "status": "ok",
            "nb_lignes": len(lignes),
            "nb_erreurs": sum(1 for l in lignes if l["erreur"]),
        }

    except Exception as e:
        print(f"⚠️ Sonde de publication échouée : {e}")
        return {"status": "error", "error": str(e)[:500]}


def enregistrer_sonde(scheduler, engine, job_status, env=None,
                      creer_declencheur=None) -> bool:
    """
    Ajoute le job horaire au scheduler si SONDE_PUBLICATION vaut "on".

    Retourne True si le job est planifié. Sans la variable, le module est
    déployé mais inerte : on arrête la mesure en retirant la variable sur
    Railway, sans nouveau déploiement de code (⚠️ la modifier redéploie le
    service : hors fenêtre 00h55–02h30 UTC).
    """
    env = os.environ if env is None else env
    if str(env.get(VARIABLE_ACTIVATION, "")).strip().lower() != "on":
        print(f"🛰️ Sonde de publication inactive ({VARIABLE_ACTIVATION} ≠ on)")
        return False

    if creer_declencheur is None:
        from apscheduler.triggers.cron import CronTrigger

        def creer_declencheur():
            return CronTrigger(minute=MINUTE_PASSAGE, timezone="UTC")

    scheduler.add_job(lambda: executer_sonde(engine, job_status),
                      creer_declencheur(),
                      id=JOB_ID, replace_existing=True, misfire_grace_time=600)
    print(f"🛰️ Sonde de publication active — {len(INSTRUMENTS_SONDE)} instruments, "
          f"toutes les heures à :{MINUTE_PASSAGE:02d} UTC")
    return True
