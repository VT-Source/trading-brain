# ============================================================
# scheduler.py — Orchestration APScheduler v3.3
# Trading Brain | VT-Source
# ============================================================
# Jobs planifiés (automatiques) :
#   05h30 lun-ven — Sync prix quotidien yfinance (remplace n8n)
#   06h00 lun-ven — Analyse incrémentale AT + ML
#                   (charge 220j de contexte, sauvegarde 5j)
#   06h15 lun-ven — Sync ETF sectoriels + Force Relative (30j)
#   06h30 lun-ven — Sync metadata Yahoo Finance
#   06h45 lund - Job Ranking Hebdo
#   Dimanche 02h00 — Réentraînement ML hebdomadaire
#
# ⚠️  /run-analysis-full N'EST PAS planifié ici.
#     À appeler manuellement depuis Railway ou Postman :
#       - Après un changement de logique AT
#       - Après l'ajout d'un nouvel indicateur
#       - Pour initialiser un nouveau ticker
#
# ⚠️  /sync-etf-sectoriels?full=true N'EST PAS planifié ici.
#     À appeler manuellement une seule fois après déploiement
#     pour initialiser 5 ans d'historique ETF.
# ============================================================

import logging
import os
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

try:
    from main import run_analysis_logic, sync_metadata_logic, sync_secteurs_etf_logic, compute_and_store_ranking, sync_prix_logic
except ImportError as e:
    log.error(f"❌ Impossible d'importer main.py : {e}")
    run_analysis_logic      = lambda **kw: log.error("run_analysis_logic non disponible")
    sync_metadata_logic     = lambda: log.error("sync_metadata_logic non disponible")
    sync_secteurs_etf_logic = lambda **kw: log.error("sync_secteurs_etf_logic non disponible")
    compute_and_store_ranking = lambda **kw: log.error("compute_and_store_ranking non disponible")
    sync_prix_logic           = lambda **kw: log.error("sync_prix_logic non disponible")

try:
    from train_model import train_brain
except ImportError as e:
    log.error(f"❌ Impossible d'importer train_model.py : {e}")
    train_brain = lambda: log.error("train_brain non disponible")


# ============================================================
# WRAPPERS
# ============================================================

def job_sync_prix():
    """
    Sync quotidienne des prix OHLCV (remplace n8n).
    Lance à 05h30, AVANT l'analyse (06h00) pour que les indicateurs soient calculés sur des données fraîches.
    NE PAS appeler avec full=True ici — utiliser /sync-prix?full=true manuellement.
    """
    log.info("💰 Job : Sync prix quotidien yfinance (incrémental 30j)")
    try:
        sync_prix_logic(full=False)
        log.info("✅ Job sync prix terminé.")
    except Exception as e:
        log.error(f"❌ Erreur job sync prix : {e}")
        
def job_analyse():
    """
    Analyse incrémentale quotidienne.
    Charge 220 jours de contexte par ticker, ne sauvegarde que les 5 derniers jours.
    NE PAS modifier pour appeler full=True — utiliser /run-analysis-full manuellement.
    """
    log.info("🚀 Job : Analyse incrémentale AT + ML (full=False)")
    try:
        run_analysis_logic(full=False)
        log.info("✅ Job analyse terminé.")
    except Exception as e:
        log.error(f"❌ Erreur job analyse : {e}")


def job_sync_etf():
    """
    Sync quotidienne des ETF sectoriels (mode incrémental — 30 derniers jours).
    Lance à 06h15, après l'analyse (06h00) et avant la sync metadata (06h30).
    NE PAS appeler avec full=True ici — utiliser /sync-etf-sectoriels?full=true manuellement.
    """
    log.info("📊 Job : Sync ETF sectoriels + Force Relative (incrémental 30j)")
    try:
        sync_secteurs_etf_logic(full=False)
        log.info("✅ Job sync ETF sectoriels terminé.")
    except Exception as e:
        log.error(f"❌ Erreur job sync ETF : {e}")


def job_sync_metadata():
    log.info("🔄 Job : Sync metadata Yahoo Finance")
    try:
        sync_metadata_logic()
        log.info("✅ Job sync metadata terminé.")
    except Exception as e:
        log.error(f"❌ Erreur job sync metadata : {e}")

def job_compute_ranking():
    """
    Pré-calcul du ranking hebdomadaire.
    Lancé le lundi à 06h45, après analyse (06h00), ETF (06h15), metadata (06h30).
    Résultat stocké dans ranking_hebdo, lu instantanément par le dashboard.
    """
    log.info("📊 Job : Calcul ranking hebdomadaire")
    try:
        result = compute_and_store_ranking(top_n=20)
        if "error" in result:
            log.error(f"❌ Erreur ranking : {result['error']}")
        else:
            log.info(f"✅ Ranking calculé : {result.get('nb_ranked', 0)} tickers")
    except Exception as e:
        log.error(f"❌ Erreur job ranking : {e}")

def job_train_model():
    log.info("🧠 Job : Réentraînement ML hebdomadaire")
    try:
        train_brain()
        log.info("✅ Job réentraînement terminé.")
    except Exception as e:
        log.error(f"❌ Erreur job réentraînement : {e}")


# ============================================================
# SCHEDULER
# ============================================================

def main():
    scheduler = BlockingScheduler(timezone="Europe/Brussels")

    # Job 0 : Sync prix quotidien — lun-ven à 05h30
    scheduler.add_job(
        job_sync_prix,
        trigger=CronTrigger(day_of_week="mon-fri", hour=5, minute=30),
        id="sync_prix",
        name="Sync prix quotidien yfinance",
        replace_existing=True,
        misfire_grace_time=600
    )

    # Job 1 : Analyse incrémentale — lun-ven à 06h00
    scheduler.add_job(
        job_analyse,
        trigger=CronTrigger(day_of_week="mon-fri", hour=6, minute=0),
        id="analyse_incrementale",
        name="Analyse incrémentale AT + ML",
        replace_existing=True,
        misfire_grace_time=600
    )

    # Job 2 : Sync ETF sectoriels — lun-ven à 06h15
    scheduler.add_job(
        job_sync_etf,
        trigger=CronTrigger(day_of_week="mon-fri", hour=6, minute=15),
        id="sync_etf_sectoriels",
        name="Sync ETF sectoriels + Force Relative",
        replace_existing=True,
        misfire_grace_time=600
    )

    # Job 3 : Sync metadata — lun-ven à 06h30
    scheduler.add_job(
        job_sync_metadata,
        trigger=CronTrigger(day_of_week="mon-fri", hour=6, minute=30),
        id="sync_metadata",
        name="Sync metadata Yahoo Finance",
        replace_existing=True,
        misfire_grace_time=600
    )

    # Job 4 : Réentraînement ML — dimanche à 02h00
    scheduler.add_job(
        job_train_model,
        trigger=CronTrigger(day_of_week="sun", hour=2, minute=0),
        id="reentrainement_ml",
        name="Réentraînement ML hebdomadaire",
        replace_existing=True,
        misfire_grace_time=3600
    )

    # Job 5 : Calcul ranking hebdomadaire — lundi à 06h45
    scheduler.add_job(
        job_compute_ranking,
        trigger=CronTrigger(day_of_week="mon", hour=6, minute=45),
        id="compute_ranking",
        name="Calcul ranking hebdomadaire",
        replace_existing=True,
        misfire_grace_time=600
    )

    log.info("⏰ Scheduler démarré — Jobs planifiés :")
    log.info("   05h30 (lun-ven) → Sync prix quotidien yfinance")
    log.info("   06h00 (lun-ven) → Analyse incrémentale AT + ML")
    log.info("   06h15 (lun-ven) → Sync ETF sectoriels + Force Relative")
    log.info("   06h30 (lun-ven) → Sync metadata Yahoo Finance")
    log.info("   06h45 (lundi)   → Calcul ranking hebdomadaire")
    log.info("   02h00 (dimanche) → Réentraînement ML")
    log.info("")
    log.info("   ⚠️  Rattrapage prix (5 ans)    : GET /sync-prix?full=true             (MANUEL)")
    log.info("   ⚠️  Initialisation ETF (5 ans) : GET /sync-etf-sectoriels?full=true  (MANUEL)")
    log.info("   ⚠️  Recalcul complet AT        : GET /run-analysis-full              (MANUEL)")
    log.info("   Timezone : Europe/Brussels")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("🛑 Scheduler arrêté proprement.")


if __name__ == "__main__":
    main()
