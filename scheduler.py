# ============================================================
# scheduler.py — Orchestration APScheduler v3.1
# Trading Brain | VT-Source
# ============================================================
# Remplace le cron n8n pour les tâches Python.
# n8n reste uniquement pour les notifications Telegram.
#
# Jobs planifiés :
#   06h00 — Analyse technique + score ML (run_analysis_logic)
#   06h30 — Sync metadata Yahoo Finance (sync_metadata_logic)
#   Dimanche 02h00 — Réentraînement ML (train_brain)
# ============================================================

import logging
import os
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

load_dotenv()

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# --- Import des fonctions métier ---
# Importées ici pour éviter les imports circulaires
try:
    from main import run_analysis_logic, sync_metadata_logic
except ImportError as e:
    log.error(f"❌ Impossible d'importer main.py : {e}")
    run_analysis_logic  = lambda: log.error("run_analysis_logic non disponible")
    sync_metadata_logic = lambda: log.error("sync_metadata_logic non disponible")

try:
    from train_model import train_brain
except ImportError as e:
    log.error(f"❌ Impossible d'importer train_model.py : {e}")
    train_brain = lambda: log.error("train_brain non disponible")


# ============================================================
# WRAPPERS — Gestion des erreurs pour chaque job
# ============================================================

def job_analyse():
    log.info("🚀 Démarrage job : Analyse technique + score ML")
    try:
        run_analysis_logic()
        log.info("✅ Job analyse terminé.")
    except Exception as e:
        log.error(f"❌ Erreur job analyse : {e}")


def job_sync_metadata():
    log.info("🔄 Démarrage job : Sync metadata Yahoo Finance")
    try:
        sync_metadata_logic()
        log.info("✅ Job sync metadata terminé.")
    except Exception as e:
        log.error(f"❌ Erreur job sync metadata : {e}")


def job_train_model():
    log.info("🧠 Démarrage job : Réentraînement ML")
    try:
        train_brain()
        log.info("✅ Job réentraînement ML terminé.")
    except Exception as e:
        log.error(f"❌ Erreur job réentraînement : {e}")


# ============================================================
# SCHEDULER
# ============================================================

def main():
    scheduler = BlockingScheduler(timezone="Europe/Brussels")

    # --- Job 1 : Analyse AT + ML — chaque jour à 06h00 ---
    scheduler.add_job(
        job_analyse,
        trigger=CronTrigger(hour=6, minute=0),
        id="analyse_quotidienne",
        name="Analyse technique + score ML",
        replace_existing=True,
        misfire_grace_time=600  # Tolérance 10 min si Railway redémarre
    )

    # --- Job 2 : Sync metadata — chaque jour à 06h30 ---
    scheduler.add_job(
        job_sync_metadata,
        trigger=CronTrigger(hour=6, minute=30),
        id="sync_metadata",
        name="Sync metadata Yahoo Finance",
        replace_existing=True,
        misfire_grace_time=600
    )

    # --- Job 3 : Réentraînement ML — chaque dimanche à 02h00 ---
    # À n'activer qu'après avoir validé signal_achat + target_ml
    scheduler.add_job(
        job_train_model,
        trigger=CronTrigger(day_of_week="sun", hour=2, minute=0),
        id="reentrainement_ml",
        name="Réentraînement ML hebdomadaire",
        replace_existing=True,
        misfire_grace_time=3600
    )

    log.info("⏰ Scheduler démarré — Jobs planifiés :")
    log.info("   06h00 (lun-ven) → Analyse AT + ML")
    log.info("   06h30 (lun-ven) → Sync metadata")
    log.info("   02h00 (dimanche) → Réentraînement ML")
    log.info("   Timezone : Europe/Brussels")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("🛑 Scheduler arrêté proprement.")


if __name__ == "__main__":
    main()
