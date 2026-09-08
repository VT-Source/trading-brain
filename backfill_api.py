# ============================================================
# backfill_api.py — Trading Brain — v1.0
# ============================================================
# Endpoint de rattrapage du ranking sur une plage de dates passées.
#
# Pourquoi un module séparé plutôt que 70 lignes de plus dans main.py :
#   1. main.py est repassé sous les 100 Ko au prix du lot #5, et « rester
#      sous les 100 Ko » est devenu un critère de conception — au-delà,
#      l'API GitHub refuse l'écriture et chaque lot impose un upload web
#      manuel (incident du 07/09). Ajouter un endpoint par lot dans main.py
#      reconstruit exactement le problème qu'on vient de payer pour défaire.
#   2. C'est la direction déjà prise par la roadmap (#5 livré, #30 ouvert) :
#      ce qui peut sortir de main.py en sort.
#
# `engine` et `run_job` sont INJECTÉS via `creer_router_backfill`, jamais
# importés : un `from main import engine` créerait un import circulaire.
# Même motif que ranking.py et analysis.py.
#
# ⚠️ DÉPLOIEMENT COUPLÉ : backfill_api.py + ranking.py + main.py.
# ============================================================

from datetime import date

from fastapi import APIRouter, BackgroundTasks

from ranking import backfill_ranking, valider_plage_backfill

BACKFILL_API_VERSION = "1.0"


def creer_router_backfill(engine, run_job) -> APIRouter:
    """
    Construit le router du rattrapage.

    Args:
        engine  : SQLAlchemy engine, injecté par main.py
        run_job : le `_run_job` de main.py. Le rattrapage laisse ainsi une
                  trace dans job_status et déclenche l'alerte (a) s'il
                  échoue — ce que les autres déclenchements manuels ne font
                  toujours pas (roadmap #29).
    """
    router = APIRouter()

    @router.get("/backfill-ranking")
    async def trigger_backfill_ranking(background_tasks: BackgroundTasks,
                                       debut: str, fin: str, top_n: int = 20,
                                       overwrite: bool = False,
                                       dry_run: bool = True,
                                       inclure_seance_du_jour: bool = False):
        """
        Recalcule le ranking sur une plage de dates PASSÉES (roadmap #31).

        Comble le trou du 24-31/08 (job figé, #25) et sert de rattrapage
        générique la prochaine fois qu'un job se grippe. `/compute-ranking`
        ne peut pas le faire : il écrit toujours `date_calcul = date.today()`.

        Paramètres :
            debut, fin  : bornes incluses, format ISO (2026-08-24)
            dry_run     : TRUE PAR DÉFAUT — simule et compare, n'écrit rien.
                          Il faut `dry_run=false` explicite pour écrire.
            overwrite   : inclure les dates déjà classées. Combiné à dry_run,
                          c'est le mode CALIBRATION : recalculer des dates
                          connues et comparer au stocké, sans rien toucher.
            inclure_seance_du_jour :
                          inclut la séance de la date cible dans le scoring.
                          ⚠️ Introduit du look-ahead : un ranking daté du
                          25/08 connaîtrait la clôture du 25/08, que la prod
                          ne pouvait pas voir à 02h15. Laisser à false pour
                          toute ligne destinée à l'analyse.

        Le détail par date (avec comparaison au stocké en mode calibration)
        part dans les logs Railway ; le résumé atterrit dans job_status,
        lisible via GET /health-jobs → jobs.backfill_ranking.

        ⚠️ Action manuelle — NE PAS planifier dans le scheduler.
        """
        try:
            debut_d = date.fromisoformat(debut)
            fin_d = date.fromisoformat(fin)
        except (ValueError, TypeError):
            return {"status": "error",
                    "message": "Dates attendues au format ISO, ex. "
                               "?debut=2026-08-24&fin=2026-08-31"}

        # Règles de plage : une seule implémentation, partagée avec
        # `backfill_ranking` (cf. valider_plage_backfill, et roadmap #30 sur
        # le coût des règles dupliquées). Ici elle sert à répondre tout de
        # suite ; là-bas à se protéger d'un appel venu d'ailleurs.
        probleme = valider_plage_backfill(debut_d, fin_d)
        if probleme:
            return {"status": "error", "message": probleme}

        background_tasks.add_task(run_job, "backfill_ranking", backfill_ranking,
                                  engine, debut_d, fin_d, top_n=top_n,
                                  overwrite=overwrite, dry_run=dry_run,
                                  inclure_seance_du_jour=inclure_seance_du_jour)

        return {
            "status": "processing",
            "mode": "dry_run" if dry_run else "ecriture",
            "message": (f"Backfill ranking {debut_d} → {fin_d} lancé en "
                        f"arrière-plan ("
                        f"{'simulation, aucune écriture' if dry_run else 'ÉCRITURE en base'}, "
                        f"top {top_n}, overwrite={overwrite}). "
                        f"Suivre le résultat dans GET /health-jobs → "
                        f"jobs.backfill_ranking."),
        }

    return router
