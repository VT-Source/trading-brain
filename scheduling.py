# ============================================================
# scheduling.py — Règles d'ordonnancement pures — Trading Brain v1.0
# VT-Source/trading-brain
# ============================================================
# Règles de décision du pipeline nocturne, extraites de main.py pour être
# testables sans DB, sans réseau et sans importer FastAPI/APScheduler
# (motif `zone_priority_for` — cf. PROJECT_STATUS, règle « extraire avant
# de tester »).
#
# Ce module n'a AUCUNE dépendance externe. Il est importé sans try/except
# par main.py, volontairement : un garde-fou de sécurité qui disparaît en
# silence est pire que pas de garde-fou du tout.
# ============================================================


def analysis_should_skip(job_status: dict | None) -> tuple[bool, str]:
    """
    L'analyse doit-elle renoncer à tourner maintenant ?

    Incident du 2026-09-07 : `sync_prix` a dérivé de ~5 min (durée retenue
    au moment où le planning a été écrit) à ~33 min. L'analyse, planifiée
    30 min après, démarrait donc pendant que la sync écrivait encore dans
    `actions_prix_historique`. Comme la sync traite les tickers dans
    l'ordre, ceux qui manquaient étaient la FIN de la liste : 28 tickers
    (V→Z) sont sortis sans aucun indicateur, et le job s'est terminé en
    `ok` — la panne était invisible côté statut.

    Le vrai correctif est le chaînage (l'analyse part à la fin réelle de la
    sync, plus à une heure fixe). Cette règle couvre ce que le chaînage ne
    couvre pas : un déclenchement manuel de /run-analysis ou
    /run-analysis-full pendant que le pipeline nocturne tourne.

    Retourne (True, motif) s'il faut renoncer, (False, "") sinon.
    Ne lève jamais : un job_status absent ou malformé ne doit pas empêcher
    l'analyse de tourner.
    """
    if not job_status or not isinstance(job_status, dict):
        return False, ""

    etat_sync = job_status.get("sync_prix")
    if not isinstance(etat_sync, dict):
        return False, ""

    if etat_sync.get("status") == "running":
        return True, ("sync_prix est en cours d'exécution : lire "
                      "actions_prix_historique maintenant produirait des "
                      "indicateurs manquants sur les derniers tickers "
                      "synchronisés (incident du 2026-09-07)")

    return False, ""


def analysis_should_follow_sync(job_status: dict | None) -> tuple[bool, str]:
    """
    Après `sync_prix`, faut-il enchaîner l'analyse ?

    Non si la sync n'a pas abouti : recalculer des indicateurs sur des prix
    partiels revient à écrire de fausses valeurs plutôt qu'aucune, ce que le
    projet refuse par principe (cf. décision R4 du 2026-09-04, point 5).

    Retourne (True, "") s'il faut enchaîner, (False, motif) sinon.
    """
    if not job_status or not isinstance(job_status, dict):
        return False, "job_status indisponible"

    etat_sync = job_status.get("sync_prix")
    if not isinstance(etat_sync, dict):
        return False, "sync_prix n'a pas de statut"

    statut = etat_sync.get("status")
    if statut == "ok":
        return True, ""

    return False, f"sync_prix n'a pas abouti (statut={statut!r})"
