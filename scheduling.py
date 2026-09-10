# ============================================================
# scheduling.py — Règles d'ordonnancement pures — Trading Brain v1.2
# VT-Source/trading-brain
# ============================================================
# Règles de décision du pipeline nocturne, extraites de main.py pour être
# testables sans DB, sans réseau et sans importer FastAPI/APScheduler
# (motif `zone_priority_for` — cf. PROJECT_STATUS, règle « extraire avant
# de tester »).
#
# Ce module n'a AUCUNE dépendance tierce ni projet. Il est importé sans
# try/except par main.py, volontairement : un garde-fou de sécurité qui
# disparaît en silence est pire que pas de garde-fou du tout. Seule la
# bibliothèque standard est autorisée (v1.2, pour `typing.NamedTuple`) —
# elle est présente dans tout déploiement Python, donc l'import ne peut
# pas échouer. `tests/test_couplages.py` verrouille cette limite.
# ============================================================

from typing import NamedTuple


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


def masque_barres_closes(dates, aujourd_hui) -> list[bool]:
    """
    Quelles barres correspondent à une séance TERMINÉE ?

    Incident du 2026-09-07 : le pipeline tourne à 01h00 UTC, soit 10h00 KST
    — le KRX est ouvert depuis une heure. yfinance renvoie alors la barre
    du jour EN COURS pour les 10 tickers coréens : volume à 9-20 % de sa
    médiane, donc RVOL effondré. Le RVOL pesant 25 % du score composite,
    la zone KR disparaissait du ranking du lundi au vendredi et n'y
    réapparaissait que le samedi, seul jour calculé sur données closes
    (21 apparitions sur 39 en 60 jours, RVOL 1,06 le samedi contre 0,20
    à 0,32 du mardi au jeudi).

    La barre partielle était ensuite écrasée par la barre complète au run
    suivant : le bug ne laissait aucune trace dans la table des prix, et
    n'était visible que dans les snapshots de `ranking_hebdo`.

    Règle : on ne stocke que des séances closes. La barre du jour D entre
    en base, complète, au run de D+1. Vrai quel que soit le fuseau et quel
    que soit le marché — contrairement à un décalage d'horaire, qui
    redevient faux au prochain changement d'heure ou au prochain fuseau
    ajouté à l'univers.

    `dates` : itérable de datetime.date. `aujourd_hui` : datetime.date.
    Retourne un masque booléen de même longueur, utilisable tel quel pour
    filtrer un DataFrame.
    """
    return [d < aujourd_hui for d in dates]


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


# ============================================================
# LE PIPELINE NOCTURNE COMME DONNÉE (v1.2)
# ============================================================
# Jusqu'ici l'ordre du pipeline vivait dans six `add_job` à heure fixe, et
# la documentation le *décrivait*. Les deux ont divergé — comme la liste des
# couplages avant le lot #32. Ici l'ordre devient une donnée que main.py
# LIT pour construire l'enchaînement : la description ne peut plus mentir,
# puisqu'il n'y a plus de description.
#
# `depend_de` énonce les dépendances de DONNÉES, relues dans le code le
# 2026-09-09 (rév. ad7b8d7), pas l'ordre historique des horaires :
#
#   sync_prix        → actions_prix_historique
#   sync_etf         → secteurs_etf_prix ET indices_prix. Ne lit jamais
#                      actions_prix_historique : indépendant des prix.
#                      ⚠️ C'est lui qui alimente le régime MACRO, pas
#                      sync_prix — contre-intuitif, vérifié sync.py:496.
#   sync_metadata    → tickers_info ; lit la liste des tickers depuis
#                      actions_prix_historique (sync.py:278), d'où sa
#                      dépendance à sync_prix.
#   compute_ranking  → a besoin des prix, de la force sectorielle, de la
#                      macro et du mapping secteur : les trois amonts.
#   suivi_rendements → prix uniquement.
#   analyse          → prix uniquement, et PERSONNE n'en dépend : les trois
#                      chemins de décision recalculent leurs indicateurs au
#                      lieu de lire les colonnes qu'elle écrit (cf. CLAUDE.md,
#                      « Où vivent réellement les indicateurs de décision »).
#                      Elle passe donc en DERNIER : la faire précéder le
#                      ranking, c'était retarder une décision derrière un
#                      calcul qui n'alimente que le ML et le backtest.
#
# sync_fx garde son propre horaire (00h55 UTC) : aucune dépendance, aucun
# dépendant dans le pipeline. L'y intégrer n'apporterait rien.


class EtapePipeline(NamedTuple):
    """Une étape du pipeline nocturne et ses dépendances de données."""

    id: str
    depend_de: tuple = ()


PIPELINE_NOCTURNE = (
    EtapePipeline("sync_prix"),
    EtapePipeline("sync_etf"),
    EtapePipeline("sync_metadata", ("sync_prix",)),
    EtapePipeline("compute_ranking", ("sync_prix", "sync_etf", "sync_metadata")),
    EtapePipeline("suivi_rendements", ("sync_prix",)),
    EtapePipeline("analyse", ("sync_prix",)),
)


def ordre_est_coherent(pipeline=PIPELINE_NOCTURNE) -> tuple[bool, str]:
    """
    L'ordre déclaré respecte-t-il les dépendances ?

    L'exécution est séquentielle : chaque étape ne voit que le statut de
    celles qui l'ont précédée. Une étape dont une dépendance est déclarée
    APRÈS elle serait donc systématiquement sautée — un pipeline qui ne
    produit plus rien, chaque nuit, sans erreur. Ce test doit tomber au
    moment où quelqu'un réordonne le tuple, pas trois nuits plus tard.

    Retourne (True, "") si l'ordre est cohérent, (False, motif) sinon.
    """
    vues = set()
    connues = {etape.id for etape in pipeline}

    for etape in pipeline:
        for dependance in etape.depend_de:
            if dependance not in connues:
                return False, (
                    f"'{etape.id}' dépend de '{dependance}', qui n'est pas "
                    f"une étape du pipeline"
                )
            if dependance not in vues:
                return False, (
                    f"'{etape.id}' dépend de '{dependance}', déclarée APRÈS "
                    f"elle : l'étape serait sautée à chaque exécution"
                )
        vues.add(etape.id)

    return True, ""


def peut_demarrer(etape: EtapePipeline, job_status: dict | None) -> tuple[bool, str]:
    """
    Les dépendances de cette étape se sont-elles toutes terminées en `ok` ?

    Remplace le pari sur les horaires par une vérification d'état. Le cas
    que cette fonction existe pour empêcher : `sync_prix` échoue à 01h00,
    et `compute_ranking` se déclenche quand même à 02h15 sur des prix
    périmés — en se terminant en `ok`. Un ranking faux et silencieux est
    pire qu'un ranking absent : le second se rattrape (`backfill_ranking`,
    lot #31), le premier se croit.

    Une étape sans dépendance démarre toujours.

    Retourne (True, "") s'il faut exécuter, (False, motif) sinon. Ne lève
    jamais : un `job_status` absent ou malformé ne doit pas faire tomber le
    pipeline entier.
    """
    if not etape.depend_de:
        return True, ""

    if not job_status or not isinstance(job_status, dict):
        return False, "job_status indisponible"

    for dependance in etape.depend_de:
        etat = job_status.get(dependance)
        if not isinstance(etat, dict):
            return False, f"{dependance} n'a pas de statut"

        statut = etat.get("status")
        if statut != "ok":
            return False, f"{dependance} n'a pas abouti (statut={statut!r})"

    return True, ""
