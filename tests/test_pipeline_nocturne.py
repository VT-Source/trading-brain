# ============================================================
# tests/test_pipeline_nocturne.py — Trading Brain
# VT-Source/trading-brain
# ============================================================
# Verrouille l'ordonnancement nocturne déclaré dans scheduling.py (v1.2).
#
# Ce que ces tests protègent, concrètement : la nuit du 2026-09-07, sync_prix
# est passée de ~5 à ~33 min et l'analyse, planifiée 30 min après, a lu la
# table pendant que la sync y écrivait — 28 tickers sans indicateurs, job en
# `ok`. Une panne muette. Le chaînage supprime la cause ; ces tests
# empêchent qu'un réordonnancement la réintroduise sans bruit.
#
# Aucune DB, aucun réseau, aucun import de main.py : scheduling.py est pur.
# ============================================================

import pytest

from scheduling import (
    PIPELINE_NOCTURNE,
    EtapePipeline,
    ordre_est_coherent,
    peut_demarrer,
)

OK = {"status": "ok"}


def _tout_ok(*etapes):
    return {etape: dict(OK) for etape in etapes}


# ------------------------------------------------------------
# Le graphe déclaré
# ------------------------------------------------------------
def test_ordre_declare_est_coherent():
    """
    Toute dépendance est déclarée avant l'étape qui l'utilise.

    L'exécution est séquentielle : une étape ne voit que le statut de celles
    qui l'ont précédée. Une dépendance déclarée trop tard produirait un
    `skipped` chaque nuit, sans erreur — le pipeline s'arrêterait de produire
    en silence. C'est le mode de panne le plus coûteux du projet.
    """
    coherent, motif = ordre_est_coherent()
    assert coherent, motif


def test_ordre_incoherent_est_detecte():
    """Le contrôle rougit sur une dépendance déclarée après son étape."""
    a_lenvers = (
        EtapePipeline("compute_ranking", ("sync_prix",)),
        EtapePipeline("sync_prix"),
    )
    coherent, motif = ordre_est_coherent(a_lenvers)
    assert not coherent
    assert "sync_prix" in motif and "APRÈS" in motif


def test_dependance_inconnue_est_detectee():
    """Une dépendance qui n'est pas une étape du pipeline est une faute."""
    coherent, motif = ordre_est_coherent(
        (EtapePipeline("compute_ranking", ("sync_inexistant",)),)
    )
    assert not coherent
    assert "sync_inexistant" in motif


def test_analyse_est_la_derniere_etape():
    """
    `analyse` ferme la marche, et c'est un choix, pas un hasard.

    Personne ne dépend d'elle : les trois chemins de décision rechargent et
    recalculent leurs indicateurs au lieu de lire les colonnes qu'elle écrit
    (CLAUDE.md, « Où vivent réellement les indicateurs de décision »). La
    remonter avant compute_ranking, c'est remettre une décision de trade
    derrière un calcul qui n'alimente que le ML, /backtest-detail et
    /schema-diagnostic. Si un jour une étape dépend de `analyse`, ce test
    doit tomber pour forcer l'arbitrage.
    """
    assert PIPELINE_NOCTURNE[-1].id == "analyse"

    dependants = [
        etape.id for etape in PIPELINE_NOCTURNE if "analyse" in etape.depend_de
    ]
    assert not dependants, (
        f"{dependants} dépend(ent) désormais de `analyse` : sa place en fin "
        f"de pipeline doit être rearbitrée, pas conservée par habitude."
    )


def test_ranking_depend_bien_des_trois_amonts():
    """
    compute_ranking a besoin des prix, des ETF et des metadata.

    Le piège est `sync_etf` : il n'alimente pas seulement la force
    sectorielle mais aussi `indices_prix`, donc le régime MACRO (sync.py:496).
    Un ranking calculé sans lui perd deux entrées du score, et
    get_secteurs_en_force renvoie [] sans le dire.
    """
    ranking = next(e for e in PIPELINE_NOCTURNE if e.id == "compute_ranking")
    assert set(ranking.depend_de) == {"sync_prix", "sync_etf", "sync_metadata"}


def test_sync_etf_est_independant_des_prix():
    """
    sync_etf ne lit jamais actions_prix_historique (vérifié sync.py:433-688).

    Le déclarer dépendant de sync_prix serait une contrainte inventée : elle
    le ferait sauter inutilement les nuits où la sync des prix échoue, et
    priverait le moteur de sa macro pour rien.
    """
    etf = next(e for e in PIPELINE_NOCTURNE if e.id == "sync_etf")
    assert etf.depend_de == ()


# ------------------------------------------------------------
# Le garde
# ------------------------------------------------------------
def test_etape_sans_dependance_demarre_toujours():
    demarrer, motif = peut_demarrer(EtapePipeline("sync_prix"), {})
    assert demarrer and motif == ""


def test_etape_sans_dependance_demarre_meme_sans_job_status():
    """Un job_status absent ne doit pas bloquer une étape de tête."""
    for status in (None, {}, "cassé", 42):
        demarrer, _ = peut_demarrer(EtapePipeline("sync_prix"), status)
        assert demarrer


def test_toutes_dependances_ok_demarre():
    ranking = next(e for e in PIPELINE_NOCTURNE if e.id == "compute_ranking")
    demarrer, motif = peut_demarrer(
        ranking, _tout_ok("sync_prix", "sync_etf", "sync_metadata")
    )
    assert demarrer and motif == ""


@pytest.mark.parametrize("statut", ["error", "skipped", "running", None])
def test_dependance_non_aboutie_bloque(statut):
    """
    Le cas que tout ce lot existe pour empêcher.

    Avant le chaînage, sync_prix pouvait échouer à 01h00 et compute_ranking
    se déclencher quand même à 02h15, sur des prix périmés, en se terminant
    en `ok`. Un ranking absent se rattrape (backfill_ranking, #31) ; un
    ranking faux se croit.
    """
    ranking = next(e for e in PIPELINE_NOCTURNE if e.id == "compute_ranking")
    etat = _tout_ok("sync_prix", "sync_etf", "sync_metadata")
    etat["sync_etf"] = {"status": statut}

    demarrer, motif = peut_demarrer(ranking, etat)
    assert not demarrer
    assert "sync_etf" in motif, "le motif doit NOMMER la dépendance fautive"


def test_dependance_absente_bloque():
    """Une dépendance qui n'a jamais tourné bloque aussi — pas de bénéfice du doute."""
    ranking = next(e for e in PIPELINE_NOCTURNE if e.id == "compute_ranking")
    demarrer, motif = peut_demarrer(ranking, _tout_ok("sync_prix", "sync_etf"))
    assert not demarrer
    assert "sync_metadata" in motif


def test_job_status_malforme_bloque_une_etape_dependante():
    """
    Une étape à dépendances renonce si l'état est illisible.

    Asymétrie voulue avec les étapes de tête : ne rien savoir de l'amont
    n'autorise pas à supposer qu'il a réussi.
    """
    ranking = next(e for e in PIPELINE_NOCTURNE if e.id == "compute_ranking")
    for status in (None, {}, "cassé"):
        demarrer, motif = peut_demarrer(ranking, status)
        assert not demarrer and motif


def test_peut_demarrer_ne_leve_jamais():
    """
    Aucune forme de job_status ne doit faire tomber le pipeline entier.

    Le garde protège une étape ; s'il levait, il emporterait toutes les
    suivantes — exactement ce qu'il est censé éviter.
    """
    ranking = next(e for e in PIPELINE_NOCTURNE if e.id == "compute_ranking")
    for status in (None, {}, [], "x", 0, {"sync_prix": None}, {"sync_prix": []},
                   {"sync_prix": {"status": ["ok"]}}):
        demarrer, motif = peut_demarrer(ranking, status)
        assert isinstance(demarrer, bool) and isinstance(motif, str)


def test_echec_de_tete_saute_tout_le_dependant():
    """
    Simulation d'une nuit où sync_prix échoue.

    Attendu : sync_etf tourne quand même (indépendant), tout le reste est
    sauté explicitement. Aujourd'hui, dans le même cas, compute_ranking
    tournerait à 02h15 et se terminerait en `ok`.
    """
    job_status = {}
    executees, sautees = [], []

    for etape in PIPELINE_NOCTURNE:
        demarrer, motif = peut_demarrer(etape, job_status)
        if not demarrer:
            job_status[etape.id] = {"status": "skipped", "reason": motif}
            sautees.append(etape.id)
            continue
        executees.append(etape.id)
        job_status[etape.id] = {"status": "error"} if etape.id == "sync_prix" else dict(OK)

    assert executees == ["sync_prix", "sync_etf"]
    assert sautees == ["sync_metadata", "compute_ranking", "suivi_rendements", "analyse"]
    assert "sync_prix" in job_status["compute_ranking"]["reason"]


def test_nuit_nominale_execute_tout():
    """Toutes les étapes passent quand chacune aboutit."""
    job_status = {}
    executees = []

    for etape in PIPELINE_NOCTURNE:
        demarrer, _ = peut_demarrer(etape, job_status)
        assert demarrer, f"{etape.id} sauté alors que tout l'amont est ok"
        executees.append(etape.id)
        job_status[etape.id] = dict(OK)

    assert executees == [etape.id for etape in PIPELINE_NOCTURNE]


# ------------------------------------------------------------
# Le câblage main.py ↔ scheduling.py
# ------------------------------------------------------------
def test_main_expose_un_executable_par_etape():
    """
    Chaque étape déclarée a un exécutable dans `_pipeline_nocturne`.

    Sans ce contrôle, ajouter une étape à PIPELINE_NOCTURNE sans câbler sa
    fonction lève un KeyError à 01h00 UTC — et, comme _run_job n'entoure pas
    la boucle, la nuit entière s'arrête là. Le genre de panne qu'on découvre
    au matin, sur un ranking manquant.

    Lu avec `ast` : importer main.py exigerait une DB, FastAPI et le réseau,
    ce qui rendrait ce test impossible à faire tourner ici.
    """
    import ast
    import pathlib

    source = (pathlib.Path(__file__).resolve().parent.parent / "main.py").read_text(
        encoding="utf-8"
    )

    fonction = next(
        (n for n in ast.walk(ast.parse(source))
         if isinstance(n, ast.FunctionDef) and n.name == "_pipeline_nocturne"),
        None,
    )
    assert fonction is not None, "main.py n'expose plus _pipeline_nocturne"

    dictionnaire = next(
        (n.value for n in fonction.body
         if isinstance(n, ast.Assign)
         and isinstance(n.value, ast.Dict)
         and any(getattr(c, "id", None) == "executables" for c in n.targets)),
        None,
    )
    assert dictionnaire is not None, (
        "_pipeline_nocturne n'assigne plus un dict `executables` — le "
        "contrôle de câblage ne peut plus s'appliquer, l'adapter."
    )

    cables = {cle.value for cle in dictionnaire.keys if isinstance(cle, ast.Constant)}
    declarees = {etape.id for etape in PIPELINE_NOCTURNE}

    assert declarees - cables == set(), (
        f"Étapes déclarées sans exécutable : {sorted(declarees - cables)} — "
        f"KeyError garanti à la prochaine exécution nocturne."
    )
    assert cables - declarees == set(), (
        f"Exécutables câblés hors du pipeline : {sorted(cables - declarees)} — "
        f"code mort, ou étape oubliée dans PIPELINE_NOCTURNE."
    )
