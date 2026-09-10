# ============================================================
# tests/test_couplages.py — Cohérence doc ↔ code — Trading Brain
# VT-Source/trading-brain
# ============================================================
# Ce fichier est la SOURCE de la liste des déploiements couplés. La section
# « Déploiements couplés » de CLAUDE.md en est une copie lisible ; en cas de
# divergence, c'est ici qui fait foi, parce qu'ici seul un test peut échouer.
#
# Pourquoi ce test existe : la liste de CLAUDE.md est restée à trois modules
# pendant que le code en comptait sept (constat du 2026-09-09). Elle était
# pourtant relue à chaque session. La discipline n'a pas manqué — elle ne
# suffit pas. Une liste que rien ne vérifie diverge en silence, et un couplage
# dur oublié, c'est une API qui ne redémarre pas après un déploiement partiel.
#
# Ce module n'importe RIEN du projet : il lit les sources avec `ast`. Il ne
# peut donc pas être mis en échec par une dépendance manquante, une DB
# absente ou un import circulaire — ce qui est précisément le genre de panne
# qu'il surveille.
# ============================================================

import ast
import pathlib
import sys

import pytest

RACINE = pathlib.Path(__file__).resolve().parent.parent


# ------------------------------------------------------------
# Le graphe déclaré — à mettre à jour EN MÊME TEMPS que le code
# ------------------------------------------------------------
# "dur"    : import au niveau module, NON gardé. Le module manquant ou
#            incompatible fait crasher au démarrage. Un seul commit atomique.
# "souple" : import sous try/except. L'absence dégrade une fonctionnalité
#            sans tuer l'application.
#
# ⚠️ `backtest_ranking` est déclaré souple pour `main` — c'est ce que dit
# l'import. En pratique il porte les TROIS chemins de décision : sans lui,
# evaluate_open_positions renvoie une erreur et le moteur ne décide plus rien.
# Souple à l'import, dur à l'usage. Aucun test ne peut attraper cette
# nuance-là ; elle est écrite ici pour qu'elle ne se perde pas.
#
# ⚠️ `scheduling` est non gardé VOLONTAIREMENT (main.py, ranking du
# 2026-09-07) : un garde-fou de sécurité qui disparaît en silence est pire
# que son absence. Le passer en souple serait une régression, pas une
# robustification.
COUPLAGES = {
    "main": {
        "dur": {
            "ai_opinion",
            "analysis",
            "backfill_api",
            "models_api",
            "ranking",
            "scheduling",
            "sync",
        },
        "souple": {"alerting", "backtest", "backtest_ranking", "train_model"},
    },
    "analysis": {"dur": {"scheduling"}, "souple": set()},
    "backfill_api": {"dur": {"ranking"}, "souple": set()},
    "ranking": {"dur": {"freshness"}, "souple": {"alerting", "backtest_ranking"}},
    "sync": {"dur": {"scheduling"}, "souple": {"alerting"}},
}

# Modules garantis sans dépendance TIERCE ni PROJET : bibliothèque standard
# uniquement. C'est ce qui les rend testables sans DB, sans réseau et sans
# FastAPI — et ce qui autorise `main.py` à importer `scheduling` sans garde,
# puisque la stdlib est présente dans tout déploiement Python et ne peut
# donc pas manquer.
#
# La règle a été DESSERRÉE au lot #33 (elle exigeait zéro import) pour que
# scheduling.py puisse déclarer PIPELINE_NOCTURNE avec typing.NamedTuple.
# Desserrer un test pour faire passer son propre code est un geste à
# surveiller : ici il est délibéré, et la propriété qui compte — « l'import
# ne peut pas échouer » — est préservée telle quelle. Une dépendance tierce
# (pandas, sqlalchemy, fastapi) reste interdite et fait toujours rougir.
MODULES_AUTONOMES = {"scheduling", "freshness"}

# `dashboard.py` ne parle à `main.py` qu'en HTTP. Couplage de CONTRAT :
# changer la forme d'une réponse d'API ne lève aucune erreur d'import, la
# page affiche faux. Ce test verrouille l'absence d'import ; il ne dit
# évidemment rien de la forme des réponses, qui reste non protégée.
MODULES_SANS_IMPORT_LOCAL = {"dashboard"}


# ------------------------------------------------------------
# Analyse statique
# ------------------------------------------------------------
def modules_locaux() -> set[str]:
    """Tout fichier .py à la racine du repo est un module local."""
    return {p.stem for p in RACINE.glob("*.py")}


def _noms_importes(noeud) -> set[str]:
    """Noms de modules racines importés par un noeud Import / ImportFrom."""
    if isinstance(noeud, ast.ImportFrom):
        # `from . import x` (level > 0) n'existe pas ici : projet à plat.
        if noeud.module and noeud.level == 0:
            return {noeud.module.split(".")[0]}
        return set()
    if isinstance(noeud, ast.Import):
        return {alias.name.split(".")[0] for alias in noeud.names}
    return set()


def imports_locaux(module: str) -> tuple[set[str], set[str]]:
    """
    Retourne (durs, souples) : les imports de modules LOCAUX au niveau
    module, séparés selon qu'ils sont ou non protégés par un try/except.

    Seuls les imports de premier niveau comptent. Un import à l'intérieur
    d'une fonction (`from ai_opinion import generate_position_opinion` dans
    un endpoint, par exemple) est différé : il ne peut pas faire échouer le
    démarrage, donc il ne crée pas de couplage de déploiement.
    """
    source = (RACINE / f"{module}.py").read_text(encoding="utf-8")
    arbre = ast.parse(source)
    locaux = modules_locaux()
    durs, souples = set(), set()

    for instruction in arbre.body:
        if isinstance(instruction, (ast.Import, ast.ImportFrom)):
            durs |= _noms_importes(instruction) & locaux
        elif isinstance(instruction, ast.Try):
            for interne in instruction.body:
                souples |= _noms_importes(interne) & locaux

    return durs - {module}, souples - {module}


def tous_imports(module: str) -> set[str]:
    """Tous les modules importés au niveau module, locaux ou non."""
    source = (RACINE / f"{module}.py").read_text(encoding="utf-8")
    arbre = ast.parse(source)
    noms = set()
    for instruction in arbre.body:
        if isinstance(instruction, (ast.Import, ast.ImportFrom)):
            noms |= _noms_importes(instruction)
        elif isinstance(instruction, ast.Try):
            for interne in instruction.body:
                noms |= _noms_importes(interne)
    return noms


# ------------------------------------------------------------
# Tests
# ------------------------------------------------------------
@pytest.mark.parametrize("module", sorted(COUPLAGES))
def test_couplages_declares_conformes(module):
    """Le graphe déclaré correspond exactement aux imports réels."""
    durs, souples = imports_locaux(module)
    attendu = COUPLAGES[module]

    assert durs == attendu["dur"], (
        f"{module}.py : couplage DUR divergent.\n"
        f"  déclaré : {sorted(attendu['dur'])}\n"
        f"  réel    : {sorted(durs)}\n"
        f"Mettre à jour COUPLAGES ici ET la section « Déploiements couplés » "
        f"de CLAUDE.md — un import non gardé oublié, c'est une API qui ne "
        f"redémarre pas après un déploiement partiel."
    )
    assert souples == attendu["souple"], (
        f"{module}.py : couplage SOUPLE divergent.\n"
        f"  déclaré : {sorted(attendu['souple'])}\n"
        f"  réel    : {sorted(souples)}\n"
        f"Un module passé de gardé à non gardé change son mode de panne : "
        f"vérifier que c'est voulu avant de mettre à jour la déclaration."
    )


def test_aucun_couplage_non_declare():
    """
    Aucun module du projet n'a de couplage local hors du graphe déclaré.

    C'est ce test qui attrape le cas réellement vécu : un nouveau module
    extrait de main.py, importé sans garde, et une documentation qui reste
    à la version précédente.
    """
    oublies = {}
    for module in sorted(modules_locaux()):
        if module in COUPLAGES:
            continue
        durs, souples = imports_locaux(module)
        if durs or souples:
            oublies[module] = {"dur": sorted(durs), "souple": sorted(souples)}

    assert not oublies, (
        f"Modules avec un couplage local absent de COUPLAGES : {oublies}\n"
        f"Les ajouter au graphe déclaré ici et dans CLAUDE.md."
    )


@pytest.mark.parametrize("module", sorted(MODULES_AUTONOMES))
def test_modules_autonomes_sans_dependance(module):
    """
    `scheduling` et `freshness` n'importent que la bibliothèque standard.

    C'est la propriété qui les rend testables sans DB, sans réseau et sans
    FastAPI. Le jour où l'un d'eux importe une dépendance tierce, la raison
    même de son extraction disparaît — et, pour `scheduling`, la
    justification de son import non gardé dans main.py avec : un module qui
    peut échouer à l'import ne peut pas servir de garde-fou.
    """
    hors_stdlib = {
        nom for nom in tous_imports(module)
        if nom not in sys.stdlib_module_names
    }
    assert not hors_stdlib, (
        f"{module}.py ne doit dépendre que de la bibliothèque standard, or "
        f"il importe {sorted(hors_stdlib)}. Si l'ajout est délibéré, c'est "
        f"une décision d'architecture à arbitrer, pas un détail "
        f"d'implémentation."
    )


@pytest.mark.parametrize("module", sorted(MODULES_SANS_IMPORT_LOCAL))
def test_couplage_de_contrat_reste_sans_import(module):
    """
    `dashboard.py` ne doit importer aucun module du projet.

    Il communique avec l'API en HTTP. Y ajouter un import ferait passer un
    couplage de contrat (silencieux, visible à l'écran) en couplage
    d'import (bruyant, visible au démarrage) : ce n'est pas forcément une
    mauvaise idée, mais c'est un changement d'architecture qui doit être
    décidé, pas subi.
    """
    durs, souples = imports_locaux(module)
    assert not (durs | souples), (
        f"{module}.py importe désormais {sorted(durs | souples)} depuis le "
        f"projet. Arbitrer : soit l'import est voulu et le graphe doit "
        f"l'accueillir, soit le dashboard doit rester en HTTP pur."
    )
