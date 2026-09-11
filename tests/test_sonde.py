# ============================================================
# tests/test_sonde.py — Sonde de publication Yahoo (#35) — Trading Brain
# VT-Source/trading-brain
# ============================================================
# Sans DB ni réseau : le téléchargement, l'engine et le déclencheur
# APScheduler sont injectés. Si sqlalchemy est absent de l'environnement de
# test, un substitut minimal de `text` est posé le temps du test.
#
# Ce que ces tests NE couvrent PAS : la forme réelle d'une réponse yfinance
# du jour (seulement les formes connues, reproduites à la main), la
# création de la table sur PostgreSQL, et le déclenchement horaire réel.
# La vérification réelle est le premier passage en prod.
# ============================================================

import ast
import pathlib
import re
import sys
import types
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

import sonde
from freshness import place_de_cotation
from scheduling import PIPELINE_NOCTURNE

RACINE = pathlib.Path(__file__).resolve().parent.parent

# Jour UTC des passages simulés : la séance du 11/09 est « en cours ».
AUJOURDHUI = date(2026, 9, 11)


# ------------------------------------------------------------
# Paramètres : la sonde doit voir ce que voit sync_prix
# ------------------------------------------------------------
def _fonction(arbre, nom):
    for noeud in ast.walk(arbre):
        if isinstance(noeud, ast.FunctionDef) and noeud.name == nom:
            return noeud
    return None


def test_parametres_identiques_a_sync_prix():
    """
    Mêmes arguments de yf.download que sync_prix_logic en mode quotidien.

    Si la sync change un paramètre (plage, ajustement, timeout), la sonde
    mesurerait autre chose que ce que la sync reçoit : ce test tombe.
    """
    arbre = ast.parse((RACINE / "sync.py").read_text(encoding="utf-8"))
    fonction = _fonction(arbre, "sync_prix_logic")
    assert fonction is not None, "sync.py n'expose plus sync_prix_logic"

    appels = [
        n for n in ast.walk(fonction)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute) and n.func.attr == "download"
    ]
    assert len(appels) == 1, "sync_prix_logic devrait contenir un seul yf.download"
    mots_cles = {k.arg: k.value for k in appels[0].keywords}

    assert set(mots_cles) == set(sonde.PARAMETRES_TELECHARGEMENT), (
        f"Arguments divergents — sync : {sorted(mots_cles)}, "
        f"sonde : {sorted(sonde.PARAMETRES_TELECHARGEMENT)}"
    )
    for nom, valeur in sonde.PARAMETRES_TELECHARGEMENT.items():
        if nom == "period":
            continue
        assert isinstance(mots_cles[nom], ast.Constant), f"{nom} n'est plus une constante"
        assert mots_cles[nom].value == valeur, f"{nom} : sync={mots_cles[nom].value!r}, sonde={valeur!r}"

    # period = period_override or ("5y" if full else "1mo")
    affectations = [
        n for n in ast.walk(fonction)
        if isinstance(n, ast.Assign)
        and any(isinstance(c, ast.Name) and c.id == "period" for c in n.targets)
    ]
    assert len(affectations) == 1, "sync_prix_logic devrait affecter `period` une seule fois"
    periodes_quotidiennes = [
        n.orelse.value for n in ast.walk(affectations[0].value)
        if isinstance(n, ast.IfExp) and isinstance(n.orelse, ast.Constant)
    ]
    assert periodes_quotidiennes == [sonde.PARAMETRES_TELECHARGEMENT["period"]]


# ------------------------------------------------------------
# sonde_peut_tourner
# ------------------------------------------------------------
@pytest.mark.parametrize("job_status", [None, {}, "pas un dict", {"sync_prix": "ok"}])
def test_etat_absent_ou_mal_forme_laisse_passer(job_status):
    assert sonde.sonde_peut_tourner(job_status) == (True, "")


@pytest.mark.parametrize("etape", [e.id for e in PIPELINE_NOCTURNE])
def test_une_etape_du_pipeline_en_cours_bloque(etape):
    peut, motif = sonde.sonde_peut_tourner({etape: {"status": "running"}})
    assert peut is False
    assert etape in motif


@pytest.mark.parametrize("statut", ["ok", "skipped", "error"])
def test_pipeline_termine_laisse_passer(statut):
    etat = {e.id: {"status": statut} for e in PIPELINE_NOCTURNE}
    assert sonde.sonde_peut_tourner(etat)[0] is True


def test_un_job_hors_pipeline_en_cours_ne_bloque_pas():
    assert sonde.sonde_peut_tourner({"poll_ai_opinions": {"status": "running"}})[0] is True


# ------------------------------------------------------------
# resumer_reponse
# ------------------------------------------------------------
def _reponse(dates, clotures, volumes=None, multi_index=False, ticker="AAPL"):
    volumes = volumes if volumes is not None else [1_000_000.0] * len(dates)
    df = pd.DataFrame(
        {"Open": clotures, "High": clotures, "Low": clotures,
         "Close": clotures, "Volume": volumes},
        index=pd.DatetimeIndex(pd.to_datetime(dates), name="Date"),
    )
    if multi_index:
        # Forme de yf.download récent : colonnes (Price, Ticker)
        df.columns = pd.MultiIndex.from_tuples([(c, ticker) for c in df.columns],
                                               names=["Price", "Ticker"])
    return df


@pytest.mark.parametrize("df", [None, pd.DataFrame()])
def test_reponse_vide(df):
    r = sonde.resumer_reponse(df, AUJOURDHUI)
    assert r["erreur"] == "réponse vide"
    assert r["nb_barres"] == 0
    assert r["derniere_date_brute"] is None and r["derniere_date_valide"] is None
    assert r["derniere_date_close"] is None


@pytest.mark.parametrize("multi_index", [False, True])
def test_reponse_nominale(multi_index):
    df = _reponse(["2026-09-08", "2026-09-09", "2026-09-10"], [10.0, 11.0, 12.5],
                  volumes=[100.0, 200.0, 300.0], multi_index=multi_index)
    r = sonde.resumer_reponse(df, AUJOURDHUI)
    assert r == {
        "derniere_date_brute": date(2026, 9, 10),
        "derniere_date_valide": date(2026, 9, 10),
        "derniere_date_close": date(2026, 9, 10),
        "derniere_cloture": 12.5,
        "dernier_volume": 300.0,
        "nb_barres": 3,
        "erreur": None,
    }
    assert type(r["derniere_cloture"]) is float and type(r["nb_barres"]) is int


def test_derniere_cloture_nan_separe_brute_et_valide():
    """Le cas que le dropna de sync_prix rendrait invisible."""
    df = _reponse(["2026-09-08", "2026-09-09", "2026-09-10"], [10.0, 11.0, np.nan])
    r = sonde.resumer_reponse(df, AUJOURDHUI)
    assert r["derniere_date_brute"] == date(2026, 9, 10)
    assert r["derniere_date_valide"] == date(2026, 9, 9)
    assert r["derniere_date_close"] == date(2026, 9, 9)
    assert r["derniere_cloture"] is None
    assert r["erreur"] is None


def test_aucune_cloture_valide():
    r = sonde.resumer_reponse(_reponse(["2026-09-09", "2026-09-10"], [np.nan, np.nan]), AUJOURDHUI)
    assert r["derniere_date_brute"] == date(2026, 9, 10)
    assert r["derniere_date_valide"] is None
    assert r["derniere_date_close"] is None


def test_colonnes_inattendues():
    df = pd.DataFrame({"Prix": [1.0]},
                      index=pd.DatetimeIndex(pd.to_datetime(["2026-09-10"]), name="Date"))
    r = sonde.resumer_reponse(df, AUJOURDHUI)
    assert r["erreur"].startswith("colonnes inattendues")
    assert r["nb_barres"] == 0


# ------------------------------------------------------------
# Séance close (v1.1, #35b) — ce que sync_prix aurait stocké
# ------------------------------------------------------------
def test_barre_du_jour_exclue_de_la_date_close():
    """
    Place ouverte pendant le passage : yfinance renvoie la séance en cours.
    La date brute et la date valide la montrent ; la date close, non.
    """
    df = _reponse(["2026-09-09", "2026-09-10", "2026-09-11"], [10.0, 11.0, 11.4],
                  volumes=[1e6, 1e6, 1.4e5])
    r = sonde.resumer_reponse(df, AUJOURDHUI)
    assert r["derniere_date_brute"] == date(2026, 9, 11)
    assert r["derniere_date_valide"] == date(2026, 9, 11)
    assert r["derniere_date_close"] == date(2026, 9, 10)
    assert r["dernier_volume"] == 1.4e5   # la barre en cours reste reconnaissable


def test_barre_en_cours_ne_masque_pas_une_seance_absente():
    """
    Le défaut de la v1.0 : séance du 10/09 non publiée, barre en cours du
    11/09 présente. La dernière ligne disait « à jour » ; la sync, elle,
    n'aurait eu que le 09/09.
    """
    df = _reponse(["2026-09-08", "2026-09-09", "2026-09-11"], [10.0, 11.0, 11.4])
    r = sonde.resumer_reponse(df, AUJOURDHUI)
    assert r["derniere_date_valide"] == date(2026, 9, 11)
    assert r["derniere_date_close"] == date(2026, 9, 9)


def test_regle_de_seance_close_unique():
    """
    La règle « séance close » ne s'écrit qu'une fois : sonde et sync_prix
    appellent la même fonction de scheduling.py.
    """
    import scheduling
    assert sonde.masque_barres_closes is scheduling.masque_barres_closes

    arbre = ast.parse((RACINE / "sonde.py").read_text(encoding="utf-8"))
    appels = {
        n.func.id for n in ast.walk(_fonction(arbre, "resumer_reponse"))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "masque_barres_closes" in appels


# ------------------------------------------------------------
# Instruments
# ------------------------------------------------------------
def test_instruments_uniques_et_types_connus():
    tickers = [i.ticker for i in sonde.INSTRUMENTS_SONDE]
    assert len(tickers) == len(set(tickers))
    assert {i.type_instrument for i in sonde.INSTRUMENTS_SONDE} <= {"indice", "etf", "action"}


@pytest.mark.parametrize("instrument",
                         [i for i in sonde.INSTRUMENTS_SONDE if i.type_instrument != "indice"],
                         ids=lambda i: i.ticker)
def test_place_conforme_a_freshness(instrument):
    """Une seule définition de la place : celle de freshness.py."""
    assert instrument.place == place_de_cotation(instrument.ticker)


def test_tailles_compatibles_avec_la_table():
    for i in sonde.INSTRUMENTS_SONDE:
        assert len(i.ticker) <= 20 and len(i.place) <= 40 and len(i.type_instrument) <= 10


# ------------------------------------------------------------
# executer_sonde — engine et téléchargement simulés
# ------------------------------------------------------------
class _ConnexionFactice:
    def __init__(self, journal, echec):
        self.journal, self.echec = journal, echec

    def __enter__(self):
        if self.echec:
            raise RuntimeError("DB indisponible")
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, instruction, parametres=None):
        self.journal.append((str(instruction), parametres))


class _EngineFactice:
    def __init__(self, echec=False):
        self.journal, self.echec = [], echec

    def begin(self):
        return _ConnexionFactice(self.journal, self.echec)


@pytest.fixture
def sqlalchemy_disponible(monkeypatch):
    """Substitut minimal de sqlalchemy.text si la bibliothèque est absente."""
    try:
        import sqlalchemy  # noqa: F401
    except ImportError:
        faux = types.ModuleType("sqlalchemy")
        faux.text = lambda sql: sql
        monkeypatch.setitem(sys.modules, "sqlalchemy", faux)


MAINTENANT = datetime(2026, 9, 11, 13, 20, tzinfo=timezone.utc)


def _telecharger_ok(ticker):
    return _reponse(["2026-09-09", "2026-09-10"], [1.0, 2.0], ticker=ticker)


def test_passage_saute_pendant_le_pipeline(sqlalchemy_disponible):
    engine, appels = _EngineFactice(), []
    res = sonde.executer_sonde(engine, {"sync_prix": {"status": "running"}},
                               telecharger=lambda t: appels.append(t),
                               maintenant=MAINTENANT, pause_s=0)
    assert res["status"] == "skipped"
    assert appels == [] and engine.journal == []


def test_engine_absent():
    assert sonde.executer_sonde(None, {}, pause_s=0)["status"] == "skipped"


def test_passage_nominal_une_ligne_par_instrument(sqlalchemy_disponible):
    engine = _EngineFactice()
    res = sonde.executer_sonde(engine, {}, telecharger=_telecharger_ok,
                               maintenant=MAINTENANT, pause_s=0)
    assert res == {"status": "ok", "nb_lignes": len(sonde.INSTRUMENTS_SONDE), "nb_erreurs": 0}

    (ddl, _), (insert, lignes) = engine.journal
    assert "CREATE TABLE IF NOT EXISTS sonde_publication" in ddl
    assert "INSERT INTO sonde_publication" in insert
    assert [l["ticker"] for l in lignes] == [i.ticker for i in sonde.INSTRUMENTS_SONDE]

    # Chaque paramètre nommé de l'INSERT est fourni, et rien de plus.
    attendus = set(re.findall(r":(\w+)", sonde.INSERT_SONDE))
    for ligne in lignes:
        assert set(ligne) == attendus
        assert ligne["sonde_at"] == MAINTENANT
        assert ligne["derniere_date_valide"] == date(2026, 9, 10)
        assert ligne["derniere_date_close"] == date(2026, 9, 10)

    # La colonne existe dans le DDL, sinon l'INSERT échouerait en prod.
    for colonne in attendus:
        assert colonne in sonde.DDL_SONDE


def test_passage_pendant_une_seance_ouverte(sqlalchemy_disponible):
    """Barre en cours du 11/09 renvoyée à 13h20 UTC le 11/09 : exclue."""
    def telecharger(ticker):
        return _reponse(["2026-09-09", "2026-09-10", "2026-09-11"], [1.0, 2.0, 2.1], ticker=ticker)

    engine = _EngineFactice()
    sonde.executer_sonde(engine, {}, telecharger=telecharger, maintenant=MAINTENANT, pause_s=0)
    for ligne in engine.journal[1][1]:
        assert ligne["derniere_date_valide"] == date(2026, 9, 11)
        assert ligne["derniere_date_close"] == date(2026, 9, 10)


def test_le_jour_de_reference_est_le_jour_utc_du_passage(sqlalchemy_disponible):
    """
    00h20 à Bruxelles le 11/09 = 22h20 UTC le 10/09 : pour sync_prix,
    qui raisonne en UTC, la séance du 10/09 est encore « du jour ».
    """
    bruxelles_ete = timezone(timedelta(hours=2))
    engine = _EngineFactice()
    sonde.executer_sonde(engine, {}, telecharger=_telecharger_ok,
                         maintenant=datetime(2026, 9, 11, 0, 20, tzinfo=bruxelles_ete),
                         pause_s=0)
    for ligne in engine.journal[1][1]:
        assert ligne["derniere_date_close"] == date(2026, 9, 9)


def test_une_exception_sur_un_instrument_n_arrete_pas_le_passage(sqlalchemy_disponible):
    def telecharger(ticker):
        if ticker == "SAP.DE":
            raise TimeoutError("délai dépassé")
        return _telecharger_ok(ticker)

    engine = _EngineFactice()
    res = sonde.executer_sonde(engine, {}, telecharger=telecharger,
                               maintenant=MAINTENANT, pause_s=0)
    assert res["status"] == "ok" and res["nb_erreurs"] == 1
    lignes = engine.journal[1][1]
    sap = next(l for l in lignes if l["ticker"] == "SAP.DE")
    assert sap["erreur"].startswith("exception : ") and sap["nb_barres"] == 0
    assert sum(1 for l in lignes if l["erreur"] is None) == len(lignes) - 1


def test_ne_leve_jamais_si_la_db_tombe(sqlalchemy_disponible):
    res = sonde.executer_sonde(_EngineFactice(echec=True), {}, telecharger=_telecharger_ok,
                               maintenant=MAINTENANT, pause_s=0)
    assert res["status"] == "error"
    assert "DB indisponible" in res["error"]


# ------------------------------------------------------------
# enregistrer_sonde
# ------------------------------------------------------------
class _SchedulerFactice:
    def __init__(self):
        self.jobs = []

    def add_job(self, fonction, declencheur, **options):
        self.jobs.append((fonction, declencheur, options))


@pytest.mark.parametrize("env", [{}, {"SONDE_PUBLICATION": ""}, {"SONDE_PUBLICATION": "off"},
                                 {"SONDE_PUBLICATION": "true"}, {"SONDE_PUBLICATION": "1"}])
def test_inerte_sans_activation_explicite(env):
    scheduler = _SchedulerFactice()
    assert sonde.enregistrer_sonde(scheduler, object(), {}, env=env,
                                   creer_declencheur=lambda: "cron") is False
    assert scheduler.jobs == []


@pytest.mark.parametrize("valeur", ["on", " ON "])
def test_active_planifie_un_job(valeur):
    scheduler = _SchedulerFactice()
    assert sonde.enregistrer_sonde(scheduler, object(), {}, env={"SONDE_PUBLICATION": valeur},
                                   creer_declencheur=lambda: "cron") is True
    (fonction, declencheur, options), = scheduler.jobs
    assert declencheur == "cron"
    assert options["id"] == sonde.JOB_ID and options["replace_existing"] is True
    assert callable(fonction)


# ------------------------------------------------------------
# Câblage main.py
# ------------------------------------------------------------
def test_main_enregistre_la_sonde_au_demarrage():
    """
    Lu avec `ast` : importer main.py exigerait une DB, FastAPI et le réseau.
    Le caractère gardé de l'import est vérifié par test_couplages.py.
    """
    arbre = ast.parse((RACINE / "main.py").read_text(encoding="utf-8"))
    demarrage = _fonction(arbre, "start_scheduler")
    assert demarrage is not None, "main.py n'expose plus start_scheduler"
    appels = {
        n.func.id for n in ast.walk(demarrage)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "enregistrer_sonde" in appels
