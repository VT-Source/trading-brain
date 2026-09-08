"""
Tests du garde-fou de fraîcheur par place de cotation (roadmap #28).

Chaque incident documenté a son test de régression (règle du projet) :
  - 2026-09-07 : trou européen (EUR à 1 barre sur 65) — doit être détecté ;
  - 2026-09-07 : Labor Day, US fermés — ne doit PAS être détecté ;
  - 14 juillet : Euronext Paris fermé, Amsterdam et Bruxelles cotent — ne
    doit PAS être détecté, ce qui est la raison d'être du découpage par
    place plutôt que par zone.

Aucune DB, aucun réseau.
"""

from datetime import date

import pytest

from freshness import (
    COMPLETE,
    FERMEE,
    PARTIELLE,
    PERIMEE,
    PLACE_US,
    derniere_seance_commune,
    diagnostic_fraicheur,
    evaluer_places,
    place_de_cotation,
)


# ============================================================
# place_de_cotation
# ============================================================

def test_place_sans_suffixe_est_us():
    assert place_de_cotation("AAPL") == PLACE_US
    assert place_de_cotation("BRK-B") == PLACE_US


def test_place_adr_us_malgre_siege_europeen():
    # NXPI est néerlandaise, ASML (sans suffixe) est l'ADR : cotées à New York.
    # Même règle que zone_priority_for : la place prime sur le pays du siège.
    assert place_de_cotation("NXPI") == PLACE_US
    assert place_de_cotation("ASML") == PLACE_US
    assert place_de_cotation("ASML.AS") == "Euronext Amsterdam"


def test_places_europeennes_distinctes():
    assert place_de_cotation("AI.PA") == "Euronext Paris"
    assert place_de_cotation("ABI.BR") == "Euronext Bruxelles"
    assert place_de_cotation("SAP.DE") == "Xetra"
    assert place_de_cotation("NESN.SW") == "SIX Zurich"
    assert place_de_cotation("SAN.MC") == "BME Madrid"
    assert place_de_cotation("ENI.MI") == "Borsa Italiana"
    assert place_de_cotation("HM-B.ST") == "Nasdaq Stockholm"


def test_place_coreenne():
    assert place_de_cotation("005930.KS") == "KRX"


def test_suffixe_inconnu_reste_visible():
    # Ne doit surtout pas retomber sur US : un ticker exotique diluerait la
    # couverture de la plus grosse place et rendrait la règle muette.
    assert place_de_cotation("XYZ.TO") == "suffixe:TO"


# ============================================================
# Jeux de données
# ============================================================

def _univers(nb_us=20, nb_pa=8, nb_as=5, nb_de=6, nb_sw=5, nb_mc=5, nb_ks=10):
    """Construit un univers de tickers proche de la répartition réelle."""
    tickers = {}
    tickers.update({f"US{i}": PLACE_US for i in range(nb_us)})
    tickers.update({f"P{i}.PA": None for i in range(nb_pa)})
    tickers.update({f"A{i}.AS": None for i in range(nb_as)})
    tickers.update({f"D{i}.DE": None for i in range(nb_de)})
    tickers.update({f"S{i}.SW": None for i in range(nb_sw)})
    tickers.update({f"M{i}.MC": None for i in range(nb_mc)})
    tickers.update({f"{i:06d}.KS": None for i in range(nb_ks)})
    return list(tickers)


# Cinq séances consécutives (lun→ven), plus le lundi suivant.
SEANCES = [date(2026, 8, 31), date(2026, 9, 1), date(2026, 9, 2),
           date(2026, 9, 3), date(2026, 9, 4), date(2026, 9, 7)]

VEILLE = date(2026, 9, 4)
JOUR = date(2026, 9, 7)


def _dernieres(regle):
    """regle : callable(ticker) -> date de la dernière barre."""
    return {t: regle(t) for t in _univers()}


# ============================================================
# Cas nominal
# ============================================================

def test_journee_normale_toutes_places_a_jour():
    diag = diagnostic_fraicheur(_dernieres(lambda t: JOUR), SEANCES)

    assert diag["ok"] is True
    assert diag["places_douteuses"] == []
    assert diag["tickers_exclus"] == []
    assert diag["date_reference"] == JOUR
    assert all(p["verdict"] == COMPLETE for p in diag["places"].values())


def test_seuil_de_couverture_est_inclusif():
    # 4 tickers sur 5 = 80 % pile : la place reste exploitable.
    dernieres = {"S0.SW": JOUR, "S1.SW": JOUR, "S2.SW": JOUR,
                 "S3.SW": JOUR, "S4.SW": VEILLE}
    places = evaluer_places(dernieres, SEANCES)

    assert places["SIX Zurich"]["verdict"] == COMPLETE
    assert places["SIX Zurich"]["taux_couverture"] == 0.8


# ============================================================
# Fériés : ce que la règle doit LAISSER PASSER
# ============================================================

def test_labor_day_us_ferme_ne_declenche_pas():
    # 2026-09-07 : NYSE fermé, les autres places cotent. Le dernier cours US
    # du 04/09 est le vrai dernier cours : rien à exclure.
    diag = diagnostic_fraicheur(
        _dernieres(lambda t: VEILLE if place_de_cotation(t) == PLACE_US else JOUR),
        SEANCES,
    )

    assert diag["ok"] is True
    assert diag["places"][PLACE_US]["verdict"] == FERMEE
    assert diag["places"][PLACE_US]["retard_seances"] == 1
    assert diag["tickers_exclus"] == []


def test_quatorze_juillet_paris_ferme_amsterdam_ouvert():
    # Le cas qui condamne un découpage par ZONE : Euronext Paris ferme le
    # 14 juillet, Amsterdam et Bruxelles cotent. Vu « zone EU », la couverture
    # tomberait à ~70 % et la règle crierait au loup tous les 14 juillet.
    diag = diagnostic_fraicheur(
        _dernieres(lambda t: VEILLE if t.endswith(".PA") else JOUR),
        SEANCES,
    )

    assert diag["ok"] is True
    assert diag["places"]["Euronext Paris"]["verdict"] == FERMEE
    assert diag["places"]["Euronext Amsterdam"]["verdict"] == COMPLETE


def test_weekend_ne_compte_pas_comme_du_retard():
    # Le retard se compte en SÉANCES. Entre le vendredi 04/09 et le lundi
    # 07/09 il n'y a qu'une séance d'écart, pas trois jours.
    diag = diagnostic_fraicheur(
        _dernieres(lambda t: VEILLE if place_de_cotation(t) == "KRX" else JOUR),
        SEANCES,
    )

    assert diag["places"]["KRX"]["retard_seances"] == 1
    assert diag["ok"] is True


# ============================================================
# Incidents : ce que la règle doit ATTRAPER
# ============================================================

def test_trou_europeen_du_7_septembre_2026():
    # Incident réel : 64 tickers EUR sur 65 sans barre au lundi 07/09, alors
    # que la couverture est à 65/65 tous les jours ouvrés des trois semaines
    # précédentes. BME avait 1 ticker sur 5, SIX 3 sur 5, Euronext et Xetra
    # aucun. Les 10 coréens étaient bien là.
    def regle(t):
        place = place_de_cotation(t)
        if place == "BME Madrid":
            return JOUR if t == "M0.MC" else VEILLE      # 1/5
        if place == "SIX Zurich":
            return JOUR if t in ("S0.SW", "S1.SW", "S2.SW") else VEILLE  # 3/5
        if place in ("Euronext Paris", "Euronext Amsterdam", "Xetra"):
            return VEILLE                                 # 0 %
        if place == PLACE_US:
            return VEILLE                                 # Labor Day
        return JOUR                                       # KRX

    zones = {t: ("KR" if t.endswith(".KS")
                 else "US" if place_de_cotation(t) == PLACE_US else "EU")
             for t in _univers()}

    diag = diagnostic_fraicheur(_dernieres(regle), SEANCES, zones=zones)

    assert diag["ok"] is False
    assert set(diag["places_douteuses"]) == {"BME Madrid", "SIX Zurich"}
    assert diag["places"]["BME Madrid"]["verdict"] == PARTIELLE
    assert diag["places"]["SIX Zurich"]["verdict"] == PARTIELLE
    assert diag["zones_touchees"] == ["EU"]
    # Toutes les valeurs des deux places sont écartées, y compris celles qui
    # ont une barre : un ranking à moitié à jour sur une place est pire qu'un
    # ranking sans elle, car la normalisation min-max est globale.
    assert len(diag["tickers_exclus"]) == 10
    assert "M0.MC" in diag["tickers_exclus"]
    # Les fériés légitimes du même jour ne sont pas touchés : les 20 valeurs
    # américaines restent dans le ranking, scorées sur leur clôture du 04/09.
    assert diag["places"][PLACE_US]["verdict"] == FERMEE
    assert not any(t.startswith("US") for t in diag["tickers_exclus"])


def test_place_absente_trop_longtemps_est_perimee():
    # Au-delà d'un férié, une absence totale n'est plus explicable par le
    # calendrier : 3 séances de retard.
    diag = diagnostic_fraicheur(
        _dernieres(lambda t: date(2026, 9, 2) if t.endswith(".DE") else JOUR),
        SEANCES,
    )

    assert diag["ok"] is False
    assert diag["places"]["Xetra"]["verdict"] == PERIMEE
    assert diag["places"]["Xetra"]["retard_seances"] == 3
    assert all(t.endswith(".DE") for t in diag["tickers_exclus"])


def test_tolerance_parametrable():
    # Avec une tolérance de 0 séance, même un férié devient un motif
    # d'exclusion. Le paramètre existe pour pouvoir durcir sans retoucher
    # le code appelant.
    diag = diagnostic_fraicheur(
        _dernieres(lambda t: VEILLE if place_de_cotation(t) == PLACE_US else JOUR),
        SEANCES,
        tolerance_seances=0,
    )

    assert diag["ok"] is False
    assert diag["places"][PLACE_US]["verdict"] == PERIMEE


def test_resume_nomme_les_places_et_les_taux():
    diag = diagnostic_fraicheur(
        _dernieres(lambda t: date(2026, 9, 2) if t.endswith(".DE") else JOUR),
        SEANCES,
    )

    assert "Xetra" in diag["resume"]
    assert "0/6" in diag["resume"]
    assert str(JOUR) in diag["resume"]


# ============================================================
# Robustesse — la règle ne doit jamais faire tomber le ranking
# ============================================================

def test_entrees_vides_ne_levent_pas():
    assert evaluer_places({}, []) == {}

    diag = diagnostic_fraicheur({}, [])
    assert diag["ok"] is True
    assert diag["tickers_exclus"] == []
    assert diag["date_reference"] is None


def test_calendrier_absent_se_deduit_des_dernieres_dates():
    # Si l'appelant ne fournit pas de calendrier, on retombe sur les dates
    # observées plutôt que de planter.
    diag = diagnostic_fraicheur({"AAPL": JOUR, "AI.PA": VEILLE}, [])

    assert diag["date_reference"] == JOUR
    assert diag["places"]["Euronext Paris"]["verdict"] == FERMEE


def test_univers_mono_place():
    diag = diagnostic_fraicheur({f"US{i}": JOUR for i in range(5)}, SEANCES)

    assert diag["ok"] is True
    assert list(diag["places"]) == [PLACE_US]


def test_ticker_sans_date_est_traite_comme_absent():
    dernieres = {f"US{i}": JOUR for i in range(5)}
    dernieres.update({"D0.DE": None, "D1.DE": None})

    diag = diagnostic_fraicheur(dernieres, SEANCES)

    assert diag["places"]["Xetra"]["derniere_date"] is None
    assert diag["places"]["Xetra"]["verdict"] == PERIMEE


def test_date_reference_explicite_est_respectee():
    # On peut demander « classe-moi au 04/09 » même si des barres du 07/09
    # existent : utile pour rejouer un incident.
    diag = diagnostic_fraicheur(_dernieres(lambda t: JOUR), SEANCES,
                                date_reference=VEILLE)

    assert diag["date_reference"] == VEILLE
    # Toutes les places sont en avance sur la référence : aucune n'est à jour
    # AU SENS de la date demandée, mais aucune n'est en retard non plus.
    assert all(p["retard_seances"] == 0 for p in diag["places"].values())


@pytest.mark.parametrize("nb_a_jour,attendu", [
    (5, COMPLETE),    # 100 %
    (4, COMPLETE),    # 80 % — seuil inclusif
    (3, PARTIELLE),   # 60 %
    (1, PARTIELLE),   # 20 % — le cas BME du 07/09
    (0, FERMEE),      # aucune barre, 1 séance de retard
])
def test_frontiere_du_seuil(nb_a_jour, attendu):
    dernieres = {f"S{i}.SW": (JOUR if i < nb_a_jour else VEILLE)
                 for i in range(5)}
    places = evaluer_places(dernieres, SEANCES)

    assert places["SIX Zurich"]["verdict"] == attendu


# ============================================================
# Contrat d'appel avec compute_and_store_ranking
# ============================================================

def test_contrat_appel_depuis_ticker_data():
    """
    Reproduit exactement la conversion faite dans main.py (~l. 2170) :
    `ticker_data` est un dict de DataFrames indexés par pd.Timestamp, alors
    que la règle raisonne en `datetime.date`. Ce test verrouille le passage
    d'un type à l'autre — c'est là que le câblage peut casser en silence.
    """
    pd = pytest.importorskip("pandas")

    idx = pd.to_datetime(["2026-09-02", "2026-09-03", "2026-09-04", "2026-09-07"])
    ticker_data = {
        "AAPL":   pd.DataFrame({"prix_ajuste": [1, 2, 3]}, index=idx[:3]),   # Labor Day
        "MSFT":   pd.DataFrame({"prix_ajuste": [1, 2, 3]}, index=idx[:3]),
        "AI.PA":  pd.DataFrame({"prix_ajuste": [1, 2, 3]}, index=idx[:3]),   # trou EU
        "BNP.PA": pd.DataFrame({"prix_ajuste": [1, 2, 3, 4]}, index=idx),
        "005930.KS": pd.DataFrame({"prix_ajuste": [1, 2, 3, 4]}, index=idx),
    }

    all_dates = set()
    for df in ticker_data.values():
        all_dates.update(df.index)

    dernieres_dates = {t: df.index.max().date()
                       for t, df in ticker_data.items() if not df.empty}
    diag = diagnostic_fraicheur(dernieres_dates,
                                calendrier={d.date() for d in all_dates},
                                zones={"AI.PA": "EU", "BNP.PA": "EU"})

    assert diag["date_reference"] == date(2026, 9, 7)
    # Euronext Paris : 1 ticker sur 2 → couverture partielle, pas un férié.
    assert diag["places"]["Euronext Paris"]["verdict"] == PARTIELLE
    assert diag["tickers_exclus"] == ["AI.PA", "BNP.PA"]
    assert diag["zones_touchees"] == ["EU"]
    # NYSE totalement absent d'une seule séance → férié, rien à exclure.
    assert diag["places"][PLACE_US]["verdict"] == FERMEE


# ============================================================
# derniere_seance_commune — politique « aligner »
# ============================================================

def test_seance_commune_recule_au_ferie():
    # Labor Day : NYSE arrêté au 04/09, le reste au 07/09. Aligner revient à
    # classer tout le monde au 04/09 plutôt que de comparer deux séances.
    diag = diagnostic_fraicheur(
        _dernieres(lambda t: VEILLE if place_de_cotation(t) == PLACE_US else JOUR),
        SEANCES,
    )

    assert derniere_seance_commune(diag) == VEILLE


def test_seance_commune_ignore_les_places_douteuses():
    # Xetra est périmé de 3 séances : il ne doit pas tirer le ranking au 02/09.
    diag = diagnostic_fraicheur(
        _dernieres(lambda t: date(2026, 9, 2) if t.endswith(".DE") else JOUR),
        SEANCES,
    )

    assert "Xetra" in diag["places_douteuses"]
    assert derniere_seance_commune(diag) == JOUR


def test_seance_commune_journee_normale():
    diag = diagnostic_fraicheur(_dernieres(lambda t: JOUR), SEANCES)
    assert derniere_seance_commune(diag) == JOUR


def test_seance_commune_sans_donnees():
    assert derniere_seance_commune({}) is None
    assert derniere_seance_commune(diagnostic_fraicheur({}, [])) is None
