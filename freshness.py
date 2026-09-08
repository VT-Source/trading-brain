# ============================================================
# freshness.py — Fraîcheur des données par place de cotation — Trading Brain v1.0
# VT-Source/trading-brain
# ============================================================
# Règle de cohérence temporelle du ranking, extraite pour être testable sans
# DB, sans réseau et sans importer pandas/FastAPI (motif `zone_priority_for`,
# puis `scheduling.py` — cf. PROJECT_STATUS, règle « extraire avant de tester »).
#
# Ce module n'a AUCUNE dépendance externe. Il est importé sans try/except,
# comme `scheduling.py` : un garde-fou de sécurité qui disparaît en silence
# est pire que pas de garde-fou du tout.
# ============================================================

# Suffixe Yahoo → place de cotation. La place, et non la zone, est l'unité
# pertinente : c'est elle qui porte un calendrier de bourse. Euronext Paris
# ferme le 14 juillet pendant qu'Amsterdam et Bruxelles cotent ; SIX ferme
# le 1er août pendant que le reste de l'Europe cote. Raisonner « zone EU »
# transformerait chacun de ces jours en fausse alerte.
PLACES_PAR_SUFFIXE = {
    "AS": "Euronext Amsterdam",
    "BR": "Euronext Bruxelles",
    "PA": "Euronext Paris",
    "DE": "Xetra",
    "SW": "SIX Zurich",
    "MC": "BME Madrid",
    "MI": "Borsa Italiana",
    "ST": "Nasdaq Stockholm",
    "KS": "KRX",
    "KQ": "KOSDAQ",
    "L":  "London Stock Exchange",
    "CO": "Nasdaq Copenhague",
    "OL": "Oslo Børs",
    "HE": "Nasdaq Helsinki",
    "VI": "Wiener Börse",
    "LS": "Euronext Lisbonne",
}

PLACE_US = "NYSE/NASDAQ"

# Couverture en dessous de laquelle une place est considérée comme cassée.
# Le 2026-09-07, BME était à 1 ticker sur 5 et SIX à 3 sur 5 : une place qui
# cote publie pour la quasi-totalité de ses valeurs, une place trouée non.
SEUIL_COUVERTURE = 0.80

# Nombre de séances de retard tolérées pour une place TOTALEMENT absente.
# 1 séance = un jour férié local (Labor Day côté US, 1er août côté SIX).
TOLERANCE_SEANCES = 1

# Verdicts possibles pour une place, du plus sain au plus grave.
COMPLETE = "complete"       # la place a coté et publié
FERMEE   = "fermee"         # aucune barre, mais l'écart est celui d'un férié
PARTIELLE = "partielle"     # la place a publié pour une minorité de ses valeurs
PERIMEE  = "perimee"        # la place est absente depuis plus d'un férié


def place_de_cotation(ticker: str) -> str:
    """
    Place de cotation d'un ticker, déduite de son suffixe Yahoo.

    Même principe que `zone_priority_for` : la place de cotation prime sur le
    pays du siège. Un ticker sans suffixe est coté NYSE/NASDAQ, quelle que
    soit l'incorporation de la société (ADR : NXPI, ACN, LIN, TEL, MTD…).

    Un suffixe inconnu retourne "suffixe:<XX>" plutôt que de tomber sur US :
    une place non répertoriée doit rester visible, pas être fondue dans le
    groupe le plus gros — sinon un ticker exotique dilue la couverture US et
    la règle devient muette.
    """
    if "." not in ticker:
        return PLACE_US
    suffixe = ticker.rsplit(".", 1)[-1].upper()
    return PLACES_PAR_SUFFIXE.get(suffixe, f"suffixe:{suffixe}")


def evaluer_places(dernieres_dates, calendrier, date_reference=None,
                   seuil_couverture=SEUIL_COUVERTURE,
                   tolerance_seances=TOLERANCE_SEANCES) -> dict:
    """
    Diagnostique chaque place de cotation à la date de référence.

    Le problème (roadmap #28) : `compute_composite_score` accepte pour chaque
    ticker sa dernière barre disponible jusqu'à 5 jours d'écart, en silence.
    Le score composite compare donc des tickers arrêtés à des dates
    différentes, et la normalisation min-max étant calculée sur l'ensemble
    des candidats, une zone décalée ne fausse pas seulement ses propres
    lignes : elle déplace le score de tout le monde.

    La difficulté est de distinguer « la place était fermée » de « la donnée
    manque ». Sans calendrier de bourse, le discriminant est la COUVERTURE :
      - une place qui cote publie pour la quasi-totalité de ses valeurs ;
      - un férié donne 0 % ;
      - une panne fournisseur donne un taux intermédiaire (le 2026-09-07,
        l'EUR était à 1 barre sur 65, dont BME 1/5 et SIX 3/5).
    Reste l'angle mort assumé : une panne parfaitement totale sur une place
    est indiscernable d'un férié tant qu'elle ne dure qu'une séance. Au-delà,
    l'écart la trahit ; en deçà, l'alerte (b) sur tickers en retard prend le
    relais.

    `dernieres_dates` : {ticker: date de sa dernière barre}.
    `calendrier` : itérable des dates de séance observées dans l'univers
                   (typiquement l'union des index de prix sur la fenêtre
                   récente) — sert à compter en SÉANCES et non en jours
                   calendaires, un week-end ne comptant pas comme un retard.
    `date_reference` : date à laquelle on prétend classer. Par défaut, la
                   dernière date du calendrier.

    Retourne {place: {verdict, taux_couverture, nb_a_jour, nb_tickers,
                      derniere_date, retard_seances, tickers}}.
    Ne lève jamais sur des entrées vides : retourne {}.
    """
    if not dernieres_dates:
        return {}

    seances = sorted({d for d in calendrier if d is not None})
    if not seances:
        seances = sorted({d for d in dernieres_dates.values() if d is not None})
    if not seances:
        return {}

    if date_reference is None:
        date_reference = seances[-1]

    # Séances retenues pour compter le retard : celles qui vont jusqu'à la
    # date de référence incluse. Compter en séances est ce qui rend la règle
    # insensible aux week-ends et aux fériés globaux.
    seances_utiles = [d for d in seances if d <= date_reference]

    groupes = {}
    for ticker, dern in dernieres_dates.items():
        groupes.setdefault(place_de_cotation(ticker), {})[ticker] = dern

    resultat = {}
    for place, membres in groupes.items():
        nb_tickers = len(membres)
        nb_a_jour = sum(1 for d in membres.values() if d == date_reference)
        taux = nb_a_jour / nb_tickers if nb_tickers else 0.0

        dates_place = [d for d in membres.values() if d is not None]
        derniere = max(dates_place) if dates_place else None

        # Retard = nombre de séances de l'univers strictement postérieures à
        # la dernière barre de la place et antérieures ou égales à la date de
        # référence. 0 si la place est à jour, 1 pour un férié local.
        if derniere is None:
            retard = len(seances_utiles)
        else:
            retard = sum(1 for d in seances_utiles if d > derniere)

        if taux >= seuil_couverture:
            verdict = COMPLETE
        elif nb_a_jour == 0 and retard <= tolerance_seances:
            verdict = FERMEE
        elif nb_a_jour == 0:
            verdict = PERIMEE
        else:
            verdict = PARTIELLE

        resultat[place] = {
            "verdict":         verdict,
            "taux_couverture": round(taux, 4),
            "nb_a_jour":       nb_a_jour,
            "nb_tickers":      nb_tickers,
            "derniere_date":   derniere,
            "retard_seances":  retard,
            "tickers":         sorted(membres),
        }

    return resultat


def diagnostic_fraicheur(dernieres_dates, calendrier, zones=None,
                         date_reference=None,
                         seuil_couverture=SEUIL_COUVERTURE,
                         tolerance_seances=TOLERANCE_SEANCES) -> dict:
    """
    Verdict global : quelles places sont exploitables, lesquelles ne le sont
    pas, et quels tickers il faudrait écarter du ranking.

    `zones` : {ticker: zone} — facultatif, sert uniquement à libeller les
    zones touchées dans l'alerte. La décision, elle, se prend par place :
    une zone n'a pas de calendrier de bourse, une place si.

    Retourne :
      - `date_reference`   : la date à laquelle on prétend classer
      - `places`           : le détail par place (cf. `evaluer_places`)
      - `places_saines`    : places complètes ou légitimement fermées
      - `places_douteuses` : places partielles ou périmées
      - `tickers_exclus`   : tickers rattachés à une place douteuse
      - `zones_touchees`   : zones concernées, pour le message d'alerte
      - `ok`               : True s'il n'y a aucune place douteuse
      - `resume`           : phrase prête à logger ou à pousser en alerte
    """
    places = evaluer_places(dernieres_dates, calendrier, date_reference,
                            seuil_couverture, tolerance_seances)
    if not places:
        return {
            "date_reference": None, "places": {}, "places_saines": [],
            "places_douteuses": [], "tickers_exclus": [], "zones_touchees": [],
            "ok": True, "resume": "aucune donnée à évaluer",
        }

    if date_reference is None:
        candidates = [d for d in dernieres_dates.values() if d is not None]
        date_reference = max(candidates) if candidates else None

    saines, douteuses, exclus = [], [], []
    for place, info in places.items():
        if info["verdict"] in (COMPLETE, FERMEE):
            saines.append(place)
        else:
            douteuses.append(place)
            exclus.extend(info["tickers"])

    zones = zones or {}
    zones_touchees = sorted({zones.get(t) for t in exclus if zones.get(t)})

    if not douteuses:
        resume = f"{len(places)} places, toutes exploitables au {date_reference}"
    else:
        details = ", ".join(
            f"{p} {places[p]['nb_a_jour']}/{places[p]['nb_tickers']}"
            f" ({places[p]['verdict']})"
            for p in sorted(douteuses)
        )
        resume = (f"{len(douteuses)} place(s) non exploitable(s) au "
                  f"{date_reference} : {details} — {len(exclus)} tickers concernés")

    return {
        "date_reference":   date_reference,
        "places":           places,
        "places_saines":    sorted(saines),
        "places_douteuses": sorted(douteuses),
        "tickers_exclus":   sorted(set(exclus)),
        "zones_touchees":   zones_touchees,
        "ok":               not douteuses,
        "resume":           resume,
    }


def derniere_seance_commune(diagnostic: dict):
    """
    Dernière séance publiée par TOUTES les places saines.

    Répond à la part du problème que l'exclusion ne couvre pas. Un férié
    local est parfaitement légitime côté données — le dernier cours du NYSE
    au 04/09 EST le dernier cours — mais le score composite compare alors ce
    04/09 américain à du 07/09 coréen. Reculer à la séance commune supprime
    la comparaison décalée, au prix d'un ranking qui perd une séance chaque
    jour férié local.

    Les places douteuses sont ignorées : une place périmée depuis trois
    semaines tirerait sinon tout le ranking avec elle. Retourne None s'il n'y
    a rien à aligner.
    """
    if not diagnostic:
        return None

    places = diagnostic.get("places") or {}
    saines = diagnostic.get("places_saines") or []
    dates = [places[p]["derniere_date"] for p in saines
             if p in places and places[p].get("derniere_date")]

    return min(dates) if dates else None
