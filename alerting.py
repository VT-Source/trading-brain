# ============================================================
# alerting.py — Alerting actif (Telegram)
# Trading Brain | VT-Source
# ============================================================
# v1.0 (2026-09-01) — Roadmap Phase 1 #19.
#
# Objectif : passer de l'observabilité *pull* (/health-jobs, logs Railway)
# à un *push*. Cible les deux scénarios vécus :
#   - 2026-05-16 : corruption silencieuse (59 tickers EU non synchronisés)
#                  propagée jusqu'à une décision de trade.
#   - 2026-08-31 : batch avis IA soumis un lundi, polling planifié sam+dim
#                  → batch perdu, découvert 24h après par hasard.
#
# PRINCIPES DE CONCEPTION
# -----------------------
# 1. NE JAMAIS CASSER L'APPELANT. Toute fonction publique est enveloppée
#    dans un try/except large et retourne un dict. Une alerte qui échoue
#    ne doit jamais faire tomber un job scheduler ou un endpoint.
# 2. DÉSACTIVATION PROPRE. Sans TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID le
#    module est un no-op silencieux (log unique au démarrage). Le code
#    appelant n'a donc aucune condition à écrire.
# 3. ANTI-SPAM. Un job qui échoue toutes les 30 min ne doit pas produire
#    48 messages/jour : cooldown par clé de déduplication (défaut 6h).
# 4. LOGIQUE MÉTIER PURE ET TESTABLE. Les règles de détection
#    (check_ranking_composition, check_stale_tickers) sont des fonctions
#    pures, sans I/O — testées dans tests/test_alerting.py.
#
# CONFIGURATION (variables d'environnement Railway)
# -------------------------------------------------
#   TELEGRAM_BOT_TOKEN   token donné par @BotFather        (obligatoire)
#   TELEGRAM_CHAT_ID     id de la conversation cible        (obligatoire)
#   ALERTING_ENABLED     "0" pour couper sans dé-configurer (défaut "1")
#   ALERT_COOLDOWN_MIN   minutes entre 2 alertes identiques (défaut 360)
#
# Pour obtenir le chat_id : créer le bot via @BotFather, lui envoyer un
# message, puis GET https://api.telegram.org/bot<TOKEN>/getUpdates
# → result[0].message.chat.id
# ============================================================

import os
import time
import requests

ALERTING_VERSION = "1.0"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
ALERTING_ENABLED   = os.getenv("ALERTING_ENABLED", "1") not in ("0", "false", "False", "")

try:
    ALERT_COOLDOWN_MIN = int(os.getenv("ALERT_COOLDOWN_MIN", "360"))
except (TypeError, ValueError):
    ALERT_COOLDOWN_MIN = 360

TELEGRAM_TIMEOUT_S = 10
TELEGRAM_MAX_CHARS = 3900          # limite API = 4096, marge de sécurité

# Seuils des règles de détection (surchargeables par appel)
SEUIL_ZONE_PCT       = 0.90        # > 90 % d'une seule zone dans le ranking
MIN_TICKERS_RANKED   = 10          # moins de N tickers classés = anormal
MIN_NB_ELIGIBLE      = 20          # moins de N tickers éligibles = anormal
SEUIL_STALE_TICKERS  = 5           # nb de tickers en retard avant alerte

_PREFIXE = {"INFO": "ℹ️", "WARN": "⚠️", "CRIT": "🚨"}

# Mémoire de déduplication : {dedup_key: timestamp du dernier envoi}
# En mémoire volontairement (perdue au redéploiement Railway, comme
# job_status) : au pire une alerte est renvoyée une fois de trop après
# un redémarrage, ce qui est le bon sens de l'erreur.
_LAST_SENT = {}


# ============================================================
# TRANSPORT
# ============================================================

def is_configured() -> bool:
    """True si le module peut effectivement envoyer une alerte."""
    return bool(ALERTING_ENABLED and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def alerting_status() -> dict:
    """
    État de l'alerting, destiné à /health-jobs et à la page Système.
    Ne divulgue jamais le token.
    """
    return {
        "version":      ALERTING_VERSION,
        "enabled":      bool(ALERTING_ENABLED),
        "configured":   is_configured(),
        "token_set":    bool(TELEGRAM_BOT_TOKEN),
        "chat_id_set":  bool(TELEGRAM_CHAT_ID),
        "cooldown_min": ALERT_COOLDOWN_MIN,
        "dedup_keys":   sorted(_LAST_SENT.keys()),
    }


def _should_send(dedup_key: str, cooldown_min: int, now: float = None) -> bool:
    """
    Anti-spam : True si la clé n'a pas déjà été envoyée dans la fenêtre
    de cooldown. Fonction pure (hors _LAST_SENT) pour rester testable.
    """
    if not dedup_key:
        return True
    now = now if now is not None else time.time()
    last = _LAST_SENT.get(dedup_key)
    if last is None:
        return True
    return (now - last) >= cooldown_min * 60


def _post_telegram(message: str) -> dict:
    """
    Envoi brut. Pas de parse_mode : un ticker comme `AI.PA` ou un message
    d'erreur contenant des `_` ou `*` ferait échouer un parsing Markdown,
    et une alerte qui n'arrive pas est pire qu'une alerte moche.
    """
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    resp = requests.post(
        url,
        json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message[:TELEGRAM_MAX_CHARS],
            "disable_web_page_preview": True,
        },
        timeout=TELEGRAM_TIMEOUT_S,
    )
    ok = resp.status_code == 200
    if not ok:
        print(f"⚠️ Alerting : Telegram HTTP {resp.status_code} — {resp.text[:200]}")
    return {"sent": ok, "http_status": resp.status_code}


def send_alert(titre: str, corps: str = "", level: str = "WARN",
               dedup_key: str = None, cooldown_min: int = None) -> dict:
    """
    Point d'entrée unique. Ne lève JAMAIS.

    Retour : {"sent": bool, "reason": str}
      reason ∈ {"ok", "not_configured", "cooldown", "exception", "http_error"}
    """
    try:
        if not is_configured():
            return {"sent": False, "reason": "not_configured"}

        cd = ALERT_COOLDOWN_MIN if cooldown_min is None else cooldown_min
        if not _should_send(dedup_key, cd):
            return {"sent": False, "reason": "cooldown", "dedup_key": dedup_key}

        prefixe = _PREFIXE.get(level, "⚠️")
        horodatage = time.strftime("%d/%m %H:%M")
        message = f"{prefixe} [Trading Brain] {titre}\n{horodatage}"
        if corps:
            message += f"\n\n{corps}"

        res = _post_telegram(message)
        if res.get("sent") and dedup_key:
            _LAST_SENT[dedup_key] = time.time()
        res["reason"] = "ok" if res.get("sent") else "http_error"
        return res

    except Exception as e:
        # Volontairement avalé : l'alerting est un filet, pas une dépendance.
        print(f"⚠️ Alerting : envoi impossible ({type(e).__name__}: {e})")
        return {"sent": False, "reason": "exception", "error": str(e)}


def send_test_alert() -> dict:
    """Vérification de bout en bout (bouton page Système / endpoint /test-alert)."""
    return send_alert(
        "Test d'alerte",
        f"Si vous lisez ceci, l'alerting v{ALERTING_VERSION} est opérationnel.",
        level="INFO",
        dedup_key=None,          # jamais dédupliqué : c'est un test manuel
    )


# ============================================================
# RÈGLES DE DÉTECTION — fonctions pures, testables sans réseau
# ============================================================

def check_ranking_composition(zones: list, nb_eligible: int,
                              seuil_zone_pct: float = SEUIL_ZONE_PCT,
                              min_tickers: int = MIN_TICKERS_RANKED,
                              min_eligible: int = MIN_NB_ELIGIBLE) -> list:
    """
    Détecte une composition de ranking anormale.

    `zones` = liste des zones des tickers effectivement classés
              (ex. ["US", "US", "EU", "KR", ...]).
    `nb_eligible` = nb de tickers ayant passé les filtres avant troncature.

    Retourne une liste de messages d'anomalie (vide = ranking plausible).

    Motif : le 2026-05-16, 59 tickers EU non synchronisés ont produit un
    ranking 100 % US sans qu'aucune erreur ne soit levée. Un ranking
    mono-zone n'est pas impossible (macro bearish sur une zone), mais il
    mérite un coup d'œil — d'où WARN et non CRIT.
    """
    anomalies = []
    total = len(zones)

    if total == 0:
        anomalies.append("Ranking vide : 0 ticker classé.")
        return anomalies

    if total < min_tickers:
        anomalies.append(f"Seulement {total} ticker(s) classé(s) (seuil : {min_tickers}).")

    if nb_eligible is not None and nb_eligible < min_eligible:
        anomalies.append(f"Seulement {nb_eligible} ticker(s) éligible(s) "
                         f"après filtres (seuil : {min_eligible}).")

    compte = {}
    for z in zones:
        z = z or "?"
        compte[z] = compte.get(z, 0) + 1
    zone_dom, n_dom = max(compte.items(), key=lambda kv: kv[1])
    part = n_dom / total
    if part > seuil_zone_pct:
        detail = ", ".join(f"{z}={n}" for z, n in sorted(compte.items()))
        anomalies.append(f"Concentration zone {zone_dom} : {part:.0%} du ranking "
                         f"({detail}) — sync partiel possible.")

    return anomalies


def check_stale_tickers(stale: list, seuil_nb: int = SEUIL_STALE_TICKERS) -> bool:
    """
    True si le nombre de tickers en retard justifie une alerte.
    `stale` = liste de tuples (ticker, derniere_date) issue de l'audit
    post-sync de sync_prix_logic. Un ou deux tickers en retard est un
    incident isolé (jour férié local, feed cassé sur un titre) ; au-delà
    c'est un problème systémique.
    """
    return bool(stale) and len(stale) >= seuil_nb


# ============================================================
# HELPERS D'ALERTE — un par point d'appel
# ============================================================

def alert_job_failure(job_id: str, erreur: str, duration_s=None) -> dict:
    """(a) Job scheduler en échec — appelé depuis _run_job (main.py)."""
    duree = f" après {duration_s}s" if duration_s is not None else ""
    return send_alert(
        f"Job '{job_id}' en échec{duree}",
        f"Erreur : {str(erreur)[:800]}\n\n"
        f"Vérifier les logs Railway et /health-jobs.",
        level="CRIT",
        dedup_key=f"job_failure:{job_id}",
    )


def alert_tickers_stale(stale: list, ref_date, seuil_nb: int = SEUIL_STALE_TICKERS) -> dict:
    """(b) Audit post-sync — appelé depuis sync_prix_logic (sync.py)."""
    if not check_stale_tickers(stale, seuil_nb):
        return {"sent": False, "reason": "sous_seuil", "nb": len(stale or [])}

    apercu = "\n".join(f"  {t} → {d}" for t, d in list(stale)[:15])
    reste = len(stale) - 15
    if reste > 0:
        apercu += f"\n  ... et {reste} autre(s)"

    return send_alert(
        f"{len(stale)} ticker(s) en retard après le sync prix",
        f"Dernière date en base < {ref_date}.\n"
        f"Un ranking calculé sur ces données sera biaisé "
        f"(cf. incident du 16/05).\n\n{apercu}\n\n"
        f"Rattrapage : /sync-prix?full=true&tickers=...",
        level="CRIT",
        dedup_key="tickers_stale",
    )


def alert_batch_avis_bloque(batch_health: dict) -> dict:
    """
    (d) Batch avis IA resté 'SOUMIS' au-delà du seuil — appelé après
    poll_batch_opinions (main.py). Consomme directement le dict rendu
    par ai_opinion.get_batch_jobs_health().
    """
    if not batch_health or not batch_health.get("alerte"):
        return {"sent": False, "reason": "pas_d_alerte"}

    bloques = [j for j in batch_health.get("jobs", []) if j.get("stale")]
    detail = "\n".join(
        f"  {j.get('batch_id')} — semaine {j.get('semaine')} — {j.get('age_h')}h"
        for j in bloques[:10]
    )
    return send_alert(
        f"{len(bloques)} batch(s) avis IA bloqué(s) "
        f"(> {batch_health.get('seuil_h')}h)",
        f"Le SLA Anthropic est de 24h : au-delà, soit le polling ne tourne "
        f"pas, soit la récupération échoue.\n\n{detail}\n\n"
        f"Geste : GET /poll-ai-opinions (résultats disponibles ~29 jours, "
        f"aucune re-facturation).",
        level="CRIT",
        dedup_key="batch_avis_bloque",
        cooldown_min=720,          # 12h : le poll tourne toutes les 30 min
    )


def alert_ranking_composition(zones: list, nb_eligible: int = None,
                              data_date=None) -> dict:
    """(c) Composition du ranking anormale — appelé depuis compute_and_store_ranking."""
    anomalies = check_ranking_composition(zones, nb_eligible)
    if not anomalies:
        return {"sent": False, "reason": "ranking_plausible"}

    corps = "\n".join(f"  • {a}" for a in anomalies)
    if data_date:
        corps += f"\n\nDate des données : {data_date}"
    corps += "\n\nVérifier l'audit post-sync avant toute décision de trade."

    return send_alert(
        "Composition du ranking anormale",
        corps,
        level="WARN",
        dedup_key="ranking_composition",
    )
