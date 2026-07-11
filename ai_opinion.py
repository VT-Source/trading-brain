# ============================================================
# ai_opinion.py — Avis IA par ticker (Claude Sonnet + web search)
# Trading Brain | VT-Source | v1.3
# ============================================================
# Reçoit `engine` en paramètre (injection de dépendance).
# Appelé par main.py (endpoints + scheduler).
#
# v1.2 — Tracking performance IA :
#   - Snapshot indicateurs au moment de l'avis
#   - Colonnes rendement +1s/+2s/+4s (remplies par job scheduler)
#   - Traçabilité prompt_version
#   - Fonction update_suivi_rendements() pour compléter les rendements
#
# v1.4 — Robustesse suivi rendements :
#   - update_suivi_rendements reconstitue prix_emission via
#     actions_prix_historique si NULL en base (avis générés sans
#     snapshot indicateurs, ex: avant migration v1.2).
# ============================================================

import os
import re
import json
import anthropic
from datetime import date, timedelta
from sqlalchemy import text


ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = "claude-sonnet-5"
PROMPT_VERSION = "v2.1"

_TABLE_CREATED = False


def _ensure_avis_ia_table(engine):
    """Crée la table avis_ia si elle n'existe pas (même engine = même schéma que l'API)."""
    global _TABLE_CREATED
    if _TABLE_CREATED:
        return
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS avis_ia (
                id              SERIAL PRIMARY KEY,
                ticker          VARCHAR(20)   NOT NULL,
                semaine         VARCHAR(10)   NOT NULL,
                rang            INTEGER,
                conviction      VARCHAR(20)   NOT NULL DEFAULT 'MODÉRÉ',
                "analyse"       TEXT,
                resume          VARCHAR(500),
                source          VARCHAR(20)   DEFAULT 'manual',
                model_used      VARCHAR(50),
                tokens_used     INTEGER,
                generated_at    TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
                -- v1.2 : snapshot indicateurs au moment de l'avis
                score_composite DOUBLE PRECISION,
                mom_r2          DOUBLE PRECISION,
                rvol            DOUBLE PRECISION,
                obv_slope       DOUBLE PRECISION,
                prix_emission   DOUBLE PRECISION,
                sma_200         DOUBLE PRECISION,
                atr_14          DOUBLE PRECISION,
                k_adaptatif     DOUBLE PRECISION,
                secteur_force   BOOLEAN,
                macro_bullish   BOOLEAN,
                prompt_version  VARCHAR(20),
                -- v1.3 : type d'avis (ranking = avis d'achat hebdo / position = garder-vendre)
                type_avis       VARCHAR(20),
                -- v1.2 : suivi rendements (remplis par job scheduler)
                prix_1s         DOUBLE PRECISION,
                prix_2s         DOUBLE PRECISION,
                prix_4s         DOUBLE PRECISION,
                rendement_1s    DOUBLE PRECISION,
                rendement_2s    DOUBLE PRECISION,
                rendement_4s    DOUBLE PRECISION,
                UNIQUE (semaine, ticker)
            )
        """))
    _TABLE_CREATED = True
    print("✅ Table avis_ia vérifiée/créée via SQLAlchemy")


def _migrate_avis_ia_columns(engine):
    """Ajoute les colonnes v1.2/v1.3 si elles n'existent pas (migration safe)."""
    new_columns = [
        ("score_composite", "DOUBLE PRECISION"),
        ("mom_r2",          "DOUBLE PRECISION"),
        ("rvol",            "DOUBLE PRECISION"),
        ("obv_slope",       "DOUBLE PRECISION"),
        ("prix_emission",   "DOUBLE PRECISION"),
        ("sma_200",         "DOUBLE PRECISION"),
        ("atr_14",          "DOUBLE PRECISION"),
        ("k_adaptatif",     "DOUBLE PRECISION"),
        ("secteur_force",   "BOOLEAN"),
        ("macro_bullish",   "BOOLEAN"),
        ("prompt_version",  "VARCHAR(20)"),
        ("type_avis",       "VARCHAR(20)"),
        ("prix_1s",         "DOUBLE PRECISION"),
        ("prix_2s",         "DOUBLE PRECISION"),
        ("prix_4s",         "DOUBLE PRECISION"),
        ("rendement_1s",    "DOUBLE PRECISION"),
        ("rendement_2s",    "DOUBLE PRECISION"),
        ("rendement_4s",    "DOUBLE PRECISION"),
        # v2.1 : batch comparatif
        ("classement_ia",       "INTEGER"),
        ("risque_evenementiel", "BOOLEAN"),
    ]
    # IMPORTANT : on sépare chaque opération en sa propre transaction.
    # Si on regroupe ALTER TABLE + UPDATE dans le même 'with engine.begin()',
    # une erreur sur l'UPDATE rollback aussi le ALTER TABLE → la colonne
    # n'est jamais réellement créée. (Bug observé sur Railway PostgreSQL.)

    # 1. Élargir colonnes trop étroites (chacune dans sa propre transaction)
    for col, new_type in [("source", "VARCHAR(20)"), ("conviction", "VARCHAR(20)")]:
        try:
            with engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE avis_ia ALTER COLUMN {col} TYPE {new_type}"))
        except Exception:
            pass

    # 2. Ajouter chaque colonne manquante (chacune dans sa propre transaction)
    for col_name, col_type in new_columns:
        try:
            with engine.begin() as conn:
                conn.execute(text(
                    f'ALTER TABLE avis_ia ADD COLUMN {col_name} {col_type}'
                ))
                print(f"  ✅ Colonne {col_name} ajoutée")
        except Exception:
            # Colonne existe déjà — OK
            pass

    # 3. Backfill type_avis dans une transaction séparée
    #    (ainsi un éventuel échec ici ne casse pas la création des colonnes)
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE avis_ia
                SET type_avis = 'position'
                WHERE source LIKE 'position_%' AND type_avis IS NULL
            """))
            conn.execute(text("""
                UPDATE avis_ia
                SET type_avis = 'ranking'
                WHERE (source IS NULL OR source NOT LIKE 'position_%') AND type_avis IS NULL
            """))
    except Exception as e:
        print(f"  ⚠️ Backfill type_avis : {e}")

    print("✅ Migration colonnes avis_ia v1.3 terminée")

_BATCH_TABLE_CREATED = False

def _ensure_batch_jobs_table(engine):
    """Table de suivi des soumissions batch asynchrones (Message Batches API).
    Persistée en base : un redéploiement Railway entre soumission et
    récupération ne perd rien."""
    global _BATCH_TABLE_CREATED
    if _BATCH_TABLE_CREATED:
        return
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS avis_ia_batch_jobs (
                id             SERIAL PRIMARY KEY,
                batch_id       VARCHAR(64) UNIQUE,
                semaine        VARCHAR(10)  NOT NULL,
                statut         VARCHAR(20)  NOT NULL DEFAULT 'SOUMIS',
                source         VARCHAR(20)  DEFAULT 'auto',
                tickers_data   JSONB        NOT NULL,
                macro_regime   JSONB,
                prompt_version VARCHAR(20),
                model_used     VARCHAR(50),
                lecture_lot    TEXT,
                classement     VARCHAR(200),
                tokens_total   INTEGER,
                tentatives     INTEGER      DEFAULT 1,
                erreur         TEXT,
                submitted_at   TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
                completed_at   TIMESTAMP
            )
        """))
    _BATCH_TABLE_CREATED = True
    print("✅ Table avis_ia_batch_jobs vérifiée/créée")

# ============================================================
# GÉNÉRATION D'AVIS
# ============================================================

def generate_opinion(engine, ticker: str, semaine: str, rang: int = None,
                     quant_data: dict = None, source: str = "manual",
                     type_avis: str = "ranking") -> dict:
    """
    Génère un avis IA pour un ticker donné via Claude Sonnet + web search.

    Paramètres
    ----------
    engine : SQLAlchemy engine
    ticker : str — ex: "NVDA"
    semaine : str — lundi de la semaine, format YYYY-MM-DD
    rang : int — rang dans le ranking (None si appel ad hoc)
    quant_data : dict — données quantitatives (score, mom_r2, rvol, obv_slope, etc.)
    source : str — 'auto' (scheduler) ou 'manual' (dashboard)
    type_avis : str — 'ranking' (avis d'achat) ou 'position' (garder/vendre)

    Retourne
    --------
    dict avec conviction, analyse, resume, tokens_used
    """
    if not ANTHROPIC_API_KEY:
        return {"error": "ANTHROPIC_API_KEY non configurée dans les variables d'environnement"}

    if engine is None:
        return {"error": "engine non connecté"}

    # --- Construire le contexte quantitatif ---
    quant_context = _build_quant_context(ticker, rang, quant_data)

    # --- Appel Claude Sonnet avec web search ---
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        response = client.messages.create(
            model=MODEL,
            max_tokens=4000,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[
                {
                    "role": "user",
                    "content": _build_prompt(ticker, quant_context),
                }
            ],
        )

        # --- Extraire la réponse texte ---
        analyse_full = ""
        for block in response.content:
            if block.type == "text":
                analyse_full += block.text

        tokens_used = response.usage.input_tokens + response.usage.output_tokens

        # --- Parser conviction, résumé et drapeau événementiel ---
        conviction = _extract_conviction(analyse_full)
        resume = _extract_resume(analyse_full)
        risque_evt = _extract_risque_evenementiel(analyse_full)

        # --- Persister en base (UPSERT) avec snapshot ---
        _save_opinion(engine, ticker, semaine, rang, conviction,
                      analyse_full, resume, source, MODEL, tokens_used,
                      quant_data, type_avis=type_avis,
                      risque_evenementiel=risque_evt)

        return {
            "status":      "ok",
            "ticker":      ticker,
            "semaine":     semaine,
            "conviction":  conviction,
            "resume":      resume,
            "analyse":     analyse_full,
            "tokens_used": tokens_used,
            "model":       MODEL,
            "source":      source,
        }

    except anthropic.APIError as e:
        print(f"❌ Erreur API Anthropic pour {ticker} : {e}")
        return {"error": f"Erreur API Anthropic : {e}"}
    except Exception as e:
        print(f"❌ Erreur generate_opinion({ticker}) : {e}")
        return {"error": str(e)}


# ============================================================
# BATCH HEBDO ASYNCHRONE — Message Batches API (−50 % tokens)
# ============================================================

def submit_batch_opinion(engine, tickers_data: list, semaine: str,
                         source: str = "auto", macro_regime: dict = None) -> dict:
    """
    Soumet UN appel batch asynchrone pour le top 5 (prompt v2.1 comparatif).
    Le snapshot quant est persisté dans le job : les indicateurs sauvegardés
    refléteront l'émission, pas la récupération.
    """
    if not ANTHROPIC_API_KEY:
        return {"error": "ANTHROPIC_API_KEY non configurée"}
    if engine is None:
        return {"error": "engine non connecté"}
    if not tickers_data:
        return {"error": "tickers_data vide"}

    _ensure_avis_ia_table(engine)
    _ensure_batch_jobs_table(engine)

    prompt = _build_batch_prompt(engine, tickers_data, semaine, macro_regime)

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        batch = client.messages.batches.create(
            requests=[{
                "custom_id": f"ranking-{semaine}",
                "params": {
                    "model": MODEL,
                    "max_tokens": 16000,
                    "tools": [{"type": "web_search_20250305", "name": "web_search"}],
                    "messages": [{"role": "user", "content": prompt}],
                },
            }]
        )
    except anthropic.APIError as e:
        print(f"❌ Soumission batch avis IA : {e}")
        return {"error": f"Erreur API Anthropic : {e}"}

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO avis_ia_batch_jobs
                (batch_id, semaine, statut, source, tickers_data,
                 macro_regime, prompt_version, model_used)
            VALUES
                (:batch_id, :semaine, 'SOUMIS', :source,
                 CAST(:tickers_data AS jsonb), CAST(:macro AS jsonb),
                 :pv, :model)
        """), {
            "batch_id":     batch.id,
            "semaine":      semaine,
            "source":       source,
            "tickers_data": json.dumps(tickers_data),
            "macro":        json.dumps(macro_regime or {}),
            "pv":           PROMPT_VERSION,
            "model":        MODEL,
        })

    print(f"📤 Batch avis IA soumis : {batch.id} "
          f"({len(tickers_data)} tickers, semaine {semaine})")
    return {"status": "submitted", "batch_id": batch.id,
            "tickers": [t["ticker"] for t in tickers_data]}


def poll_batch_opinions(engine) -> dict:
    """
    Job scheduler (toutes les 30 min) : vérifie les batchs SOUMIS, récupère
    et persiste les avis quand le traitement est terminé (<1h en général,
    24h max). No-op (1 SELECT) quand rien n'est en attente.
    Resoumet automatiquement si ERREUR/EXPIRÉ (max 3 tentatives).
    """
    if engine is None:
        return {"error": "engine non connecté"}
    _ensure_batch_jobs_table(engine)

    with engine.connect() as conn:
        jobs = conn.execute(text("""
            SELECT id, batch_id, semaine, source, tickers_data,
                   macro_regime, tentatives, submitted_at
            FROM avis_ia_batch_jobs
            WHERE statut = 'SOUMIS'
            ORDER BY submitted_at ASC
        """)).fetchall()

    if not jobs:
        return {"status": "ok", "pending": 0}

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    done, still_pending = 0, 0

    for job in jobs:
        (job_id, batch_id, semaine, source,
         tickers_data, macro_regime, tentatives, submitted_at) = job
        if isinstance(tickers_data, str):
            tickers_data = json.loads(tickers_data)
        if isinstance(macro_regime, str):
            macro_regime = json.loads(macro_regime) if macro_regime else {}

        try:
            batch = client.messages.batches.retrieve(batch_id)
        except anthropic.APIError as e:
            print(f"⚠️ Poll batch {batch_id} : {e}")
            continue

        if batch.processing_status != "ended":
            still_pending += 1
            continue

        # --- Batch terminé : lire le résultat (1 seule requête par batch) ---
        analyse_full, tokens_total, result_type = "", 0, None
        try:
            for entry in client.messages.batches.results(batch_id):
                result_type = entry.result.type
                if result_type == "succeeded":
                    msg = entry.result.message
                    for block in msg.content:
                        if block.type == "text":
                            analyse_full += block.text
                    tokens_total = msg.usage.input_tokens + msg.usage.output_tokens
                    if msg.stop_reason == "max_tokens":            # AJOUT
                        print(f"  ⚠️ Batch {batch_id} : réponse TRONQUÉE (max_tokens)")  # AJOUT
        except Exception as e:
            _mark_batch_job(engine, job_id, "ERREUR", erreur=str(e))
            continue

        if result_type != "succeeded" or not analyse_full:
            statut = "EXPIRÉ" if result_type == "expired" else "ERREUR"
            print(f"❌ Batch {batch_id} : résultat {result_type}")
            _mark_batch_job(engine, job_id, statut, erreur=f"result={result_type}")
            if tentatives < 3:
                retry = submit_batch_opinion(engine, tickers_data, semaine,
                                             source=source, macro_regime=macro_regime)
                if retry.get("batch_id"):
                    with engine.begin() as conn:
                        conn.execute(text("""
                            UPDATE avis_ia_batch_jobs
                            SET tentatives = :t WHERE batch_id = :b
                        """), {"t": tentatives + 1, "b": retry["batch_id"]})
            continue

        # --- Parser et persister les avis ---
        parsed = _extract_batch_opinions(
            analyse_full, [t["ticker"] for t in tickers_data])
        tokens_par_avis = tokens_total // max(len(tickers_data), 1)
        nb_saved = 0
        for td in tickers_data:
            tk = td["ticker"].upper()
            p = parsed["tickers"].get(tk)
            if not p:
                print(f"  ⚠️ Batch {batch_id} : section absente pour {tk}")
                continue
            _save_opinion(engine, tk, semaine, td.get("rang"),
                          p["conviction"], p["analyse"], p["resume"],
                          source, MODEL, tokens_par_avis,
                          quant_data=td, type_avis="ranking",
                          classement_ia=p.get("classement_ia"),
                          risque_evenementiel=p.get("risque_evenementiel"),
                          generated_at=submitted_at)
            nb_saved += 1
            cls = p.get("classement_ia")
            print(f"  ✅ {tk} → {p['conviction']}"
                  f"{f' (classé IA #{cls})' if cls else ''}")

        statut_final = "TERMINÉ" if nb_saved == len(tickers_data) else "ERREUR"
        _mark_batch_job(engine, job_id, statut_final,
                        lecture_lot=parsed.get("lecture_lot"),
                        classement=parsed.get("classement_brut"),
                        tokens_total=tokens_total)
        print(f"📥 Batch {batch_id} récupéré : {nb_saved}/{len(tickers_data)} "
              f"avis, {tokens_total} tokens")
        done += 1

    return {"status": "ok", "completed": done, "pending": still_pending}


def _mark_batch_job(engine, job_id, statut, lecture_lot=None,
                    classement=None, tokens_total=None, erreur=None):
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE avis_ia_batch_jobs
            SET statut = :s, lecture_lot = :l, classement = :c,
                tokens_total = :t, erreur = :e, completed_at = NOW()
            WHERE id = :id
        """), {"s": statut, "l": lecture_lot, "c": classement,
               "t": tokens_total, "e": erreur, "id": job_id})


def _build_batch_prompt(engine, tickers_data, semaine, macro_regime=None) -> str:
    """Prompt v2.1 « batch comparatif » (REFONTE_AVIS_IA §4)."""
    n = len(tickers_data)
    tickers_list = ", ".join(td["ticker"].upper() for td in tickers_data)
    exemple = ", ".join(td["ticker"].upper() for td in reversed(tickers_data))

    macro_ctx = _format_macro_context(macro_regime)
    secteurs_ctx = _load_secteurs_context(engine)
    concentration = _format_concentration_lot(tickers_data)

    blocs = []
    for td in tickers_data:
        tk = td["ticker"].upper()
        bloc = _build_quant_context(tk, td.get("rang"), td)
        dernier = _load_dernier_avis(engine, tk, semaine)
        if dernier:
            bloc += f"\n{dernier}"
        blocs.append(bloc)
    candidats = "\n\n".join(blocs)

    return f"""Tu es membre d'un comité d'investissement momentum. Le système quantitatif a
sélectionné {n} candidats à l'achat pour lundi prochain ({tickers_list}).
Ton rôle : les ORDONNER par préférence d'allocation et détecter ce que les
chiffres ne voient pas (catalyseurs, risques événementiels, earnings). Tu n'as
AUCUNE obligation de répartir les convictions : si les {n} dossiers sont
solides, dis-le ; si aucun ne l'est, dis-le aussi.

RÈGLES DE LA STRATÉGIE (contexte d'exécution) :
- Entrée : lundi à l'ouverture, au marché. Aucun timing d'entrée à recommander.
- Sortie : pilotée par 5 conditions absolues (trailing stop ATR, SMA200,
  momentum R², force sectorielle, macro). Hold moyen 29 jours, parfois 60-90.
- Un ticker n'est jamais vendu parce qu'un autre est devenu meilleur.

CONTEXTE MARCHÉ (données système) :
{macro_ctx}
{secteurs_ctx}
{concentration}

LES {n} CANDIDATS (données quantitatives + historique) :
{candidats}

INSTRUCTIONS WEB SEARCH :
1. Pour CHAQUE ticker : date du prochain earnings (obligatoire).
2. Actualités ≤ 14 jours uniquement. Chaque fait cité doit porter sa date.
   Hiérarchie : communiqué officiel > presse financière > analyse > forum
   (à ignorer). Ne jamais présenter une rumeur comme un fait.
3. Si tu ne trouves rien de fiable et daté sur un ticker : c'est INSUFFISANT,
   pas une raison d'inventer une narration.

FORMAT DE RÉPONSE OBLIGATOIRE :

LECTURE DU LOT : [2-3 phrases descriptives : quel thème porte ces {n} candidats,
quels événements de marché arrivent cette semaine (CPI, FOMC, saison des
earnings…), le lot repose-t-il sur un seul pari sectoriel ? Descriptif
uniquement — aucune recommandation d'exposition globale.]

CLASSEMENT : [les tickers séparés par des virgules, du préféré au moins
préféré, ex. {exemple}]

Puis pour chaque ticker, dans l'ordre du classement :

=== TICKER : {{ticker}} ===
CONVICTION : [FORT | MODÉRÉ | FAIBLE | INSUFFISANT]
RISQUE ÉVÉNEMENTIEL : [OUI — {{événement daté <10 jours de bourse}} | NON |
                       EARNINGS NON TROUVÉ]
RÉSUMÉ : [1-2 phrases]
POUR : [catalyseurs datés]
CONTRE : [minimum 2 risques concrets et datés — obligatoire même si FORT]
CONCLUSION : [1 phrase actionnable]

DÉFINITIONS (absolues — aucune répartition imposée) :
- FORT = catalyseur positif daté <14j ET pas d'earnings <10j de bourse ET
  aucun risque matériel identifié. Cinq FORT sont possibles si chacun remplit
  ces trois conditions ; zéro aussi.
- MODÉRÉ = dossier correct sans catalyseur daté, ou signaux mixtes.
- FAIBLE = au moins un risque matériel documenté et daté.
- INSUFFISANT = couverture d'information trop pauvre pour juger.
- Sois factuel et concis. Pas de disclaimers."""


def _format_macro_context(macro_regime) -> str:
    if not macro_regime:
        return "Régime macro : non disponible."
    parts = [f"{zone}: {'BULLISH' if bull else 'BEARISH'}"
             for zone, bull in macro_regime.items()]
    return "Régime macro par zone (indice vs SMA200) : " + " | ".join(parts)


def _load_secteurs_context(engine) -> str:
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT zone, secteur_yahoo, ROUND(ratio_vs_mm50::numeric, 3)
                FROM v_secteurs_en_force
                ORDER BY zone, ratio_vs_mm50 DESC
            """)).fetchall()
        if not rows:
            return "Secteurs en force relative : aucun."
        parts = [f"{r[0]}/{r[1]} ({r[2]})" for r in rows]
        return "Secteurs en force relative (ratio vs MM50) : " + ", ".join(parts)
    except Exception as e:
        print(f"⚠️ _load_secteurs_context : {e}")
        return "Secteurs en force relative : non disponibles."


def _format_concentration_lot(tickers_data) -> str:
    secteurs = {}
    for td in tickers_data:
        s = td.get("secteur") or "Inconnu"
        secteurs[s] = secteurs.get(s, 0) + 1
    n = len(tickers_data)
    detail = ", ".join(f"{v}/{n} {k}" for k, v in
                       sorted(secteurs.items(), key=lambda x: -x[1]))
    return f"Concentration du lot : {detail}."


def _load_dernier_avis(engine, ticker, semaine_courante) -> str:
    """Mémoire inter-semaines : dernier avis émis + rendement réalisé depuis."""
    try:
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT semaine, conviction, rendement_4s, rendement_2s, rendement_1s
                FROM avis_ia
                WHERE ticker = :t AND type_avis = 'ranking' AND semaine < :s
                ORDER BY semaine DESC LIMIT 1
            """), {"t": ticker.upper(), "s": semaine_courante}).fetchone()
        if not row:
            return ""
        rdt = next((r for r in (row[2], row[3], row[4]) if r is not None), None)
        rdt_txt = f", rendement réalisé depuis : {rdt:+.1%}" if rdt is not None else ""
        return f"Dernier avis émis : {row[1]} (semaine du {row[0]}{rdt_txt})"
    except Exception as e:
        print(f"⚠️ _load_dernier_avis({ticker}) : {e}")
        return ""


def _extract_batch_opinions(analyse_full: str, expected_tickers: list) -> dict:
    """
    Découpe la réponse batch v2.1 :
      - LECTURE DU LOT (entre 'LECTURE DU LOT' et 'CLASSEMENT')
      - CLASSEMENT (liste de tickers → classement_ia 1..n)
      - une section par ticker sur '=== TICKER : XXX ==='
    """
    out = {"lecture_lot": None, "classement_brut": None, "tickers": {}}

    m = re.search(r"LECTURE DU LOT[\W_]{0,10}(.*?)[\n\r]+[\W_]{0,5}CLASSEMENT",
                  analyse_full, re.DOTALL | re.IGNORECASE)
    if m:
        out["lecture_lot"] = m.group(1).strip()[:2000]

    classement = {}
    m = re.search(r"CLASSEMENT[\W_]{0,10}([^\n]+)", analyse_full, re.IGNORECASE)
    if m:
        out["classement_brut"] = m.group(1).strip()[:200]
        ordre = re.findall(r"[A-Z0-9][A-Z0-9.\-]{0,14}", m.group(1).upper())
        expected_up = [t.upper() for t in expected_tickers]
        rang_ia = 0
        for tk in ordre:
            if tk in expected_up and tk not in classement:
                rang_ia += 1
                classement[tk] = rang_ia

    sections = re.split(r"===\s*TICKER\s*:\s*", analyse_full)
    for sec in sections[1:]:
        m = re.match(r"([A-Z0-9.\-]+)\s*===", sec.strip())
        if not m:
            continue
        tk = m.group(1).upper()
        body = sec.strip()[m.end():].strip()
        out["tickers"][tk] = {
            "conviction":          _extract_conviction(body),
            "resume":              _extract_resume(body),
            "analyse":             body,
            "classement_ia":       classement.get(tk),
            "risque_evenementiel": _extract_risque_evenementiel(body),
        }

    # Fallback : si la ligne CLASSEMENT est absente ou inexploitable,
    # l'ordre d'apparition des sections fait foi (le prompt impose
    # « dans l'ordre du classement »).
    if not classement and out["tickers"]:
        for i, tk in enumerate(out["tickers"], start=1):
            out["tickers"][tk]["classement_ia"] = i
    return out

def generate_position_opinion(engine, position_data: dict, 
                               eval_data: dict, source: str = "manual") -> dict:
    """
    Génère un avis IA 'garder ou vendre' pour une position ouverte.
    
    Paramètres
    ----------
    position_data : dict — données position (ticker, prix_achat, quantite, date_achat, etc.)
    eval_data : dict — évaluation v4.1 (conditions, feux, P&L, prix_actuel, etc.)
    source : str — 'manual' (bouton dashboard) ou 'auto' (futur batch)
    """
    import traceback
    ticker = position_data.get("ticker", "?")
    print(f"🤖 [position] Début analyse IA pour {ticker}...")

    if not ANTHROPIC_API_KEY:
        print(f"❌ [position] ANTHROPIC_API_KEY non configurée")
        return {"error": "ANTHROPIC_API_KEY non configurée"}
    if engine is None:
        print(f"❌ [position] engine non connecté")
        return {"error": "engine non connecté"}

    try:
        _ensure_avis_ia_table(engine)

        # --- Construire le contexte position ---
        pos_context = _build_position_context(position_data, eval_data)
        print(f"   [position] Contexte construit, appel Claude en cours...")
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[
                {
                    "role": "user",
                    "content": _build_position_prompt(ticker, pos_context),
                }
            ],
        )

        analyse_full = ""
        for block in response.content:
            if block.type == "text":
                analyse_full += block.text

        tokens_used = response.usage.input_tokens + response.usage.output_tokens
        conviction = _extract_conviction(analyse_full)
        resume = _extract_resume(analyse_full)
        print(f"   [position] Réponse Claude reçue — {tokens_used} tokens, conviction={conviction}")

        # --- Persister avec source='position' et type_avis='position' ---
        today = date.today()
        semaine = str(today - timedelta(days=today.weekday()))

        _save_opinion(engine, ticker, semaine, rang=None,
                      conviction=conviction, analyse=analyse_full,
                      resume=resume, source=f"position_{source}",
                      model_used=MODEL, tokens_used=tokens_used,
                      quant_data={
                          "prix": eval_data.get("prix_actuel"),
                          "sma_200": eval_data.get("conditions", {}).get("sma", {}).get("sma_200"),
                          "atr_14": eval_data.get("conditions", {}).get("trailing_stop", {}).get("atr_14"),
                      },
                      type_avis="position")
        print(f"   ✅ [position] Avis {ticker} sauvegardé en base (semaine={semaine})")

        return {
            "status": "ok",
            "ticker": ticker,
            "conviction": conviction,
            "resume": resume,
            "analyse": analyse_full,
            "tokens_used": tokens_used,
            "model": MODEL,
        }
    except anthropic.APIError as e:
        print(f"❌ Erreur API Anthropic pour position {ticker} : {e}")
        traceback.print_exc()
        return {"error": f"Erreur API Anthropic : {e}"}
    except Exception as e:
        print(f"❌ Erreur generate_position_opinion({ticker}) : {e}")
        traceback.print_exc()
        return {"error": str(e)}


def _build_position_context(position_data: dict, eval_data: dict) -> str:
    """Formate le contexte d'une position ouverte pour le prompt IA."""
    lines = []
    
    ticker = position_data.get("ticker", "?")
    lines.append(f"Ticker : {ticker}")
    lines.append(f"Date d'achat : {position_data.get('date_achat', '?')}")
    lines.append(f"Prix d'achat : {position_data.get('prix_achat', '?')}€")
    lines.append(f"Quantité : {position_data.get('quantite', '?')}")
    lines.append(f"Montant investi : {position_data.get('montant_investi', '?')}€")
    lines.append(f"Jours de détention : {eval_data.get('jours_detention', '?')}")
    lines.append("")
    
    # P&L
    pnl_pct = eval_data.get("pnl_pct", 0)
    pnl_eur = eval_data.get("pnl_eur", 0)
    pnl_sign = "+" if pnl_pct >= 0 else ""
    lines.append(f"Prix actuel : {eval_data.get('prix_actuel', '?')}€")
    lines.append(f"P&L : {pnl_sign}{pnl_pct}% ({pnl_sign}{pnl_eur:.2f}€)")
    lines.append("")
    
    # Conditions de sortie v4.1
    lines.append("CONDITIONS DE SORTIE v4.1 :")
    conditions = eval_data.get("conditions", {})
    
    cond_labels = {
        "trailing_stop": "Trailing Stop ATR",
        "sma": "Prix vs SMA 200",
        "momentum": "Momentum R²",
        "secteur": "Secteur Force Relative",
        "macro": "Macro (indice zone)",
    }
    
    for key, label in cond_labels.items():
        cond = conditions.get(key, {})
        feu = cond.get("feu", "⚪")
        detail = cond.get("detail", "")
        lines.append(f"  {feu} {label} : {detail}")
    
    # Alerte globale
    alerte = eval_data.get("alerte_globale", "INCONNU")
    lines.append("")
    lines.append(f"ALERTE GLOBALE : {alerte}")
    
    violated = eval_data.get("violated", [])
    warnings = eval_data.get("warnings", [])
    if violated:
        lines.append(f"Conditions VIOLÉES : {', '.join(violated)}")
    if warnings:
        lines.append(f"Conditions en VIGILANCE : {', '.join(warnings)}")
    
    return "\n".join(lines)


def _build_position_prompt(ticker: str, pos_context: str) -> str:
    """Prompt spécifique pour l'analyse garder/vendre d'une position ouverte."""
    return f"""Tu es un analyste quantitatif senior. Tu dois produire un avis 
"GARDER ou VENDRE" pour une position ouverte, dans le cadre d'une stratégie momentum.

DONNÉES DE LA POSITION :
{pos_context}

CONTEXTE STRATÉGIE :
- Stratégie momentum v4.1 hybrid : entrée ranking cross-ticker, sortie sur conditions absolues
- 5 conditions de sortie : trailing stop ATR, prix < SMA200, momentum R² < 0, 
  secteur hors force, macro bearish. N'importe laquelle suffit pour vendre.
- Durée moyenne de détention cible : ~30 jours
- L'objectif n'est PAS de timer le top — c'est de rester tant que la tendance est intacte

INSTRUCTIONS :
1. Utilise le web search pour chercher les actualités récentes sur {ticker}
   (earnings, guidance, catalyseurs, risques, news sectorielles).
2. Croise les données quantitatives (feux v4.1) avec le contexte fondamental.
3. Donne un avis structuré en français.

FORMAT DE RÉPONSE (respecter exactement) :
CONVICTION : [GARDER / VENDRE / RENFORCER]
RÉSUMÉ : [1-2 phrases max — la conclusion actionnable]

ANALYSE :
- Situation technique : [état des 5 conditions v4.1, tendance prix]
- Catalyseurs / Risques : [actualités récentes trouvées via web search]
- Contexte position : [P&L, durée, cohérence avec la stratégie]
- Recommandation : [garder / vendre / renforcer, avec justification et éventuellement un seuil de sortie]
"""

# ============================================================
# LECTURE AVIS
# ============================================================

def get_opinions(engine, semaine: str = None, ticker: str = None, all: bool = False) -> dict:
    """
    Lit les avis IA depuis la base.
    - all=True               → tous les avis (toutes semaines)
    - semaine seule           → tous les avis de la semaine
    - ticker seul             → dernier avis pour ce ticker
    - les deux                → avis spécifique
    - aucun                   → dernière semaine disponible
    """
    if engine is None:
        return {"error": "engine non connecté"}

    _ensure_avis_ia_table(engine)

    # Colonnes à lire (v1.3 enrichi)
    cols = """ticker, semaine, rang, conviction, "analyse", resume,
              source, model_used, tokens_used, generated_at,
              score_composite, mom_r2, rvol, obv_slope,
              prix_emission, sma_200, atr_14, k_adaptatif,
              secteur_force, macro_bullish, prompt_version,
              prix_1s, prix_2s, prix_4s,
              rendement_1s, rendement_2s, rendement_4s,
              type_avis, classement_ia, risque_evenementiel"""

    try:
        with engine.connect() as conn:
            if all:
                rows = conn.execute(text(f"""
                    SELECT {cols} FROM avis_ia
                    ORDER BY semaine DESC, rang ASC NULLS LAST
                """)).fetchall()
            elif semaine and ticker:
                rows = conn.execute(text(f"""
                    SELECT {cols} FROM avis_ia
                    WHERE semaine = :semaine AND ticker = :ticker
                """), {"semaine": semaine, "ticker": ticker.upper()}).fetchall()
            elif semaine:
                rows = conn.execute(text(f"""
                    SELECT {cols} FROM avis_ia
                    WHERE semaine = :semaine
                    ORDER BY rang ASC NULLS LAST
                """), {"semaine": semaine}).fetchall()
            elif ticker:
                rows = conn.execute(text(f"""
                    SELECT {cols} FROM avis_ia
                    WHERE ticker = :ticker
                    ORDER BY semaine DESC
                    LIMIT 1
                """), {"ticker": ticker.upper()}).fetchall()
            else:
                rows = conn.execute(text(f"""
                    SELECT {cols} FROM avis_ia
                    WHERE semaine = (SELECT MAX(semaine) FROM avis_ia)
                    ORDER BY rang ASC NULLS LAST
                """)).fetchall()

        avis = []
        for r in rows:
            avis.append({
                "ticker":          r[0],
                "semaine":         str(r[1]),
                "rang":            r[2],
                "conviction":      r[3],
                "analyse":         r[4],
                "resume":          r[5],
                "source":          r[6],
                "model_used":      r[7],
                "tokens_used":     r[8],
                "generated_at":    str(r[9]) if r[9] else None,
                # Snapshot indicateurs
                "score_composite": r[10],
                "mom_r2":          r[11],
                "rvol":            r[12],
                "obv_slope":       r[13],
                "prix_emission":   r[14],
                "sma_200":         r[15],
                "atr_14":          r[16],
                "k_adaptatif":     r[17],
                "secteur_force":   r[18],
                "macro_bullish":   r[19],
                "prompt_version":  r[20],
                # Suivi rendements
                "prix_1s":         r[21],
                "prix_2s":         r[22],
                "prix_4s":         r[23],
                "rendement_1s":    r[24],
                "rendement_2s":    r[25],
                "rendement_4s":    r[26],
                # v1.3 : type d'avis
                "type_avis":       r[27],
                "classement_ia":        r[28],
                "risque_evenementiel":  r[29],
            })

        return {"nb_avis": len(avis), "avis": avis}

    except Exception as e:
        return {"error": str(e)}


# ============================================================
# SUIVI RENDEMENTS — Job scheduler
# ============================================================

def update_suivi_rendements(engine) -> dict:
    """
    Complète/révise les rendements +1s/+2s/+4s des avis 'ranking'.

    v1.5 — Corrige deux bugs (cf. REFONTE_AVIS_IA) :
      A. Les deux extrémités du rendement sont lues dans la MÊME série
         `prix_ajuste` (avant : émission=prix_ajuste, cible=prix_cloture →
         0,00 % exact sur les US sans dividende récent).
      B. La barre cible doit être STRICTEMENT postérieure à la barre
         d'émission ; sinon on ne calcule pas (avant : +1s mesuré le lundi
         W2 quand la dernière barre dispo était encore le vendredi W1 →
         horizon effectif nul, puis figé par un skip-si-non-NULL).
      + Le snapshot figé `prix_emission` n'est plus dénominateur (immunise
        contre les splits, cf. cas KLAC).
      + Révisable : tant qu'une barre postérieure à la cible n'existe pas,
        la valeur reste recalculable (plus de write-once définitif).
    """
    if engine is None:
        return {"error": "engine non connecté"}

    _ensure_avis_ia_table(engine)

    today = date.today()
    updated = {"1s": 0, "2s": 0, "4s": 0, "skipped": 0, "errors": 0}

    def _prix_ajuste_le(conn, ticker, d):
        """Dernière clôture ajustée à une date <= d (barre + sa date)."""
        r = conn.execute(text("""
            SELECT date, prix_ajuste FROM actions_prix_historique
            WHERE ticker = :t AND date <= :d AND prix_ajuste IS NOT NULL
              AND prix_ajuste <> 'NaN'::numeric
            ORDER BY date DESC LIMIT 1
        """), {"t": ticker, "d": d}).fetchone()
        return (r[0], float(r[1])) if r else (None, None)

    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT a.id, a.ticker, a.semaine, a.generated_at,
                       a.rendement_1s, a.rendement_2s, a.rendement_4s
                FROM avis_ia a
                WHERE (a.type_avis = 'ranking' OR a.type_avis IS NULL)
                  AND (a.rendement_1s IS NULL OR a.rendement_2s IS NULL
                       OR a.rendement_4s IS NULL)
                ORDER BY a.semaine ASC
            """)).fetchall()

        print(f"📊 Suivi rendements : {len(rows)} avis ranking à (re)vérifier")

        for row in rows:
            avis_id, ticker, semaine_raw, generated_at = row[0], row[1], row[2], row[3]
            try:
                semaine_date = date.fromisoformat(str(semaine_raw))
            except ValueError:
                print(f"  ⚠️ ID {avis_id} : semaine invalide '{semaine_raw}'")
                updated["errors"] += 1
                continue

            # Ancre d'émission = dernière barre <= date de génération réelle
            # (le prix d'émission = clôture du vendredi précédant le samedi 06h00).
            # Fallback : semaine + 6 jours si generated_at absent (avis anciens).
            gen_date = generated_at.date() if generated_at else (semaine_date + timedelta(days=6))
            with engine.connect() as conn:
                emission_date, prix_emission = _prix_ajuste_le(conn, ticker, gen_date)
            if emission_date is None or not prix_emission or prix_emission <= 0:
                updated["skipped"] += 1
                continue

            horizons = [
                ("1s", 7,  "prix_1s", "rendement_1s"),
                ("2s", 14, "prix_2s", "rendement_2s"),
                ("4s", 28, "prix_4s", "rendement_4s"),
            ]
            for label, days_offset, col_prix, col_rdt in horizons:
                target_date = emission_date + timedelta(days=days_offset)
                if target_date > today:
                    continue

                with engine.connect() as conn:
                    cible_date, prix_cible = _prix_ajuste_le(conn, ticker, target_date)

                # Bug B : la cible doit être STRICTEMENT après l'émission.
                # Sinon horizon effectif nul → on laisse NULL, révisé au prochain run.
                if cible_date is None or cible_date <= emission_date:
                    updated["skipped"] += 1
                    continue

                rendement = (prix_cible - prix_emission) / prix_emission
                with engine.begin() as conn:
                    conn.execute(text(f"""
                        UPDATE avis_ia SET {col_prix} = :prix, {col_rdt} = :rdt
                        WHERE id = :id
                    """), {"prix": prix_cible, "rdt": rendement, "id": avis_id})
                print(f"  ✅ {ticker} ({semaine_date}) +{label} : {rendement:+.2%}")
                updated[label] += 1

        print(f"📊 Suivi rendements terminé : {updated}")
        return {"status": "ok", "updated": updated, "total_checked": len(rows)}

    except Exception as e:
        print(f"❌ Erreur update_suivi_rendements : {e}")
        return {"error": str(e)}


# ============================================================
# FONCTIONS INTERNES
# ============================================================

def _build_quant_context(ticker: str, rang: int, quant_data: dict) -> str:
    """Formate les données quantitatives pour le prompt, avec résumé en langage naturel."""
    if not quant_data:
        return f"Ticker : {ticker}\nAucune donnée quantitative disponible."

    lines = [f"Ticker : {ticker}"]
    if rang:
        lines.append(f"Rang dans le ranking hebdomadaire : #{rang}")

    field_labels = {
        "score":     "Score composite",
        "mom_r2":    "Momentum R²",
        "rvol":      "RVOL (volume relatif)",
        "obv_slope": "OBV slope (accumulation)",
        "prix":      "Prix actuel",
        "sma_200":   "SMA 200",
        "atr_14":    "ATR 14",
        "k":         "K adaptatif (trailing stop)",
        "zone":      "Zone géographique",
        "secteur":   "Secteur",
    }

    for key, label in field_labels.items():
        val = quant_data.get(key)
        if val is not None:
            lines.append(f"{label} : {val}")

    # --- Résumé en langage naturel (feux) ---
    lines.append("")
    lines.append("DIAGNOSTIC RAPIDE :")

    prix = quant_data.get("prix")
    sma = quant_data.get("sma_200")
    mom = quant_data.get("mom_r2")
    rvol = quant_data.get("rvol")
    obv = quant_data.get("obv_slope")

    if prix and sma:
        pct_above = (prix - sma) / sma * 100
        if prix > sma:
            lines.append(f"  Tendance : 🟢 Prix {pct_above:+.1f}% au-dessus de la SMA200")
        else:
            lines.append(f"  Tendance : 🔴 Prix {pct_above:+.1f}% en dessous de la SMA200")

    if mom is not None:
        if mom > 0.3:
            lines.append(f"  Momentum : 🟢 Fort (R²={mom:.4f})")
        elif mom > 0:
            lines.append(f"  Momentum : 🟡 Positif mais modéré (R²={mom:.4f})")
        else:
            lines.append(f"  Momentum : 🔴 Négatif (R²={mom:.4f})")

    if rvol is not None:
        if rvol > 2.0:
            lines.append(f"  Volume : 🟢 Élevé (RVOL={rvol:.2f}x)")
        elif rvol > 1.0:
            lines.append(f"  Volume : 🟡 Normal (RVOL={rvol:.2f}x)")
        else:
            lines.append(f"  Volume : 🔴 Faible (RVOL={rvol:.2f}x)")

    if obv is not None:
        if obv > 0:
            lines.append(f"  Accumulation : 🟢 OBV en hausse (slope={obv:,.0f})")
        else:
            lines.append(f"  Accumulation : 🔴 OBV en baisse (slope={obv:,.0f})")

    if not any(v is not None for v in [mom, rvol, obv]):
        lines.append("  ⚠️ Indicateurs momentum non disponibles (ticker hors univers ranking)")

    return "\n".join(lines)


def _build_prompt(ticker: str, quant_context: str) -> str:
    """Prompt ad hoc mono-ticker v2.1 — tronc commun avec le batch :
    définitions absolues, bear case obligatoire, drapeau événementiel,
    web search contrainte. Sans classement (rien à comparer)."""
    return f"""Tu es membre d'un comité d'investissement momentum. Le système quantitatif
suit ce ticker ; ton rôle : détecter ce que les chiffres ne voient pas
(catalyseurs, risques événementiels, earnings) et qualifier le dossier.

RÈGLES DE LA STRATÉGIE (contexte d'exécution) :
- Entrée éventuelle : lundi à l'ouverture, au marché. Aucun timing à recommander.
- Sortie : pilotée par 5 conditions absolues (trailing stop ATR, SMA200,
  momentum R², force sectorielle, macro). Hold moyen 29 jours, parfois 60-90.

DONNÉES QUANTITATIVES DU SYSTÈME :
{quant_context}

INSTRUCTIONS WEB SEARCH :
1. Date du prochain earnings : obligatoire.
2. Actualités ≤ 14 jours uniquement. Chaque fait cité doit porter sa date.
   Hiérarchie : communiqué officiel > presse financière > analyse > forum
   (à ignorer). Ne jamais présenter une rumeur comme un fait.
3. Si tu ne trouves rien de fiable et daté : c'est INSUFFISANT,
   pas une raison d'inventer une narration.

FORMAT DE RÉPONSE OBLIGATOIRE :

CONVICTION : [FORT | MODÉRÉ | FAIBLE | INSUFFISANT]
RISQUE ÉVÉNEMENTIEL : [OUI — {{événement daté <10 jours de bourse}} | NON |
                       EARNINGS NON TROUVÉ]
RÉSUMÉ : [1-2 phrases]
POUR : [catalyseurs datés]
CONTRE : [minimum 2 risques concrets et datés — obligatoire même si FORT]
CONCLUSION : [1 phrase actionnable]

DÉFINITIONS (absolues) :
- FORT = catalyseur positif daté <14j ET pas d'earnings <10j de bourse ET
  aucun risque matériel identifié.
- MODÉRÉ = dossier correct sans catalyseur daté, ou signaux mixtes.
- FAIBLE = au moins un risque matériel documenté et daté.
- INSUFFISANT = couverture d'information trop pauvre pour juger.
- Sois factuel et concis. Pas de disclaimers."""


def _extract_conviction(analyse: str) -> str:
    """Extrait le niveau de conviction depuis la réponse.
    Supporte les deux formats :
      - Ranking : FORT / MODÉRÉ / FAIBLE
      - Position : GARDER / VENDRE / RENFORCER
    Tolère les variantes Markdown (**FORT**, ## CONVICTION, ###),
    les emojis intercalés (🟢, 🟡, 🔴) et les espaces multiples
    entre 'CONVICTION' et le mot-clé.
    """
    analyse_upper = analyse.upper()

    # Regex : "CONVICTION" suivi (jusqu'à 30 caractères de tolérance) du mot-clé.
    # \W couvre la ponctuation, les espaces, les emojis et les ** markdown.
    # Position d'abord (vocabulaire spécifique, moins ambigu que ranking).
    patterns_position = [
        (r"CONVICTION[\W_]{0,30}?GARDER",    "GARDER"),
        (r"CONVICTION[\W_]{0,30}?VENDRE",    "VENDRE"),
        (r"CONVICTION[\W_]{0,30}?RENFORCER", "RENFORCER"),
    ]
    patterns_ranking = [
        (r"CONVICTION[\W_]{0,30}?INSUFFISANT", "INSUFFISANT"),
        (r"CONVICTION[\W_]{0,30}?FORT",      "FORT"),
        (r"CONVICTION[\W_]{0,30}?FAIBLE",    "FAIBLE"),
        (r"CONVICTION[\W_]{0,30}?MODÉRÉ",    "MODÉRÉ"),
        (r"CONVICTION[\W_]{0,30}?MODERE",    "MODÉRÉ"),  # variante sans accent
    ]

    for pattern, label in patterns_position:
        if re.search(pattern, analyse_upper):
            return label
    for pattern, label in patterns_ranking:
        if re.search(pattern, analyse_upper):
            return label

    # Fallback explicite — log pour repérer les analyses non parsables
    print(f"⚠️ _extract_conviction : aucun pattern matché, fallback MODÉRÉ. "
          f"Début analyse : {analyse[:200]!r}")
    return "MODÉRÉ"

def _extract_risque_evenementiel(analyse: str):
    """Drapeau RISQUE ÉVÉNEMENTIEL v2.1 → True (OUI) / False (NON) /
    None (EARNINGS NON TROUVÉ ou absent)."""
    m = re.search(r"RISQUE\s+ÉVÉNEMENTIEL[\W_]{0,10}(OUI|NON|EARNINGS NON TROUVÉ)",
                  analyse, re.IGNORECASE)
    if not m:
        return None
    val = m.group(1).upper()
    return True if val == "OUI" else (False if val == "NON" else None)


def _extract_resume(analyse: str) -> str:
    """Extrait le résumé depuis la réponse."""
    for marker in ["RÉSUMÉ :", "RÉSUMÉ:", "RESUME :", "RESUME:"]:
        if marker in analyse:
            after = analyse.split(marker, 1)[1]
            lines = after.strip().split("\n")
            resume_lines = []
            for line in lines:
                stripped = line.strip()
                if (stripped.startswith("ANALYSE") or stripped.startswith("**")
                        or stripped.startswith("POUR") or stripped.startswith("CONTRE")
                        or stripped.startswith("CONCLUSION")):
                    break
                if stripped:
                    resume_lines.append(stripped)
            if resume_lines:
                return " ".join(resume_lines)[:500]
    return ""


def _save_opinion(engine, ticker, semaine, rang, conviction,
                  analyse, resume, source, model_used, tokens_used,
                  quant_data=None, type_avis="ranking",
                  classement_ia=None, risque_evenementiel=None,
                  generated_at=None):
    """Persiste l'avis en base (UPSERT sur semaine+ticker) avec snapshot indicateurs."""
    _ensure_avis_ia_table(engine)
    # Nettoyage caractères nuls (incompatibles PostgreSQL)
    if analyse:
        analyse = analyse.replace("\x00", "")
    if resume:
        resume = resume.replace("\x00", "")

    # Extraire le snapshot des indicateurs depuis quant_data
    qd = quant_data or {}

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO avis_ia
                (ticker, semaine, rang, conviction, "analyse", resume,
                 source, model_used, tokens_used, generated_at,
                 score_composite, mom_r2, rvol, obv_slope,
                 prix_emission, sma_200, atr_14, k_adaptatif,
                 secteur_force, macro_bullish, prompt_version, type_avis,
                 classement_ia, risque_evenementiel)
            VALUES
                (:ticker, :semaine, :rang, :conviction, :analyse, :resume,
                 :source, :model_used, :tokens_used, COALESCE(:generated_at, NOW()),
                 :score_composite, :mom_r2, :rvol, :obv_slope,
                 :prix_emission, :sma_200, :atr_14, :k_adaptatif,
                 :secteur_force, :macro_bullish, :prompt_version, :type_avis,
                 :classement_ia, :risque_evenementiel)
            ON CONFLICT (semaine, ticker)
            DO UPDATE SET
                rang             = EXCLUDED.rang,
                conviction       = EXCLUDED.conviction,
                "analyse"        = EXCLUDED."analyse",
                resume           = EXCLUDED.resume,
                source           = EXCLUDED.source,
                model_used       = EXCLUDED.model_used,
                tokens_used      = EXCLUDED.tokens_used,
                generated_at     = EXCLUDED.generated_at,
                score_composite  = EXCLUDED.score_composite,
                mom_r2           = EXCLUDED.mom_r2,
                rvol             = EXCLUDED.rvol,
                obv_slope        = EXCLUDED.obv_slope,
                prix_emission    = EXCLUDED.prix_emission,
                sma_200          = EXCLUDED.sma_200,
                atr_14           = EXCLUDED.atr_14,
                k_adaptatif      = EXCLUDED.k_adaptatif,
                secteur_force    = EXCLUDED.secteur_force,
                macro_bullish    = EXCLUDED.macro_bullish,
                prompt_version   = EXCLUDED.prompt_version,
                type_avis        = EXCLUDED.type_avis,
                classement_ia        = EXCLUDED.classement_ia,
                risque_evenementiel  = EXCLUDED.risque_evenementiel
        """), {
            "ticker":          ticker.upper(),
            "semaine":         semaine,
            "rang":            rang,
            "conviction":      conviction,
            "analyse":         analyse,
            "resume":          resume,
            "source":          source,
            "model_used":      model_used,
            "tokens_used":     tokens_used,
            "score_composite": qd.get("score"),
            "mom_r2":          qd.get("mom_r2"),
            "rvol":            qd.get("rvol"),
            "obv_slope":       qd.get("obv_slope"),
            "prix_emission":   qd.get("prix"),
            "sma_200":         qd.get("sma_200"),
            "atr_14":          qd.get("atr_14"),
            "k_adaptatif":     qd.get("k"),
            "secteur_force":   None,  # TODO: passer l'info depuis le ranking
            "macro_bullish":   None,  # TODO: passer l'info depuis le ranking
            "prompt_version":  PROMPT_VERSION,
            "type_avis":       type_avis,
            "classement_ia":       classement_ia,
            "risque_evenementiel": risque_evenementiel,
            "generated_at":        generated_at,
        })
