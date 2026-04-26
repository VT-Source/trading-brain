# ============================================================
# ai_opinion.py — Avis IA par ticker (Claude Sonnet + web search)
# Trading Brain | VT-Source | v1.2
# ============================================================
# Reçoit `engine` en paramètre (injection de dépendance).
# Appelé par main.py (endpoints + scheduler).
#
# v1.2 — Tracking performance IA :
#   - Snapshot indicateurs au moment de l'avis
#   - Colonnes rendement +1s/+2s/+4s (remplies par job scheduler)
#   - Traçabilité prompt_version
#   - Fonction update_suivi_rendements() pour compléter les rendements
# ============================================================

import os
import anthropic
from datetime import date, timedelta
from sqlalchemy import text


ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = "claude-sonnet-4-6"
PROMPT_VERSION = "v1.0"

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
                conviction      VARCHAR(10)   NOT NULL DEFAULT 'MODÉRÉ',
                "analyse"       TEXT,
                resume          VARCHAR(500),
                source          VARCHAR(10)   DEFAULT 'manual',
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
    """Ajoute les colonnes v1.2 si elles n'existent pas (migration safe)."""
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
        ("prix_1s",         "DOUBLE PRECISION"),
        ("prix_2s",         "DOUBLE PRECISION"),
        ("prix_4s",         "DOUBLE PRECISION"),
        ("rendement_1s",    "DOUBLE PRECISION"),
        ("rendement_2s",    "DOUBLE PRECISION"),
        ("rendement_4s",    "DOUBLE PRECISION"),
    ]
    with engine.begin() as conn:
        for col_name, col_type in new_columns:
            try:
                conn.execute(text(
                    f'ALTER TABLE avis_ia ADD COLUMN {col_name} {col_type}'
                ))
                print(f"  ✅ Colonne {col_name} ajoutée")
            except Exception:
                # Colonne existe déjà — OK
                pass
    print("✅ Migration colonnes avis_ia v1.2 terminée")


# ============================================================
# GÉNÉRATION D'AVIS
# ============================================================

def generate_opinion(engine, ticker: str, semaine: str, rang: int = None,
                     quant_data: dict = None, source: str = "manual") -> dict:
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
            max_tokens=2000,
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

        # --- Parser conviction et résumé ---
        conviction = _extract_conviction(analyse_full)
        resume = _extract_resume(analyse_full)

        # --- Persister en base (UPSERT) avec snapshot ---
        _save_opinion(engine, ticker, semaine, rang, conviction,
                      analyse_full, resume, source, MODEL, tokens_used,
                      quant_data)

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


def generate_opinions_batch(engine, tickers_data: list, semaine: str,
                            source: str = "auto") -> list:
    """
    Génère les avis IA pour plusieurs tickers (top 5 du ranking).

    Paramètres
    ----------
    tickers_data : list of dict — chaque dict contient ticker, rang, et données quant
    semaine : str — lundi de la semaine
    source : str — 'auto' pour le scheduler

    Retourne
    --------
    list de résultats (un par ticker)
    """
    results = []
    for td in tickers_data:
        print(f"🤖 Génération avis IA : {td['ticker']} (#{td.get('rang', '?')})...")
        result = generate_opinion(
            engine=engine,
            ticker=td["ticker"],
            semaine=semaine,
            rang=td.get("rang"),
            quant_data=td,
            source=source,
        )
        results.append(result)

        if result.get("error"):
            print(f"   ⚠️ {td['ticker']} : {result['error']}")
        else:
            print(f"   ✅ {td['ticker']} → {result['conviction']} ({result['tokens_used']} tokens)")

    return results


# ============================================================
# LECTURE AVIS
# ============================================================

def get_opinions(engine, semaine: str = None, ticker: str = None) -> dict:
    """
    Lit les avis IA depuis la base.
    - semaine seule → tous les avis de la semaine
    - ticker seul → dernier avis pour ce ticker
    - les deux → avis spécifique
    - aucun → dernière semaine disponible
    """
    if engine is None:
        return {"error": "engine non connecté"}

    _ensure_avis_ia_table(engine)

    # Colonnes à lire (v1.2 enrichi)
    cols = """ticker, semaine, rang, conviction, "analyse", resume,
              source, model_used, tokens_used, generated_at,
              score_composite, mom_r2, rvol, obv_slope,
              prix_emission, sma_200, atr_14, k_adaptatif,
              secteur_force, macro_bullish, prompt_version,
              prix_1s, prix_2s, prix_4s,
              rendement_1s, rendement_2s, rendement_4s"""

    try:
        with engine.connect() as conn:
            if semaine and ticker:
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
            })

        return {"nb_avis": len(avis), "avis": avis}

    except Exception as e:
        return {"error": str(e)}


# ============================================================
# SUIVI RENDEMENTS — Job scheduler
# ============================================================

def update_suivi_rendements(engine) -> dict:
    """
    Complète les rendements à +1s, +2s, +4s pour les avis passés.

    Logique :
    - Cherche les avis qui ont un prix_emission mais des rendements manquants
    - Pour chaque horizon (1s, 2s, 4s), vérifie si la date cible est atteinte
    - Récupère le prix de clôture à la date cible depuis actions_prix_historique
    - Calcule et stocke le rendement

    Appelé par le scheduler chaque lundi après le sync prix.
    """
    if engine is None:
        return {"error": "engine non connecté"}

    _ensure_avis_ia_table(engine)

    today = date.today()
    updated = {"1s": 0, "2s": 0, "4s": 0, "errors": 0}

    try:
        with engine.connect() as conn:
            # Récupérer tous les avis avec prix_emission mais au moins un rendement manquant
            rows = conn.execute(text("""
                SELECT id, ticker, semaine, prix_emission
                FROM avis_ia
                WHERE prix_emission IS NOT NULL
                  AND (rendement_1s IS NULL OR rendement_2s IS NULL OR rendement_4s IS NULL)
                ORDER BY semaine ASC
            """)).fetchall()

        print(f"📊 Suivi rendements : {len(rows)} avis à vérifier")

        for row in rows:
            avis_id = row[0]
            ticker = row[1]
            semaine_str = row[2]
            prix_emission = row[3]

            try:
                semaine_date = date.fromisoformat(semaine_str)
            except ValueError:
                print(f"  ⚠️ ID {avis_id} : semaine invalide '{semaine_str}'")
                updated["errors"] += 1
                continue

            # Pour chaque horizon, vérifier et compléter
            horizons = [
                ("1s", 7,  "prix_1s", "rendement_1s"),
                ("2s", 14, "prix_2s", "rendement_2s"),
                ("4s", 28, "prix_4s", "rendement_4s"),
            ]

            for label, days_offset, col_prix, col_rdt in horizons:
                target_date = semaine_date + timedelta(days=days_offset)

                # Pas encore atteint ? On skip
                if target_date > today:
                    continue

                # Déjà rempli ? On skip
                with engine.connect() as conn:
                    existing = conn.execute(text(f"""
                        SELECT {col_rdt} FROM avis_ia WHERE id = :id
                    """), {"id": avis_id}).fetchone()
                    if existing and existing[0] is not None:
                        continue

                # Chercher le prix de clôture le plus proche de target_date
                # (on prend le dernier jour de bourse <= target_date)
                with engine.connect() as conn:
                    prix_row = conn.execute(text("""
                        SELECT prix_cloture FROM actions_prix_historique
                        WHERE ticker = :ticker
                          AND date <= :target_date
                        ORDER BY date DESC
                        LIMIT 1
                    """), {"ticker": ticker, "target_date": target_date}).fetchone()

                if prix_row and prix_row[0] and prix_emission > 0:
                    prix_cible = float(prix_row[0])
                    rendement = (prix_cible - prix_emission) / prix_emission

                    with engine.begin() as conn:
                        conn.execute(text(f"""
                            UPDATE avis_ia
                            SET {col_prix} = :prix, {col_rdt} = :rdt
                            WHERE id = :id
                        """), {"prix": prix_cible, "rdt": rendement, "id": avis_id})

                    print(f"  ✅ {ticker} ({semaine_str}) +{label} : {rendement:+.2%}")
                    updated[label] += 1
                else:
                    # Pas de donnée prix — peut arriver si sync pas encore fait
                    pass

        print(f"📊 Suivi rendements terminé : {updated}")
        return {"status": "ok", "updated": updated, "total_checked": len(rows)}

    except Exception as e:
        print(f"❌ Erreur update_suivi_rendements : {e}")
        return {"error": str(e)}


# ============================================================
# FONCTIONS INTERNES
# ============================================================

def _build_quant_context(ticker: str, rang: int, quant_data: dict) -> str:
    """Formate les données quantitatives pour le prompt."""
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

    return "\n".join(lines)


def _build_prompt(ticker: str, quant_context: str) -> str:
    """Construit le prompt complet pour Claude Sonnet."""
    return f"""Tu es un analyste quantitatif senior. Tu dois produire un avis d'investissement
court terme (1-4 semaines) pour l'action suivante, dans le cadre d'une stratégie momentum.

DONNÉES QUANTITATIVES DU SYSTÈME :
{quant_context}

INSTRUCTIONS :
1. Utilise le web search pour chercher les actualités récentes sur ce ticker
   (earnings, guidance, catalyseurs, risques, actualités sectorielles).
2. Croise les données quantitatives avec le contexte fondamental trouvé.
3. Produis une analyse structurée en français.

FORMAT DE RÉPONSE OBLIGATOIRE :

CONVICTION : [FORT | MODÉRÉ | FAIBLE]

RÉSUMÉ : [1-2 phrases synthétiques justifiant la conviction]

ANALYSE :
- **Momentum technique** : [interprétation des signaux quantitatifs]
- **Contexte fondamental** : [actualités récentes, earnings, catalyseurs]
- **Risques identifiés** : [ce qui pourrait invalider la thèse]
- **Conclusion** : [recommandation synthétique pour la stratégie momentum]

IMPORTANT :
- FORT = les signaux techniques ET fondamentaux convergent positivement
- MODÉRÉ = signaux mixtes ou manque de catalyseur clair
- FAIBLE = divergence entre technique et fondamental, ou risques élevés
- Sois factuel et concis. Pas de disclaimers juridiques."""


def _extract_conviction(analyse: str) -> str:
    """Extrait le niveau de conviction depuis la réponse."""
    analyse_upper = analyse.upper()
    if "CONVICTION : FORT" in analyse_upper or "CONVICTION: FORT" in analyse_upper:
        return "FORT"
    elif "CONVICTION : FAIBLE" in analyse_upper or "CONVICTION: FAIBLE" in analyse_upper:
        return "FAIBLE"
    else:
        return "MODÉRÉ"


def _extract_resume(analyse: str) -> str:
    """Extrait le résumé depuis la réponse."""
    for marker in ["RÉSUMÉ :", "RÉSUMÉ:", "RESUME :", "RESUME:"]:
        if marker in analyse:
            after = analyse.split(marker, 1)[1]
            lines = after.strip().split("\n")
            resume_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("ANALYSE") or stripped.startswith("**"):
                    break
                if stripped:
                    resume_lines.append(stripped)
            if resume_lines:
                return " ".join(resume_lines)[:500]
    return ""


def _save_opinion(engine, ticker, semaine, rang, conviction,
                  analyse, resume, source, model_used, tokens_used,
                  quant_data=None):
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
                 secteur_force, macro_bullish, prompt_version)
            VALUES
                (:ticker, :semaine, :rang, :conviction, :analyse, :resume,
                 :source, :model_used, :tokens_used, NOW(),
                 :score_composite, :mom_r2, :rvol, :obv_slope,
                 :prix_emission, :sma_200, :atr_14, :k_adaptatif,
                 :secteur_force, :macro_bullish, :prompt_version)
            ON CONFLICT (semaine, ticker)
            DO UPDATE SET
                rang             = EXCLUDED.rang,
                conviction       = EXCLUDED.conviction,
                "analyse"        = EXCLUDED."analyse",
                resume           = EXCLUDED.resume,
                source           = EXCLUDED.source,
                model_used       = EXCLUDED.model_used,
                tokens_used      = EXCLUDED.tokens_used,
                generated_at     = NOW(),
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
                prompt_version   = EXCLUDED.prompt_version
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
        })
