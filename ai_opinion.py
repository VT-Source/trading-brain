# ============================================================
# ai_opinion.py — Avis IA par ticker (Claude Sonnet + web search)
# Trading Brain | VT-Source
# ============================================================
# Reçoit `engine` en paramètre (injection de dépendance).
# Appelé par main.py (endpoints + scheduler).
# ============================================================

import os
import anthropic
from datetime import date
from sqlalchemy import text


ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = "claude-sonnet-4-6"


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

        # --- Persister en base (UPSERT) ---
        _save_opinion(engine, ticker, semaine, rang, conviction,
                      analyse_full, resume, source, MODEL, tokens_used)

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

    try:
        with engine.connect() as conn:
            if semaine and ticker:
                rows = conn.execute(text("""
                    SELECT ticker, semaine, rang, conviction, "analyse", resume,
                           source, model_used, tokens_used, generated_at
                    FROM avis_ia
                    WHERE semaine = :semaine AND ticker = :ticker
                """), {"semaine": semaine, "ticker": ticker.upper()}).fetchall()
            elif semaine:
                rows = conn.execute(text("""
                    SELECT ticker, semaine, rang, conviction, "analyse", resume,
                           source, model_used, tokens_used, generated_at
                    FROM avis_ia
                    WHERE semaine = :semaine
                    ORDER BY rang ASC NULLS LAST
                """), {"semaine": semaine}).fetchall()
            elif ticker:
                rows = conn.execute(text("""
                    SELECT ticker, semaine, rang, conviction, "analyse", resume,
                           source, model_used, tokens_used, generated_at
                    FROM avis_ia
                    WHERE ticker = :ticker
                    ORDER BY semaine DESC
                    LIMIT 1
                """), {"ticker": ticker.upper()}).fetchall()
            else:
                rows = conn.execute(text("""
                    SELECT ticker, semaine, rang, conviction, "analyse", resume,
                           source, model_used, tokens_used, generated_at
                    FROM avis_ia
                    WHERE semaine = (SELECT MAX(semaine) FROM avis_ia)
                    ORDER BY rang ASC NULLS LAST
                """)).fetchall()

        avis = [
            {
                "ticker":       r[0],
                "semaine":      str(r[1]),
                "rang":         r[2],
                "conviction":   r[3],
                "analyse":      r[4],
                "resume":       r[5],
                "source":       r[6],
                "model_used":   r[7],
                "tokens_used":  r[8],
                "generated_at": str(r[9]),
            }
            for r in rows
        ]

        return {"nb_avis": len(avis), "avis": avis}

    except Exception as e:
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
                  analyse, resume, source, model_used, tokens_used):
    """Persiste l'avis en base (UPSERT sur semaine+ticker)."""
    # Nettoyage caractères nuls (incompatibles PostgreSQL)
    if analyse:
        analyse = analyse.replace("\x00", "")
    if resume:
        resume = resume.replace("\x00", "")
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO avis_ia
                (ticker, semaine, rang, conviction, "analyse", resume,
                 source, model_used, tokens_used, generated_at)
            VALUES
                (:ticker, :semaine, :rang, :conviction, :analyse, :resume,
                 :source, :model_used, :tokens_used, NOW())
            ON CONFLICT (semaine, ticker)
            DO UPDATE SET
                rang         = EXCLUDED.rang,
                conviction   = EXCLUDED.conviction,
                "analyse"    = EXCLUDED."analyse",
                resume       = EXCLUDED.resume,
                source       = EXCLUDED.source,
                model_used   = EXCLUDED.model_used,
                tokens_used  = EXCLUDED.tokens_used,
                generated_at = NOW()
        """), {
            "ticker":      ticker.upper(),
            "semaine":     semaine,
            "rang":        rang,
            "conviction":  conviction,
            "analyse":     analyse,
            "resume":      resume,
            "source":      source,
            "model_used":  model_used,
            "tokens_used": tokens_used,
        })
