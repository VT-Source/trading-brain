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
# v1.3 — Séparation ranking vs position :
#   - Nouvelle colonne type_avis ('ranking' | 'position')
#   - update_suivi_rendements ne traite que les avis 'ranking'
#     (les avis position se mesurent via la P&L réelle des positions)
#   - generate_opinion : type_avis='ranking' par défaut
#   - generate_position_opinion : force type_avis='position'
# ============================================================

import os
import re
import anthropic
from datetime import date, timedelta
from sqlalchemy import text


ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = "claude-sonnet-4-6"
PROMPT_VERSION = "v1.1"

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
                      quant_data, type_avis=type_avis)

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
            type_avis="ranking",
        )
        results.append(result)

        if result.get("error"):
            print(f"   ⚠️ {td['ticker']} : {result['error']}")
        else:
            print(f"   ✅ {td['ticker']} → {result['conviction']} ({result['tokens_used']} tokens)")

    return results

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
              type_avis"""

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
    - Filtre uniquement type_avis='ranking' (les avis 'position' se mesurent
      via la P&L réelle des positions, pas via un rendement théorique)
    - Pour chaque horizon (1s, 2s, 4s), vérifie si la date cible est atteinte
    - Récupère le prix de clôture le plus proche (≤ target_date) depuis
      actions_prix_historique (gère week-ends et jours fériés)
    - Calcule et stocke le rendement

    Appelé par le scheduler chaque jour ouvré à 23h45 (ou samedi selon config).
    """
    if engine is None:
        return {"error": "engine non connecté"}

    _ensure_avis_ia_table(engine)

    today = date.today()
    updated = {"1s": 0, "2s": 0, "4s": 0, "errors": 0}

    try:
        with engine.connect() as conn:
            # v1.3 : ne traiter que les avis 'ranking'
            # (les avis 'position' = NULL hérité historique sont aussi inclus
            #  pour ne rien casser sur les avis pré-migration)
            rows = conn.execute(text("""
                SELECT id, ticker, semaine, prix_emission
                FROM avis_ia
                WHERE prix_emission IS NOT NULL
                  AND (type_avis = 'ranking' OR type_avis IS NULL)
                  AND (rendement_1s IS NULL OR rendement_2s IS NULL OR rendement_4s IS NULL)
                ORDER BY semaine ASC
            """)).fetchall()

        print(f"📊 Suivi rendements : {len(rows)} avis ranking à vérifier")

        for row in rows:
            avis_id = row[0]
            ticker = row[1]
            semaine_str = str(row[2])
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
                  quant_data=None, type_avis="ranking"):
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
                 secteur_force, macro_bullish, prompt_version, type_avis)
            VALUES
                (:ticker, :semaine, :rang, :conviction, :analyse, :resume,
                 :source, :model_used, :tokens_used, NOW(),
                 :score_composite, :mom_r2, :rvol, :obv_slope,
                 :prix_emission, :sma_200, :atr_14, :k_adaptatif,
                 :secteur_force, :macro_bullish, :prompt_version, :type_avis)
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
                prompt_version   = EXCLUDED.prompt_version,
                type_avis        = EXCLUDED.type_avis
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
        })
