# ============================================================
# dashboard.py — Trading Brain Dashboard v4.0
# VT-Source/trading-brain
# ============================================================
# Streamlit dashboard pour visualiser :
#   - Ranking hebdomadaire live (signaux v4.1)
#   - Régime macro & force sectorielle
#   - Résultats backtest historiques
#
# Lancement local  : streamlit run dashboard.py
# Lancement Railway: voir Procfile / commande de déploiement
# ============================================================

import os
import streamlit as st
import pandas as pd
import requests
from datetime import date, datetime

# ============================================================
# CONFIG
# ============================================================

API_BASE = os.getenv("API_BASE_URL", "https://trading-brain-production-f082.up.railway.app")

st.set_page_config(
    page_title="Trading Brain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# AUTH — Protection par mot de passe
# ============================================================

DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "")

def check_password() -> bool:
    """Gate d'authentification simple."""
    if not DASHBOARD_PASSWORD:
        # Pas de mot de passe configuré → accès libre (dev local)
        return True

    if st.session_state.get("authenticated"):
        return True

    st.markdown("# 🧠 Trading Brain")
    st.markdown("### 🔒 Accès protégé")
    password = st.text_input("Mot de passe", type="password", key="pw_input")

    if st.button("Se connecter", use_container_width=True):
        if password == DASHBOARD_PASSWORD:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Mot de passe incorrect.")

    st.stop()

check_password()

# ============================================================
# STYLE
# ============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=DM+Sans:wght@400;500;700&display=swap');

    /* --- Global --- */
    .stApp {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'JetBrains Mono', monospace !important;
        letter-spacing: -0.02em;
    }

    /* --- Metric cards --- */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-card .value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #e94560;
    }
    .metric-card .label {
        font-size: 0.85rem;
        color: #8892b0;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* --- Ranking table --- */
    .ranking-row {
        border-left: 3px solid transparent;
        padding: 8px 12px;
        margin: 2px 0;
        border-radius: 6px;
        transition: all 0.2s ease;
    }
    .ranking-row:hover {
        background: rgba(233, 69, 96, 0.05);
        border-left-color: #e94560;
    }
    .rank-badge {
        display: inline-block;
        width: 28px;
        height: 28px;
        line-height: 28px;
        text-align: center;
        border-radius: 50%;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .rank-1 { background: #ffd700; color: #1a1a2e; }
    .rank-2 { background: #c0c0c0; color: #1a1a2e; }
    .rank-3 { background: #cd7f32; color: #1a1a2e; }
    .rank-other { background: #2a2a4a; color: #8892b0; }

    /* --- Status pills --- */
    .pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    .pill-bull { background: #0d4b2e; color: #34d399; }
    .pill-bear { background: #4b0d1a; color: #f87171; }
    .pill-force { background: #0d3b4b; color: #38bdf8; }
    .pill-hors { background: #3b3b0d; color: #a3a322; }

    /* --- Future placeholders --- */
    .placeholder-box {
        border: 2px dashed #333;
        border-radius: 10px;
        padding: 24px;
        text-align: center;
        color: #555;
        font-style: italic;
    }

    /* --- Sidebar --- */
/* --- Sidebar --- */
    section[data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid #1a1a2e;
    }
section[data-testid="stSidebar"] *:not(button *):not(button) {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] button p,
    section[data-testid="stSidebar"] button span,
    section[data-testid="stSidebar"] button {
        color: #0d1117 !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        color: #e94560 !important;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] small {
        color: #8892b0 !important;
    }
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# API HELPERS
# ============================================================

@st.cache_data(ttl=300)  # cache 5 min
def api_get(endpoint: str, params: dict = None) -> dict | None:
    """Appel GET à l'API Trading Brain avec cache."""
    try:
        r = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        st.error(f"Timeout sur {endpoint} — le calcul prend plus de 2 min.")
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"Connexion impossible à l'API ({API_BASE})")
        return None
    except Exception as e:
        st.error(f"Erreur API {endpoint}: {e}")
        return None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("# 🧠 Trading Brain")
    st.caption(f"v4.1 hybrid • {date.today().strftime('%d %b %Y')}")
    st.divider()

    page = st.radio(
        "Navigation",
        ["📊 Ranking Hebdo", "🌍 Macro & Secteurs", "💼 Portefeuille", "📈 Backtest & Analyse", "📋 Décisions", "⚙️ Système"],
        label_visibility="collapsed",
    )

    st.divider()

    # --- Refresh button ---
    if st.button("🔄 Rafraîchir les données", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption("Trading Brain © 2026")
    st.caption("Les signaux sont des suggestions.")
    st.caption("L'humain décide.")


# ============================================================
# PAGE 1 — RANKING HEBDO
# ============================================================

if page == "📊 Ranking Hebdo":

    st.markdown("# 📊 Ranking Hebdomadaire")
    st.caption("Top candidats du momentum ranking v4.1 — calculé chaque lundi sur ~400 tickers")

    # --- Fetch ranking ---
    with st.spinner("Calcul du ranking en cours... (~30s sur 400 tickers)"):
        data = api_get("/ranking-live")

    if data and "ranking" in data:
        meta = data.get("meta", {})
        ranking = data["ranking"]
        macro_regime = data.get("macro_regime", {})

        # --- Metrics row ---
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tickers éligibles", meta.get("nb_eligible", "—"))
        with col2:
            st.metric("Tickers scorés", meta.get("nb_total_tickers", "—"))
        with col3:
            zones = []
            for zone, bull in macro_regime.items():
                zones.append(f"{zone} {'🟢' if bull else '🔴'}")
            st.metric("Macro", " / ".join(zones) if zones else "—")
        with col4:
            st.metric("Date données", meta.get("data_date", "—"))

        st.divider()

        # --- Ranking table ---
        if ranking:
            df_rank = pd.DataFrame(ranking)

            # Display columns
            display_cols = {
                "rank":      "#",
                "ticker":    "Ticker",
                "score":     "Score",
                "mom_r2":    "Mom R²",
                "rvol":      "RVOL",
                "obv_slope": "OBV Slope",
                "prix":      "Prix",
                "sma_200":   "SMA 200",
                "atr_14":    "ATR 14",
                "k":         "K adapt.",
                "zone":      "Zone",
                "secteur":   "Secteur",
            }

            # Only keep columns that exist
            available = [c for c in display_cols if c in df_rank.columns]
            df_display = df_rank[available].copy()
            df_display = df_display.rename(columns={c: display_cols[c] for c in available})

            # Format numbers
            for col in ["Score", "Mom R²"]:
                if col in df_display.columns:
                    df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
            for col in ["RVOL"]:
                if col in df_display.columns:
                    df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
            for col in ["Prix", "SMA 200", "ATR 14"]:
                if col in df_display.columns:
                    df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
            for col in ["K adapt."]:
                if col in df_display.columns:
                    df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
            for col in ["OBV Slope"]:
                if col in df_display.columns:
                    df_display[col] = df_display[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "—")

            # Highlight top 5
            st.markdown("### 🏆 Top 20 Candidats")
            st.caption("Les 5 premiers rempliraient les slots disponibles (si positions libres)")

            st.dataframe(
                df_display,
                use_container_width=True,
                hide_index=True,
                height=min(len(df_display) * 38 + 40, 800),
            )
        else:
            st.warning("Aucun ticker éligible au ranking — vérifier filtres macro/sectoriels.")

# --- Section unifiée : Analyse par ticker ---
        st.divider()
        st.markdown("### 🔍 Analyse par ticker")
        st.caption("Avis IA + décision humaine pour chaque candidat")
 
        # Calcul de la semaine courante (lundi)
        today = date.today()
        semaine_courante = str(today - pd.Timedelta(days=today.weekday()))
 
        # Chargement des décisions existantes
        decisions_existantes = {}
        dec_data = api_get("/decisions", params={"semaine": semaine_courante})
        if dec_data and "decisions" in dec_data:
            for d in dec_data["decisions"]:
                decisions_existantes[d["ticker"]] = d
 
        # Chargement des avis IA existants
        avis_existants = {}
        avis_data = api_get("/ai-opinions", params={"semaine": semaine_courante})
        if avis_data and "avis" in avis_data:
            for a in avis_data["avis"]:
                avis_existants[a["ticker"]] = a
 
        # Résumé semaine en haut
        nb_fort   = sum(1 for a in avis_existants.values() if a.get("conviction") == "FORT")
        nb_modere = sum(1 for a in avis_existants.values() if a.get("conviction") == "MODÉRÉ")
        nb_faible = sum(1 for a in avis_existants.values() if a.get("conviction") == "FAIBLE")
        nb_avis   = len(avis_existants)
 
        if nb_avis > 0:
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            col_s1.metric("Avis IA générés", nb_avis)
            col_s2.metric("🟢 Fort", nb_fort)
            col_s3.metric("🟡 Modéré", nb_modere)
            col_s4.metric("🔴 Faible", nb_faible)
        else:
            st.info("Aucun avis IA généré pour cette semaine. Utilise le bouton ci-dessous ou lance `/generate-ai-opinion` via l'API.")
 
        # Bouton pour générer les avis du top 5
        col_gen, col_space = st.columns([1, 3])
        with col_gen:
            if st.button("🤖 Générer avis IA (top 5)", use_container_width=True):
                with st.spinner("Génération en cours... (~2 min pour 5 tickers)"):
                    result = api_get("/generate-ai-opinion")
                if result:
                    st.success(result.get("message", "Avis IA lancé en arrière-plan"))
                    st.caption("Rafraîchis la page dans ~2 min pour voir les résultats.")
                else:
                    st.error("Erreur lors du lancement")
 
        st.divider()
 
        # --- Expanders par ticker ---
        for ticker_row in ranking:
            ticker = ticker_row["ticker"]
            rang   = ticker_row["rank"]
            dec_ex = decisions_existantes.get(ticker, {})
            avis   = avis_existants.get(ticker, {})
 
            # Construire le label de l'expander
            conviction = avis.get("conviction", "")
            conv_emoji = {"FORT": "🟢", "MODÉRÉ": "🟡", "FAIBLE": "🔴"}.get(conviction, "⚪")
            dec_emoji  = {"suivi": "✅", "ignore": "❌", "modifie": "🔄"}.get(dec_ex.get("decision", ""), "")
 
            label = f"#{rang} — {ticker}  {conv_emoji} {conviction}  {dec_emoji}"
 
            with st.expander(label):
 
                # --- Ligne 1 : Indicateurs clés ---
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Score", f"{ticker_row.get('score', 0):.4f}")
                c2.metric("Mom R²", f"{ticker_row.get('mom_r2', 0):.4f}")
                c3.metric("RVOL", f"{ticker_row.get('rvol', 0):.2f}")
                c4.metric("Prix", f"${ticker_row.get('prix', 0):.2f}")
                c5.metric("K adapt.", f"{ticker_row.get('k', 0):.2f}")
 
                # --- Avis IA ---
                if avis:
                    st.markdown("---")
                    st.markdown(f"**🤖 Avis IA** — Conviction : **{conv_emoji} {conviction}**")
 
                    resume = avis.get("resume", "")
                    if resume:
                        st.info(resume)
 
                    analyse = avis.get("analyse", "")
                    if analyse:
                        with st.expander("📄 Analyse complète", expanded=False):
                            st.markdown(analyse)
 
                    st.caption(f"Modèle : {avis.get('model_used', '?')} — {avis.get('tokens_used', '?')} tokens — {avis.get('generated_at', '?')[:16]}")
                else:
                    st.markdown("---")
                    st.caption("Pas d'avis IA pour cette semaine.")
                    if st.button(f"🤖 Analyser {ticker}", key=f"ai_{ticker}"):
                        with st.spinner(f"Analyse IA de {ticker} en cours..."):
                            result = api_get("/generate-ai-opinion", params={"ticker": ticker})
                        if result:
                            st.success(f"Analyse de {ticker} lancée en arrière-plan. Rafraîchis dans ~30s.")
                        else:
                            st.error("Erreur lors du lancement")
 
                # --- Décision humaine ---
                st.markdown("---")
                st.markdown("**🧑‍💼 Décision**")
 
                col_dec, col_com, col_btn = st.columns([2, 4, 1])
 
                options    = ["—", "suivi", "ignore", "modifie"]
                labels_dec = ["— (non décidé)", "✅ Suivi", "❌ Ignoré", "🔄 Modifié"]
                current    = dec_ex.get("decision", "—")
                current_idx = options.index(current) if current in options else 0
 
                with col_dec:
                    choix = st.selectbox(
                        "Décision",
                        options=labels_dec,
                        index=current_idx,
                        key=f"dec_{ticker}",
                        label_visibility="collapsed",
                    )
 
                with col_com:
                    commentaire = st.text_input(
                        "Commentaire",
                        value=dec_ex.get("commentaire", ""),
                        placeholder="Commentaire libre (optionnel)...",
                        key=f"com_{ticker}",
                        label_visibility="collapsed",
                    )
 
                with col_btn:
                    if st.button("💾", key=f"save_{ticker}", help="Sauvegarder"):
                        decision_val = options[labels_dec.index(choix)]
                        if decision_val == "—":
                            st.warning("Sélectionne une décision avant de sauvegarder.")
                        else:
                            payload = {
                                "semaine":     semaine_courante,
                                "ticker":      ticker,
                                "rang":        rang,
                                "decision":    decision_val,
                                "commentaire": commentaire or None,
                            }
                            try:
                                r = requests.post(f"{API_BASE}/decisions", json=payload, timeout=10)
                                if r.status_code == 200:
                                    st.success(f"Décision sauvegardée pour {ticker}")
                                    st.cache_data.clear()
                                    st.rerun()
                                else:
                                    st.error(f"Erreur API : {r.text}")
                            except Exception as e:
                                st.error(f"Erreur : {e}")
 
    elif data and "error" in data:
        st.error(f"Erreur API: {data['error']}")
    else:
        st.info("Impossible de charger le ranking. Vérifier que l'API est accessible.")

# ============================================================
# PAGE 2 — MACRO & SECTEURS
# ============================================================

elif page == "🌍 Macro & Secteurs":

    st.markdown("# 🌍 Régime Macro & Secteurs")

    # --- Macro status ---
    with st.spinner("Chargement macro..."):
        macro = api_get("/macro-status")

    if macro:
        st.markdown("### 📡 Régime Macro")
        st.caption("Indice de zone vs SMA 200 — détermine si les achats sont autorisés")

        cols = st.columns(len(macro.get("zones", [])) or 1)
        for i, zone_info in enumerate(macro.get("zones", [])):
            with cols[i]:
                zone_name = zone_info.get("zone", "?")
                is_bull = zone_info.get("bullish", False)
                indice = zone_info.get("indice", "?")
                prix = zone_info.get("prix_indice", 0)
                sma = zone_info.get("sma_200", 0)

                status = "🟢 BULLISH" if is_bull else "🔴 BEARISH"
                st.markdown(f"**{zone_name}** — {indice}")
                st.markdown(f"### {status}")
                if prix and sma:
                    pct_vs_sma = ((prix - sma) / sma) * 100
                    st.caption(f"Prix: {prix:,.2f} | SMA 200: {sma:,.2f} | {pct_vs_sma:+.1f}%")

        st.divider()

    # --- Secteurs ---
    st.markdown("### 🏭 Force Relative Sectorielle")
    st.caption("Ratio ETF sectoriel vs indice de référence, comparé à sa MM50")

    with st.spinner("Chargement secteurs..."):
        secteurs_data = api_get("/secteurs-actifs")

    if secteurs_data:
        secteurs = secteurs_data.get("secteurs", [])
        nb = secteurs_data.get("nb_secteurs_actifs", 0)

        st.metric("Secteurs en force", f"{nb} / 24")

        if secteurs:
            df_sect = pd.DataFrame(secteurs)
            df_sect["statut"] = "✅ En force"
            df_sect["ratio_vs_mm50"] = df_sect["ratio_vs_mm50"].apply(
                lambda x: f"{x:.4f}" if x else "—"
            )

            # Split by zone
            for zone in ["US", "EU"]:
                zone_df = df_sect[df_sect["zone"] == zone]
                if not zone_df.empty:
                    st.markdown(f"**{zone}** — {len(zone_df)} secteurs en force")
                    st.dataframe(
                        zone_df[["secteur_yahoo", "ticker_etf", "ratio_vs_mm50", "date"]].rename(
                            columns={
                                "secteur_yahoo": "Secteur",
                                "ticker_etf": "ETF",
                                "ratio_vs_mm50": "Ratio vs MM50",
                                "date": "Date",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
        else:
            st.warning("Aucun secteur en force relative — marché potentiellement risk-off.")

# ============================================================
# PAGE 3 — PORTEFEUILLE
# ============================================================

elif page == "💼 Portefeuille":

    st.markdown("# 💼 Portefeuille")

    tab_ouvertes, tab_ouvrir, tab_historique = st.tabs([
        "📊 Positions ouvertes", "➕ Ouvrir une position", "📜 Historique fermées"
    ])

    # ----------------------------------------------------------
    # TAB 1 — Positions ouvertes + évaluation conditions v4.1
    # ----------------------------------------------------------
    with tab_ouvertes:
        st.markdown("### 📊 Positions ouvertes — Conditions de sortie v4.1")

        with st.spinner("Évaluation des conditions de sortie..."):
            eval_data = api_get("/positions-ouvertes-eval")

        if eval_data and "positions" in eval_data:
            nb_pos = eval_data.get("nb_positions", 0)

            if nb_pos == 0:
                st.info("Aucune position ouverte. Utilise l'onglet ➕ pour en ouvrir une.")
            else:
                # --- Stats globales ---
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Positions", nb_pos)
                c2.metric("🟢 Saines", eval_data.get("nb_sain", 0))
                c3.metric("🟡 Vigilance", eval_data.get("nb_vigilance", 0))
                c4.metric("🔴 Sortie reco.", eval_data.get("nb_sortie_reco", 0))

                st.caption(f"Données au {eval_data.get('date_evaluation', '?')}")
                st.divider()

                # --- Détail par position ---
                for pos in eval_data["positions"]:
                    alerte = pos.get("alerte_globale", "INCONNU")
                    if alerte == "SORTIE_RECOMMANDÉE":
                        feu_global = "🔴"
                        border_color = "#e94560"
                    elif alerte == "VIGILANCE":
                        feu_global = "🟡"
                        border_color = "#f59e0b"
                    else:
                        feu_global = "🟢"
                        border_color = "#34d399"

                    pnl_pct = pos.get("pnl_pct", 0)
                    pnl_eur = pos.get("pnl_eur", 0)
                    pnl_color = "#34d399" if pnl_pct >= 0 else "#e94560"
                    pnl_sign = "+" if pnl_pct >= 0 else ""

                    with st.container(border=True):
                        # Header : ticker + alerte + P&L
                        h1, h2, h3, h4 = st.columns([2, 2, 3, 3])
                        h1.markdown(f"### {feu_global} {pos['ticker']}")
                        h2.metric("Jours", pos.get("jours_detention", "?"))
                        h3.metric("P&L %", f"{pnl_sign}{pnl_pct}%")
                        h4.metric("P&L €", f"{pnl_sign}{pnl_eur:.2f}€")

                        # Info position
                        i1, i2, i3 = st.columns(3)
                        i1.caption(f"Achat : {pos.get('prix_achat', '?')}€ × {pos.get('quantite', '?')}")
                        i2.caption(f"Prix actuel : {pos.get('prix_actuel', '?')}€")
                        i3.caption(f"Data : {pos.get('data_date', '?')}")

                        # --- Feux conditions ---
                        conditions = pos.get("conditions", {})

                        cols = st.columns(5)

                        # 1. Trailing stop
                        ts = conditions.get("trailing_stop", {})
                        cols[0].markdown(f"**{ts.get('feu', '⚪')} Trailing Stop**")
                        cols[0].caption(
                            f"Stop: {ts.get('stop_level', '?')}€ "
                            f"(k={ts.get('k', '?')}, dist: {ts.get('distance_pct', '?')}%)"
                        )

                        # 2. Trend SMA 200
                        tr = conditions.get("trend_sma200", {})
                        cols[1].markdown(f"**{tr.get('feu', '⚪')} SMA 200**")
                        cols[1].caption(
                            f"SMA: {tr.get('sma_200', '?')}€ "
                            f"(dist: {tr.get('distance_pct', '?')}%)"
                        )

                        # 3. Momentum R²
                        mr = conditions.get("momentum_r2", {})
                        cols[2].markdown(f"**{mr.get('feu', '⚪')} Mom R²**")
                        cols[2].caption(f"Valeur: {mr.get('value', '?')}")

                        # 4. Secteur
                        sc = conditions.get("secteur", {})
                        cols[3].markdown(f"**{sc.get('feu', '⚪')} Secteur**")
                        cols[3].caption(f"{sc.get('secteur_name', '?')}")

                        # 5. Macro
                        mc = conditions.get("macro", {})
                        cols[4].markdown(f"**{mc.get('feu', '⚪')} Macro**")
                        cols[4].caption(f"Zone: {mc.get('zone', '?')}")

                        # --- Violated / warnings summary ---
                        violated = pos.get("violated", [])
                        warnings = pos.get("warnings", [])
                        if violated:
                            st.error(f"⚠️ Conditions violées : {', '.join(violated)}")
                        elif warnings:
                            st.warning(f"Conditions en vigilance : {', '.join(warnings)}")

                        # --- Bouton fermer ---
                        with st.expander("🔒 Fermer cette position"):
                            with st.form(key=f"close_{pos['id']}"):
                                fc1, fc2 = st.columns(2)
                                prix_vente = fc1.number_input(
                                    "Prix de vente",
                                    min_value=0.01,
                                    value=float(pos.get("prix_actuel", 0)),
                                    step=0.01,
                                    key=f"pv_{pos['id']}",
                                )
                                date_vente = fc2.date_input(
                                    "Date de vente",
                                    value=date.today(),
                                    key=f"dv_{pos['id']}",
                                )
                                raison_vente = st.selectbox(
                                    "Raison",
                                    ["TRAILING_STOP", "TREND_BROKEN", "MOMENTUM_LOST",
                                     "SECTOR_WEAK", "MACRO_BEARISH", "MANUEL"],
                                    index=5,
                                    key=f"rv_{pos['id']}",
                                )

                                if st.form_submit_button("Confirmer la fermeture", use_container_width=True):
                                    try:
                                        resp = requests.post(
                                            f"{API_BASE}/positions/{pos['id']}/close",
                                            json={
                                                "prix_vente":   prix_vente,
                                                "date_vente":   str(date_vente),
                                                "raison_vente": raison_vente,
                                            },
                                            timeout=15,
                                        )
                                        result = resp.json()
                                        if result.get("status") == "ok":
                                            st.success(result.get("message", "Position fermée."))
                                            st.cache_data.clear()
                                            st.rerun()
                                        else:
                                            st.error(result.get("error", "Erreur inconnue"))
                                    except Exception as e:
                                        st.error(f"Erreur : {e}")

        elif eval_data and "error" in eval_data:
            st.error(f"Erreur API : {eval_data['error']}")
        else:
            st.error("Impossible de charger l'évaluation des positions.")

    # ----------------------------------------------------------
    # TAB 2 — Ouvrir une position
    # ----------------------------------------------------------
    with tab_ouvrir:
        st.markdown("### ➕ Ouvrir une nouvelle position")
        st.caption("Saisie manuelle après exécution sur ton broker")

        with st.form("open_position_form"):
            o1, o2 = st.columns(2)
            ticker = o1.text_input("Ticker", placeholder="ex: NVDA").strip().upper()
            prix_achat = o2.number_input("Prix d'achat", min_value=0.01, step=0.01)

            o3, o4 = st.columns(2)
            quantite = o3.number_input("Quantité", min_value=0.01, step=0.01, value=1.0)
            date_achat = o4.date_input("Date d'achat", value=date.today())

            o5, o6 = st.columns(2)
            source = o5.selectbox("Source", ["ranking", "manuel"])
            commentaire = o6.text_input("Commentaire (optionnel)", placeholder="ex: top 1 semaine 14")

            submitted = st.form_submit_button("Ouvrir la position", use_container_width=True)

            if submitted:
                if not ticker:
                    st.error("Le ticker est obligatoire.")
                elif prix_achat <= 0:
                    st.error("Le prix d'achat doit être > 0.")
                else:
                    try:
                        resp = requests.post(
                            f"{API_BASE}/positions",
                            json={
                                "ticker":      ticker,
                                "prix_achat":  prix_achat,
                                "quantite":    quantite,
                                "date_achat":  str(date_achat),
                                "source":      source,
                                "commentaire": commentaire or None,
                                "decision_id": None,
                            },
                            timeout=15,
                        )
                        result = resp.json()
                        if result.get("status") == "ok":
                            st.success(f"✅ {result.get('message', 'Position ouverte.')}")
                            st.cache_data.clear()
                        else:
                            st.error(result.get("error", "Erreur inconnue"))
                    except Exception as e:
                        st.error(f"Erreur : {e}")

    # ----------------------------------------------------------
    # TAB 3 — Historique des positions fermées
    # ----------------------------------------------------------
    with tab_historique:
        st.markdown("### 📜 Positions fermées")

        closed_data = api_get("/positions", params={"status": "closed"})

        if closed_data and "positions" in closed_data:
            fermees = closed_data["positions"]

            if not fermees:
                st.info("Aucune position fermée pour l'instant.")
            else:
                # Stats résumé
                df_f = pd.DataFrame(fermees)
                nb_win  = len(df_f[df_f["resultat_pct"].apply(lambda x: (x or 0) > 0)])
                nb_loss = len(df_f[df_f["resultat_pct"].apply(lambda x: (x or 0) <= 0)])
                total_eur = df_f["resultat_eur"].apply(lambda x: x or 0).sum()

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total fermées", len(fermees))
                c2.metric("Gagnantes", nb_win)
                c3.metric("Perdantes", nb_loss)
                c4.metric("P&L total", f"{total_eur:+,.2f}€")

                st.divider()

                # Tableau détaillé
                df_display = df_f[[
                    "ticker", "date_achat", "prix_achat", "date_vente",
                    "prix_vente", "resultat_pct", "resultat_eur", "raison_vente",
                ]].copy()
                df_display.columns = [
                    "Ticker", "Achat", "Prix achat", "Vente",
                    "Prix vente", "P&L %", "P&L €", "Raison",
                ]

                for col in ["Prix achat", "Prix vente", "P&L %", "P&L €"]:
                    df_display[col] = df_display[col].apply(
                        lambda x: f"{x:+.2f}" if x is not None else "—"
                    )

                st.dataframe(df_display, use_container_width=True, hide_index=True)
        else:
            st.error("Impossible de charger les positions fermées.")

# ============================================================
# PAGE 4 — BACKTEST
# ============================================================

elif page == "📈 Backtest & Perf IA":

    st.markdown("# 📈 Résultats Backtest v4.1")
    st.caption("Derniers résultats validés — mai 2022 → fév 2026 — 400 tickers")

    # --- Static metrics from last validated backtest ---
    st.markdown("### 🏆 Métriques v4.1 Hybrid (5 positions)")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sharpe", "0.969")
    c2.metric("Return", "+84.4%")
    c3.metric("Max DD", "15.8%")
    c4.metric("Trades", "172")
    c5.metric("Profit Factor", "2.03")

    c6, c7, c8, c9, c10 = st.columns(5)
    c6.metric("Win Rate", "46.5%")
    c7.metric("Avg Win", "+11.9%")
    c8.metric("Avg Loss", "-5.8%")
    c9.metric("Avg Duration", "29j")
    c10.metric("Avg Positions", "4.1")

    st.divider()

    # --- Exit reasons ---
    st.markdown("### 🚪 Sorties par raison")

    exit_data = pd.DataFrame([
        {"Raison": "Sector Weak",   "Trades": 85, "Avg Return": "+4.6%", "Win Rate": "57.6%"},
        {"Raison": "Trailing Stop", "Trades": 58, "Avg Return": "+1.4%", "Win Rate": "37.9%"},
        {"Raison": "Macro Bearish", "Trades": 20, "Avg Return": "-2.4%", "Win Rate": "35.0%"},
        {"Raison": "Trend Broken",  "Trades": 4,  "Avg Return": "-2.7%", "Win Rate": "0.0%"},
        {"Raison": "End Backtest",  "Trades": 5,  "Avg Return": "-0.0%", "Win Rate": "40.0%"},
    ])
    st.dataframe(exit_data, use_container_width=True, hide_index=True)

    st.divider()

    # --- Version comparison ---
    st.markdown("### 📊 Évolution des versions")

    versions = pd.DataFrame([
        {"Version": "v3.5 filtre binaire",      "Sharpe": 0.44,  "Return": "var.",    "MaxDD": "var.",  "Trades": 18,  "PF": "—"},
        {"Version": "v3.5b filtre assoupli",     "Sharpe": "var.", "Return": "var.",   "MaxDD": "var.",  "Trades": 43,  "PF": "—"},
        {"Version": "v4.0 k=3.0 fixe",          "Sharpe": 0.895, "Return": "+101.9%", "MaxDD": "25.9%", "Trades": 457, "PF": 1.45},
        {"Version": "v4.0 k=adaptive, macro",    "Sharpe": 0.932, "Return": "+80.9%",  "MaxDD": "16.4%", "Trades": 400, "PF": 1.43},
        {"Version": "v4.1 hybrid, 5 pos ✅",     "Sharpe": 0.969, "Return": "+84.4%",  "MaxDD": "15.8%", "Trades": 172, "PF": 2.03},
        {"Version": "v4.1 hybrid, 10 pos",       "Sharpe": 0.890, "Return": "+62.1%",  "MaxDD": "14.2%", "Trades": 348, "PF": 1.86},
    ])
    st.dataframe(versions, use_container_width=True, hide_index=True)

    st.divider()

    # --- Launch new backtest ---
    st.markdown("### 🚀 Lancer un nouveau backtest")

    col_a, col_b = st.columns(2)
    with col_a:
        sma_choice = st.selectbox("SMA période", [200, 150, "Comparer 200 vs 150"])
    with col_b:
        top_n = st.selectbox("Max positions", [5, 10], index=0)

    if st.button("▶️ Lancer le backtest", use_container_width=True):
        sma_param = None if sma_choice == "Comparer 200 vs 150" else int(sma_choice)
        with st.spinner("Backtest lancé en arrière-plan..."):
            result = api_get("/run-backtest-ranking", params={"top_n": top_n, "sma": sma_param})
        if result:
            st.success(f"Backtest lancé ! {result.get('message', '')}")
            st.caption("Les résultats apparaîtront dans les logs Railway.")
        else:
            st.error("Erreur lors du lancement du backtest.")

    # ── Performance IA ──────────────────────────────────────────

    st.divider()
    st.markdown("## 🎯 Performance des Avis IA")
    st.caption("Suivi de la qualité prédictive — rendements réels à +1s / +2s / +4s")

    with st.spinner("Chargement des avis IA..."):
        perf_data = api_get("/ai-opinions", params={"all": "true"})

    if not perf_data or "error" in perf_data:
        st.warning("Impossible de charger les avis IA.")
    else:
        avis_list = perf_data.get("avis", [])

        if not avis_list:
            st.info("Aucun avis IA enregistré. Les données se rempliront progressivement chaque semaine.")
        else:
            df_ia = pd.DataFrame(avis_list)
            df_rend = df_ia.dropna(subset=["rendement_1s", "rendement_2s", "rendement_4s"], how="all")

            nb_total = len(df_ia)
            nb_avec = len(df_rend)

            c1, c2, c3 = st.columns(3)
            c1.metric("Avis générés", nb_total)
            c2.metric("Avec rendement", nb_avec)
            c3.metric("Semaines couvertes", df_ia["semaine"].nunique())

            if nb_avec == 0:
                st.info("Aucun rendement encore disponible — mise à jour automatique chaque lundi 07h30.")
            else:
                # --- Par conviction ---
                st.markdown("#### Rendement moyen par conviction")

                convictions = ["FORT", "MODÉRÉ", "FAIBLE"]
                horizons = [("rendement_1s", "+1s"), ("rendement_2s", "+2s"), ("rendement_4s", "+4s")]

                rows_conv = []
                for conv in convictions:
                    dc = df_rend[df_rend["conviction"] == conv]
                    row = {"Conviction": conv, "Nb": len(dc)}
                    for col, lbl in horizons:
                        vals = dc[col].dropna()
                        if len(vals) > 0:
                            row[f"Rend {lbl}"] = f"{vals.mean():.2%}"
                            row[f"Hit {lbl}"] = f"{(vals > 0).mean():.0%}"
                        else:
                            row[f"Rend {lbl}"] = "—"
                            row[f"Hit {lbl}"] = "—"
                    rows_conv.append(row)

                st.dataframe(pd.DataFrame(rows_conv), use_container_width=True, hide_index=True)

                # --- Global ---
                st.markdown("#### Rendement global")

                rows_global = []
                for col, lbl in horizons:
                    vals = df_rend[col].dropna()
                    if len(vals) > 0:
                        rows_global.append({
                            "Horizon": lbl,
                            "Nb": len(vals),
                            "Moy": f"{vals.mean():.2%}",
                            "Médiane": f"{vals.median():.2%}",
                            "Hit rate": f"{(vals > 0).mean():.0%}",
                            "Best": f"{vals.max():.2%}",
                            "Worst": f"{vals.min():.2%}",
                        })

                if rows_global:
                    st.dataframe(pd.DataFrame(rows_global), use_container_width=True, hide_index=True)

                # --- Détail ---
                with st.expander("📋 Détail par avis"):
                    df_det = df_rend[["semaine", "ticker", "rang", "conviction", "score_composite",
                                      "rendement_1s", "rendement_2s", "rendement_4s"]].copy()
                    df_det = df_det.sort_values(["semaine", "rang"], ascending=[False, True])

                    for col in ["rendement_1s", "rendement_2s", "rendement_4s"]:
                        df_det[col] = df_det[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
                    if "score_composite" in df_det.columns:
                        df_det["score_composite"] = df_det["score_composite"].apply(
                            lambda x: f"{x:.3f}" if pd.notna(x) else "—"
                        )

                    df_det.columns = ["Semaine", "Ticker", "Rang", "Conviction", "Score",
                                      "Rend +1s", "Rend +2s", "Rend +4s"]
                    st.dataframe(df_det, use_container_width=True, hide_index=True)

                # --- Bouton MAJ manuelle ---
                if st.button("🔄 Mettre à jour les rendements", use_container_width=True):
                    r = api_get("/update-suivi-rendements")
                    if r:
                        st.success(r.get("message", "Mise à jour lancée"))
                    else:
                        st.error("Erreur lors de la mise à jour")

# ============================================================
# PAGE 5 — DÉCISIONS HUMAINES
# ============================================================

elif page == "📋 Décisions":

    st.markdown("# 📋 Historique des Décisions")
    st.caption("Toutes tes décisions hebdomadaires, semaine par semaine")

    # Chargement de toutes les décisions
    all_dec = api_get("/decisions")

    if all_dec and "decisions" in all_dec:
        semaines_dispo = sorted(all_dec["semaines"].keys(), reverse=True)

        if not semaines_dispo:
            st.info("Aucune décision enregistrée pour l'instant.")
        else:
            # Sélecteur de semaine
            semaine_sel = st.selectbox(
                "Semaine",
                semaines_dispo,
                format_func=lambda s: f"Semaine du {s}",
            )

            stats = all_dec["semaines"].get(semaine_sel, {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total décisions", stats.get("total", 0))
            c2.metric("✅ Suivis",        stats.get("suivi", 0))
            c3.metric("❌ Ignorés",       stats.get("ignore", 0))
            c4.metric("🔄 Modifiés",      stats.get("modifie", 0))

            st.divider()

            # Tableau des décisions de la semaine sélectionnée
            dec_semaine = [d for d in all_dec["decisions"] if d["semaine"] == semaine_sel]

            if dec_semaine:
                df_dec = pd.DataFrame(dec_semaine)
                df_dec = df_dec.rename(columns={
                    "rang":        "#",
                    "ticker":      "Ticker",
                    "decision":    "Décision",
                    "commentaire": "Commentaire",
                    "updated_at":  "Modifié le",
                })
                df_dec["Décision"] = df_dec["Décision"].map({
                    "suivi":   "✅ Suivi",
                    "ignore":  "❌ Ignoré",
                    "modifie": "🔄 Modifié",
                })
                cols_show = ["#", "Ticker", "Décision", "Commentaire", "Modifié le"]
                cols_show = [c for c in cols_show if c in df_dec.columns]
                st.dataframe(df_dec[cols_show], use_container_width=True, hide_index=True)
            else:
                st.info("Aucune décision pour cette semaine.")
    else:
        st.info("Aucune décision enregistrée pour l'instant.")

# ============================================================
# PAGE 6 — SYSTÈME
# ============================================================

elif page == "⚙️ Système":

    st.markdown("# ⚙️ État du Système")

    # --- API health ---
    st.markdown("### 🔌 API")
    with st.spinner("Vérification..."):
        health = api_get("/")
    if health:
        st.success(f"API active — v{health.get('version', '?')} — DB {'✅' if health.get('engine_connected') else '❌'}")
    else:
        st.error("API injoignable")

    st.divider()

    # --- Model status ---
    st.markdown("### 🤖 Modèle ML")
    with st.spinner("Vérification modèle..."):
        model = api_get("/check-model")
    if model:
        if model.get("model_found"):
            st.success(f"Modèle trouvé — accuracy {model.get('accuracy', '?')} — entraîné le {model.get('updated_at', '?')}")
            st.caption("⚠️ Cible ML invalide — NE PAS réentraîner")
        else:
            st.warning(model.get("message", "Aucun modèle"))

    st.divider()

    # --- Manual actions ---
    st.markdown("### 🔧 Actions manuelles")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("💰 Sync Prix", use_container_width=True):
            r = api_get("/sync-prix")
            st.info(r.get("message", "Lancé") if r else "Erreur")

    with col2:
        if st.button("🔄 Sync Metadata", use_container_width=True):
            r = api_get("/sync-metadata")
            st.info(r.get("message", "Lancé") if r else "Erreur")

    with col3:
        if st.button("📊 Sync ETF", use_container_width=True):
            r = api_get("/sync-etf-sectoriels")
            st.info(r.get("message", "Lancé") if r else "Erreur")

    with col4:
        if st.button("🧮 Analyse complète", use_container_width=True):
            r = api_get("/run-analysis-full")
            st.info(r.get("message", "Lancé") if r else "Erreur")

    st.divider()

    # --- Architecture overview ---
    st.markdown("### 📐 Architecture v4.1")
    st.code("""
ENTRÉE (hebdomadaire, relative)
  Chaque lundi → scorer ~400 tickers
  Score = 50% Mom R² + 25% RVOL + 25% OBV
  Filtres : prix > SMA200, secteur en force, mom_r2 > 0, macro bull
  → Top candidat remplit les slots libres (max 5)

SORTIE (quotidienne, absolue)
  1. Trailing stop ATR (k adaptatif = 2.0 + ATR% × 0.5, clamp [2.0, 4.0])
  2. Prix < SMA 200
  3. Momentum R² < 0
  4. Secteur hors force relative
  5. Macro bearish (indice zone < SMA 200)
    """, language="text")
