# ============================================================
# dashboard.py — Trading Brain Dashboard v1.0
# VT-Source/trading-brain
# ============================================================
# Streamlit dashboard pour visualiser :
#   - Ranking hebdomadaire live (signaux v4.1)
#   - Régime macro & force sectorielle
#   - Résultats backtest historiques
#
# Architecture préparée pour :
#   - v2 : Décisions humaines (suivi/ignoré/modifié)
#   - v3 : Avis IA complémentaire par ticker
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
        ["📊 Ranking Hebdo", "🌍 Macro & Secteurs", "📈 Backtest", "📋 Décisions", "⚙️ Système"],
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

# --- v2: Human decisions inline ---
        st.divider()
        st.markdown("### 🧑‍💼 Décisions humaines")
        st.caption("Encode ta décision pour chaque ticker — sauvegardé par semaine")

        # Calcul de la semaine courante (lundi)
        today = date.today()
        semaine_courante = str(today - pd.Timedelta(days=today.weekday()))

        # Chargement des décisions existantes pour cette semaine
        decisions_existantes = {}
        dec_data = api_get("/decisions", params={"semaine": semaine_courante})
        if dec_data and "decisions" in dec_data:
            for d in dec_data["decisions"]:
                decisions_existantes[d["ticker"]] = d

        for ticker_row in ranking:
            ticker = ticker_row["ticker"]
            rang   = ticker_row["rank"]
            dec_ex = decisions_existantes.get(ticker, {})

            with st.expander(f"#{rang} — {ticker}  {' ✅' if dec_ex.get('decision') == 'suivi' else ' ❌' if dec_ex.get('decision') == 'ignore' else ' 🔄' if dec_ex.get('decision') == 'modifie' else ''}"):
                col_dec, col_com, col_btn = st.columns([2, 4, 1])

                options    = ["—", "suivi", "ignore", "modifie"]
                labels     = ["— (non décidé)", "✅ Suivi", "❌ Ignoré", "🔄 Modifié"]
                current    = dec_ex.get("decision", "—")
                current_idx = options.index(current) if current in options else 0

                with col_dec:
                    choix = st.selectbox(
                        "Décision",
                        options=labels,
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
                        decision_val = options[labels.index(choix)]
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

        # --- v3 placeholder: AI advisor ---
        st.divider()
        st.markdown("### 🤖 Avis IA")
        st.markdown("""
        <div class="placeholder-box">
            <strong>v3 à venir</strong><br>
            Un bot IA donnera un avis complémentaire sur chaque ticker du ranking<br>
            (analyse fondamentale, news récentes, risques spécifiques)
        </div>
        """, unsafe_allow_html=True)

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
# PAGE 3 — BACKTEST
# ============================================================

elif page == "📈 Backtest":

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

# ============================================================
# PAGE 4 — DÉCISIONS HUMAINES
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
# PAGE 5 — SYSTÈME
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
