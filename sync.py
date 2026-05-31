# ============================================================
# sync.py — Logique de synchronisation des données
# Trading Brain | VT-Source
# ============================================================
# Fonctions extraites de main.py v6.4.0 pour lisibilité.
# Toutes reçoivent `engine` en paramètre (injection de dépendance).
# ============================================================
# v1.2 (2026-05-17) — sync_prix_logic : retry 3× avec backoff sur
#                     réponses vides yfinance + audit post-sync des
#                     tickers en retard de plus de 3 jours.
# v1.3 (2026-05-19) — sync_secteurs_etf_logic : MM50 calculée sur
#                     l'historique complet (concat base + yfinance)
#                     au lieu de la fenêtre téléchargée seule.
#                     min_periods=50 strict (NULL si fenêtre < 50).
#                     Correction du biais "ratio_vs_mm50 aplati vers 1".
# v1.4 (2026-05-31) — sync_prix_logic : liste des tickers = UNION
#                     tickers_info ∪ actions_prix_historique. Un nouveau
#                     ticker ajouté à tickers_info est désormais collecté
#                     automatiquement (plus d'amorçage manuel requis).
# ============================================================

import time
import pandas as pd
import yfinance as yf
from sqlalchemy import text


# ============================================================
# SYNC PRIX QUOTIDIEN (remplace n8n)
# ============================================================

def sync_prix_logic(engine, full: bool = False, period_override: str = None, tickers_filter: list = None):
    """
    Télécharge les prix OHLCV depuis yfinance et upsert dans actions_prix_historique.
    Remplace le workflow n8n désactivé le 2026-04-11.

    Paramètres
    ----------
    engine : SQLAlchemy engine
    full : bool
        False → 30 derniers jours (mode quotidien, scheduler)
        True  → 5 ans d'historique (initialisation / rattrapage)
    period_override : str, optional
        Ex: "3mo" — surcharge la période calculée automatiquement.
    """
    if engine is None:
        print("❌ sync_prix_logic : engine non connecté.")
        return

    mode   = "COMPLET (5 ans)" if full else "INCRÉMENTAL (30j)"
    period = period_override or ("5y" if full else "1mo")
    print(f"🔄 Sync prix quotidien — mode {mode}...")

    try:
        # 1. Liste des tickers
        if tickers_filter:
            # Ciblage explicite : ne synchronise que les tickers demandés
            # (ex. amorçage de nouveaux tickers sans relancer tout l'univers).
            all_tickers = sorted(set(tickers_filter))
            print(f"   🎯 Ciblage explicite : {len(all_tickers)} ticker(s).")
        else:
            # UNION de tickers_info (univers déclaré) et actions_prix_historique
            # (univers déjà collecté) — garantit la collecte des nouveaux tickers
            # tout en gardant les tickers historiques absents de tickers_info.
            # (v1.4)
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT ticker FROM tickers_info
                    UNION
                    SELECT DISTINCT ticker FROM actions_prix_historique
                    ORDER BY ticker
                """))
                all_tickers = [row[0] for row in result]

        if not all_tickers:
            print("⚠️ Aucun ticker en base.")
            return

        print(f"   {len(all_tickers)} tickers à synchroniser.")

        chunk_size     = 20
        total_chunks   = (len(all_tickers) - 1) // chunk_size + 1
        total_upserted = 0
        total_errors   = 0
        skipped_empty  = []  # tickers laissés vides après 3 tentatives

        for i in range(0, len(all_tickers), chunk_size):
            chunk = all_tickers[i:i + chunk_size]
            chunk_upserted = 0

            for ticker_symbol in chunk:
                try:
                    print(f"   ⏳ {ticker_symbol}...")

                    # --- Retry 3× avec backoff exponentiel ---
                    # yfinance peut retourner df_yf.empty sur rate limit silencieux
                    # ou timeout sans lever d'exception. Sans retry, le ticker est
                    # perdu pour la journée et son absence du ranking n'est détectable
                    # qu'a posteriori (cf. incident 2026-05-16 : 59 EU non synchronisés).
                    df_yf = pd.DataFrame()
                    for attempt in range(3):
                        try:
                            df_yf = yf.download(
                                ticker_symbol,
                                period=period,
                                interval="1d",
                                auto_adjust=True,
                                progress=False,
                                timeout=30
                            )
                            if not df_yf.empty:
                                break
                            wait = 2 ** attempt   # 1, 2, 4 secondes
                            print(f"   ⚠️ {ticker_symbol} : réponse vide tentative "
                                  f"{attempt+1}/3, retry dans {wait}s...")
                            time.sleep(wait)
                        except Exception as e_dl:
                            wait = 2 ** attempt
                            print(f"   ⚠️ {ticker_symbol} : exception {e_dl} tentative "
                                  f"{attempt+1}/3, retry dans {wait}s...")
                            time.sleep(wait)

                    if df_yf.empty:
                        print(f"   ❌ {ticker_symbol} : aucune donnée après 3 tentatives — SKIP")
                        skipped_empty.append(ticker_symbol)
                        total_errors += 1
                        continue

                    df_yf = df_yf.reset_index()
                    # yfinance peut retourner des MultiIndex columns
                    df_yf.columns = [c[0] if isinstance(c, tuple) else c for c in df_yf.columns]

                    df_yf = df_yf.rename(columns={
                        "Date" : "date",
                        "Open" : "prix_ouverture",
                        "High" : "prix_haut",
                        "Low"  : "prix_bas",
                        "Close": "prix_cloture",
                        "Volume": "volume",
                    })

                    # auto_adjust=True → Close est déjà ajusté
                    df_yf["prix_ajuste"] = df_yf["prix_cloture"]
                    df_yf["date"]   = pd.to_datetime(df_yf["date"]).dt.date
                    df_yf["ticker"] = ticker_symbol

                    cols = ["ticker", "date", "prix_ouverture", "prix_haut",
                            "prix_bas", "prix_cloture", "prix_ajuste", "volume"]
                    df_clean = df_yf[[c for c in cols if c in df_yf.columns]].dropna(
                        subset=["prix_cloture"]
                    )

                    if df_clean.empty:
                        continue

                    records = df_clean.to_dict("records")

                    with engine.begin() as conn:
                        conn.execute(text("""
                            INSERT INTO actions_prix_historique
                                (ticker, date, prix_ouverture, prix_haut,
                                 prix_bas, prix_cloture, prix_ajuste, volume)
                            VALUES
                                (:ticker, :date, :prix_ouverture, :prix_haut,
                                 :prix_bas, :prix_cloture, :prix_ajuste, :volume)
                            ON CONFLICT (ticker, date) DO UPDATE SET
                                prix_ouverture = EXCLUDED.prix_ouverture,
                                prix_haut      = EXCLUDED.prix_haut,
                                prix_bas       = EXCLUDED.prix_bas,
                                prix_cloture   = EXCLUDED.prix_cloture,
                                prix_ajuste    = EXCLUDED.prix_ajuste,
                                volume         = EXCLUDED.volume
                        """), records)

                    chunk_upserted += len(records)
                    time.sleep(0.5)

                except Exception as e:
                    print(f"   ❌ {ticker_symbol} : {e}")
                    total_errors += 1
                    continue

            total_upserted += chunk_upserted
            print(f"   🟢 Chunk {i // chunk_size + 1}/{total_chunks} — "
                  f"{chunk_upserted} lignes upsertées.")

        # --- Audit post-sync : tickers en retard ---
        # Détecte les tickers dont la dernière date en base est antérieure à J-3
        # (week-end inclus). Permet de repérer immédiatement les angles morts comme
        # l'incident du 2026-05-16 (59 tickers EU au 14/05 au lieu du 15/05).
        try:
            today_d = pd.Timestamp.today().date()
            ref_d   = today_d - pd.Timedelta(days=3)
            with engine.connect() as conn:
                stale = conn.execute(text("""
                    SELECT ticker, MAX(date) AS last_date
                    FROM actions_prix_historique
                    GROUP BY ticker
                    HAVING MAX(date) < :ref
                    ORDER BY MAX(date) ASC, ticker ASC
                """), {"ref": ref_d}).fetchall()
            if stale:
                print(f"⚠️ Audit post-sync : {len(stale)} ticker(s) en retard "
                      f"(dernière date < {ref_d}) :")
                for t, d in stale[:30]:
                    print(f"      {t} → {d}")
                if len(stale) > 30:
                    print(f"      ... et {len(stale) - 30} autre(s)")
            else:
                print(f"✅ Audit post-sync : tous les tickers à jour (≥ {ref_d}).")
        except Exception as e_audit:
            print(f"⚠️ Audit post-sync échoué : {e_audit}")

        if skipped_empty:
            print(f"📋 Tickers skippés (3 tentatives vides) : {len(skipped_empty)} → "
                  f"{', '.join(skipped_empty[:30])}"
                  f"{'...' if len(skipped_empty) > 30 else ''}")

        print(f"🏁 Sync prix terminée — {total_upserted} lignes upsertées, "
              f"{total_errors} erreurs.")

    except Exception as e:
        print(f"❌ Erreur sync_prix_logic : {e}")


# ============================================================
# SYNC METADATA YAHOO FINANCE
# ============================================================

def sync_metadata_logic(engine):
    """
    Met à jour les métadonnées Yahoo Finance (secteur, industrie, pays, etc.)
    pour tous les tickers en base.
    """
    if engine is None:
        return
    print("🔄 Sync Yahoo Finance (Metadata)...")
    try:
        with engine.connect() as conn:
            result  = conn.execute(text("SELECT DISTINCT ticker FROM actions_prix_historique LIMIT 500;"))
            tickers = [row[0] for row in result]

        for ticker_symbol in tickers:
            try:
                stock = yf.Ticker(ticker_symbol)
                data  = stock.info
                if "symbol" in data:
                    metadata = {
                        "ticker"    : ticker_symbol,
                        "name"      : data.get("longName"),
                        "secteur"   : (data.get("sector") or "").title() or None,
                        "industrie" : data.get("industry"),
                        "pays"      : data.get("country"),
                        "monnaie"   : data.get("currency"),
                        "market_cap": data.get("marketCap"),
                        "pe_ratio"  : data.get("trailingPE")
                    }
                    with engine.begin() as conn:
                        conn.execute(text("""
                            INSERT INTO tickers_info
                                (ticker, name, secteur, industrie, pays, monnaie,
                                 market_cap, pe_ratio, derniere_maj)
                            VALUES
                                (:ticker, :name, :secteur, :industrie, :pays, :monnaie,
                                 :market_cap, :pe_ratio, CURRENT_DATE)
                            ON CONFLICT (ticker) DO UPDATE SET
                                pe_ratio     = EXCLUDED.pe_ratio,
                                market_cap   = EXCLUDED.market_cap,
                                derniere_maj = CURRENT_DATE;
                        """), metadata)
                time.sleep(1)
            except Exception:
                continue
    except Exception as e:
        print(f"❌ Erreur Sync: {e}")


# ============================================================
# FILL HIGH/LOW — ACTION MANUELLE UNIQUE
# ============================================================

def fill_high_low_logic(engine):
    """
    Remplit prix_haut, prix_bas et prix_ouverture pour tout l'historique
    depuis yfinance. Action manuelle unique.
    Correction : utilise la même transaction (conn) pour to_sql et l'UPDATE.
    """
    if engine is None:
        print("❌ fill_high_low_logic : engine non connecté.")
        return

    period = "5y"
    print("📥 Remplissage prix_haut / prix_bas / prix_ouverture — historique complet...")

    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT DISTINCT ticker FROM actions_prix_historique ORDER BY ticker"))
            all_tickers = [row[0] for row in result]

        print(f"   {len(all_tickers)} tickers à traiter.")

        chunk_size    = 20
        total_chunks  = (len(all_tickers) - 1) // chunk_size + 1
        total_updated = 0

        for i in range(0, len(all_tickers), chunk_size):
            chunk = all_tickers[i:i + chunk_size]
            chunk_updated = 0

            for ticker_symbol in chunk:
                try:
                    df_yf = yf.download(
                        ticker_symbol,
                        period=period,
                        interval="1d",
                        auto_adjust=True,
                        progress=False,
                        timeout=30
                    )

                    if df_yf.empty:
                        continue

                    df_yf = df_yf.reset_index()
                    df_yf.columns = [c[0] if isinstance(c, tuple) else c for c in df_yf.columns]
                    df_yf = df_yf.rename(columns={
                        "Date": "date",
                        "High": "prix_haut",
                        "Low": "prix_bas",
                        "Open": "prix_ouverture",
                    })

                    df_yf["date"]   = pd.to_datetime(df_yf["date"]).dt.date
                    df_yf["ticker"] = ticker_symbol

                    cols_needed = ["ticker", "date", "prix_haut", "prix_bas", "prix_ouverture"]
                    df_to_save = df_yf[[c for c in cols_needed if c in df_yf.columns]].dropna(
                        subset=["prix_haut", "prix_bas"]
                    )

                    if df_to_save.empty:
                        continue

                    # Utiliser une vraie TEMPORARY TABLE Postgres (auto-drop en fin de session)
                    with engine.begin() as conn:
                        # Créer une vraie temp table (tronquée et supprimée auto en fin de transaction)
                        conn.execute(text("""
                            CREATE TEMP TABLE tmp_hl_data (
                                ticker         TEXT,
                                date           DATE,
                                prix_haut      DOUBLE PRECISION,
                                prix_bas       DOUBLE PRECISION,
                                prix_ouverture DOUBLE PRECISION
                            ) ON COMMIT DROP
                        """))
                        
                        # Insérer les données via to_sql sur la temp table
                        df_to_save.to_sql(
                            "tmp_hl_data", conn, 
                            if_exists="append", index=False
                        )
                        
                        conn.execute(text("""
                            UPDATE actions_prix_historique a SET
                                prix_haut      = t.prix_haut,
                                prix_bas       = t.prix_bas,
                                prix_ouverture = t.prix_ouverture
                            FROM tmp_hl_data t
                            WHERE a.ticker = t.ticker
                              AND a.date   = t.date
                        """))
                        # Pas besoin de DROP : ON COMMIT DROP s'en charge

                    chunk_updated += len(df_to_save)
                    time.sleep(0.5)

                except Exception as e:
                    print(f"   ❌ {ticker_symbol} : {e}")
                    continue

            total_updated += chunk_updated
            print(f"   🟢 Chunk {i // chunk_size + 1}/{total_chunks} terminé.")

        print(f"🏁 Fill high/low terminé — {total_updated} lignes mises à jour.")

    except Exception as e:
        print(f"❌ Erreur fill_high_low_logic : {e}")


# ============================================================
# SYNC ETF SECTORIELS + FORCE RELATIVE
# ============================================================

def sync_secteurs_etf_logic(engine, full: bool = False):
    """
    Télécharge et persiste les prix des ETF sectoriels + indices de référence.
    Calcule le ratio de force relative et le ratio vs MM50 pour chaque ETF.

    Paramètres
    ----------
    engine : SQLAlchemy engine
    full : bool
        False → 30 derniers jours (mode quotidien, scheduler)
        True  → 5 ans d'historique (initialisation manuelle)
    """
    if engine is None:
        print("❌ sync_secteurs_etf_logic : engine non connecté.")
        return

    mode   = "COMPLET (5 ans)" if full else "INCRÉMENTAL (30j)"
    period = "5y" if full else "1mo"
    print(f"🔄 Sync ETF sectoriels — mode {mode}...")

    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT ticker_etf, indice_reference
                FROM secteurs_etf
                WHERE actif = TRUE
            """)).fetchall()

        if not rows:
            print("⚠️ Aucun ETF trouvé dans secteurs_etf — exécuter secteurs_etf.sql d'abord.")
            return

        etf_list   = [r[0] for r in rows]
        etf_to_idx = {r[0]: r[1] for r in rows}
        indices    = list(set(r[1] for r in rows))

        print(f"   {len(etf_list)} ETF sectoriels + {len(indices)} indices à synchroniser.")

        # --- Indices de référence ---
        print("   📥 Téléchargement indices de référence...")
        for idx_ticker in indices:
            try:
                df_idx = yf.download(
                    idx_ticker, period=period, interval="1d",
                    auto_adjust=True, progress=False, timeout=30
                )
                if df_idx.empty:
                    print(f"   ⚠️ Aucune donnée pour l'indice {idx_ticker}")
                    continue

                df_idx = df_idx.reset_index()
                df_idx.columns = [c[0] if isinstance(c, tuple) else c for c in df_idx.columns]
                df_idx = df_idx.rename(columns={
                    "Date": "date", "Close": "prix_cloture", "Adj Close": "prix_ajuste"
                })
                if "prix_ajuste" not in df_idx.columns:
                    df_idx["prix_ajuste"] = df_idx["prix_cloture"]
                df_idx["date"]          = pd.to_datetime(df_idx["date"]).dt.date
                df_idx["ticker_indice"] = idx_ticker
                records = df_idx[["ticker_indice", "date", "prix_cloture", "prix_ajuste"]].to_dict("records")

                with engine.begin() as conn:
                    conn.execute(text("""
                        INSERT INTO indices_prix (ticker_indice, date, prix_cloture, prix_ajuste)
                        VALUES (:ticker_indice, :date, :prix_cloture, :prix_ajuste)
                        ON CONFLICT (ticker_indice, date) DO UPDATE SET
                            prix_cloture = EXCLUDED.prix_cloture,
                            prix_ajuste  = EXCLUDED.prix_ajuste
                    """), records)

                print(f"   ✅ Indice {idx_ticker} — {len(records)} jours enregistrés.")
                time.sleep(1)

            except Exception as e:
                print(f"   ❌ Erreur indice {idx_ticker} : {e}")
                continue

        # --- ETF sectoriels ---
        print("   📥 Téléchargement ETF sectoriels...")
        for etf_ticker in etf_list:
            try:
                idx_ticker = etf_to_idx[etf_ticker]

                # Télécharger ETF
                df_etf = yf.download(
                    etf_ticker, period=period, interval="1d",
                    auto_adjust=True, progress=False, timeout=30
                )
                if df_etf.empty:
                    print(f"   ⚠️ Aucune donnée pour {etf_ticker}")
                    continue

                df_etf = df_etf.reset_index()
                df_etf.columns = [c[0] if isinstance(c, tuple) else c for c in df_etf.columns]
                df_etf = df_etf.rename(columns={
                    "Date": "date", "Close": "prix_cloture",
                    "Adj Close": "prix_ajuste", "Volume": "volume"
                })
                if "prix_ajuste" not in df_etf.columns:
                    df_etf["prix_ajuste"] = df_etf["prix_cloture"]
                df_etf["date"]       = pd.to_datetime(df_etf["date"]).dt.date
                df_etf["ticker_etf"] = etf_ticker

                # Filtre défensif : ignorer les jours où yfinance retourne NaN
                # (typiquement jours fériés Xetra sur les EX*.DE). Sans ce filtre,
                # les NaN se propagent dans ratio_force_relative / ratio_vs_mm50
                # et tous les tickers de la zone EU sont filtrés du ranking.
                nb_before = len(df_etf)
                df_etf = df_etf.dropna(subset=["prix_ajuste"])
                df_etf = df_etf[df_etf["prix_ajuste"] > 0]
                nb_dropped = nb_before - len(df_etf)
                if nb_dropped > 0:
                    print(f"   ℹ️ {etf_ticker} — {nb_dropped} jour(s) NaN ignoré(s) "
                          f"(yfinance/Xetra fermé)")

                if df_etf.empty:
                    print(f"   ⚠️ {etf_ticker} — toutes les lignes sont NaN, skip.")
                    continue

                # Charger l'indice de référence depuis la base
                with engine.connect() as conn:
                    df_indice = pd.read_sql(text("""
                        SELECT date, prix_ajuste AS prix_indice
                        FROM indices_prix
                        WHERE ticker_indice = :idx
                        ORDER BY date ASC
                    """), conn, params={"idx": idx_ticker})

                if df_indice.empty:
                    print(f"   ⚠️ Aucune donnée indice {idx_ticker} pour {etf_ticker}")
                    continue

                df_indice["date"] = pd.to_datetime(df_indice["date"]).dt.date

                # ============================================================
                # FIX v1.3 (2026-05-19) — MM50 sur historique complet
                # ============================================================
                # En mode incrémental, df_etf ne contient que ~22 jours. Calculer
                # la MM50 dessus produit en réalité une MM<50 avec min_periods=1,
                # ce qui aplatit le ratio vers 1.0 et bascule des ETF borderline
                # côté en_force=true à tort (cf. diagnostic 2026-05-19 : 6% de
                # classifications erronées, biaisées vers en_force=true).
                #
                # Solution : charger l'historique complet ETF en base, le merger
                # avec les nouvelles données yfinance, calculer la MM50 sur
                # l'ensemble, puis ne ré-écrire que les lignes correspondant à
                # la fenêtre téléchargée.
                # ============================================================
                with engine.connect() as conn:
                    df_etf_hist = pd.read_sql(text("""
                        SELECT date, prix_cloture, prix_ajuste, volume
                        FROM secteurs_etf_prix
                        WHERE ticker_etf = :etf
                        ORDER BY date ASC
                    """), conn, params={"etf": etf_ticker})

                if not df_etf_hist.empty:
                    df_etf_hist["date"] = pd.to_datetime(df_etf_hist["date"]).dt.date
                    # Concat hist + yfinance, dédupliquer (yfinance écrase l'historique
                    # sur les dates communes — données plus fraîches/corrigées)
                    df_etf_full = pd.concat([df_etf_hist, df_etf], ignore_index=True)
                    df_etf_full = df_etf_full.drop_duplicates(
                        subset=["date"], keep="last"
                    ).sort_values("date").reset_index(drop=True)
                else:
                    # Premier sync de cet ETF (ne devrait arriver qu'à l'init)
                    df_etf_full = df_etf.copy()

                # Dates effectivement présentes dans la fenêtre téléchargée
                # (= dates à ré-upserter dans secteurs_etf_prix)
                dates_to_write = set(df_etf["date"].tolist())

                # Merge avec indice sur l'historique complet
                df_merged = df_etf_full.merge(df_indice, on="date", how="inner")

                if df_merged.empty:
                    print(f"   ⚠️ {etf_ticker} — merge ETF+indice vide, skip.")
                    continue

                # Calcul ratio de force relative (base = première date commune)
                first_etf = df_merged["prix_ajuste"].iloc[0]
                first_idx = df_merged["prix_indice"].iloc[0]
                df_merged["ratio_force_relative"] = (
                    (df_merged["prix_ajuste"] / first_etf) /
                    (df_merged["prix_indice"] / first_idx)
                )
                # min_periods=50 strict : NULL avant 50 points (au lieu de fenêtre tronquée
                # biaisée vers la valeur courante).
                df_merged["mm50_ratio"]        = df_merged["ratio_force_relative"].rolling(50, min_periods=50).mean()
                df_merged["ratio_vs_mm50"]     = df_merged["ratio_force_relative"] / df_merged["mm50_ratio"]
                df_merged["en_force_relative"] = df_merged["ratio_vs_mm50"] > 1.0

                # Restreindre l'upsert aux seules dates téléchargées
                df_merged = df_merged[df_merged["date"].isin(dates_to_write)]
                # Exclure les lignes où la MM50 n'est pas encore définie
                # (cas init : 49 premières dates de l'historique)
                df_merged = df_merged.dropna(subset=["mm50_ratio"])

                # ticker_etf manquant pour les lignes venues de l'historique base
                df_merged["ticker_etf"] = etf_ticker

                # Upsert
                cols_to_save = [
                    "ticker_etf", "date", "prix_cloture", "prix_ajuste",
                    "volume", "prix_indice", "ratio_force_relative",
                    "ratio_vs_mm50", "en_force_relative"
                ]
                records = df_merged[cols_to_save].to_dict("records")

                for r in records:
                    r["en_force_relative"] = bool(r["en_force_relative"])
                    r["volume"]            = int(r["volume"]) if r["volume"] is not None else 0

                with engine.begin() as conn:
                    conn.execute(text("""
                        INSERT INTO secteurs_etf_prix
                            (ticker_etf, date, prix_cloture, prix_ajuste, volume,
                             prix_indice, ratio_force_relative, ratio_vs_mm50, en_force_relative)
                        VALUES
                            (:ticker_etf, :date, :prix_cloture, :prix_ajuste, :volume,
                             :prix_indice, :ratio_force_relative, :ratio_vs_mm50, :en_force_relative)
                        ON CONFLICT (ticker_etf, date) DO UPDATE SET
                            prix_cloture         = EXCLUDED.prix_cloture,
                            prix_ajuste          = EXCLUDED.prix_ajuste,
                            volume               = EXCLUDED.volume,
                            prix_indice          = EXCLUDED.prix_indice,
                            ratio_force_relative = EXCLUDED.ratio_force_relative,
                            ratio_vs_mm50        = EXCLUDED.ratio_vs_mm50,
                            en_force_relative    = EXCLUDED.en_force_relative,
                            updated_at           = CURRENT_TIMESTAMP
                    """), records)

                if df_merged.empty:
                    print(f"   ⚠️ {etf_ticker} — aucune ligne ré-upsertée "
                          f"(historique < 50 points ?).")
                else:
                    statut_fr = "OUI ✅" if df_merged["en_force_relative"].iloc[-1] else "NON ❌"
                    ratio_val = df_merged["ratio_vs_mm50"].iloc[-1]
                    print(f"   ✅ {etf_ticker} — {len(records)} jours | "
                          f"Force relative : {statut_fr} | "
                          f"ratio_vs_mm50 = {ratio_val:.3f}")
                time.sleep(1)

            except Exception as e:
                print(f"   ❌ Erreur ETF {etf_ticker} : {e}")
                continue

        print("🏁 Sync ETF sectoriels terminée.")

    except Exception as e:
        print(f"❌ Erreur sync_secteurs_etf_logic : {e}")
