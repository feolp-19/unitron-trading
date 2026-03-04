"""Dagens Rekommendationer -- auto-scan all assets with AI as primary decision maker."""

import json
from datetime import datetime

import streamlit as st

from config import ALL_ASSETS_FLAT, Asset
from data.market_data import fetch_ohlc, get_52_week_range
from data.news_data import get_news_for_asset
from analysis.technical import analyze as technical_analyze
from analysis.sentiment import analyze_sentiment, dict_to_signal, run_full_analysis
from analysis.synergy import decide
from avanza.certificates import search_certificates
from storage.history import save_recommendation
from ui.translations import T


def render_daily_picks():
    """Render the auto-scan daily picks view."""
    today_str = datetime.now().strftime("%A %d %B %Y")

    col_title, col_btn = st.columns([4, 1])
    with col_title:
        st.markdown(
            f"""
            <div style="padding: 16px 0 8px 0;">
                <h1 style="margin-bottom: 0;">Unitron Handelsanalys</h1>
                <p style="color: #888; font-size: 18px;">{today_str}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_btn:
        st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
        rescan = st.button("🔄 Skanna nu", type="primary", use_container_width=True)

    if rescan:
        st.session_state.pop("scan_data", None)

    if "scan_data" not in st.session_state:
        progress_text = st.empty()
        progress_bar = st.progress(0)
        scan_data = {"results": [], "report": []}
        total = len(ALL_ASSETS_FLAT)

        for i, asset in enumerate(ALL_ASSETS_FLAT):
            progress_bar.progress((i + 1) / total)
            progress_text.caption(f"Analyserar {asset.display_name} ({i+1}/{total})...")

            entry = {
                "name": asset.display_name,
                "ticker": asset.ticker,
                "status": "?",
                "rsi": "-",
                "vix": "-",
                "volume": "-",
                "sentiment": "-",
                "action": "-",
                "ai_verdict": "-",
                "reason": "",
            }

            try:
                df = fetch_ohlc(asset.ticker)
                if df.empty:
                    entry["status"] = "Ingen data"
                    entry["reason"] = "yfinance returnerade ingen data"
                    scan_data["report"].append(entry)
                    continue

                tech = technical_analyze(df, ticker=asset.ticker)
                if tech is None:
                    entry["status"] = "For lite data"
                    entry["reason"] = f"Bara {len(df)} dagar"
                    scan_data["report"].append(entry)
                    continue

                entry["rsi"] = f"{tech.rsi_value:.1f}"
                entry["vix"] = f"{tech.vix_value:.1f}" if tech.vix_value else "-"
                entry["volume"] = f"{tech.volume_ratio:.1f}x"

                headlines = get_news_for_asset(asset)
                headlines_json = json.dumps(headlines)
                sent_dict = analyze_sentiment(asset.display_name, headlines_json)
                sent = dict_to_signal(sent_dict.copy())

                entry["sentiment"] = f"{sent.direction} ({sent.confidence:.0%})"

                # AI is the primary decision maker
                ai_result = run_full_analysis(
                    asset_name=asset.display_name,
                    price=tech.current_price,
                    sma_200=tech.sma_200,
                    price_vs_sma=tech.price_vs_sma,
                    sma_50w=tech.sma_50w,
                    price_vs_weekly_sma=tech.price_vs_weekly_sma,
                    rsi=tech.rsi_value,
                    atr=tech.atr_value,
                    rsi_trend=tech.rsi_trend_2d,
                    atr_ratio=tech.atr_ratio,
                    volume_ratio=tech.volume_ratio,
                    vix_value=tech.vix_value,
                    vix_level=tech.vix_level,
                    headlines_json=headlines_json,
                )

                if ai_result:
                    verdict = ai_result.get("verdict", "NO_TRADE")
                    confidence = ai_result.get("confidence", 0)
                    entry["ai_verdict"] = verdict

                    if verdict == "BUY_BULL":
                        action = "BULL"
                    elif verdict == "BUY_BEAR":
                        action = "BEAR"
                    else:
                        action = "NONE"

                    entry["action"] = action

                    if action != "NONE":
                        entry["status"] = "SIGNAL"
                        entry["reason"] = ai_result.get("analysis", "")[:120]

                        save_recommendation(
                            ticker=asset.ticker,
                            asset_name=asset.display_name,
                            action=action,
                            confidence=confidence,
                            entry_price=tech.current_price,
                            stop_loss=0,
                            take_profit=0,
                            reasoning=ai_result.get("key_factors", []),
                        )

                        scan_data["results"].append({
                            "asset": asset,
                            "ai_result": ai_result,
                            "tech": tech,
                            "sent": sent,
                        })
                    else:
                        entry["status"] = "Ingen signal"
                        entry["reason"] = ai_result.get("analysis", "")[:120]
                else:
                    # Fallback to rule-based engine
                    week_range = get_52_week_range(df)
                    w52_low = week_range[0] if week_range else None
                    decision = decide(
                        tech=tech, sent=sent,
                        week_52_low=w52_low,
                        is_crypto=asset.asset_type == "crypto",
                    )
                    entry["action"] = decision.action
                    entry["ai_verdict"] = "fallback"

                    if decision.action != "NONE":
                        entry["status"] = "SIGNAL"
                        entry["reason"] = decision.reasoning[0] if decision.reasoning else ""
                        scan_data["results"].append({
                            "asset": asset,
                            "ai_result": None,
                            "tech": tech,
                            "sent": sent,
                            "decision": decision,
                        })
                    else:
                        entry["status"] = "Ingen signal"
                        entry["reason"] = decision.reasoning[0] if decision.reasoning else ""

            except Exception as e:
                entry["status"] = "Fel"
                entry["reason"] = str(e)[:80]

            scan_data["report"].append(entry)

        scan_data["results"].sort(
            key=lambda x: x.get("ai_result", {}).get("confidence", 0)
                          if x.get("ai_result") else x.get("decision", {}).get("confidence_score", 0) if isinstance(x.get("decision"), dict) else 0,
            reverse=True,
        )
        st.session_state["scan_data"] = scan_data
        progress_bar.empty()
        progress_text.empty()

    scan_data = st.session_state["scan_data"]

    results = scan_data["results"]
    report = scan_data["report"]

    # --- Scan summary stats ---
    total = len(report)
    signals = len([r for r in report if r["status"] == "SIGNAL"])
    no_signal = len([r for r in report if r["status"] == "Ingen signal"])
    failed = len([r for r in report if r["status"] in ("Ingen data", "For lite data", "Fel")])

    # VIX banner
    vix_entries = [r for r in report if r.get("vix", "-") != "-"]
    if vix_entries:
        vix_val = float(vix_entries[0]["vix"])
        if vix_val > 30:
            vix_color, vix_text = "#FF1744", "EXTREM RADSLA"
        elif vix_val > 25:
            vix_color, vix_text = "#FF9100", "Forhojd radsla"
        elif vix_val > 20:
            vix_color, vix_text = "#FFD600", "Forsiktighet"
        elif vix_val > 15:
            vix_color, vix_text = "#888", "Normal"
        else:
            vix_color, vix_text = "#00C853", "Lugn marknad"

        st.markdown(
            f"""
            <div style="
                background: {vix_color}15;
                border: 1px solid {vix_color}44;
                border-radius: 8px;
                padding: 8px 16px;
                margin-bottom: 12px;
                text-align: center;
                font-size: 14px;
            ">
                <strong>VIX:</strong> {vix_val:.1f} — {vix_text}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div style="
            background: #1A1D23;
            border: 1px solid #333;
            border-radius: 12px;
            padding: 12px 24px;
            margin-bottom: 20px;
            display: flex;
            gap: 32px;
            justify-content: center;
        ">
            <div style="text-align: center;">
                <div style="font-size: 24px; font-weight: 700;">{total}</div>
                <div style="color: #888; font-size: 13px;">Skannade</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 24px; font-weight: 700; color: #00C853;">{signals}</div>
                <div style="color: #888; font-size: 13px;">Signaler</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 24px; font-weight: 700; color: #888;">{no_signal}</div>
                <div style="color: #888; font-size: 13px;">Ingen signal</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 24px; font-weight: 700; color: #FF1744;">{failed}</div>
                <div style="color: #888; font-size: 13px;">Misslyckades</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Results ---
    if results:
        st.markdown(
            f"""
            <div style="
                background: #00C85315;
                border: 1px solid #00C85344;
                border-radius: 12px;
                padding: 16px 24px;
                text-align: center;
                margin-bottom: 24px;
            ">
                <span style="font-size: 18px;">
                    AI-motorn hittade <strong>{len(results)}</strong> handelsmöjlighet{'er' if len(results) > 1 else ''} idag
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for i, r in enumerate(results):
            _render_pick_card(r, rank=i + 1)
    else:
        st.markdown(
            """
            <div style="
                background: #1A1D23;
                border: 1px solid #333;
                border-radius: 16px;
                padding: 48px 32px;
                text-align: center;
                margin: 16px 0 24px 0;
            ">
                <div style="font-size: 64px; margin-bottom: 16px;">😴</div>
                <h2 style="color: #888;">Inga handelsmöjligheter idag</h2>
                <p style="color: #666; font-size: 16px;">
                    AI-motorn hittade ingen tillgång där alla faktorer
                    (trend, momentum, volym, sentiment) pekar åt samma håll.<br>
                    Bäst att stå utanför marknaden idag.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Detailed scan report ---
    with st.expander(f"📋 Skanningsrapport — {total} tillgångar analyserade", expanded=False):
        import pandas as pd
        report_df = pd.DataFrame(report)
        report_df = report_df.rename(columns={
            "name": "Tillgång",
            "ticker": "Ticker",
            "status": "Status",
            "rsi": "RSI",
            "vix": "VIX",
            "volume": "Volym",
            "sentiment": "Sentiment",
            "ai_verdict": "AI Beslut",
            "action": "Signal",
            "reason": "Detaljer",
        })

        def _color_status(val):
            if val == "SIGNAL":
                return "background-color: #00C85333; color: #00C853"
            elif val == "Ingen signal":
                return "color: #888"
            elif val in ("Ingen data", "For lite data", "Fel"):
                return "background-color: #FF174433; color: #FF1744"
            return ""

        styled = report_df.style.map(_color_status, subset=["Status"])
        st.dataframe(styled, hide_index=True, use_container_width=True)


def _render_pick_card(result: dict, rank: int):
    """Render a single daily pick as a prominent card."""
    asset = result["asset"]
    tech = result["tech"]
    sent = result["sent"]
    ai_result = result.get("ai_result")

    if ai_result:
        verdict = ai_result.get("verdict", "NO_TRADE")
        confidence = ai_result.get("confidence", 0)
        analysis_text = ai_result.get("analysis", "")
        key_factors = ai_result.get("key_factors", [])
        risks = ai_result.get("risks", [])
        provider = ai_result.get("provider", "")
    else:
        decision = result.get("decision")
        verdict = f"BUY_{decision.action}" if decision and decision.action != "NONE" else "NO_TRADE"
        confidence = decision.confidence_score if decision else 0
        analysis_text = "; ".join(decision.reasoning) if decision else ""
        key_factors = []
        risks = []
        provider = "regelbaserad"

    if verdict == "BUY_BULL":
        color = "#00C853"
        icon = "📈"
        action_text = "KÖP BULL-CERTIFIKAT"
        direction_sv = "uppgång"
        action = "BULL"
    elif verdict == "BUY_BEAR":
        color = "#FF1744"
        icon = "📉"
        action_text = "KÖP BEAR-CERTIFIKAT"
        direction_sv = "nedgång"
        action = "BEAR"
    else:
        return

    confidence_pct = f"{confidence:.0%}"

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {color}15, {color}08);
            border: 1px solid {color}44;
            border-radius: 16px;
            padding: 24px 32px;
            margin-bottom: 20px;
        ">
            <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 12px;">
                <span style="font-size: 36px;">{icon}</span>
                <div>
                    <div style="font-size: 24px; font-weight: 700; color: {color};">
                        #{rank} {asset.display_name}
                    </div>
                    <div style="font-size: 14px; color: #888;">
                        {asset.ticker} — {asset.category} — via {provider}
                    </div>
                </div>
                <div style="margin-left: auto; text-align: right;">
                    <div style="
                        background: {color};
                        color: #000;
                        font-weight: 700;
                        padding: 8px 20px;
                        border-radius: 8px;
                        font-size: 16px;
                    ">{action_text}</div>
                </div>
            </div>
            <div style="font-size: 15px; color: #ccc; margin-bottom: 8px;">
                <strong>Konfidens:</strong> {confidence_pct} —
                <strong>Pris:</strong> {tech.current_price:,.2f} —
                <strong>RSI:</strong> {tech.rsi_value:.1f} —
                <strong>Volym:</strong> {tech.volume_ratio:.1f}x —
                <strong>VIX:</strong> {tech.vix_value if tech.vix_value else 'N/A'}
            </div>
            <div style="font-size: 14px; color: #aaa; line-height: 1.5;">
                {analysis_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander(f"Detaljer — {asset.display_name}"):
        if key_factors:
            st.markdown("**Avgörande faktorer:**")
            for f in key_factors:
                st.markdown(f"- {f}")

        if risks:
            st.markdown("**Risker:**")
            for r in risks:
                st.caption(f"⚠️ {r}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(T["price_label"], f"{tech.current_price:,.2f}")
        with col2:
            st.metric(T["sma_label"], f"{tech.sma_200:,.2f}")
        with col3:
            if tech.sma_50w:
                st.metric("50v SMA", f"{tech.sma_50w:,.2f}")
            else:
                st.metric("50v SMA", "N/A")

        certs = search_certificates(asset.ticker, action)
        if certs:
            st.markdown(f"**{T['avanza_title']}:**")
            for cert in certs[:3]:
                if cert["url"]:
                    st.markdown(f"- [{cert['name']}]({cert['url']}) ({T['leverage_label']}: {cert['leverage']})")
                else:
                    st.write(f"- {cert['name']}")
