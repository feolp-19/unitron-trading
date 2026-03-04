"""Dagens Rekommendationer -- auto-scan all assets and show today's best trades."""

import json
from datetime import datetime

import streamlit as st

from config import ALL_ASSETS_FLAT, Asset
from data.market_data import fetch_ohlc, get_52_week_range
from data.news_data import get_news_for_asset
from analysis.technical import analyze as technical_analyze
from analysis.sentiment import analyze_sentiment, dict_to_signal
from analysis.synergy import decide
from avanza.certificates import search_certificates
from storage.history import save_recommendation
from ui.translations import T


@st.cache_data(ttl=3600, show_spinner=False)
def _run_daily_scan() -> list[dict]:
    """Scan ALL curated assets and return actionable results sorted by confidence."""
    results = []

    for asset in ALL_ASSETS_FLAT:
        try:
            df = fetch_ohlc(asset.ticker)
            if df.empty:
                continue

            tech = technical_analyze(df)
            if tech is None:
                continue

            headlines = get_news_for_asset(asset)
            sent_dict = analyze_sentiment(asset.display_name, json.dumps(headlines))
            sent = dict_to_signal(sent_dict.copy())

            week_range = get_52_week_range(df)
            w52_low = week_range[0] if week_range else None

            decision = decide(
                tech=tech,
                sent=sent,
                week_52_low=w52_low,
                is_crypto=asset.asset_type == "crypto",
            )

            if decision.action != "NONE":
                save_recommendation(
                    ticker=asset.ticker,
                    asset_name=asset.display_name,
                    action=decision.action,
                    confidence=decision.confidence_score,
                    entry_price=decision.current_price,
                    stop_loss=decision.stop_loss_price,
                    take_profit=decision.take_profit_price,
                    reasoning=decision.reasoning,
                )

                results.append({
                    "asset": asset,
                    "decision": decision,
                    "tech": tech,
                    "sent": sent,
                })
        except Exception:
            continue

    results.sort(key=lambda x: x["decision"].confidence_score, reverse=True)
    return results


def render_daily_picks():
    """Render the auto-scan daily picks view."""
    today_str = datetime.now().strftime("%A %d %B %Y")

    st.markdown(
        f"""
        <div style="text-align: center; padding: 16px 0 8px 0;">
            <h1 style="margin-bottom: 0;">Unitron Handelsanalys</h1>
            <p style="color: #888; font-size: 18px;">{today_str}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Skannar alla marknader..."):
        results = _run_daily_scan()

    if not results:
        st.markdown(
            """
            <div style="
                background: #1A1D23;
                border: 1px solid #333;
                border-radius: 16px;
                padding: 48px 32px;
                text-align: center;
                margin: 32px 0;
            ">
                <div style="font-size: 64px; margin-bottom: 16px;">😴</div>
                <h2 style="color: #888;">Inga handelsmöjligheter idag</h2>
                <p style="color: #666; font-size: 16px;">
                    Synergy Engine hittade ingen tillgång där teknisk analys
                    och makrosentiment är synkroniserade.<br>
                    Bäst att stå utanför marknaden idag.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

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
                Motorn hittade <strong>{len(results)}</strong> handelsmöjlighet{'er' if len(results) > 1 else ''} idag
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for i, r in enumerate(results):
        _render_pick_card(r, rank=i + 1)


def _render_pick_card(result: dict, rank: int):
    """Render a single daily pick as a prominent card."""
    asset = result["asset"]
    decision = result["decision"]
    tech = result["tech"]
    sent = result["sent"]

    if decision.action == "BULL":
        color = "#00C853"
        icon = "📈"
        action_text = "KÖP BULL-CERTIFIKAT"
        direction_sv = "uppgång"
    else:
        color = "#FF1744"
        icon = "📉"
        action_text = "KÖP BEAR-CERTIFIKAT"
        direction_sv = "nedgång"

    confidence_pct = f"{decision.confidence_score:.0%}"

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
                        {asset.ticker} — {asset.category}
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
                <strong>Sentiment:</strong> {sent.direction}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander(f"Varför {direction_sv} på {asset.display_name}?"):
        for reason in decision.reasoning:
            st.markdown(f"- {reason}")

        if decision.uncertainty_factors:
            st.markdown("**Osäkerhetsfaktorer:**")
            for uf in decision.uncertainty_factors:
                st.caption(f"⚠️ {uf}")

        if decision.warnings:
            st.markdown("**Varningar:**")
            for w in decision.warnings:
                st.warning(w)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(T["entry_label"], f"{decision.current_price:,.2f}")
        with col2:
            st.metric(T["stop_loss_label"], f"{decision.stop_loss_price:,.2f}")
        with col3:
            st.metric(T["take_profit_label"], f"{decision.take_profit_price:,.2f}")

        certs = search_certificates(asset.ticker, decision.action)
        if certs:
            st.markdown(f"**{T['avanza_title']}:**")
            for cert in certs[:3]:
                if cert["url"]:
                    st.markdown(f"- [{cert['name']}]({cert['url']}) ({T['leverage_label']}: {cert['leverage']})")
                else:
                    st.write(f"- {cert['name']}")
