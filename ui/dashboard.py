import json

import streamlit as st

from config import Asset
from data.market_data import fetch_ohlc, get_52_week_range
from data.news_data import get_news_for_asset
from analysis.technical import analyze as technical_analyze
from analysis.sentiment import analyze_sentiment, dict_to_signal, run_full_analysis
from analysis.synergy import decide
from risk.risk_manager import assess_risks
from risk.calendar_check import check_macro_events_today
from avanza.certificates import search_certificates
from storage.history import save_recommendation
from ui.translations import T
from ui.components import (
    render_traffic_light,
    render_confidence_bar,
    render_price_chart,
    render_headline_table,
    render_warning_box,
)


def render_dashboard(asset: Asset):
    """Render the full single-asset deep-dive dashboard."""
    st.header(f"{asset.display_name} ({asset.ticker})")

    with st.spinner("Hamtar marknadsdata..."):
        df = fetch_ohlc(asset.ticker)

    if df.empty:
        st.error(T["no_data"])
        return

    tech = technical_analyze(df)
    if tech is None:
        st.error(T["insufficient_data"])
        return

    with st.spinner("Analyserar nyheter och sentiment..."):
        headlines = get_news_for_asset(asset)
        headlines_json = json.dumps(headlines)
        sent_dict = analyze_sentiment(asset.display_name, headlines_json)
        sent = dict_to_signal(sent_dict.copy())

    week_range = get_52_week_range(df)
    w52_low = week_range[0] if week_range else None
    w52_high = week_range[1] if week_range else None

    decision = decide(
        tech=tech, sent=sent,
        week_52_low=w52_low, week_52_high=w52_high,
        is_crypto=asset.asset_type == "crypto",
    )

    risk = assess_risks(
        tech=tech, sent=sent,
        action=decision.action,
        is_crypto=asset.asset_type == "crypto",
    )

    macro_warnings = check_macro_events_today()

    # Full AI analysis
    with st.spinner("Genererar AI-analys..."):
        ai_analysis = run_full_analysis(
            asset_name=asset.display_name,
            price=tech.current_price,
            sma=tech.sma_value,
            price_vs_sma=tech.price_vs_sma,
            rsi=tech.rsi_value,
            atr=tech.atr_value,
            rsi_trend=tech.rsi_trend_2d,
            atr_ratio=tech.atr_ratio,
            headlines_json=headlines_json,
        )

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

    # === 1. AI ANALYSIS (the star of the show) ===
    if ai_analysis:
        verdict = ai_analysis.get("verdict", "NO_TRADE")
        if verdict == "BUY_BULL":
            v_color = "#00C853"
            v_icon = "📈"
            v_text = "KOP BULL"
        elif verdict == "BUY_BEAR":
            v_color = "#FF1744"
            v_icon = "📉"
            v_text = "KOP BEAR"
        else:
            v_color = "#888"
            v_icon = "⏸️"
            v_text = "AVVAKTA"

        ai_confidence = ai_analysis.get("confidence", 0)
        provider = ai_analysis.get("provider", "")

        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, {v_color}18, {v_color}08);
                border: 2px solid {v_color}66;
                border-radius: 16px;
                padding: 24px 32px;
                margin-bottom: 20px;
            ">
                <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 12px;">
                    <span style="font-size: 40px;">{v_icon}</span>
                    <div>
                        <div style="font-size: 28px; font-weight: 700; color: {v_color};">
                            {v_text} — {ai_confidence:.0%} konfidens
                        </div>
                        <div style="font-size: 13px; color: #666;">AI-analys via {provider}</div>
                    </div>
                </div>
                <div style="font-size: 16px; color: #ddd; line-height: 1.6;">
                    {ai_analysis.get('analysis', '')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if ai_analysis.get("risks"):
            st.markdown("**Risker att bevaka:**")
            for r in ai_analysis["risks"]:
                st.caption(f"⚠️ {r}")

        if ai_analysis.get("outlook"):
            st.info(f"**Utsikt:** {ai_analysis['outlook']}")
    else:
        render_traffic_light(decision.action, decision.confidence_score)
        if decision.action != "NONE":
            render_confidence_bar(decision.confidence_score)
        with st.expander(T["why_title"], expanded=True):
            for reason in decision.reasoning:
                st.markdown(f"- {reason}")

    # === 2. Technical Analysis ===
    with st.expander(T["tech_title"], expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(T["price_label"], f"{tech.current_price:,.2f}")
        with col2:
            rsi_delta = "Oversalt" if tech.rsi_value < 30 else ("Overkopt" if tech.rsi_value > 70 else "Neutral")
            st.metric(T["rsi_label"], f"{tech.rsi_value:.1f}", delta=rsi_delta)
        with col3:
            sma_delta = T[f"price_{tech.price_vs_sma}_sma"]
            st.metric(T["sma_label"], f"{tech.sma_value:,.2f}", delta=sma_delta)
        with col4:
            st.metric(T["atr_label"], f"{tech.atr_value:,.2f}")

        render_price_chart(df, asset.display_name)

    # === 3. News & Sentiment ===
    with st.expander(T["sentiment_title"], expanded=False):
        sent_col1, sent_col2, sent_col3 = st.columns(3)
        with sent_col1:
            direction_sv = T.get(f"sentiment_{sent.direction.lower()}", sent.direction)
            st.metric("Riktning", direction_sv)
        with sent_col2:
            st.metric(T["confidence_label"], f"{sent.confidence:.0%}")
        with sent_col3:
            provider_used = getattr(sent, 'ai_provider_used', 'unknown')
            st.metric("AI-leverantor", provider_used)

        if sent.summary:
            st.info(f"**{T['ai_summary']}:** {sent.summary}")

        render_headline_table(sent.headline_details)

    # === 4. Risk Management ===
    if decision.action != "NONE":
        with st.expander(T["risk_title"], expanded=True):
            r_col1, r_col2, r_col3, r_col4 = st.columns(4)
            with r_col1:
                st.metric(T["entry_label"], f"{risk.exit_plan['entry']:,.2f}")
            with r_col2:
                st.metric(T["stop_loss_label"], f"{risk.exit_plan['stop_loss']:,.2f}")
            with r_col3:
                st.metric(T["take_profit_label"], f"{risk.exit_plan['take_profit']:,.2f}")
            with r_col4:
                st.metric(T["risk_reward_label"], risk.exit_plan["risk_reward"])

            st.markdown(f"**{T['exit_strategy_label']}:** {risk.exit_plan['strategy']}")

    # === 5. Avanza Certificates ===
    if decision.action in ("BULL", "BEAR"):
        with st.expander(T["avanza_title"], expanded=True):
            certs = search_certificates(asset.ticker, decision.action)
            if certs:
                for cert in certs:
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        if cert["url"]:
                            st.markdown(f"[{cert['name']}]({cert['url']})")
                        else:
                            st.write(cert["name"])
                    with col_b:
                        if cert["leverage"] != "-":
                            st.write(f"{T['leverage_label']}: {cert['leverage']}")
            else:
                st.info(T["avanza_no_certs"])

    # === 6. Warnings ===
    all_warnings = decision.warnings + macro_warnings
    if risk.bias_warnings:
        all_warnings.extend(risk.bias_warnings)
    if all_warnings:
        with st.expander(T["warnings_title"], expanded=True):
            render_warning_box(all_warnings)
