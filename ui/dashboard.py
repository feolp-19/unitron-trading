import json

import streamlit as st

from config import Asset
from data.market_data import fetch_ohlc, get_52_week_range
from data.news_data import get_news_for_asset
from analysis.technical import analyze as technical_analyze
from analysis.sentiment import analyze_sentiment, dict_to_signal
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

    # Fetch data
    with st.spinner("Hämtar marknadsdata..."):
        df = fetch_ohlc(asset.ticker)

    if df.empty:
        st.error(T["no_data"])
        return

    # Technical analysis
    tech = technical_analyze(df)
    if tech is None:
        st.error(T["insufficient_data"])
        return

    # Sentiment analysis
    with st.spinner("Analyserar nyhetssentiment..."):
        headlines = get_news_for_asset(asset)
        sent_dict = analyze_sentiment(asset.display_name, json.dumps(headlines))
        sent = dict_to_signal(sent_dict.copy())

    # 52-week range
    week_range = get_52_week_range(df)
    w52_low = week_range[0] if week_range else None
    w52_high = week_range[1] if week_range else None

    # Synergy decision
    decision = decide(
        tech=tech,
        sent=sent,
        week_52_low=w52_low,
        week_52_high=w52_high,
        is_crypto=asset.asset_type == "crypto",
    )

    # Risk assessment
    risk = assess_risks(
        tech=tech,
        sent=sent,
        action=decision.action,
        is_crypto=asset.asset_type == "crypto",
    )

    # Macro events
    macro_warnings = check_macro_events_today()

    # Log recommendation
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

    # === RENDER SECTIONS ===

    # 1. Signal
    render_traffic_light(decision.action, decision.confidence_score)
    if decision.action != "NONE":
        render_confidence_bar(decision.confidence_score)

    # 2. Why?
    with st.expander(T["why_title"], expanded=True):
        for reason in decision.reasoning:
            st.markdown(f"- {reason}")

    # 3. Technical Analysis
    with st.expander(T["tech_title"], expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(T["price_label"], f"{tech.current_price:,.2f}")
        with col2:
            rsi_delta = "Översålt" if tech.rsi_value < 30 else ("Överköpt" if tech.rsi_value > 70 else "Neutral")
            st.metric(T["rsi_label"], f"{tech.rsi_value:.1f}", delta=rsi_delta)
        with col3:
            sma_delta = T[f"price_{tech.price_vs_sma}_sma"]
            st.metric(T["sma_label"], f"{tech.sma_value:,.2f}", delta=sma_delta)
        with col4:
            st.metric(T["atr_label"], f"{tech.atr_value:,.2f}")

        render_price_chart(df, asset.display_name)

    # 4. Macro Sentiment
    with st.expander(T["sentiment_title"], expanded=True):
        sent_col1, sent_col2, sent_col3 = st.columns(3)
        with sent_col1:
            direction_sv = T.get(f"sentiment_{sent.direction.lower()}", sent.direction)
            st.metric("Riktning", direction_sv)
        with sent_col2:
            st.metric(T["confidence_label"], f"{sent.confidence:.0%}")
        with sent_col3:
            st.metric(T["relevant_headlines"], f"{sent.relevant_count}/{sent.total_count}")

        if sent.low_data_quality:
            st.warning(T["low_data_warning"])

        if sent.summary:
            st.info(f"**{T['ai_summary']}:** {sent.summary}")

        render_headline_table(sent.headline_details)

    # 5. Uncertainty Factors
    with st.expander(T["uncertainty_title"], expanded=decision.action != "NONE"):
        if decision.uncertainty_factors:
            for factor in decision.uncertainty_factors:
                st.markdown(
                    f"""<div style="
                        background: #FFD60022;
                        border-left: 4px solid #FFD600;
                        padding: 10px 16px;
                        border-radius: 4px;
                        margin-bottom: 8px;
                    ">⚠️ {factor}</div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.success(T["no_uncertainty"])

    # 6. Risk Management
    with st.expander(T["risk_title"], expanded=decision.action != "NONE"):
        if decision.action != "NONE":
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

        if risk.bias_warnings:
            st.markdown(f"### {T['bias_warnings_title']}")
            for bw in risk.bias_warnings:
                st.warning(bw)

    # 7. Avanza Certificates
    if decision.action in ("BULL", "BEAR"):
        with st.expander(T["avanza_title"], expanded=True):
            with st.spinner("Söker certifikat på Avanza..."):
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

    # 8. Warnings
    all_warnings = decision.warnings + macro_warnings
    if all_warnings:
        with st.expander(T["warnings_title"], expanded=True):
            render_warning_box(all_warnings)
