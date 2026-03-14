import html
import json

import streamlit as st

from config import Asset
from data.market_data import fetch_ohlc, get_52_week_range
from data.news_data import get_news_for_asset
from analysis.technical import analyze as technical_analyze
from analysis.sentiment import analyze_sentiment, dict_to_signal, run_full_analysis
from analysis.synergy import decide
from analysis.exit_strategy import generate_trading_plan
from analysis.verification import verify_recommendation
from analysis.sentiment import _format_sr_text
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
    render_trading_plan,
)


def _sma_alignment_sv(alignment: str) -> str:
    labels = {
        "bullish_stack": "Hausse-linjering (Pris > SMA20 > SMA50 > SMA200)",
        "bearish_stack": "Baisse-linjering (Pris < SMA20 < SMA50 < SMA200)",
        "mixed": "Blandad (ingen tydlig ordning)",
    }
    return labels.get(alignment, alignment)


def render_dashboard(asset: Asset):
    """Render the full single-asset deep-dive dashboard."""
    st.header(f"{asset.display_name} ({asset.ticker})")

    with st.spinner("Hamtar marknadsdata..."):
        df = fetch_ohlc(asset.ticker)

    if df.empty:
        st.error(T["no_data"])
        return

    tech = technical_analyze(df, ticker=asset.ticker)
    if tech is None:
        st.error(T["insufficient_data"])
        return

    sr = tech.support_resistance

    with st.spinner("Analyserar nyheter och sentiment..."):
        headlines = get_news_for_asset(asset)
        headlines_json = json.dumps(headlines)
        sent_dict = analyze_sentiment(asset.display_name, headlines_json)
        sent = dict_to_signal(sent_dict.copy())

    week_range = get_52_week_range(df)
    w52_low = week_range[0] if week_range else None
    w52_high = week_range[1] if week_range else None

    # AI is the primary decision maker
    with st.spinner("Genererar AI-analys..."):
        ai_analysis = run_full_analysis(
            asset_name=asset.display_name,
            price=tech.current_price,
            sma_20=tech.sma_20,
            sma_50=tech.sma_50,
            sma_200=tech.sma_200,
            price_vs_sma=tech.price_vs_sma,
            sma_50w=tech.sma_50w,
            price_vs_weekly_sma=tech.price_vs_weekly_sma,
            sma_alignment=tech.sma_alignment,
            sma_bias=tech.sma_bias,
            rsi=tech.rsi_value,
            atr=tech.atr_value,
            rsi_trend=tech.rsi_trend_2d,
            atr_ratio=tech.atr_ratio,
            volume_ratio=tech.volume_ratio,
            vix_value=tech.vix_value,
            vix_level=tech.vix_level,
            supports_json=json.dumps(sr.supports),
            resistances_json=json.dumps(sr.resistances),
            near_resistance=tech.near_resistance,
            near_support=tech.near_support,
            headlines_json=headlines_json,
            macd_value=tech.macd_value,
            macd_signal=tech.macd_signal,
            macd_histogram=tech.macd_histogram,
            macd_cross=tech.macd_cross,
            bb_upper=tech.bb_upper,
            bb_lower=tech.bb_lower,
            bb_middle=tech.bb_middle,
            bb_position=tech.bb_position,
            bb_width=tech.bb_width,
        )

    if ai_analysis:
        verdict = ai_analysis.get("verdict", "NO_TRADE")
        if verdict == "BUY_BULL":
            final_action = "BULL"
        elif verdict == "BUY_BEAR":
            final_action = "BEAR"
        else:
            final_action = "NONE"
        final_confidence = ai_analysis.get("confidence", 0)
    else:
        decision = decide(
            tech=tech, sent=sent,
            week_52_low=w52_low, week_52_high=w52_high,
            is_crypto=asset.asset_type == "crypto",
        )
        final_action = decision.action
        final_confidence = decision.confidence_score

    risk = assess_risks(
        tech=tech, sent=sent,
        action=final_action,
        is_crypto=asset.asset_type == "crypto",
    )

    macro_warnings = check_macro_events_today()

    save_recommendation(
        ticker=asset.ticker,
        asset_name=asset.display_name,
        action=final_action,
        confidence=final_confidence,
        entry_price=tech.current_price,
        stop_loss=risk.exit_plan.get("stop_loss", 0),
        take_profit=risk.exit_plan.get("take_profit", 0),
        reasoning=ai_analysis.get("key_factors", []) if ai_analysis else [],
    )

    # === 1. AI ANALYSIS (the primary decision maker) ===
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
                    {html.escape(ai_analysis.get('analysis', ''))}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if ai_analysis.get("key_factors"):
            st.markdown("**Avgörande faktorer:**")
            for f in ai_analysis["key_factors"]:
                st.markdown(f"- {f}")

        if ai_analysis.get("risks"):
            st.markdown("**Risker att bevaka:**")
            for r in ai_analysis["risks"]:
                st.caption(f"⚠️ {r}")

        if ai_analysis.get("outlook"):
            st.info(f"**Utsikt:** {ai_analysis['outlook']}")

        # Stage 2: Verification for actionable trades
        if final_action != "NONE":
            st.divider()
            with st.spinner("Steg 2: Verifierar med andra AI + djävulens advokat..."):
                provider = ai_analysis.get("provider", "Groq")
                verdict = ai_analysis.get("verdict", "NO_TRADE")
                tp = generate_trading_plan(tech, final_action)

                sr_text = _format_sr_text(
                    sr.supports, sr.resistances,
                    tech.current_price, tech.near_resistance, tech.near_support,
                )
                headlines_text = "\n".join(
                    f"- {h['headline']}" for h in headlines
                ) if headlines else "Inga rubriker."

                try:
                    verification = verify_recommendation(
                        asset_name=asset.display_name,
                        first_provider=provider,
                        verdict=verdict,
                        confidence=final_confidence,
                        tech=tech,
                        sr_text=sr_text,
                        headlines_text=headlines_text,
                        stop_loss=tp.stop_loss if tp else 0,
                        take_profit=tp.take_profit if tp else 0,
                        news_keywords=asset.news_keywords,
                    )
                except Exception:
                    verification = None

            if verification:
                if verification.verified:
                    st.success("✅ **VERIFIERAD** — Båda AI-modellerna och djävulens advokat godkänner denna trade.")
                else:
                    st.warning("⚠️ **EJ VERIFIERAD** — En eller flera kontroller misslyckades. Var extra försiktig.")

                v_col1, v_col2 = st.columns(2)
                with v_col1:
                    agree_icon = "✅" if verification.second_ai_agrees else "❌"
                    st.markdown(
                        f"**Andra AI:n ({html.escape(verification.second_ai_provider)}):** "
                        f"{agree_icon} {'Håller med' if verification.second_ai_agrees else 'Håller INTE med'} "
                        f"({verification.second_ai_confidence:.0%} konfidens)"
                    )
                    if verification.second_ai_reasoning:
                        st.caption(verification.second_ai_reasoning)
                    if verification.disagreement_points:
                        for dp in verification.disagreement_points:
                            st.caption(f"⚠️ {dp}")

                with v_col2:
                    risk_colors = {"LOW": "#00C853", "MEDIUM": "#FFD600", "HIGH": "#FF9100", "CRITICAL": "#FF1744"}
                    rc = risk_colors.get(verification.devils_advocate_risk, "#888")
                    st.markdown(
                        f"**Djävulens Advokat:** "
                        f"<span style='color:{rc};font-weight:700;'>"
                        f"{verification.devils_advocate_risk}</span> risk",
                        unsafe_allow_html=True,
                    )
                    if verification.biggest_risk:
                        st.caption(f"🎯 Största risken: {verification.biggest_risk}")
                    if verification.devils_advocate_recommendation:
                        st.caption(f"💡 {verification.devils_advocate_recommendation}")

                if verification.counter_arguments:
                    with st.expander("Motargument (Djävulens Advokat)", expanded=False):
                        for ca in verification.counter_arguments:
                            st.markdown(f"- {ca}")

                if verification.risk_headlines:
                    with st.expander("Ytterligare risknyheter", expanded=False):
                        for rh in verification.risk_headlines[:5]:
                            st.caption(f"📰 {rh.get('headline', '')}")
                            if rh.get("url"):
                                st.caption(f"   [{rh['source']}]({rh['url']})")
            else:
                st.info("Verifiering kunde inte köras — bara en AI tillgänglig.")

    else:
        decision = decide(
            tech=tech, sent=sent,
            week_52_low=w52_low, week_52_high=w52_high,
            is_crypto=asset.asset_type == "crypto",
        )
        render_traffic_light(decision.action, decision.confidence_score)
        if decision.action != "NONE":
            render_confidence_bar(decision.confidence_score)
        with st.expander(T["why_title"], expanded=True):
            for reason in decision.reasoning:
                st.markdown(f"- {reason}")

    # === 2. Teknisk Granskning & Handelsplan ===
    with st.expander("Teknisk Granskning & Handelsplan", expanded=True):

        # --- Indikatorer ---
        st.subheader("Indikatorer")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(T["price_label"], f"{tech.current_price:,.2f}")
        with col2:
            rsi_delta = "Oversalt" if tech.rsi_value < 30 else ("Overkopt" if tech.rsi_value > 70 else "Neutral")
            st.metric(T["rsi_label"], f"{tech.rsi_value:.1f}", delta=rsi_delta)
        with col3:
            st.metric(T["atr_label"], f"{tech.atr_value:,.2f}",
                       delta=f"{tech.atr_ratio:.1f}x" if tech.atr_ratio != 1.0 else None)
        with col4:
            vix_display = f"{tech.vix_value:.1f}" if tech.vix_value else "N/A"
            vix_delta = T.get(f"vix_{tech.vix_level}", tech.vix_level)
            st.metric("VIX", vix_display, delta=vix_delta)

        # SMA status row
        st.markdown("**SMA-status:**")
        sma_col1, sma_col2, sma_col3, sma_col4 = st.columns(4)
        with sma_col1:
            st.metric("SMA 20", f"{tech.sma_20:,.2f}",
                       delta="Pris over" if tech.current_price > tech.sma_20 else "Pris under")
        with sma_col2:
            st.metric("SMA 50", f"{tech.sma_50:,.2f}",
                       delta="Pris over" if tech.current_price > tech.sma_50 else "Pris under")
        with sma_col3:
            sma_delta = T[f"price_{tech.price_vs_sma}_sma"]
            st.metric("SMA 200", f"{tech.sma_200:,.2f}", delta=sma_delta)
        with sma_col4:
            if tech.sma_50w:
                w_delta = T.get(f"price_{tech.price_vs_weekly_sma}_weekly", tech.price_vs_weekly_sma)
                st.metric("50v SMA", f"{tech.sma_50w:,.2f}", delta=w_delta)
            else:
                st.metric("50v SMA", "N/A")

        st.caption(f"SMA-linjering: {_sma_alignment_sv(tech.sma_alignment)}")

        # Volume
        vol_col1, vol_col2 = st.columns(2)
        with vol_col1:
            if tech.volume_ratio < 0.01:
                st.metric("Volym vs 20d snitt", "N/A")
            else:
                vol_delta = "Hog" if tech.volume_ratio > 1.5 else ("Lag" if tech.volume_ratio < 0.7 else "Normal")
                st.metric("Volym vs 20d snitt", f"{tech.volume_ratio:.1f}x", delta=vol_delta)

        st.divider()

        # --- Stöd & Motstånd ---
        st.subheader("Stöd & Motstånd")
        sr_col1, sr_col2 = st.columns(2)
        with sr_col1:
            st.markdown("**Stödnivåer (Support):**")
            if sr.supports:
                for i, s in enumerate(sr.supports):
                    dist_pct = ((s - tech.current_price) / tech.current_price) * 100
                    st.markdown(f"S{i+1}: **{s:,.2f}** ({dist_pct:+.1f}%)")
            else:
                st.caption("Inga tydliga stödnivåer identifierade")
        with sr_col2:
            st.markdown("**Motståndsnivåer (Resistance):**")
            if sr.resistances:
                for i, r in enumerate(sr.resistances):
                    dist_pct = ((r - tech.current_price) / tech.current_price) * 100
                    st.markdown(f"R{i+1}: **{r:,.2f}** ({dist_pct:+.1f}%)")
            else:
                st.caption("Inga tydliga motståndsnivåer identifierade")

        if tech.near_resistance:
            st.warning(
                f"Ankrings-bias: Priset ligger nära motstånd ({sr.resistances[0]:,.2f}). "
                f"Risk för avvisning — överväg att vänta på bekräftat utbrott."
            )
        if tech.near_support:
            st.info(
                f"Priset ligger nära stöd ({sr.supports[0]:,.2f}). "
                f"Potentiell studszon — bekräftelse krävs."
            )

        st.divider()

        # --- Chart with S/R ---
        render_price_chart(
            df, asset.display_name,
            supports=sr.supports,
            resistances=sr.resistances,
        )

        st.divider()

        # --- EXIT-STRATEGI (Handelsplan) ---
        if final_action != "NONE" and risk.trading_plan:
            st.subheader("EXIT-STRATEGI")
            render_trading_plan(risk.trading_plan, final_action)
        elif final_action == "NONE":
            st.info("Ingen aktiv handelsplan — motorn rekommenderar att avvakta.")

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

    # === 4. Avanza Certificates ===
    if final_action in ("BULL", "BEAR"):
        with st.expander(T["avanza_title"], expanded=True):
            certs = search_certificates(asset.ticker, final_action)
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

    # === 5. Warnings ===
    all_warnings = macro_warnings[:]
    if risk.bias_warnings:
        all_warnings.extend(risk.bias_warnings)
    if all_warnings:
        with st.expander(T["warnings_title"], expanded=True):
            render_warning_box(all_warnings)
