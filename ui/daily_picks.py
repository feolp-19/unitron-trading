"""Dagens Rekommendationer -- auto-scan all assets with AI as primary decision maker."""

import html
import json
from datetime import datetime

import streamlit as st

from config import ALL_ASSETS_FLAT, Asset
from data.market_data import fetch_ohlc, get_52_week_range
from data.news_data import get_news_for_asset
from analysis.technical import analyze as technical_analyze
from analysis.sentiment import analyze_sentiment, dict_to_signal, run_full_analysis
from analysis.synergy import decide
from analysis.exit_strategy import generate_trading_plan
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

                sr = tech.support_resistance
                entry["rsi"] = f"{tech.rsi_value:.1f}"
                entry["vix"] = f"{tech.vix_value:.1f}" if tech.vix_value else "-"
                entry["volume"] = "N/A" if tech.volume_ratio < 0.01 else f"{tech.volume_ratio:.1f}x"

                headlines = get_news_for_asset(asset)
                headlines_json = json.dumps(headlines)
                sent_dict = analyze_sentiment(asset.display_name, headlines_json)
                sent = dict_to_signal(sent_dict.copy())

                entry["sentiment"] = f"{sent.direction} ({sent.confidence:.0%})"

                ai_result = run_full_analysis(
                    asset_name=asset.display_name,
                    price=tech.current_price,
                    sma_20=tech.sma_20,
                    sma_50=tech.sma_50,
                    sma_200=tech.sma_200,
                    price_vs_sma=tech.price_vs_sma,
                    sma_50w=tech.sma_50w,
                    price_vs_weekly_sma=tech.price_vs_weekly_sma,
                    sma_alignment=tech.sma_alignment,
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

                    # Quality gate: reject weak signals
                    MIN_CONFIDENCE = 0.55
                    if action != "NONE" and confidence < MIN_CONFIDENCE:
                        entry["action"] = "NONE"
                        entry["status"] = "Svag signal"
                        entry["reason"] = (
                            f"AI sa {verdict} men konfidens ({confidence:.0%}) "
                            f"under minimum ({MIN_CONFIDENCE:.0%})"
                        )
                        scan_data["report"].append(entry)
                        continue

                    trading_plan = None
                    if action != "NONE":
                        trading_plan = generate_trading_plan(tech, action)

                        # R/R quality gate: reject trades with poor risk/reward
                        if trading_plan:
                            rr_val = (trading_plan.reward_amount / trading_plan.risk_amount
                                      if trading_plan.risk_amount > 0 else 0)
                            if rr_val < 1.5:
                                entry["action"] = "NONE"
                                entry["status"] = "Svag signal"
                                entry["reason"] = (
                                    f"Risk/reward {trading_plan.risk_reward_ratio} "
                                    f"under minimum (1:1.5)"
                                )
                                scan_data["report"].append(entry)
                                continue

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
                            stop_loss=trading_plan.stop_loss if trading_plan else 0,
                            take_profit=trading_plan.take_profit if trading_plan else 0,
                            reasoning=ai_result.get("key_factors", []),
                        )

                        scan_data["results"].append({
                            "asset": asset,
                            "ai_result": ai_result,
                            "tech": tech,
                            "sent": sent,
                            "trading_plan": trading_plan,
                        })
                    else:
                        entry["status"] = "Ingen signal"
                        entry["reason"] = ai_result.get("analysis", "")[:120]
                else:
                    # Fallback to rule-based engine with same quality gates
                    week_range = get_52_week_range(df)
                    w52_low = week_range[0] if week_range else None
                    decision = decide(
                        tech=tech, sent=sent,
                        week_52_low=w52_low,
                        is_crypto=asset.asset_type == "crypto",
                    )
                    entry["ai_verdict"] = "fallback"

                    if decision.action != "NONE" and decision.confidence_score < MIN_CONFIDENCE:
                        decision = decision.__class__(
                            action="NONE", confidence_score=decision.confidence_score,
                            current_price=decision.current_price,
                            stop_loss_price=0, take_profit_price=0,
                            reasoning=[f"Konfidens ({decision.confidence_score:.0%}) under minimum"],
                            warnings=[], uncertainty_factors=[],
                        )

                    fallback_plan = None
                    if decision.action != "NONE":
                        fallback_plan = generate_trading_plan(tech, decision.action)
                        if fallback_plan:
                            fb_rr = (fallback_plan.reward_amount / fallback_plan.risk_amount
                                     if fallback_plan.risk_amount > 0 else 0)
                            if fb_rr < 1.5:
                                decision = decision.__class__(
                                    action="NONE", confidence_score=decision.confidence_score,
                                    current_price=decision.current_price,
                                    stop_loss_price=0, take_profit_price=0,
                                    reasoning=[f"Risk/reward ({fallback_plan.risk_reward_ratio}) under minimum"],
                                    warnings=[], uncertainty_factors=[],
                                )
                                fallback_plan = None

                    entry["action"] = decision.action

                    if decision.action != "NONE":
                        entry["status"] = "SIGNAL"
                        entry["reason"] = decision.reasoning[0] if decision.reasoning else ""
                        scan_data["results"].append({
                            "asset": asset,
                            "ai_result": None,
                            "tech": tech,
                            "sent": sent,
                            "decision": decision,
                            "trading_plan": fallback_plan,
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
                          if x.get("ai_result") else 0,
            reverse=True,
        )
        st.session_state["scan_data"] = scan_data
        progress_bar.empty()
        progress_text.empty()

    scan_data = st.session_state["scan_data"]

    results = scan_data["results"]
    report = scan_data["report"]

    total = len(report)
    signals = len([r for r in report if r["status"] == "SIGNAL"])
    weak = len([r for r in report if r["status"] == "Svag signal"])
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
                <div style="font-size: 24px; font-weight: 700; color: #FFD600;">{weak}</div>
                <div style="color: #888; font-size: 13px;">Avvisade</div>
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
                <div style="font-size: 64px; margin-bottom: 16px;">🛡️</div>
                <h2 style="color: #888;">Inga starka handelsmöjligheter idag</h2>
                <p style="color: #666; font-size: 16px;">
                    AI-motorn hittade ingen tillgång med tillräckligt stark signal
                    (konfidens ≥55%, R/R ≥1:1.5, SMA-linjering + sentiment i linje).<br>
                    <strong>Att stå utanför marknaden är ett klokt beslut — en missad trade kostar ingenting.</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
            elif val == "Svag signal":
                return "background-color: #FFD60033; color: #FFD600"
            elif val == "Ingen signal":
                return "color: #888"
            elif val in ("Ingen data", "For lite data", "Fel"):
                return "background-color: #FF174433; color: #FF1744"
            return ""

        styled = report_df.style.map(_color_status, subset=["Status"])
        st.dataframe(styled, hide_index=True, use_container_width=True)


def _render_pick_card(result: dict, rank: int):
    """Render a single daily pick with exit strategy."""
    asset = result["asset"]
    tech = result["tech"]
    ai_result = result.get("ai_result")
    trading_plan = result.get("trading_plan")

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
        action = "BULL"
    elif verdict == "BUY_BEAR":
        color = "#FF1744"
        icon = "📉"
        action_text = "KÖP BEAR-CERTIFIKAT"
        action = "BEAR"
    else:
        return

    confidence_pct = f"{confidence:.0%}"

    # Build exit strategy summary for the card
    exit_summary = ""
    if trading_plan:
        exit_summary = (
            f"<strong>SL:</strong> {trading_plan.stop_loss:,.2f} — "
            f"<strong>TP:</strong> {trading_plan.take_profit:,.2f} — "
            f"<strong>R/R:</strong> {trading_plan.risk_reward_ratio}"
        )

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
                        #{rank} {html.escape(asset.display_name)}
                    </div>
                    <div style="font-size: 14px; color: #888;">
                        {html.escape(asset.ticker)} — {html.escape(asset.category)} — via {html.escape(provider)}
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
                <strong>Volym:</strong> {'N/A' if tech.volume_ratio < 0.01 else f'{tech.volume_ratio:.1f}x'} —
                <strong>VIX:</strong> {tech.vix_value if tech.vix_value else 'N/A'}
            </div>
            <div style="font-size: 14px; color: #aaa; line-height: 1.5; margin-bottom: 8px;">
                {html.escape(analysis_text)}
            </div>
            {"<div style='font-size: 14px; color: #ccc; border-top: 1px solid #444; padding-top: 10px; margin-top: 4px; display: flex; gap: 24px; flex-wrap: wrap;'><div><span style=\"color: #888;\">Ingång:</span> <strong>" + f"{trading_plan.entry_price:,.2f}" + "</strong></div><div><span style=\"color: #888;\">Stop-Loss:</span> <strong style=\"color: #FF6B6B;\">" + f"{trading_plan.stop_loss:,.2f}" + "</strong> <span style=\"font-size:11px;color:#666;\">(" + trading_plan.stop_loss_method + ")</span></div><div><span style=\"color: #888;\">Målkurs:</span> <strong style=\"color: #69F0AE;\">" + f"{trading_plan.take_profit:,.2f}" + "</strong> <span style=\"font-size:11px;color:#666;\">(" + trading_plan.take_profit_method + ")</span></div><div><span style=\"color: #888;\">R/R:</span> <strong>" + trading_plan.risk_reward_ratio + "</strong></div></div>" if trading_plan else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander(f"Detaljer & Handelsplan — {html.escape(asset.display_name)}"):
        if key_factors:
            st.markdown("**Avgörande faktorer:**")
            for f in key_factors:
                st.markdown(f"- {f}")

        if risks:
            st.markdown("**Risker:**")
            for r in risks:
                st.caption(f"⚠️ {r}")

        # Exit strategy details
        if trading_plan:
            st.divider()
            st.markdown("**Handelsplan (Exit-strategi):**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Ingång", f"{trading_plan.entry_price:,.2f}")
            with col2:
                st.metric(f"Stop-Loss ({trading_plan.stop_loss_method})",
                           f"{trading_plan.stop_loss:,.2f}")
            with col3:
                st.metric(f"Målkurs ({trading_plan.take_profit_method})",
                           f"{trading_plan.take_profit:,.2f}")
            with col4:
                st.metric("Risk/Reward", trading_plan.risk_reward_ratio)

            st.caption(f"Stop-Loss: {trading_plan.stop_loss_reasoning}")
            st.caption(f"Målkurs: {trading_plan.take_profit_reasoning}")
            st.caption(f"Trailing Stop: {trading_plan.trailing_stop_reasoning}")

        # S/R levels
        sr = tech.support_resistance
        if sr.supports or sr.resistances:
            st.divider()
            sr_col1, sr_col2 = st.columns(2)
            with sr_col1:
                st.markdown("**Stöd:**")
                for i, s in enumerate(sr.supports[:3]):
                    st.caption(f"S{i+1}: {s:,.2f}")
            with sr_col2:
                st.markdown("**Motstånd:**")
                for i, r_val in enumerate(sr.resistances[:3]):
                    st.caption(f"R{i+1}: {r_val:,.2f}")

        certs = search_certificates(asset.ticker, action)
        if certs:
            st.divider()
            st.markdown(f"**{T['avanza_title']}:**")
            for cert in certs[:3]:
                if cert["url"]:
                    st.markdown(f"- [{cert['name']}]({cert['url']}) ({T['leverage_label']}: {cert['leverage']})")
                else:
                    st.write(f"- {cert['name']}")
