"""Dagens Rekommendationer -- auto-scan all assets with AI as primary decision maker."""

import html
import json
import time
from datetime import datetime

import streamlit as st

from config import ALL_ASSETS_FLAT, Asset
from data.market_data import fetch_ohlc, get_52_week_range
from data.news_data import get_news_for_asset
from analysis.technical import analyze as technical_analyze
from analysis.sentiment import (
    analyze_sentiment, dict_to_signal, run_full_analysis,
    run_analysis_with_provider, run_risk_assessment, run_macro_context,
    _format_sr_text,
)
from analysis.synergy import decide
from analysis.exit_strategy import generate_trading_plan
from analysis.verification import verify_recommendation, VerificationResult
from avanza.certificates import search_certificates
from storage.history import save_recommendation
from ui.translations import T


from storage.usage_tracker import get_usage, get_scan_count, can_scan, track_scan
from storage.scan_results import save_scan, load_scan


def _render_usage_bar(usage: dict):
    """Show live API credit usage using native Streamlit components."""
    providers = [
        ("Groq", "groq"),
        ("Gemini", "gemini"),
        ("Tavily", "tavily"),
    ]
    cols = st.columns(len(providers))
    for col, (label, key) in zip(cols, providers):
        u = usage[key]
        with col:
            st.caption(f"{label}: {u['used']}/{u['limit']}")
            st.progress(min(u["pct"] / 100, 1.0))


def render_daily_picks():
    """Render the auto-scan daily picks view."""
    today_str = datetime.now().strftime("%A %d %B %Y")
    usage = get_usage()
    scan_count = usage["scans"]["used"]
    scans_left = usage["scans"]["remaining"]

    col_title, col_btn = st.columns([4, 1])
    with col_title:
        st.markdown(
            f"""
            <div style="padding: 16px 0 8px 0;">
                <h1 style="margin-bottom: 0;">Unitron Handelsanalys</h1>
                <p style="color: #888; font-size: 18px;">
                    {today_str} &nbsp;·&nbsp;
                    Skanningar: {scan_count}/{usage['scans']['limit']}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_btn:
        st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
        if scans_left > 0:
            rescan = st.button(
                "🔄 Starta dagens skanning",
                type="primary", use_container_width=True,
            )
        else:
            st.button(
                "✅ Dagens skanning klar",
                type="secondary", use_container_width=True, disabled=True,
            )
            rescan = False

    # Live API usage dashboard
    _render_usage_bar(usage)

    if rescan:
        st.cache_data.clear()
        st.session_state.pop("scan_data", None)
        st.session_state.pop("scan_from_file", None)
        st.session_state["run_scan"] = True

    if st.session_state.get("run_scan") and "scan_data" not in st.session_state:
        MIN_CONFIDENCE = 0.55
        MIN_RR_RATIO = 1.5

        st.markdown("#### ⚙️ Steg 1 — Dubbel AI-screening (Groq + Gemini)")
        progress_text = st.empty()
        progress_bar = st.progress(0)
        live_log = st.empty()
        scan_data = {"results": [], "report": [], "log": []}
        total = len(ALL_ASSETS_FLAT)

        def _update_log(log_lines, container):
            container.markdown("\n".join(log_lines), unsafe_allow_html=False)

        def _run_analysis_args(tech, headlines_json, sr):
            return dict(
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

        for i, asset in enumerate(ALL_ASSETS_FLAT):
            progress_bar.progress((i + 1) / total)
            progress_text.caption(f"Analyserar {asset.display_name} ({i+1}/{total}) — Groq + Gemini...")

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
                    scan_data["log"].append(f"❌ {asset.display_name} — Ingen data")
                    _update_log(scan_data["log"], live_log)
                    continue

                tech = technical_analyze(df, ticker=asset.ticker)
                if tech is None:
                    entry["status"] = "For lite data"
                    entry["reason"] = f"Bara {len(df)} dagar"
                    scan_data["report"].append(entry)
                    scan_data["log"].append(f"❌ {asset.display_name} — För lite data")
                    _update_log(scan_data["log"], live_log)
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

                analysis_kwargs = _run_analysis_args(tech, headlines_json, sr)

                sr_text = _format_sr_text(
                    sr.supports, sr.resistances,
                    tech.current_price, tech.near_resistance, tech.near_support,
                )
                headlines_text = "\n".join(
                    f"{j+1}. {h['headline']}" for j, h in enumerate(headlines)
                ) if headlines else "No recent headlines available."

                # --- PASS 1: Groq full analysis ---
                groq_result = run_analysis_with_provider(
                    provider="Groq", asset_name=asset.display_name, **analysis_kwargs,
                )
                time.sleep(2)

                # --- PASS 2: Risk assessment (Groq) ---
                risk_data = run_risk_assessment(
                    asset_name=asset.display_name,
                    price=tech.current_price,
                    sma_alignment=tech.sma_alignment,
                    rsi=tech.rsi_value,
                    volume_ratio=tech.volume_ratio,
                    vix_value=tech.vix_value,
                    atr=tech.atr_value,
                    atr_ratio=tech.atr_ratio,
                    near_resistance=tech.near_resistance,
                    near_support=tech.near_support,
                    sr_text=sr_text,
                    headlines_text=headlines_text,
                )
                time.sleep(2)

                # --- PASS 3: Macro context (Groq) ---
                macro_data = run_macro_context(
                    asset_name=asset.display_name,
                    asset_type=asset.asset_type,
                    price=tech.current_price,
                    sma_alignment=tech.sma_alignment,
                    vix_value=tech.vix_value,
                    headlines_text=headlines_text,
                )
                time.sleep(2)

                # --- PASS 4: Gemini independent analysis ---
                gemini_result = run_analysis_with_provider(
                    provider="Gemini", asset_name=asset.display_name, **analysis_kwargs,
                )

                groq_verdict = groq_result.get("verdict", "NO_TRADE") if groq_result else "NO_TRADE"
                gemini_verdict = gemini_result.get("verdict", "NO_TRADE") if gemini_result else "NO_TRADE"
                groq_conf = groq_result.get("confidence", 0) if groq_result else 0
                gemini_conf = gemini_result.get("confidence", 0) if gemini_result else 0

                # Apply risk/macro modifiers to confidence
                risk_penalty = 0
                if risk_data:
                    risk_level = risk_data.get("overall_risk", "MEDIUM")
                    if risk_level == "CRITICAL":
                        risk_penalty = -0.20
                    elif risk_level == "HIGH":
                        risk_penalty = -0.10
                    elif risk_level == "LOW":
                        risk_penalty = 0.05
                    if not risk_data.get("safe_to_trade", True):
                        risk_penalty = min(risk_penalty, -0.15)

                macro_modifier = 0
                if macro_data:
                    macro_modifier = float(macro_data.get("recommendation_modifier", 0))
                    macro_modifier = max(-0.15, min(0.15, macro_modifier))

                groq_conf = max(0, min(1, groq_conf + risk_penalty + macro_modifier))
                gemini_conf = max(0, min(1, gemini_conf + risk_penalty + macro_modifier))

                groq_ok = groq_result is not None
                gemini_ok = gemini_result is not None

                # Determine consensus
                if groq_ok and gemini_ok:
                    same_direction = (
                        (groq_verdict == gemini_verdict and groq_verdict != "NO_TRADE")
                        or (groq_verdict in ("BUY_BULL", "BUY_BEAR") and gemini_verdict == groq_verdict)
                    )
                    if same_direction:
                        avg_conf = (groq_conf + gemini_conf) / 2
                        ai_result = groq_result if groq_conf >= gemini_conf else gemini_result
                        ai_result["confidence"] = round(avg_conf, 2)
                        ai_result["provider"] = f"Groq ({groq_conf:.0%}) + Gemini ({gemini_conf:.0%})"
                        ai_result["groq_analysis"] = groq_result.get("analysis", "")
                        ai_result["gemini_analysis"] = gemini_result.get("analysis", "")
                        ai_result["risk_data"] = risk_data
                        ai_result["macro_data"] = macro_data
                        consensus_tag = "KONSENSUS"
                    elif groq_verdict == "NO_TRADE" and gemini_verdict == "NO_TRADE":
                        ai_result = groq_result
                        ai_result["provider"] = "Groq + Gemini"
                        consensus_tag = "ENIGA: NO_TRADE"
                    else:
                        ai_result = {"verdict": "NO_TRADE", "confidence": 0, "provider": "Groq + Gemini"}
                        ai_result["analysis"] = (
                            f"AI:erna är oense: Groq={groq_verdict} ({groq_conf:.0%}), "
                            f"Gemini={gemini_verdict} ({gemini_conf:.0%}). Ingen konsensus = ingen trade."
                        )
                        ai_result["key_factors"] = []
                        ai_result["risks"] = [f"Groq: {groq_verdict}, Gemini: {gemini_verdict} — ingen konsensus"]
                        consensus_tag = "OENIGA"
                elif groq_ok:
                    ai_result = groq_result
                    consensus_tag = "ENBART GROQ"
                elif gemini_ok:
                    ai_result = gemini_result
                    consensus_tag = "ENBART GEMINI"
                else:
                    ai_result = None
                    consensus_tag = "BÅDA MISSLYCKADES"

                # Build detailed log line showing all 4 passes
                risk_tag = ""
                if risk_data:
                    r_level = risk_data.get("overall_risk", "?")
                    risk_tag = f" | Risk: {r_level}"
                macro_tag = ""
                if macro_data:
                    m_bias = macro_data.get("macro_bias", "?")
                    m_mod = macro_data.get("recommendation_modifier", 0)
                    macro_tag = f" | Makro: {m_bias} ({m_mod:+.2f})"

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

                    if action != "NONE" and confidence < MIN_CONFIDENCE:
                        entry["action"] = "NONE"
                        entry["status"] = "Svag signal"
                        entry["reason"] = (
                            f"Konfidens ({confidence:.0%}) under minimum ({MIN_CONFIDENCE:.0%})"
                        )
                        scan_data["report"].append(entry)
                        scan_data["log"].append(
                            f"🟡 {asset.display_name} — {verdict} avvisad ({consensus_tag}{risk_tag}{macro_tag}, "
                            f"konfidens {confidence:.0%} < 55%)"
                        )
                        _update_log(scan_data["log"], live_log)
                        time.sleep(2)
                        continue

                    trading_plan = None
                    if action != "NONE":
                        trading_plan = generate_trading_plan(tech, action)
                        if trading_plan:
                            rr_val = (trading_plan.reward_amount / trading_plan.risk_amount
                                      if trading_plan.risk_amount > 0 else 0)
                            if rr_val < MIN_RR_RATIO:
                                entry["action"] = "NONE"
                                entry["status"] = "Svag signal"
                                entry["reason"] = (
                                    f"Risk/reward {trading_plan.risk_reward_ratio} under minimum (1:1.5)"
                                )
                                scan_data["report"].append(entry)
                                scan_data["log"].append(
                                    f"🟡 {asset.display_name} — {verdict} avvisad (R/R {trading_plan.risk_reward_ratio} < 1:1.5)"
                                )
                                _update_log(scan_data["log"], live_log)
                                time.sleep(3)
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
                            "headlines": headlines,
                        })
                        scan_data["log"].append(
                            f"🟢 {asset.display_name} — **{verdict}** ({confidence:.0%}) "
                            f"✅ {consensus_tag}{risk_tag}{macro_tag}"
                        )
                    else:
                        entry["status"] = "Ingen signal"
                        entry["reason"] = ai_result.get("analysis", "")[:120]
                        scan_data["log"].append(
                            f"⚪ {asset.display_name} — NO_TRADE ({consensus_tag}{risk_tag}{macro_tag})"
                        )
                else:
                    # Both AIs failed — fallback to rule-based engine
                    week_range = get_52_week_range(df)
                    w52_low = week_range[0] if week_range else None
                    decision = decide(
                        tech=tech, sent=sent,
                        week_52_low=w52_low,
                        is_crypto=asset.asset_type == "crypto",
                    )
                    entry["ai_verdict"] = "fallback"

                    if decision.action != "NONE" and decision.confidence_score < MIN_CONFIDENCE:
                        decision.action = "NONE"
                        decision.reasoning = [f"Konfidens ({decision.confidence_score:.0%}) under minimum"]

                    fallback_plan = None
                    if decision.action != "NONE":
                        fallback_plan = generate_trading_plan(tech, decision.action)
                        if fallback_plan:
                            fb_rr = (fallback_plan.reward_amount / fallback_plan.risk_amount
                                     if fallback_plan.risk_amount > 0 else 0)
                            if fb_rr < MIN_RR_RATIO:
                                decision.action = "NONE"
                                decision.reasoning = [f"Risk/reward ({fallback_plan.risk_reward_ratio}) under minimum"]
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
                            "headlines": headlines,
                        })
                        scan_data["log"].append(
                            f"🟢 {asset.display_name} — **{decision.action}** ({decision.confidence_score:.0%}) ✅ KANDIDAT (fallback)"
                        )
                    else:
                        entry["status"] = "Ingen signal"
                        entry["reason"] = decision.reasoning[0] if decision.reasoning else ""
                        scan_data["log"].append(
                            f"⚪ {asset.display_name} — Ingen signal (fallback)"
                        )

            except Exception as e:
                entry["status"] = "Fel"
                entry["reason"] = str(e)[:80]
                scan_data["log"].append(
                    f"🔴 {asset.display_name} — FEL: {str(e)[:50]}"
                )

            scan_data["report"].append(entry)
            _update_log(scan_data["log"], live_log)
            time.sleep(3)

        scan_data["results"].sort(
            key=lambda x: x.get("ai_result", {}).get("confidence", 0)
                          if x.get("ai_result") else 0,
            reverse=True,
        )

        # --- Stage 2: Deep dive on candidates (Devil's Advocate + risk news) ---
        if scan_data["results"]:
            n_candidates = len(scan_data["results"])
            scan_data["log"].append("")
            scan_data["log"].append(f"#### ⚙️ Steg 2 — Djupanalys av {n_candidates} kandidat{'er' if n_candidates > 1 else ''}")
            _update_log(scan_data["log"], live_log)

            progress_text.caption("Steg 2: Djupanalys — Devil's Advocate + risksökning...")
            progress_bar.progress(0)

            verified_results = []
            for idx, result in enumerate(scan_data["results"]):
                asset = result["asset"]
                ai_result = result.get("ai_result", {})
                tech = result["tech"]
                tp = result.get("trading_plan")
                provider = ai_result.get("provider", "Groq") if ai_result else "fallback"
                verdict = ai_result.get("verdict", "BUY_BULL") if ai_result else "BUY_BULL"

                progress_text.caption(
                    f"Djupanalys {asset.display_name} ({idx+1}/{n_candidates})..."
                )
                progress_bar.progress((idx + 1) / n_candidates)
                scan_data["log"].append(f"🔍 Djupanalys {asset.display_name}...")
                _update_log(scan_data["log"], live_log)

                sr = tech.support_resistance
                sr_text = _format_sr_text(
                    sr.supports, sr.resistances,
                    tech.current_price, tech.near_resistance, tech.near_support,
                )

                raw_headlines = result.get("headlines", [])
                headlines_text = "\n".join(
                    f"- {h.get('headline', '')}" for h in raw_headlines
                ) if raw_headlines else "Inga rubriker tillgängliga."

                try:
                    verification = verify_recommendation(
                        asset_name=asset.display_name,
                        first_provider="Groq",
                        verdict=verdict,
                        confidence=ai_result.get("confidence", 0) if ai_result else 0,
                        tech=tech,
                        sr_text=sr_text,
                        headlines_text=headlines_text,
                        stop_loss=tp.stop_loss if tp else 0,
                        take_profit=tp.take_profit if tp else 0,
                        news_keywords=asset.news_keywords,
                    )
                except Exception as ve:
                    verification = None
                    scan_data["log"].append(
                        f"   🔴 Djupanalys kraschade: {str(ve)[:60]}"
                    )
                    _update_log(scan_data["log"], live_log)

                result["verification"] = verification

                if verification and verification.verified:
                    verified_results.append(result)
                    scan_data["log"].append(
                        f"   ✅ {asset.display_name} — VERIFIERAD "
                        f"(Devil's Advocate: {verification.devils_advocate_risk} risk)"
                    )
                elif verification and not verification.verified:
                    result["rejected_reason"] = "verification_failed"
                    verified_results.append(result)
                    scan_data["log"].append(
                        f"   ⚠️ {asset.display_name} — EJ VERIFIERAD "
                        f"(risk: {verification.devils_advocate_risk})"
                    )
                else:
                    verified_results.append(result)
                    scan_data["log"].append(
                        f"   🔄 {asset.display_name} — Djupanalys misslyckades"
                    )

                _update_log(scan_data["log"], live_log)

            scan_data["results"] = verified_results
        else:
            scan_data["log"].append("")
            scan_data["log"].append("ℹ️ Inga kandidater — steg 2 hoppades över")

        # Final summary
        n_signals = len([r for r in scan_data["results"]
                         if r.get("verification") and r["verification"].verified])
        n_unverified = len([r for r in scan_data["results"]
                            if r.get("verification") and not r["verification"].verified])
        scan_data["log"].append("")
        scan_data["log"].append(
            f"**Klart!** {total} tillgångar screenade med dubbel AI → "
            f"{len(scan_data['results'])} kandidater → "
            f"{n_signals} verifierade, {n_unverified} ej verifierade"
        )
        _update_log(scan_data["log"], live_log)

        st.session_state["scan_data"] = scan_data
        save_scan(scan_data)
        track_scan()
        progress_bar.empty()
        progress_text.empty()

    if "scan_data" not in st.session_state:
        saved = load_scan()
        if saved:
            st.session_state["scan_data"] = saved
            st.session_state["scan_from_file"] = True
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
                    <div style="font-size: 64px; margin-bottom: 16px;">📊</div>
                    <h2 style="color: #ccc;">Välkommen till Unitron</h2>
                    <p style="color: #888; font-size: 16px;">
                        Tryck på <strong>"Skanna nu"</strong> för att starta dagens marknadsanalys.<br>
                        AI-motorn screnar alla tillgångar i två steg och visar bara de bästa möjligheterna.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

    scan_data = st.session_state["scan_data"]
    is_saved = st.session_state.get("scan_from_file", False)

    if is_saved:
        scan_time = scan_data.get("scan_time", "")
        st.caption(f"Visar sparade resultat från dagens skanning ({scan_time})")

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
                    Tvåstegsverifiering: AI #1 screenade alla tillgångar, AI #2 verifierade kandidaterna.<br>
                    Ingen tillgång klarade båda stegen (konfidens ≥55%, R/R ≥1:1.5, AI-konsensus).<br>
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


def _get(obj, key, default=None):
    """Access attribute or dict key — works for both dataclass objects and loaded dicts."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _render_pick_card(result: dict, rank: int):
    """Render a single daily pick with exit strategy and verification."""
    asset = result["asset"]
    tech = result["tech"]
    ai_result = result.get("ai_result")
    tp = result.get("trading_plan")
    verif = result.get("verification")

    asset_name = _get(asset, "display_name", "")
    asset_ticker = _get(asset, "ticker", "")
    asset_category = _get(asset, "category", "")

    price = _get(tech, "current_price", 0)
    rsi = _get(tech, "rsi_value", 0)
    vol = _get(tech, "volume_ratio", 0)
    vix = _get(tech, "vix_value")

    if ai_result:
        verdict = ai_result.get("verdict", "NO_TRADE")
        confidence = ai_result.get("confidence", 0)
        analysis_text = ai_result.get("analysis", "")
        key_factors = ai_result.get("key_factors", [])
        risks = ai_result.get("risks", [])
        provider = ai_result.get("provider", "")
    else:
        decision = result.get("decision")
        if isinstance(decision, dict):
            d_action = decision.get("action", "NONE")
            d_conf = decision.get("confidence_score", 0)
            d_reason = decision.get("reasoning", [])
        elif decision:
            d_action = decision.action
            d_conf = decision.confidence_score
            d_reason = decision.reasoning
        else:
            d_action, d_conf, d_reason = "NONE", 0, []
        verdict = f"BUY_{d_action}" if d_action != "NONE" else "NO_TRADE"
        confidence = d_conf
        analysis_text = "; ".join(d_reason) if d_reason else ""
        key_factors = []
        risks = []
        provider = "regelbaserad"

    if verdict == "BUY_BULL":
        color, icon, action_text, action = "#00C853", "📈", "KÖP BULL-CERTIFIKAT", "BULL"
    elif verdict == "BUY_BEAR":
        color, icon, action_text, action = "#FF1744", "📉", "KÖP BEAR-CERTIFIKAT", "BEAR"
    else:
        return

    confidence_pct = f"{confidence:.0%}"

    v_verified = _get(verif, "verified", False) if verif else None
    if verif:
        badge = "✅ VERIFIERAD" if v_verified else "⚠️ EJ VERIFIERAD"
        badge_color = "#00C853" if v_verified else "#FF9100"
    else:
        badge, badge_color = "🔄 Ej verifierad", "#888"

    tp_entry = _get(tp, "entry_price", 0) if tp else 0
    tp_sl = _get(tp, "stop_loss", 0) if tp else 0
    tp_tp = _get(tp, "take_profit", 0) if tp else 0
    tp_sl_method = _get(tp, "stop_loss_method", "") if tp else ""
    tp_tp_method = _get(tp, "take_profit_method", "") if tp else ""
    tp_rr = _get(tp, "risk_reward_ratio", "") if tp else ""

    tp_html = ""
    if tp:
        tp_html = (
            f"<div style='font-size: 14px; color: #ccc; border-top: 1px solid #444; "
            f"padding-top: 10px; margin-top: 4px; display: flex; gap: 24px; flex-wrap: wrap;'>"
            f"<div><span style=\"color: #888;\">Ingång:</span> <strong>{tp_entry:,.2f}</strong></div>"
            f"<div><span style=\"color: #888;\">Stop-Loss:</span> <strong style=\"color: #FF6B6B;\">{tp_sl:,.2f}</strong>"
            f" <span style=\"font-size:11px;color:#666;\">({tp_sl_method})</span></div>"
            f"<div><span style=\"color: #888;\">Målkurs:</span> <strong style=\"color: #69F0AE;\">{tp_tp:,.2f}</strong>"
            f" <span style=\"font-size:11px;color:#666;\">({tp_tp_method})</span></div>"
            f"<div><span style=\"color: #888;\">R/R:</span> <strong>{tp_rr}</strong></div></div>"
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
                        #{rank} {html.escape(asset_name)}
                    </div>
                    <div style="font-size: 14px; color: #888;">
                        {html.escape(asset_ticker)} — {html.escape(asset_category)} — via {html.escape(provider)}
                        &nbsp;&nbsp;<span style="color: {badge_color}; font-weight: 600;">{badge}</span>
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
                <strong>Pris:</strong> {price:,.2f} —
                <strong>RSI:</strong> {rsi:.1f} —
                <strong>Volym:</strong> {'N/A' if vol < 0.01 else f'{vol:.1f}x'} —
                <strong>VIX:</strong> {vix if vix else 'N/A'}
            </div>
            <div style="font-size: 14px; color: #aaa; line-height: 1.5; margin-bottom: 8px;">
                {html.escape(analysis_text)}
            </div>
            {tp_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander(f"Detaljer & Handelsplan — {html.escape(asset_name)}"):
        if key_factors:
            st.markdown("**Avgörande faktorer:**")
            for f in key_factors:
                st.markdown(f"- {f}")

        if risks:
            st.markdown("**Risker:**")
            for r in risks:
                st.caption(f"⚠️ {r}")

        # Risk Assessment details (from dedicated risk pass)
        risk_data = ai_result.get("risk_data") if ai_result else None
        if risk_data:
            st.divider()
            risk_colors = {"LOW": "#00C853", "MEDIUM": "#FFD600", "HIGH": "#FF9100", "CRITICAL": "#FF1744"}
            rl = risk_data.get("overall_risk", "UNKNOWN")
            st.markdown(f"**🛡️ Riskbedömning:** <span style='color:{risk_colors.get(rl, '#888')}'>{rl}</span>", unsafe_allow_html=True)
            if risk_data.get("biggest_threat"):
                st.caption(f"Största hot: {risk_data['biggest_threat']}")
            for rd in risk_data.get("risks", [])[:5]:
                sev = rd.get("severity", "")
                st.caption(f"{'🔴' if sev == 'critical' else '🟡' if sev == 'high' else '⚪'} [{rd.get('category', '')}] {rd.get('description', '')}")

        # Macro Context details (from dedicated macro pass)
        macro_data = ai_result.get("macro_data") if ai_result else None
        if macro_data:
            st.divider()
            m_bias = macro_data.get("macro_bias", "neutral")
            m_conf = macro_data.get("macro_confidence", 0)
            st.markdown(f"**🌍 Makroanalys:** {m_bias.upper()} ({m_conf:.0%} konfidens)")
            if macro_data.get("institutional_view"):
                st.caption(f"Institutionell vy: {macro_data['institutional_view']}")
            if macro_data.get("best_case"):
                st.caption(f"Bästa scenario: {macro_data['best_case']}")
            if macro_data.get("worst_case"):
                st.caption(f"Värsta scenario: {macro_data['worst_case']}")
            for evt in macro_data.get("key_events", [])[:3]:
                st.caption(f"📅 {evt}")

        if tp:
            st.divider()
            st.markdown("**Handelsplan (Exit-strategi):**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Ingång", f"{tp_entry:,.2f}")
            with col2:
                st.metric(f"Stop-Loss ({tp_sl_method})", f"{tp_sl:,.2f}")
            with col3:
                st.metric(f"Målkurs ({tp_tp_method})", f"{tp_tp:,.2f}")
            with col4:
                st.metric("Risk/Reward", tp_rr)

            st.caption(f"Stop-Loss: {_get(tp, 'stop_loss_reasoning', '')}")
            st.caption(f"Målkurs: {_get(tp, 'take_profit_reasoning', '')}")
            st.caption(f"Trailing Stop: {_get(tp, 'trailing_stop_reasoning', '')}")

        supports = _get(tech, "supports", []) if isinstance(tech, dict) else tech.support_resistance.supports
        resistances = _get(tech, "resistances", []) if isinstance(tech, dict) else tech.support_resistance.resistances
        if supports or resistances:
            st.divider()
            sr_col1, sr_col2 = st.columns(2)
            with sr_col1:
                st.markdown("**Stöd:**")
                for i, s in enumerate(supports[:3]):
                    st.caption(f"S{i+1}: {s:,.2f}")
            with sr_col2:
                st.markdown("**Motstånd:**")
                for i, r_val in enumerate(resistances[:3]):
                    st.caption(f"R{i+1}: {r_val:,.2f}")

        if verif:
            st.divider()
            st.markdown("**🔍 Steg 2 — Verifiering:**")

            v_agrees = _get(verif, "second_ai_agrees", False)
            v_provider = _get(verif, "second_ai_provider", "")
            v_conf = _get(verif, "second_ai_confidence", 0)
            v_reasoning = _get(verif, "second_ai_reasoning", "")
            v_disagree = _get(verif, "disagreement_points", [])
            v_da_risk = _get(verif, "devils_advocate_risk", "UNKNOWN")
            v_biggest = _get(verif, "biggest_risk", "")
            v_da_rec = _get(verif, "devils_advocate_recommendation", "")
            v_counter = _get(verif, "counter_arguments", [])
            v_risk_news = _get(verif, "risk_headlines", [])

            v_col1, v_col2 = st.columns(2)
            with v_col1:
                agree_icon = "✅" if v_agrees else "❌"
                st.markdown(
                    f"**Andra AI:n ({html.escape(v_provider)}):** "
                    f"{agree_icon} {'Håller med' if v_agrees else 'Håller INTE med'} "
                    f"({v_conf:.0%} konfidens)"
                )
                if v_reasoning:
                    st.caption(v_reasoning)
                if v_disagree:
                    for dp in v_disagree:
                        st.caption(f"⚠️ {dp}")

            with v_col2:
                risk_colors = {"LOW": "#00C853", "MEDIUM": "#FFD600", "HIGH": "#FF9100", "CRITICAL": "#FF1744"}
                rc = risk_colors.get(v_da_risk, "#888")
                st.markdown(
                    f"**Djävulens Advokat:** "
                    f"<span style='color:{rc};font-weight:700;'>{v_da_risk}</span> risk",
                    unsafe_allow_html=True,
                )
                if v_biggest:
                    st.caption(f"🎯 Största risken: {v_biggest}")
                if v_da_rec:
                    st.caption(f"💡 {v_da_rec}")

            if v_counter:
                st.markdown("**Motargument:**")
                for ca in v_counter:
                    st.caption(f"❗ {ca}")

            if v_risk_news:
                st.markdown("**Ytterligare risknyheter:**")
                for rh in v_risk_news[:3]:
                    headline = rh.get("headline", "") if isinstance(rh, dict) else str(rh)
                    st.caption(f"📰 {headline}")

        certs = search_certificates(asset_ticker, action)
        if certs:
            st.divider()
            st.markdown(f"**{T['avanza_title']}:**")
            for cert in certs[:3]:
                if cert["url"]:
                    st.markdown(f"- [{cert['name']}]({cert['url']}) ({T['leverage_label']}: {cert['leverage']})")
                else:
                    st.write(f"- {cert['name']}")
