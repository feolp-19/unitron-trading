"""Unitron Morgonrapport — 3-phase deep scan with Master Report UI."""

import html
import json
import time
from datetime import datetime

import streamlit as st

from config import ALL_ASSETS_FLAT
from analysis.deep_scan import run_deep_scan, DeepScanResult
from analysis.exit_strategy import generate_trading_plan
from avanza.certificates import search_certificates
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
    """Render the deep-scan Master Report view."""
    today_str = datetime.now().strftime("%A %d %B %Y")
    usage = get_usage()
    scan_count = usage["scans"]["used"]
    scans_left = usage["scans"]["remaining"]

    col_title, col_btn = st.columns([4, 1])
    with col_title:
        st.markdown(
            f"""
            <div style="padding: 16px 0 8px 0;">
                <h1 style="margin-bottom: 0;">Unitron Djupanalys</h1>
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
                f"🔬 Djupskanna ({scans_left} kvar)",
                type="primary", use_container_width=True,
            )
        else:
            st.button(
                "✅ Dagens analys klar",
                type="secondary", use_container_width=True, disabled=True,
            )
            rescan = False

    _render_usage_bar(usage)

    if rescan:
        st.cache_data.clear()
        st.session_state.pop("scan_data", None)
        st.session_state.pop("scan_from_file", None)
        st.session_state["run_scan"] = True

    if "scan_data" not in st.session_state and not st.session_state.get("run_scan"):
        saved = load_scan()
        if saved:
            st.session_state["scan_data"] = saved
            st.session_state["scan_from_file"] = True
        elif scans_left > 0:
            st.session_state["run_scan"] = True

    # --- RUN DEEP SCAN ---
    if st.session_state.get("run_scan") and "scan_data" not in st.session_state:
        st.markdown(
            """
            <div style="
                background: #1A1D2390;
                border: 1px solid #444;
                border-radius: 12px;
                padding: 16px 24px;
                margin-bottom: 16px;
                text-align: center;
            ">
                <div style="font-size: 32px; margin-bottom: 8px;">🔬</div>
                <div style="font-size: 18px; font-weight: 600;">Djupanalys pågår...</div>
                <div style="color: #888; font-size: 14px;">
                    3-fas pipeline: Kvantitativ Granskning → Djävulens Advokat → Makrosyntes<br>
                    ~5 min — analyserar 17+ tillgångar med maximal noggrannhet
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        progress_bar = st.progress(0)
        progress_text = st.empty()
        live_log = st.empty()
        log_lines = []

        total_steps = len(ALL_ASSETS_FLAT) + 10  # phase1 + phase2 + phase3
        step_counter = [0]

        def ui_log(phase, msg):
            step_counter[0] += 1
            progress_bar.progress(min(step_counter[0] / total_steps, 1.0))
            if phase in ("phase1", "phase2", "phase3"):
                progress_text.caption(msg)
            log_lines.append(msg)
            live_log.markdown("\n\n".join(log_lines[-15:]), unsafe_allow_html=False)

        result = run_deep_scan(log_fn=ui_log, max_top=5, api_delay=5)

        scan_data = _serialize_deep_result(result)
        st.session_state["scan_data"] = scan_data
        save_scan(scan_data)
        track_scan()

        progress_bar.empty()
        progress_text.empty()

    if "scan_data" not in st.session_state:
        if scans_left > 0:
            st.info("Startar djupanalys automatiskt...")
            st.session_state["run_scan"] = True
            st.rerun()
        else:
            st.markdown(
                """
                <div style="
                    background: #1A1D23; border: 1px solid #333;
                    border-radius: 16px; padding: 48px 32px;
                    text-align: center; margin: 16px 0 24px 0;
                ">
                    <div style="font-size: 64px; margin-bottom: 16px;">⛔</div>
                    <h2 style="color: #888;">Dagens analyser är slut</h2>
                    <p style="color: #666; font-size: 16px;">
                        Resultat från den automatiska morgonskanningen saknas.<br>
                        Kom tillbaka imorgon kl 07:00 för ny djupanalys.
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
        st.caption(f"Visar sparade resultat från {scan_data.get('scan_date', 'idag')} ({scan_time})")

    _render_master_report(scan_data)


def _serialize_deep_result(result: DeepScanResult) -> dict:
    """Convert DeepScanResult to a JSON-serializable dict."""
    def _ser_candidate(c):
        asset = c["asset"]
        tech = c["tech"]
        sr = tech.support_resistance if hasattr(tech, "support_resistance") else None

        asset_d = {
            "ticker": asset.ticker, "display_name": asset.display_name,
            "news_keywords": asset.news_keywords, "asset_type": asset.asset_type,
            "category": asset.category,
        } if hasattr(asset, "ticker") else asset

        tech_d = {
            "current_price": tech.current_price, "rsi_value": tech.rsi_value,
            "sma_20": tech.sma_20, "sma_50": tech.sma_50, "sma_200": tech.sma_200,
            "sma_50w": tech.sma_50w, "atr_value": tech.atr_value,
            "price_vs_sma": tech.price_vs_sma,
            "price_vs_weekly_sma": tech.price_vs_weekly_sma,
            "sma_alignment": tech.sma_alignment, "sma_bias": tech.sma_bias,
            "rsi_trend_2d": tech.rsi_trend_2d, "atr_ratio": tech.atr_ratio,
            "volume_ratio": tech.volume_ratio,
            "vix_value": tech.vix_value, "vix_level": tech.vix_level,
            "near_resistance": tech.near_resistance, "near_support": tech.near_support,
            "supports": sr.supports if sr else [],
            "resistances": sr.resistances if sr else [],
            "macd_value": tech.macd_value, "macd_signal": tech.macd_signal,
            "macd_histogram": tech.macd_histogram, "macd_cross": tech.macd_cross,
            "bb_upper": tech.bb_upper, "bb_lower": tech.bb_lower,
            "bb_middle": tech.bb_middle, "bb_position": tech.bb_position,
            "bb_width": tech.bb_width,
        } if hasattr(tech, "current_price") else tech

        out = {
            "asset": asset_d, "tech": tech_d,
            "phase1": c.get("phase1"), "phase2": c.get("phase2"),
            "phase3": c.get("phase3"), "headlines": c.get("headlines", []),
        }

        tp = c.get("trading_plan")
        if tp and not isinstance(tp, dict):
            out["trading_plan"] = {
                "entry_price": tp.entry_price, "stop_loss": tp.stop_loss,
                "stop_loss_method": tp.stop_loss_method,
                "stop_loss_reasoning": tp.stop_loss_reasoning,
                "take_profit": tp.take_profit,
                "take_profit_method": tp.take_profit_method,
                "take_profit_reasoning": tp.take_profit_reasoning,
                "risk_reward_ratio": tp.risk_reward_ratio,
                "risk_amount": tp.risk_amount, "reward_amount": tp.reward_amount,
                "trailing_stop_level": tp.trailing_stop_level,
                "trailing_stop_reasoning": tp.trailing_stop_reasoning,
            }
        else:
            out["trading_plan"] = tp

        if "final_verdict" in c:
            out["final_verdict"] = c["final_verdict"]
            out["final_confidence"] = c.get("final_confidence", 0)

        return out

    return {
        "scan_date": result.scan_date,
        "scan_time": result.scan_time,
        "vix_value": result.vix_value,
        "dxy_value": result.dxy_value,
        "global_sentiment": result.global_sentiment,
        "all_scores": result.all_scores,
        "top5": [_ser_candidate(c) for c in result.top5],
        "final_picks": [_ser_candidate(c) for c in result.final_picks],
        "log": result.log,
        "total_assets": result.total_assets,
        "phase1_calls": result.phase1_calls,
        "phase2_calls": result.phase2_calls,
        "phase3_calls": result.phase3_calls,
    }


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# ---------------------------------------------------------------------------
# MASTER REPORT
# ---------------------------------------------------------------------------

def _render_master_report(scan_data: dict):
    """Render the complete Unitron Master Report."""

    vix = scan_data.get("vix_value")
    dxy = scan_data.get("dxy_value")
    global_sent = scan_data.get("global_sentiment", "")
    all_scores = scan_data.get("all_scores", [])
    top5 = scan_data.get("top5", [])
    final_picks = scan_data.get("final_picks", [])
    total = scan_data.get("total_assets", 0)

    # --- "Morgonens Hjärtslag" ---
    _render_heartbeat(vix, dxy, global_sent, total, len(all_scores), len(top5), len(final_picks))

    # --- Score Overview ---
    if all_scores:
        _render_scores_overview(all_scores)

    # --- Final Picks ---
    if final_picks:
        st.markdown(
            f"""
            <div style="
                background: #00C85315; border: 1px solid #00C85344;
                border-radius: 12px; padding: 16px 24px;
                text-align: center; margin-bottom: 24px;
            ">
                <span style="font-size: 18px;">
                    Djupanalysen rekommenderar <strong>{len(final_picks)}</strong> trade{'s' if len(final_picks) > 1 else ''}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for i, pick in enumerate(final_picks):
            _render_deep_pick(pick, rank=i + 1)
    else:
        st.markdown(
            """
            <div style="
                background: #1A1D23; border: 1px solid #333;
                border-radius: 16px; padding: 48px 32px;
                text-align: center; margin: 16px 0 24px 0;
            ">
                <div style="font-size: 64px; margin-bottom: 16px;">🛡️</div>
                <h2 style="color: #888;">Inga rekommendationer idag</h2>
                <p style="color: #666; font-size: 16px;">
                    3-fas djupanalysen fann inga tillgångar som klarade alla granskningssteg.<br>
                    <strong>Att stå utanför marknaden är ofta det klokaste beslutet.</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Deep-Dive Expanders for all Top 5 ---
    if top5:
        st.divider()
        st.markdown("### 🔍 Djupanalys — Topp 5 Kandidater")
        for candidate in top5:
            _render_candidate_expander(candidate)

    # --- Scan Log ---
    log_lines = scan_data.get("log", [])
    if log_lines:
        with st.expander("📋 Fullständig skanningslogg", expanded=False):
            st.markdown("\n\n".join(log_lines))

    # --- Footer ---
    p1 = scan_data.get("phase1_calls", 0)
    p2 = scan_data.get("phase2_calls", 0)
    p3 = scan_data.get("phase3_calls", 0)
    st.markdown(
        f"""
        <div style="
            background: #1A1D23; border-top: 1px solid #333;
            padding: 12px 24px; margin-top: 32px;
            text-align: center; border-radius: 8px;
        ">
            <span style="color: #00C853; font-weight: 700;">Analysens komplexitetsgrad: MAXIMERAD</span><br>
            <span style="color: #666; font-size: 12px;">
                Fas 1: {p1} kvantitativa granskningar · Fas 2: {p2} skeptiska granskningar · Fas 3: {p3} makrosynteser
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_heartbeat(vix, dxy, global_sent, total, scored, top5_count, picks_count):
    """Render the 'Morgonens Hjärtslag' section."""
    if vix is not None:
        if vix > 30:
            vix_color, vix_label = "#FF1744", "EXTREM RÄDSLA"
        elif vix > 25:
            vix_color, vix_label = "#FF9100", "Förhöjd rädsla"
        elif vix > 20:
            vix_color, vix_label = "#FFD600", "Försiktighet"
        elif vix > 15:
            vix_color, vix_label = "#888", "Normal"
        else:
            vix_color, vix_label = "#00C853", "Lugn marknad"
    else:
        vix_color, vix_label = "#888", "N/A"

    dxy_text = f"{dxy:.2f}" if dxy else "N/A"
    vix_text = f"{vix:.1f}" if vix else "N/A"

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1A1D2380, #0D1117);
            border: 1px solid #333; border-radius: 16px;
            padding: 24px 32px; margin-bottom: 20px;
        ">
            <h3 style="margin: 0 0 16px 0; color: #fff;">💓 Morgonens Hjärtslag</h3>
            <div style="display: flex; gap: 32px; flex-wrap: wrap; justify-content: center;">
                <div style="text-align: center; min-width: 120px;">
                    <div style="font-size: 28px; font-weight: 700; color: {vix_color};">{vix_text}</div>
                    <div style="color: #888; font-size: 13px;">VIX — {vix_label}</div>
                </div>
                <div style="text-align: center; min-width: 120px;">
                    <div style="font-size: 28px; font-weight: 700;">{dxy_text}</div>
                    <div style="color: #888; font-size: 13px;">Dollar Index (DXY)</div>
                </div>
                <div style="text-align: center; min-width: 120px;">
                    <div style="font-size: 28px; font-weight: 700;">{total}</div>
                    <div style="color: #888; font-size: 13px;">Tillgångar skannade</div>
                </div>
                <div style="text-align: center; min-width: 120px;">
                    <div style="font-size: 28px; font-weight: 700; color: #4285F4;">{top5_count}</div>
                    <div style="color: #888; font-size: 13px;">Djupanalyserade</div>
                </div>
                <div style="text-align: center; min-width: 120px;">
                    <div style="font-size: 28px; font-weight: 700; color: #00C853;">{picks_count}</div>
                    <div style="color: #888; font-size: 13px;">Rekommenderade</div>
                </div>
            </div>
            <div style="color: #aaa; font-size: 14px; margin-top: 16px; text-align: center;">
                {html.escape(global_sent)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_scores_overview(all_scores: list[dict]):
    """Render the ranked score table for all assets."""
    with st.expander("📊 Kvantitativt Betyg — Alla Tillgångar", expanded=False):
        import pandas as pd
        scores_sorted = sorted(all_scores, key=lambda x: x.get("score", 0), reverse=True)
        rows = []
        for s in scores_sorted:
            score = s.get("score", 0)
            if score >= 7:
                emoji = "🟢"
            elif score >= 5:
                emoji = "🟡"
            else:
                emoji = "⚪"
            rows.append({
                "": emoji,
                "Tillgång": s.get("name", ""),
                "Betyg": f"{score}/10",
                "Riktning": s.get("direction", ""),
                "Konfidens": f"{s.get('confidence', 0):.0%}",
                "Sammanfattning": s.get("summary", "")[:80],
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True)


def _render_deep_pick(pick: dict, rank: int):
    """Render a final recommendation card with full depth."""
    asset = pick.get("asset", {})
    tech = pick.get("tech", {})
    p1 = pick.get("phase1", {}) or {}
    p2 = pick.get("phase2", {}) or {}
    p3 = pick.get("phase3", {}) or {}
    tp = pick.get("trading_plan")

    name = _get(asset, "display_name", "")
    ticker = _get(asset, "ticker", "")
    category = _get(asset, "category", "")

    verdict = pick.get("final_verdict", p3.get("verdict", "NO_TRADE"))
    confidence = pick.get("final_confidence", p3.get("final_confidence", 0))

    if verdict == "BUY_BULL":
        color, icon, action_text, action = "#00C853", "📈", "KÖP BULL-CERTIFIKAT", "BULL"
    elif verdict == "BUY_BEAR":
        color, icon, action_text, action = "#FF1744", "📉", "KÖP BEAR-CERTIFIKAT", "BEAR"
    else:
        return

    price = _get(tech, "current_price", 0)
    rsi = _get(tech, "rsi_value", 0)
    vol = _get(tech, "volume_ratio", 0)
    vix = _get(tech, "vix_value")

    # DA risk badge
    risk_rating = p2.get("risk_rating", "?")
    risk_colors = {"LOW": "#00C853", "MEDIUM": "#FFD600", "HIGH": "#FF9100", "CRITICAL": "#FF1744"}
    risk_c = risk_colors.get(risk_rating, "#888")

    key_catalyst = p3.get("key_catalyst", "")
    time_horizon = p3.get("time_horizon", "")

    tp_entry = _get(tp, "entry_price", p3.get("entry_price", 0))
    tp_sl = _get(tp, "stop_loss", p3.get("stop_loss", 0))
    tp_tp = _get(tp, "take_profit", p3.get("take_profit", 0))
    tp_rr = _get(tp, "risk_reward_ratio", p3.get("risk_reward", ""))
    tp_sl_method = _get(tp, "stop_loss_method", "")
    tp_tp_method = _get(tp, "take_profit_method", "")

    tp_html = ""
    if tp_entry and tp_sl and tp_tp:
        tp_html = (
            f"<div style='font-size: 14px; color: #ccc; border-top: 1px solid #444; "
            f"padding-top: 10px; margin-top: 4px; display: flex; gap: 24px; flex-wrap: wrap;'>"
            f"<div><span style='color: #888;'>Ingång:</span> <strong>{tp_entry:,.2f}</strong></div>"
            f"<div><span style='color: #888;'>Stop-Loss:</span> <strong style='color: #FF6B6B;'>{tp_sl:,.2f}</strong>"
            f" <span style='font-size:11px;color:#666;'>({tp_sl_method})</span></div>"
            f"<div><span style='color: #888;'>Målkurs:</span> <strong style='color: #69F0AE;'>{tp_tp:,.2f}</strong>"
            f" <span style='font-size:11px;color:#666;'>({tp_tp_method})</span></div>"
            f"<div><span style='color: #888;'>R/R:</span> <strong>{tp_rr}</strong></div>"
            f"<div><span style='color: #888;'>Tidshorisont:</span> <strong>{time_horizon}</strong></div></div>"
        )

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {color}15, {color}08);
            border: 1px solid {color}44;
            border-radius: 16px; padding: 24px 32px; margin-bottom: 20px;
        ">
            <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 12px;">
                <span style="font-size: 36px;">{icon}</span>
                <div>
                    <div style="font-size: 24px; font-weight: 700; color: {color};">
                        #{rank} {html.escape(str(name))}
                    </div>
                    <div style="font-size: 14px; color: #888;">
                        {html.escape(str(ticker))} — {html.escape(str(category))} —
                        Risk: <span style="color: {risk_c}; font-weight: 600;">{risk_rating}</span>
                    </div>
                </div>
                <div style="margin-left: auto; text-align: right;">
                    <div style="
                        background: {color}; color: #000; font-weight: 700;
                        padding: 8px 20px; border-radius: 8px; font-size: 16px;
                    ">{action_text}</div>
                </div>
            </div>
            <div style="font-size: 15px; color: #ccc; margin-bottom: 8px;">
                <strong>Konfidens:</strong> {confidence:.0%} —
                <strong>Pris:</strong> {price:,.2f} —
                <strong>RSI:</strong> {rsi:.1f} —
                <strong>Volym:</strong> {'N/A' if vol < 0.01 else f'{vol:.1f}x'} —
                <strong>Setup:</strong> {p1.get('setup_score', '?')}/10
            </div>
            <div style="font-size: 14px; color: #FFD600; margin-bottom: 8px;">
                <strong>Katalysator:</strong> {html.escape(str(key_catalyst))}
            </div>
            {tp_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Detailed expander
    with st.expander(f"🔬 Fullständig Djupanalys — {html.escape(str(name))}"):
        _render_candidate_details(pick)

    # Avanza certificates
    certs = search_certificates(str(ticker), action)
    if certs:
        with st.expander(f"💰 Avanza-certifikat — {html.escape(str(name))}"):
            for cert in certs[:5]:
                if cert["url"]:
                    st.markdown(f"- [{cert['name']}]({cert['url']}) ({T['leverage_label']}: {cert['leverage']})")
                else:
                    st.write(f"- {cert['name']}")


def _render_candidate_expander(candidate: dict):
    """Render a detailed expander for a top-5 candidate (even non-picked ones)."""
    asset = candidate.get("asset", {})
    name = _get(asset, "display_name", "?")
    p1 = candidate.get("phase1", {}) or {}
    p3 = candidate.get("phase3", {}) or {}

    verdict = p3.get("verdict", "NO_TRADE") if p3 else "Ej analyserad"
    score = p1.get("setup_score", 0)
    direction = p1.get("direction", "?")

    emoji = "🟢" if score >= 7 else ("🟡" if score >= 5 else "⚪")

    with st.expander(f"{emoji} {name} — Betyg {score}/10, Riktning: {direction}, Slutdom: {verdict}"):
        _render_candidate_details(candidate)


def _render_candidate_details(candidate: dict):
    """Render the 3-phase detail sections for a candidate."""
    p1 = candidate.get("phase1", {}) or {}
    p2 = candidate.get("phase2", {}) or {}
    p3 = candidate.get("phase3", {}) or {}
    tech = candidate.get("tech", {})
    tp = candidate.get("trading_plan")

    # --- Phase 1: Technical Status ---
    st.markdown("#### 📊 Fas 1: Kvantitativ Granskning (Groq)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Setup-betyg", f"{p1.get('setup_score', 0)}/10")
    with col2:
        st.metric("Riktning", p1.get("direction", "?"))
    with col3:
        st.metric("Konfidens", f"{p1.get('confidence', 0):.0%}")

    if p1.get("trend_analysis"):
        st.markdown(f"**Trend:** {p1['trend_analysis']}")
    if p1.get("momentum_analysis"):
        st.markdown(f"**Momentum:** {p1['momentum_analysis']}")
    if p1.get("volatility_analysis"):
        st.markdown(f"**Volatilitet:** {p1['volatility_analysis']}")
    if p1.get("volume_analysis"):
        st.markdown(f"**Volym:** {p1['volume_analysis']}")
    if p1.get("setup_summary"):
        st.info(p1["setup_summary"])

    # --- Phase 2: Skeptical Risks ---
    st.divider()
    st.markdown("#### 😈 Fas 2: Djävulens Advokat")

    if p2:
        risk_rating = p2.get("risk_rating", "?")
        risk_colors = {"LOW": "#00C853", "MEDIUM": "#FFD600", "HIGH": "#FF9100", "CRITICAL": "#FF1744"}
        rc = risk_colors.get(risk_rating, "#888")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"**Riskbetyg:** <span style='color:{rc}; font-weight:700; font-size:18px;'>{risk_rating}</span>",
                unsafe_allow_html=True,
            )
            proceed = p2.get("should_proceed", False)
            st.markdown(f"**Rekommendation:** {'✅ Fortsätt' if proceed else '⛔ Avbryt'}")

        with col2:
            if p2.get("recommendation"):
                st.markdown(f"**Slutsats:** {p2['recommendation']}")
            if p2.get("invalidation_level"):
                st.markdown(f"**Invalideringsnivå:** {p2['invalidation_level']:,.2f}")

        failure_reasons = p2.get("failure_reasons", [])
        if failure_reasons:
            st.markdown("**Riskfaktorer:**")
            for fr in failure_reasons:
                sev = fr.get("severity", "")
                ftype = fr.get("type", "")
                icon = "🔴" if sev == "critical" else ("🟠" if sev == "high" else "🟡")
                st.markdown(f"{icon} **[{ftype}]** {fr.get('reason', '')}")

        if p2.get("worst_case_scenario"):
            st.error(f"**Värsta scenario:** {p2['worst_case_scenario']}")
    else:
        st.caption("Djävulens Advokat-granskning ej genomförd.")

    # --- Phase 3: Macro Synthesis ---
    st.divider()
    st.markdown("#### 🌍 Fas 3: Makrosyntes & Slutgiltig Dom (Gemini)")

    if p3:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Slutgiltig dom", p3.get("verdict", "?"))
        with col2:
            st.metric("Konfidens", f"{p3.get('final_confidence', 0):.0%}")
        with col3:
            st.metric("Tidshorisont", p3.get("time_horizon", "?"))

        cot = p3.get("chain_of_thought", "")
        if cot:
            st.markdown("**Resonemang (Chain-of-Thought):**")
            st.markdown(f"> {cot}")

        if p3.get("key_catalyst"):
            st.success(f"**Nyckelkatalysator:** {p3['key_catalyst']}")
        if p3.get("biggest_risk"):
            st.warning(f"**Största risk:** {p3['biggest_risk']}")
        if p3.get("exit_strategy"):
            st.info(f"**Exit-strategi:** {p3['exit_strategy']}")
    else:
        st.caption("Makrosyntes ej genomförd.")

    # --- Trading Plan ---
    if tp:
        st.divider()
        st.markdown("#### 📋 Handelsplan (Exit-strategi)")
        tp_entry = _get(tp, "entry_price", 0)
        tp_sl = _get(tp, "stop_loss", 0)
        tp_tp = _get(tp, "take_profit", 0)
        tp_rr = _get(tp, "risk_reward_ratio", "")
        tp_sl_method = _get(tp, "stop_loss_method", "")
        tp_tp_method = _get(tp, "take_profit_method", "")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Ingång", f"{tp_entry:,.2f}")
        with col2:
            st.metric(f"Stop-Loss ({tp_sl_method})", f"{tp_sl:,.2f}")
        with col3:
            st.metric(f"Målkurs ({tp_tp_method})", f"{tp_tp:,.2f}")
        with col4:
            st.metric("Risk/Reward", tp_rr)

        if _get(tp, "stop_loss_reasoning"):
            st.caption(f"SL: {_get(tp, 'stop_loss_reasoning', '')}")
        if _get(tp, "take_profit_reasoning"):
            st.caption(f"TP: {_get(tp, 'take_profit_reasoning', '')}")
        if _get(tp, "trailing_stop_reasoning"):
            st.caption(f"Trailing: {_get(tp, 'trailing_stop_reasoning', '')}")

    # --- S/R levels ---
    supports = _get(tech, "supports", [])
    resistances = _get(tech, "resistances", [])
    if isinstance(tech, dict):
        pass
    elif hasattr(tech, "support_resistance"):
        supports = tech.support_resistance.supports
        resistances = tech.support_resistance.resistances

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
