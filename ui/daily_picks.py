"""Unitron 8-Stage Intelligence Pipeline — Master Report UI."""

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
    today_str = datetime.now().strftime("%A %d %B %Y")
    usage = get_usage()
    scan_count = usage["scans"]["used"]
    scans_left = usage["scans"]["remaining"]

    col_title, col_btn = st.columns([4, 1])
    with col_title:
        st.markdown(
            f"""
            <div style="padding: 16px 0 8px 0;">
                <h1 style="margin-bottom: 0;">Unitron 8-Stegs Djupanalys</h1>
                <p style="color: #888; font-size: 18px;">
                    {today_str} &nbsp;&middot;&nbsp;
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
                f"Djupskanna ({scans_left} kvar)",
                type="primary", use_container_width=True,
            )
        else:
            st.button(
                "Dagens analys klar",
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

    if st.session_state.get("run_scan") and "scan_data" not in st.session_state:
        _run_scan_ui()

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
                    <div style="font-size: 64px; margin-bottom: 16px;">---</div>
                    <h2 style="color: #888;">Dagens analyser ar slut</h2>
                    <p style="color: #666; font-size: 16px;">
                        Kom tillbaka imorgon kl 07:00 for ny djupanalys.
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
        st.caption(f"Visar sparade resultat fran {scan_data.get('scan_date', 'idag')} ({scan_time})")

    _render_master_report(scan_data)


def _run_scan_ui():
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
            <div style="font-size: 32px; margin-bottom: 8px;">&#128300;</div>
            <div style="font-size: 18px; font-weight: 600;">8-stegs djupanalys pagar...</div>
            <div style="color: #888; font-size: 14px;">
                Steg 0-8: Datafundament &rarr; Makroankare &rarr; Flerlins Skanning &rarr;
                Djupforskning &rarr; Devil's Advocate &rarr; Korsvalidering<br>
                ~15-25 min — maximalt djup med 5s mellan varje AI-anrop
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    progress_bar = st.progress(0)
    progress_text = st.empty()
    live_log = st.empty()
    log_lines = []

    total_steps = len(ALL_ASSETS_FLAT) * 6 + 40
    step_counter = [0]

    def ui_log(phase, msg):
        step_counter[0] += 1
        progress_bar.progress(min(step_counter[0] / total_steps, 1.0))
        stage_names = {
            "stage0": "Steg 0", "stage1": "Steg 1", "stage2": "Steg 2",
            "stage3": "Steg 3-4", "stage4": "Steg 3-4",
            "stage5": "Steg 5", "stage6": "Steg 6", "stage7": "Steg 7",
            "stage8": "Steg 8", "synthesis": "Syntes",
        }
        if phase in stage_names:
            progress_text.caption(f"[{stage_names[phase]}] {msg}")
        log_lines.append(msg)
        live_log.markdown("\n\n".join(log_lines[-20:]), unsafe_allow_html=False)

    result = run_deep_scan(log_fn=ui_log, max_top=5, api_delay=5)

    scan_data = _serialize_deep_result(result)
    st.session_state["scan_data"] = scan_data
    save_scan(scan_data)
    track_scan()

    progress_bar.empty()
    progress_text.empty()


def _serialize_deep_result(result: DeepScanResult) -> dict:
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
            "stage2": c.get("stage2"),
            "stage5": c.get("stage5"),
            "stage6": c.get("stage6"),
            "stage7": c.get("stage7"),
            "synthesis": c.get("synthesis"),
            "headlines": c.get("headlines", []),
            "final_verdict": c.get("final_verdict", "NO_TRADE"),
            "final_confidence": c.get("final_confidence", 0),
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

        return out

    return {
        "scan_date": result.scan_date,
        "scan_time": result.scan_time,
        "vix_value": result.vix_value,
        "dxy_value": result.dxy_value,
        "us10y_value": result.us10y_value,
        "market_regime": result.market_regime,
        "regime_report": result.regime_report,
        "global_sentiment": result.global_sentiment,
        "all_scores": result.all_scores,
        "top5": [_ser_candidate(c) for c in result.top5],
        "final_picks": [_ser_candidate(c) for c in result.final_picks],
        "yesterday_review": result.yesterday_review,
        "log": result.log,
        "total_assets": result.total_assets,
        "stage_calls": result.stage_calls,
    }


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# ---------------------------------------------------------------------------
# MASTER REPORT
# ---------------------------------------------------------------------------

def _render_master_report(scan_data: dict):
    vix = scan_data.get("vix_value")
    dxy = scan_data.get("dxy_value")
    us10y = scan_data.get("us10y_value")
    regime = scan_data.get("market_regime", "")
    global_sent = scan_data.get("global_sentiment", "")
    all_scores = scan_data.get("all_scores", [])
    top5 = scan_data.get("top5", [])
    final_picks = scan_data.get("final_picks", [])
    total = scan_data.get("total_assets", 0)
    yesterday = scan_data.get("yesterday_review")

    _render_heartbeat(vix, dxy, us10y, regime, global_sent, total, len(all_scores), len(top5), len(final_picks))

    if yesterday:
        _render_yesterday_review(yesterday)

    if regime:
        _render_regime_report(scan_data.get("regime_report", "{}"), regime)

    if all_scores:
        _render_scores_overview(all_scores)

    if final_picks:
        st.markdown(
            f"""
            <div style="
                background: #00C85315; border: 1px solid #00C85344;
                border-radius: 12px; padding: 16px 24px;
                text-align: center; margin-bottom: 24px;
            ">
                <span style="font-size: 18px;">
                    8-stegs djupanalysen rekommenderar <strong>{len(final_picks)}</strong> trade{'s' if len(final_picks) > 1 else ''}
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
                <div style="font-size: 64px; margin-bottom: 16px;">&#128737;</div>
                <h2 style="color: #888;">Inga rekommendationer idag</h2>
                <p style="color: #666; font-size: 16px;">
                    8-stegs djupanalysen fann inga tillgangar som klarade alla granskningssteg.<br>
                    <strong>Att sta utanfor marknaden ar ofta det klokaste beslutet.</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if top5:
        st.divider()
        st.markdown("### Djupanalys — Topp 5 Finalister")
        for candidate in top5:
            _render_candidate_expander(candidate)

    log_lines = scan_data.get("log", [])
    if log_lines:
        with st.expander("Fullstandig skanningslogg", expanded=False):
            st.markdown("\n\n".join(log_lines))

    stage_calls = scan_data.get("stage_calls", {})
    total_calls = sum(v for v in stage_calls.values() if isinstance(v, int))
    st.markdown(
        f"""
        <div style="
            background: #1A1D23; border-top: 1px solid #333;
            padding: 12px 24px; margin-top: 32px;
            text-align: center; border-radius: 8px;
        ">
            <span style="color: #00C853; font-weight: 700;">Analysens komplexitetsgrad: 8 STEG / MAXIMERAD</span><br>
            <span style="color: #666; font-size: 12px;">
                Totalt {total_calls} AI-anrop over 8 analyssteg
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_heartbeat(vix, dxy, us10y, regime, global_sent, total, scored, top5_count, picks_count):
    vix_text = f"{vix:.1f}" if vix else "N/A"
    dxy_text = f"{dxy:.2f}" if dxy else "N/A"
    us10y_text = f"{us10y:.2f}%" if us10y else "N/A"

    if vix is not None:
        if vix > 30:
            vix_color, vix_label = "#FF1744", "EXTREM RADSLA"
        elif vix > 25:
            vix_color, vix_label = "#FF9100", "Forhojd radsla"
        elif vix > 20:
            vix_color, vix_label = "#FFD600", "Forsiktighet"
        elif vix > 15:
            vix_color, vix_label = "#888", "Normal"
        else:
            vix_color, vix_label = "#00C853", "Lugn marknad"
    else:
        vix_color, vix_label = "#888", "N/A"

    regime_colors = {
        "RISK_ON": "#00C853", "RISK_OFF": "#FF1744", "REFLATION": "#FF9100",
        "STAGFLATION": "#FF5252", "TIGHTENING": "#FFD600", "EASING": "#69F0AE",
        "TRANSITION": "#888",
    }
    regime_color = regime_colors.get(regime, "#888")

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1A1D2380, #0D1117);
            border: 1px solid #333; border-radius: 16px;
            padding: 24px 32px; margin-bottom: 20px;
        ">
            <h3 style="margin: 0 0 16px 0; color: #fff;">Morgonens Hjartslag</h3>
            <div style="display: flex; gap: 24px; flex-wrap: wrap; justify-content: center;">
                <div style="text-align: center; min-width: 100px;">
                    <div style="font-size: 24px; font-weight: 700; color: {regime_color};">{html.escape(regime or 'N/A')}</div>
                    <div style="color: #888; font-size: 12px;">Marknadsregim</div>
                </div>
                <div style="text-align: center; min-width: 100px;">
                    <div style="font-size: 24px; font-weight: 700; color: {vix_color};">{vix_text}</div>
                    <div style="color: #888; font-size: 12px;">VIX — {vix_label}</div>
                </div>
                <div style="text-align: center; min-width: 100px;">
                    <div style="font-size: 24px; font-weight: 700;">{dxy_text}</div>
                    <div style="color: #888; font-size: 12px;">DXY</div>
                </div>
                <div style="text-align: center; min-width: 100px;">
                    <div style="font-size: 24px; font-weight: 700;">{us10y_text}</div>
                    <div style="color: #888; font-size: 12px;">US 10Y Yield</div>
                </div>
                <div style="text-align: center; min-width: 80px;">
                    <div style="font-size: 24px; font-weight: 700;">{total}</div>
                    <div style="color: #888; font-size: 12px;">Skannade</div>
                </div>
                <div style="text-align: center; min-width: 80px;">
                    <div style="font-size: 24px; font-weight: 700; color: #4285F4;">{top5_count}</div>
                    <div style="color: #888; font-size: 12px;">Djupanalyserade</div>
                </div>
                <div style="text-align: center; min-width: 80px;">
                    <div style="font-size: 24px; font-weight: 700; color: #00C853;">{picks_count}</div>
                    <div style="color: #888; font-size: 12px;">Rekommenderade</div>
                </div>
            </div>
            <div style="color: #aaa; font-size: 14px; margin-top: 16px; text-align: center;">
                {html.escape(global_sent)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_yesterday_review(review: dict):
    with st.expander("Gardagens Utvardering", expanded=False):
        acc = review.get("accuracy_pct", 0)
        adj = review.get("confidence_adjustment", "maintain")
        brief = review.get("learning_brief", "")

        adj_colors = {"raise": "#00C853", "lower": "#FF1744", "maintain": "#FFD600"}
        adj_labels = {"raise": "HOJD", "lower": "SANKD", "maintain": "BIBEHALLEN"}

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Traffsakerhet", f"{acc:.0f}%")
        with col2:
            c = adj_colors.get(adj, "#888")
            l = adj_labels.get(adj, adj.upper())
            st.markdown(
                f"**Konfidensjustering:** <span style='color:{c}; font-weight:700;'>{l}</span>",
                unsafe_allow_html=True,
            )

        if brief:
            st.info(brief)

        reviews = review.get("reviews", [])
        if reviews:
            import pandas as pd
            rows = []
            for r in reviews:
                correct = r.get("correct", False)
                rows.append({
                    "Tillgang": r.get("asset", ""),
                    "Riktning": r.get("yesterday_direction", ""),
                    "Ingang": r.get("yesterday_entry", 0),
                    "Dagspris": r.get("today_price", 0),
                    "P/L": r.get("pnl_pct", ""),
                    "Ratt": "JA" if correct else "NEJ",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _render_regime_report(regime_report_json: str, regime: str):
    with st.expander(f"Steg 1: Marknadsregim — {regime}", expanded=False):
        try:
            data = json.loads(regime_report_json) if isinstance(regime_report_json, str) else regime_report_json
        except Exception:
            data = {}

        if data.get("regime_description"):
            st.markdown(f"**Regimbeskrivning:** {data['regime_description']}")

        biases = data.get("asset_biases", {})
        if biases:
            cols = st.columns(4)
            labels = {"equities": "Aktier", "commodities": "Ravaror", "crypto": "Krypto", "safe_havens": "Trygga hamnar"}
            bias_colors = {"bullish": "#00C853", "bearish": "#FF1744", "neutral": "#FFD600"}
            for col, (key, label) in zip(cols, labels.items()):
                bias = biases.get(key, "neutral")
                c = bias_colors.get(bias, "#888")
                with col:
                    st.markdown(
                        f"**{label}:** <span style='color:{c};'>{bias.upper()}</span>",
                        unsafe_allow_html=True,
                    )

        risks = data.get("macro_risks", [])
        if risks:
            st.markdown("**Topp 3 Makrorisker:**")
            for i, risk in enumerate(risks):
                st.markdown(f"{i+1}. {risk}")

        levels = data.get("key_levels", {})
        if levels:
            st.markdown("**Bevaknnigsnivaer:**")
            for key, val in levels.items():
                st.caption(f"{key}: {val}")


def _render_scores_overview(all_scores: list[dict]):
    with st.expander("Steg 2: Kvantitativt Betyg — Alla Tillgangar", expanded=False):
        import pandas as pd
        scores_sorted = sorted(all_scores, key=lambda x: x.get("score", 0), reverse=True)
        rows = []
        for s in scores_sorted:
            score = s.get("score", 0)
            if score >= 7:
                indicator = "+++"
            elif score >= 5:
                indicator = "++"
            elif score >= 3:
                indicator = "+"
            else:
                indicator = "-"
            rows.append({
                "": indicator,
                "Tillgang": s.get("name", ""),
                "Betyg": f"{score:.1f}/10",
                "Riktning": s.get("direction", ""),
                "Konfidens": f"{s.get('confidence', 0):.0%}",
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True)


def _render_deep_pick(pick: dict, rank: int):
    asset = pick.get("asset", {})
    tech = pick.get("tech", {})
    s2 = pick.get("stage2", {}) or {}
    synthesis = pick.get("synthesis", {}) or {}
    s6 = pick.get("stage6", {}) or {}
    tp = pick.get("trading_plan")

    name = _get(asset, "display_name", "")
    ticker = _get(asset, "ticker", "")
    category = _get(asset, "category", "")

    verdict = pick.get("final_verdict", synthesis.get("verdict", "NO_TRADE"))
    confidence = pick.get("final_confidence", synthesis.get("final_confidence", 0))

    if verdict == "BUY_BULL":
        color, action_text, action = "#00C853", "KOP BULL-CERTIFIKAT", "BULL"
    elif verdict == "BUY_BEAR":
        color, action_text, action = "#FF1744", "KOP BEAR-CERTIFIKAT", "BEAR"
    else:
        return

    price = _get(tech, "current_price", 0)
    rsi = _get(tech, "rsi_value", 0)
    vol = _get(tech, "volume_ratio", 0)
    composite = s2.get("composite_score", 0)

    risk_rating = s6.get("risk_rating", "?")
    risk_colors = {"LOW": "#00C853", "MEDIUM": "#FFD600", "HIGH": "#FF9100", "CRITICAL": "#FF1744"}
    risk_c = risk_colors.get(risk_rating, "#888")

    key_catalyst = synthesis.get("key_catalyst", "")
    time_horizon = synthesis.get("time_horizon", "")

    tp_entry = _get(tp, "entry_price", synthesis.get("entry_price", 0))
    tp_sl = _get(tp, "stop_loss", synthesis.get("stop_loss", 0))
    tp_tp = _get(tp, "take_profit", synthesis.get("take_profit", 0))
    tp_rr = _get(tp, "risk_reward_ratio", synthesis.get("risk_reward", ""))
    tp_sl_method = _get(tp, "stop_loss_method", "")
    tp_tp_method = _get(tp, "take_profit_method", "")

    tp_html = ""
    if tp_entry and tp_sl and tp_tp:
        tp_html = (
            f"<div style='font-size: 14px; color: #ccc; border-top: 1px solid #444; "
            f"padding-top: 10px; margin-top: 4px; display: flex; gap: 24px; flex-wrap: wrap;'>"
            f"<div><span style='color: #888;'>Ingang:</span> <strong>{tp_entry:,.2f}</strong></div>"
            f"<div><span style='color: #888;'>Stop-Loss:</span> <strong style='color: #FF6B6B;'>{tp_sl:,.2f}</strong>"
            f" <span style='font-size:11px;color:#666;'>({tp_sl_method})</span></div>"
            f"<div><span style='color: #888;'>Malkurs:</span> <strong style='color: #69F0AE;'>{tp_tp:,.2f}</strong>"
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
                <strong>Komposit:</strong> {composite:.1f}/10
            </div>
            <div style="font-size: 14px; color: #FFD600; margin-bottom: 8px;">
                <strong>Katalysator:</strong> {html.escape(str(key_catalyst))}
            </div>
            {tp_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander(f"Fullstandig 8-Stegs Djupanalys — {html.escape(str(name))}"):
        _render_candidate_details(pick)

    certs = search_certificates(str(ticker), action)
    if certs:
        with st.expander(f"Avanza-certifikat — {html.escape(str(name))}"):
            for cert in certs[:5]:
                if cert["url"]:
                    st.markdown(f"- [{cert['name']}]({cert['url']}) ({T['leverage_label']}: {cert['leverage']})")
                else:
                    st.write(f"- {cert['name']}")


def _render_candidate_expander(candidate: dict):
    asset = candidate.get("asset", {})
    name = _get(asset, "display_name", "?")
    s2 = candidate.get("stage2", {}) or {}
    verdict = candidate.get("final_verdict", "NO_TRADE")
    score = s2.get("composite_score", 0)
    direction = s2.get("consensus_direction", "?")

    indicator = "++" if score >= 7 else ("+" if score >= 5 else "-")

    with st.expander(f"[{indicator}] {name} — Betyg {score:.1f}/10, Riktning: {direction}, Slutdom: {verdict}"):
        _render_candidate_details(candidate)


def _render_candidate_details(candidate: dict):
    s2 = candidate.get("stage2", {}) or {}
    s5 = candidate.get("stage5", {}) or {}
    s6 = candidate.get("stage6", {}) or {}
    s7 = candidate.get("stage7", {}) or {}
    synthesis = candidate.get("synthesis", {}) or {}
    tech = candidate.get("tech", {})
    tp = candidate.get("trading_plan")

    # --- Stage 2: Multi-Lens Technical ---
    st.markdown("#### Steg 2: Flerlins Teknisk Analys (Groq)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Kompositbetyg", f"{s2.get('composite_score', 0):.1f}/10")
    with col2:
        st.metric("Riktning", s2.get("consensus_direction", "?"))
    with col3:
        st.metric("Konfidens", f"{s2.get('confidence', 0):.0%}")

    lenses = s2.get("lenses", {})
    if lenses:
        for lens_name, lens_data in lenses.items():
            if isinstance(lens_data, dict):
                score = lens_data.get("score", "?")
                analysis = lens_data.get("analysis", "")
                st.caption(f"**{lens_name.upper()}:** {score}/10 — {analysis[:120]}")

    # --- Stage 5: High-Dimensional Research ---
    st.divider()
    st.markdown("#### Steg 5: Hogdimensionell Djupforskning (Gemini)")

    mod_a = s5.get("A")
    mod_b = s5.get("B")
    mod_c = s5.get("C")
    mod_d = s5.get("D")

    if mod_a:
        with st.expander("Modul A: Historiska Analogier"):
            analogs = mod_a.get("analogs", [])
            for a in analogs:
                st.markdown(f"**{a.get('period', '')}** (Relevans: {a.get('relevance_score', '?')}/10)")
                st.caption(f"{a.get('description', '')} → {a.get('outcome', '')}")
            st.markdown(f"**Historisk dom:** {mod_a.get('historical_verdict', '?')}")
            if mod_a.get("key_lesson"):
                st.info(mod_a["key_lesson"])

    if mod_b:
        with st.expander("Modul B: Intermarknadsanalys"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**DXY-korrelation:** {mod_b.get('dxy_correlation', '?')}")
                st.markdown(f"**DXY-dom:** {mod_b.get('dxy_verdict', '?')}")
                st.markdown(f"**Rantepaverkan:** {mod_b.get('yield_impact', '?')}")
            with col2:
                st.markdown(f"**Verklig rorelse:** {mod_b.get('real_move_confidence', 0):.0%}")
                st.markdown(f"**Korsbekraftelse:** {mod_b.get('cross_asset_confirmation', '?')}")
                st.markdown(f"**Dom:** {mod_b.get('inter_market_verdict', '?')}")
            if mod_b.get("analysis"):
                st.caption(mod_b["analysis"])

    if mod_c:
        with st.expander("Modul C: Utbud/Efterfragan & Geopolitik"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Utbudstryck:** {mod_c.get('supply_pressure', '?')}")
                st.markdown(f"**Efterfragetrend:** {mod_c.get('demand_trend', '?')}")
                st.markdown(f"**Sasongsbias:** {mod_c.get('seasonal_bias', '?')}")
            with col2:
                st.markdown(f"**Geopolitisk risk:** {mod_c.get('geopolitical_risk', '?')}")
                st.markdown(f"**Fundamental dom:** {mod_c.get('fundamental_verdict', '?')}")
                st.markdown(f"**Konfidens:** {mod_c.get('confidence', 0):.0%}")
            factors = mod_c.get("geopolitical_factors", [])
            if factors:
                st.caption("Faktorer: " + ", ".join(factors))
            if mod_c.get("key_driver"):
                st.info(f"Nyckelfaktor: {mod_c['key_driver']}")

    if mod_d:
        with st.expander("Modul D: Scenariostresstest"):
            scenarios = mod_d.get("scenarios", [])
            for sc in scenarios:
                st.markdown(f"**{sc.get('name', '')}:** {sc.get('pct_move', '?')} ({sc.get('timeframe', '?')})")
                st.caption(sc.get("reasoning", ""))
            st.markdown(f"**Varsta pris:** {mod_d.get('worst_case_price', '?')}")
            st.markdown(f"**Max drawdown:** {mod_d.get('max_drawdown_pct', '?')}")
            tail = mod_d.get("tail_risk_verdict", "?")
            tail_c = {"MANAGEABLE": "#00C853", "ELEVATED": "#FF9100", "EXTREME": "#FF1744"}.get(tail, "#888")
            st.markdown(
                f"**Tail-risk:** <span style='color:{tail_c}; font-weight:700;'>{tail}</span>",
                unsafe_allow_html=True,
            )

    # --- Stage 6: Devil's Advocate ---
    st.divider()
    st.markdown("#### Steg 6: Djavulens Advokat (Gemini)")

    if s6:
        risk_rating = s6.get("risk_rating", "?")
        risk_colors = {"LOW": "#00C853", "MEDIUM": "#FFD600", "HIGH": "#FF9100", "CRITICAL": "#FF1744"}
        rc = risk_colors.get(risk_rating, "#888")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"**Riskbetyg:** <span style='color:{rc}; font-weight:700; font-size:18px;'>{risk_rating}</span>",
                unsafe_allow_html=True,
            )
            proceed = s6.get("should_proceed", False)
            st.markdown(f"**Rekommendation:** {'FORTSATT' if proceed else 'AVBRYT'}")
        with col2:
            if s6.get("recommendation"):
                st.markdown(f"**Slutsats:** {s6['recommendation']}")
            if s6.get("invalidation_level"):
                st.markdown(f"**Invalideringsniva:** {s6['invalidation_level']:,.2f}")

        failure_reasons = s6.get("failure_reasons", [])
        if failure_reasons:
            st.markdown("**Riskfaktorer:**")
            for fr in failure_reasons:
                sev = fr.get("severity", "")
                ftype = fr.get("type", "")
                indicator = "[!]" if sev == "critical" else ("[*]" if sev == "high" else "[~]")
                st.markdown(f"{indicator} **[{ftype}]** {fr.get('reason', '')}")

        if s6.get("worst_case_scenario"):
            st.error(f"**Varsta scenario:** {s6['worst_case_scenario']}")
    else:
        st.caption("Djavulens Advokat ej genomford.")

    # --- Stage 7: Cross-Validation ---
    st.divider()
    st.markdown("#### Steg 7: Korsvalidering (Groq)")

    if s7:
        cv_verdict = s7.get("final_verdict", "?")
        cv_conf = s7.get("adjusted_confidence", 0)
        cv_colors = {"APPROVE": "#00C853", "REJECT": "#FF1744", "REDUCE_CONFIDENCE": "#FF9100"}
        cv_c = cv_colors.get(cv_verdict, "#888")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"**Dom:** <span style='color:{cv_c}; font-weight:700;'>{cv_verdict}</span>",
                unsafe_allow_html=True,
            )
        with col2:
            st.metric("Justerad konfidens", f"{cv_conf:.0%}")

        hallucinations = s7.get("hallucinations_found", [])
        if hallucinations:
            st.warning("**Hallucinationer:**")
            for h in hallucinations:
                st.caption(f"- {h}")

        errors = s7.get("logical_errors", [])
        if errors:
            st.error("**Logiska fel:**")
            for e in errors:
                st.caption(f"- {e}")

        missing = s7.get("missing_risks", [])
        if missing:
            st.info("**Missade risker:**")
            for m in missing:
                st.caption(f"- {m}")

        if s7.get("auditor_notes"):
            st.markdown(f"**Revisionsnoteringar:** {s7['auditor_notes']}")
    else:
        st.caption("Korsvalidering ej genomford.")

    # --- Synthesis: Chain-of-Thought ---
    if synthesis:
        st.divider()
        st.markdown("#### Slutgiltig Syntes & Dom")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Slutgiltig dom", synthesis.get("verdict", "?"))
        with col2:
            st.metric("Konfidens", f"{synthesis.get('final_confidence', 0):.0%}")
        with col3:
            st.metric("Tidshorisont", synthesis.get("time_horizon", "?"))

        cot = synthesis.get("chain_of_thought", "")
        if cot:
            st.markdown("**Resonemang:**")
            st.markdown(f"> {cot}")

        if synthesis.get("key_catalyst"):
            st.success(f"**Nyckelkatalysator:** {synthesis['key_catalyst']}")
        if synthesis.get("biggest_risk"):
            st.warning(f"**Storsta risk:** {synthesis['biggest_risk']}")
        if synthesis.get("exit_strategy"):
            st.info(f"**Exit-strategi:** {synthesis['exit_strategy']}")

    # --- Trading Plan ---
    if tp:
        st.divider()
        st.markdown("#### Handelsplan (Exit-strategi)")
        tp_entry = _get(tp, "entry_price", 0)
        tp_sl = _get(tp, "stop_loss", 0)
        tp_tp = _get(tp, "take_profit", 0)
        tp_rr = _get(tp, "risk_reward_ratio", "")
        tp_sl_method = _get(tp, "stop_loss_method", "")
        tp_tp_method = _get(tp, "take_profit_method", "")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Ingang", f"{tp_entry:,.2f}")
        with col2:
            st.metric(f"Stop-Loss ({tp_sl_method})", f"{tp_sl:,.2f}")
        with col3:
            st.metric(f"Malkurs ({tp_tp_method})", f"{tp_tp:,.2f}")
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
            st.markdown("**Stod:**")
            for i, s in enumerate(supports[:3]):
                st.caption(f"S{i+1}: {s:,.2f}")
        with sr_col2:
            st.markdown("**Motstand:**")
            for i, r_val in enumerate(resistances[:3]):
                st.caption(f"R{i+1}: {r_val:,.2f}")
