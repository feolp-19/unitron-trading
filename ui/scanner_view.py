import pandas as pd
import streamlit as st

from config import CURATED_ASSETS, ALL_ASSETS_FLAT, Asset
from scanner.watchlist import scan_watchlist
from ui.translations import T


def render_scanner():
    """Render the watchlist scanner view."""
    st.header(T["watchlist_title"])
    st.caption(T["watchlist_subtitle"])

    # Let user pick which categories to scan
    categories = list(CURATED_ASSETS.keys())
    selected_cats = st.multiselect(
        "Välj kategorier att skanna",
        categories,
        default=categories[:2],
    )

    assets_to_scan = []
    for cat in selected_cats:
        assets_to_scan.extend(CURATED_ASSETS.get(cat, []))

    st.caption(f"{len(assets_to_scan)} tillgångar valda")

    if st.button(T["scan_button"], type="primary", use_container_width=True):
        if not assets_to_scan:
            st.warning("Välj minst en kategori")
            return

        results = scan_watchlist(assets_to_scan)
        st.session_state["scan_results"] = results

    # Display results
    results = st.session_state.get("scan_results", [])
    if not results:
        return

    actionable = [r for r in results if r["action"] != "NONE"]
    no_trade = [r for r in results if r["action"] == "NONE"]

    # Actionable signals
    if actionable:
        st.subheader(f"{T['actionable_title']} ({len(actionable)})")
        for result in actionable:
            _render_scan_result(result, expanded=True)
    else:
        st.info(T["signal_none"])

    # No-trade signals
    if no_trade:
        with st.expander(f"{T['no_trade_title']} ({len(no_trade)})"):
            for result in no_trade:
                _render_scan_result(result, expanded=False)


def _render_scan_result(result: dict, expanded: bool):
    """Render a single scan result as a compact card."""
    asset = result["asset"]
    action = result["action"]

    if action == "BULL":
        icon = "🟢"
        color = "#00C853"
    elif action == "BEAR":
        icon = "🔴"
        color = "#FF1744"
    else:
        icon = "⚪"
        color = "#757575"

    confidence_pct = f"{result['confidence']:.0%}"

    with st.container():
        st.markdown(
            f"""
            <div style="
                background: {color}11;
                border-left: 4px solid {color};
                border-radius: 8px;
                padding: 12px 20px;
                margin-bottom: 8px;
            ">
                <strong>{icon} {asset['display_name']}</strong>
                ({asset['ticker']}) —
                <span style="color: {color}; font-weight: bold;">{action}</span>
                | {T['confidence_label']}: {confidence_pct}
                | RSI: {result['rsi']:.1f}
                | {T['price_label']}: {result['price']:,.2f}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if expanded and result.get("reasoning"):
            cols = st.columns([2, 1, 1])
            with cols[0]:
                for r in result["reasoning"]:
                    st.caption(f"  - {r}")
            with cols[1]:
                if result["stop_loss"]:
                    st.caption(f"{T['stop_loss_label']}: {result['stop_loss']:,.2f}")
            with cols[2]:
                if result["take_profit"]:
                    st.caption(f"{T['take_profit_label']}: {result['take_profit']:,.2f}")

        if result.get("warnings"):
            for w in result["warnings"]:
                st.caption(f"⚠️ {w}")
