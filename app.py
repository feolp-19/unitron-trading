import streamlit as st

from config import CURATED_ASSETS, ALL_ASSETS_FLAT, get_asset_by_ticker, create_custom_asset, AI_PROVIDER
from ui.translations import T
from ui.dashboard import render_dashboard
from ui.scanner_view import render_scanner
from storage.history import load_history

import pandas as pd

st.set_page_config(
    page_title=T["app_title"],
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def check_password() -> bool:
    """Password gate for private access. Returns True if authenticated."""
    try:
        correct_pw = st.secrets["passwords"]["app_password"]
    except (KeyError, FileNotFoundError):
        return True  # No password configured = local dev mode

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if st.session_state["authenticated"]:
        return True

    st.title("🔒 Unitron Handelsanalys")
    pw = st.text_input("Lösenord", type="password")
    if st.button("Logga in", type="primary"):
        if pw == correct_pw:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Fel lösenord")
    return False


if not check_password():
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.title(T["app_title"])
    st.caption(T["app_subtitle"])
    st.divider()

    st.markdown(f"**{T['ai_provider_label']}:** `{AI_PROVIDER}`")
    st.divider()

    st.subheader(T["sidebar_title"])

    categories = list(CURATED_ASSETS.keys())
    selected_category = st.selectbox(T["category_label"], categories)

    assets_in_category = CURATED_ASSETS[selected_category]
    asset_names = [a.display_name for a in assets_in_category]
    selected_name = st.selectbox(T["asset_label"], asset_names)

    selected_asset = next(
        (a for a in assets_in_category if a.display_name == selected_name),
        assets_in_category[0],
    )

    st.divider()
    custom_ticker = st.text_input(T["custom_ticker_label"], value="")

    if custom_ticker.strip():
        existing = get_asset_by_ticker(custom_ticker.strip())
        if existing:
            selected_asset = existing
        else:
            selected_asset = create_custom_asset(custom_ticker.strip())
        st.success(f"Vald: {selected_asset.display_name} ({selected_asset.ticker})")

# --- Main Content ---
tab_analyze, tab_watchlist, tab_history = st.tabs([
    T["tab_analyze"],
    T["tab_watchlist"],
    T["tab_history"],
])

with tab_analyze:
    render_dashboard(selected_asset)

with tab_watchlist:
    render_scanner()

with tab_history:
    st.header(T["history_title"])
    history = load_history(days=30)
    if history:
        df = pd.DataFrame(history)
        display_cols = {
            "timestamp": T["col_date"],
            "asset_name": T["col_asset"],
            "action": T["col_action"],
            "confidence": T["col_confidence"],
            "entry_price": T["col_price"],
            "stop_loss": T["col_stop_loss"],
            "take_profit": T["col_take_profit"],
        }
        available_cols = [c for c in display_cols.keys() if c in df.columns]
        df_display = df[available_cols].rename(columns=display_cols)
        st.dataframe(df_display, hide_index=True, use_container_width=True)
    else:
        st.info(T["history_empty"])
