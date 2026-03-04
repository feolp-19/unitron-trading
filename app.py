import streamlit as st

from config import search_asset, AI_PROVIDER
from ui.daily_picks import render_daily_picks
from ui.dashboard import render_dashboard

st.set_page_config(
    page_title="Unitron Handelsanalys",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def check_password() -> bool:
    try:
        correct_pw = st.secrets["passwords"]["app_password"]
    except (KeyError, FileNotFoundError):
        return True

    if st.session_state.get("authenticated"):
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

tab_today, tab_analyze = st.tabs(["Idag", "Analysera"])

with tab_today:
    render_daily_picks()

with tab_analyze:
    st.markdown(
        """
        <div style="text-align: center; padding: 24px 0 8px 0;">
            <h1>Analysera en tillgång</h1>
            <p style="color: #888;">Sök på namn eller ticker — t.ex. "gold", "tesla", "DAX", "NVDA"</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    query = st.text_input(
        "Sök",
        placeholder="Skriv t.ex. gold, tesla, DAX, bitcoin, ERIC-B.ST...",
        label_visibility="collapsed",
    )

    if query.strip():
        asset = search_asset(query)
        if asset:
            render_dashboard(asset)
    else:
        st.markdown(
            """
            <div style="
                text-align: center;
                padding: 48px 0;
                color: #666;
            ">
                <div style="font-size: 48px; margin-bottom: 16px;">🔍</div>
                <p style="font-size: 16px;">
                    Skriv ett namn eller en ticker ovan för att starta analysen
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
