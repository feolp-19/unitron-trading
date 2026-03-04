import hmac
import time

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

MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_SECONDS = 60


def check_password() -> bool:
    if st.session_state.get("authenticated"):
        return True

    try:
        correct_pw = st.secrets["passwords"]["app_password"]
    except (KeyError, FileNotFoundError):
        st.error("Lösenord ej konfigurerat. Kontakta administratören.")
        st.stop()
        return False

    attempts = st.session_state.get("login_attempts", 0)
    locked_until = st.session_state.get("locked_until", 0)

    if time.time() < locked_until:
        remaining = int(locked_until - time.time())
        st.error(f"För många misslyckade försök. Vänta {remaining} sekunder.")
        st.stop()
        return False

    login_container = st.empty()
    with login_container.container():
        st.title("Unitron Handelsanalys")
        pw = st.text_input("Lösenord", type="password", key="pw_input")
        if st.button("Logga in", type="primary"):
            if hmac.compare_digest(pw.encode(), correct_pw.encode()):
                st.session_state["authenticated"] = True
                st.session_state["login_attempts"] = 0
                login_container.empty()
                st.rerun()
            else:
                attempts += 1
                st.session_state["login_attempts"] = attempts
                if attempts >= MAX_LOGIN_ATTEMPTS:
                    st.session_state["locked_until"] = time.time() + LOCKOUT_SECONDS
                    st.error(f"För många försök. Låst i {LOCKOUT_SECONDS} sekunder.")
                else:
                    remaining = MAX_LOGIN_ATTEMPTS - attempts
                    st.error(f"Fel lösenord ({remaining} försök kvar)")
    st.stop()
    return False


check_password()

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
