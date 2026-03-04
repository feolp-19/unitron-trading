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

tab_today, tab_analyze, tab_diag = st.tabs(["Idag", "Analysera", "Status"])

with tab_diag:
    st.header("API-status")
    st.caption("Visar vilka API-nycklar som hittades och om de fungerar.")

    from config import get_secret

    keys_to_check = [
        ("GROQ_API_KEY", "Groq"),
        ("GOOGLE_API_KEY", "Gemini"),
        ("XAI_API_KEY", "Grok"),
        ("TAVILY_API_KEY", "Tavily"),
        ("FINNHUB_API_KEY", "Finnhub"),
    ]

    for key_name, label in keys_to_check:
        val = get_secret(key_name)
        if val:
            masked = val[:6] + "..." + val[-4:] if len(val) > 10 else "***"
            st.success(f"**{label}** — Nyckel hittad ({masked})")
        else:
            st.error(f"**{label}** — Nyckel SAKNAS")

    st.divider()
    st.subheader("API-test")
    if st.button("Testa alla API:er", type="primary"):
        with st.spinner("Testar..."):
            # Test Groq
            try:
                from groq import Groq
                groq_key = get_secret("GROQ_API_KEY")
                if groq_key:
                    client = Groq(api_key=groq_key)
                    resp = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": "Say OK"}],
                        max_tokens=5,
                    )
                    st.success(f"**Groq** — Fungerar! Svar: {resp.choices[0].message.content}")
                else:
                    st.error("**Groq** — Ingen nyckel")
            except Exception as e:
                st.error(f"**Groq** — FEL: {e}")

            # Test Gemini
            try:
                from google import genai
                gem_key = get_secret("GOOGLE_API_KEY")
                if gem_key:
                    client = genai.Client(api_key=gem_key)
                    resp = client.models.generate_content(
                        model="gemini-2.5-flash-lite", contents="Say OK",
                    )
                    st.success(f"**Gemini** — Fungerar! Svar: {resp.text[:50]}")
                else:
                    st.error("**Gemini** — Ingen nyckel")
            except Exception as e:
                st.error(f"**Gemini** — FEL: {e}")

            # Test Tavily
            try:
                import requests
                tav_key = get_secret("TAVILY_API_KEY")
                if tav_key:
                    r = requests.post(
                        "https://api.tavily.com/search",
                        json={"api_key": tav_key, "query": "test", "max_results": 1},
                        timeout=10,
                    )
                    if r.status_code == 200:
                        st.success(f"**Tavily** — Fungerar!")
                    else:
                        st.error(f"**Tavily** — HTTP {r.status_code}: {r.text[:100]}")
                else:
                    st.error("**Tavily** — Ingen nyckel")
            except Exception as e:
                st.error(f"**Tavily** — FEL: {e}")

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
