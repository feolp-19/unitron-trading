from datetime import datetime

import requests
import streamlit as st

from config import get_secret

FINNHUB_API_KEY = get_secret("FINNHUB_API_KEY")


MAJOR_EVENTS = {
    "Interest Rate Decision",
    "CPI",
    "Consumer Price Index",
    "Non-Farm Payrolls",
    "NFP",
    "GDP",
    "FOMC",
    "ECB Press Conference",
    "ECB Interest Rate",
    "Riksbank",
    "Fed Chair",
    "Unemployment Rate",
    "PMI",
}


@st.cache_data(ttl=3600, show_spinner=False)
def check_macro_events_today() -> list[str]:
    """Check Finnhub economic calendar for major events today."""
    if not FINNHUB_API_KEY:
        return []

    try:
        today = datetime.now().strftime("%Y-%m-%d")
        url = "https://finnhub.io/api/v1/calendar/economic"
        params = {
            "from": today,
            "to": today,
            "token": FINNHUB_API_KEY,
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        events = data.get("economicCalendar", [])
        warnings = []
        for event in events:
            event_name = event.get("event", "")
            impact = event.get("impact", "").lower()
            if impact == "high" or any(kw.lower() in event_name.lower() for kw in MAJOR_EVENTS):
                country = event.get("country", "")
                time = event.get("time", "")
                warnings.append(
                    f"Makrohändelse idag: {event_name} ({country}) kl {time} — "
                    f"förhöjd risk för plötsliga prisrörelser"
                )

        return warnings

    except Exception:
        return []
