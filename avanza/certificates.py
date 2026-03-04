"""Search for BULL/BEAR certificates on Avanza.

Uses web scraping of Avanza's public search since there's no official API.
Falls back gracefully if unavailable.
"""

import requests
import streamlit as st


AVANZA_SEARCH_URL = "https://www.avanza.se/ab/component/orderbook_search/"

UNDERLYING_MAPPING = {
    "^GDAXI": "DAX",
    "^OMX": "OMXS30",
    "^GSPC": "S&P 500",
    "^IXIC": "NASDAQ",
    "^FTSE": "FTSE 100",
    "GC=F": "GULD",
    "SI=F": "SILVER",
    "CL=F": "OLJA",
    "NG=F": "NATURGAS",
    "BTC-USD": "BITCOIN",
    "ETH-USD": "ETHEREUM",
    "ERIC-B.ST": "ERICSSON B",
    "VOLV-B.ST": "VOLVO B",
    "HM-B.ST": "H&M B",
    "SEB-A.ST": "SEB A",
    "INVE-B.ST": "INVESTOR B",
    "AAPL": "APPLE",
    "TSLA": "TESLA",
    "MSFT": "MICROSOFT",
    "NVDA": "NVIDIA",
    "AMZN": "AMAZON",
}


@st.cache_data(ttl=3600, show_spinner=False)
def search_certificates(ticker: str, direction: str) -> list[dict]:
    """Search Avanza for BULL/BEAR certificates matching the asset and direction.

    Args:
        ticker: yfinance ticker symbol
        direction: "BULL" or "BEAR"

    Returns:
        List of certificate dicts with name, leverage, url fields.
        Empty list if unavailable.
    """
    underlying = UNDERLYING_MAPPING.get(ticker, "")
    if not underlying:
        name = ticker.replace(".ST", "").replace("-", " ").replace("=F", "").replace("^", "")
        underlying = name.upper()

    search_term = f"{direction} {underlying}"

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "application/json",
        }
        params = {
            "query": search_term,
            "instrumentType": "CERTIFICATE",
            "limit": 10,
        }
        resp = requests.get(
            "https://www.avanza.se/ab/component/orderbook_search/",
            params=params,
            headers=headers,
            timeout=10,
        )

        if resp.status_code != 200:
            return _generate_search_links(underlying, direction)

        data = resp.json()
        results = []
        for item in data.get("resultGroups", []):
            for hit in item.get("hits", []):
                name = hit.get("name", "")
                link = hit.get("link", {}).get("url", "")
                if direction.upper() in name.upper():
                    results.append({
                        "name": name,
                        "url": f"https://www.avanza.se{link}" if link else "",
                        "leverage": _extract_leverage(name),
                    })

        return results if results else _generate_search_links(underlying, direction)

    except Exception:
        return _generate_search_links(underlying, direction)


def _extract_leverage(name: str) -> str:
    """Try to extract leverage multiplier from certificate name."""
    import re
    match = re.search(r"X(\d+)", name, re.IGNORECASE)
    if match:
        return f"{match.group(1)}x"
    match = re.search(r"(\d+)X", name, re.IGNORECASE)
    if match:
        return f"{match.group(1)}x"
    return "?"


def _generate_search_links(underlying: str, direction: str) -> list[dict]:
    """Generate direct Avanza search links as fallback."""
    search_query = f"{direction}+{underlying}".replace(" ", "+")
    return [{
        "name": f"Sök '{direction} {underlying}' på Avanza",
        "url": f"https://www.avanza.se/borshandlade-produkter/certifikat-turbowarrants/lista.html?query={search_query}",
        "leverage": "-",
    }]
