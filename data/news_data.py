"""News data fetching: Tavily (primary) -> Finnhub (fallback)."""

from datetime import datetime, timedelta

import requests
import streamlit as st

from config import get_secret


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_tavily(query: str, max_results: int = 10) -> list[dict]:
    """Search the web for recent news using Tavily AI search."""
    api_key = get_secret("TAVILY_API_KEY")
    if not api_key:
        return []

    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": f"{query} latest news market analysis",
                "search_depth": "basic",
                "max_results": max_results,
                "include_answer": True,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        from storage.usage_tracker import track_call
        track_call("tavily")

        headlines = []
        for result in data.get("results", []):
            headlines.append({
                "headline": result.get("title", ""),
                "source": result.get("url", "").split("/")[2] if "/" in result.get("url", "") else "",
                "datetime": datetime.now().strftime("%Y-%m-%d"),
                "url": result.get("url", ""),
                "summary": result.get("content", "")[:200],
            })

        return headlines
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_finnhub_general(keywords: list[str], max_headlines: int = 20) -> list[dict]:
    """Fetch recent news from Finnhub general news, filtered by keywords."""
    api_key = get_secret("FINNHUB_API_KEY")
    if not api_key:
        return []

    try:
        resp = requests.get(
            "https://finnhub.io/api/v1/news",
            params={"category": "general", "token": api_key},
            timeout=10,
        )
        resp.raise_for_status()
        articles = resp.json()

        keywords_lower = [kw.lower() for kw in keywords]
        filtered = []
        for article in articles:
            text = f"{article.get('headline', '')} {article.get('summary', '')}".lower()
            if any(kw in text for kw in keywords_lower):
                filtered.append({
                    "headline": article.get("headline", ""),
                    "source": article.get("source", ""),
                    "datetime": datetime.fromtimestamp(article.get("datetime", 0)).strftime("%Y-%m-%d %H:%M"),
                    "url": article.get("url", ""),
                    "summary": article.get("summary", "")[:200],
                })
            if len(filtered) >= max_headlines:
                break

        return filtered
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_finnhub_company(symbol: str, max_headlines: int = 20) -> list[dict]:
    """Fetch company-specific news from Finnhub."""
    api_key = get_secret("FINNHUB_API_KEY")
    if not api_key:
        return []

    try:
        today = datetime.now().strftime("%Y-%m-%d")
        week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        resp = requests.get(
            "https://finnhub.io/api/v1/company-news",
            params={"symbol": symbol, "from": week_ago, "to": today, "token": api_key},
            timeout=10,
        )
        resp.raise_for_status()
        articles = resp.json()

        return [
            {
                "headline": a.get("headline", ""),
                "source": a.get("source", ""),
                "datetime": datetime.fromtimestamp(a.get("datetime", 0)).strftime("%Y-%m-%d %H:%M"),
                "url": a.get("url", ""),
                "summary": a.get("summary", "")[:200],
            }
            for a in articles[:max_headlines]
        ]
    except Exception:
        return []


def _build_search_query(asset) -> str:
    """Build a targeted search query based on asset type."""
    base = " ".join(asset.news_keywords[:2])

    type_context = {
        "index": f"{base} stock market outlook today trading",
        "commodity": f"{base} price forecast supply demand today",
        "crypto": f"{base} crypto market sentiment regulation today",
        "stock": f"{base} stock earnings outlook analyst rating",
    }
    return type_context.get(asset.asset_type, f"{base} latest news market analysis")


def get_news_for_screening(asset) -> list[dict]:
    """Get news for initial screening — Finnhub first (free, unlimited)."""
    if asset.asset_type == "stock" and not asset.ticker.startswith("^"):
        clean_symbol = asset.ticker.replace(".ST", "").replace("-", ".")
        headlines = fetch_finnhub_company(clean_symbol)
        if headlines:
            return headlines

    headlines = fetch_finnhub_general(asset.news_keywords)
    if headlines:
        return headlines

    search_query = _build_search_query(asset)
    return fetch_tavily(search_query)


def get_deep_news(asset) -> list[dict]:
    """Get premium news for Top 5 finalists — Tavily first (deep, AI-curated)."""
    search_query = _build_search_query(asset)
    headlines = fetch_tavily(search_query)
    if headlines:
        return headlines

    if asset.asset_type == "stock" and not asset.ticker.startswith("^"):
        clean_symbol = asset.ticker.replace(".ST", "").replace("-", ".")
        headlines = fetch_finnhub_company(clean_symbol)
        if headlines:
            return headlines

    return fetch_finnhub_general(asset.news_keywords)


def get_news_for_asset(asset) -> list[dict]:
    """Backward-compatible: uses deep news (for Analysera tab)."""
    return get_deep_news(asset)
