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


def get_news_for_asset(asset) -> list[dict]:
    """Get news using Tavily first, then Finnhub as fallback."""
    # Try Tavily first (best coverage)
    search_terms = " ".join(asset.news_keywords[:2])
    headlines = fetch_tavily(search_terms)
    if headlines:
        return headlines

    # Fallback: Finnhub company news for stocks
    if asset.asset_type == "stock" and not asset.ticker.startswith("^"):
        clean_symbol = asset.ticker.replace(".ST", "").replace("-", ".")
        headlines = fetch_finnhub_company(clean_symbol)
        if headlines:
            return headlines

    # Fallback: Finnhub general news
    return fetch_finnhub_general(asset.news_keywords)
