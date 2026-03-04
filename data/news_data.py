from datetime import datetime, timedelta

import requests
import streamlit as st

from config import get_secret

FINNHUB_API_KEY = get_secret("FINNHUB_API_KEY")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_headlines(keywords: list[str], max_headlines: int = 20) -> list[dict]:
    """Fetch recent news headlines from Finnhub general news, filtered by keywords."""
    if not FINNHUB_API_KEY:
        return []

    try:
        url = "https://finnhub.io/api/v1/news"
        params = {
            "category": "general",
            "token": FINNHUB_API_KEY,
        }
        resp = requests.get(url, params=params, timeout=10)
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
                    "summary": article.get("summary", ""),
                })
            if len(filtered) >= max_headlines:
                break

        return filtered

    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_company_news(symbol: str, max_headlines: int = 20) -> list[dict]:
    """Fetch company-specific news from Finnhub for stock tickers."""
    if not FINNHUB_API_KEY:
        return []

    try:
        today = datetime.now().strftime("%Y-%m-%d")
        week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        url = "https://finnhub.io/api/v1/company-news"
        params = {
            "symbol": symbol,
            "from": week_ago,
            "to": today,
            "token": FINNHUB_API_KEY,
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        articles = resp.json()

        results = []
        for article in articles[:max_headlines]:
            results.append({
                "headline": article.get("headline", ""),
                "source": article.get("source", ""),
                "datetime": datetime.fromtimestamp(article.get("datetime", 0)).strftime("%Y-%m-%d %H:%M"),
                "url": article.get("url", ""),
                "summary": article.get("summary", ""),
            })

        return results

    except Exception:
        return []


def get_news_for_asset(asset) -> list[dict]:
    """Route to the right news fetcher based on asset type."""
    if asset.asset_type == "stock" and not asset.ticker.startswith("^"):
        clean_symbol = asset.ticker.replace(".ST", "").replace("-", ".")
        headlines = fetch_company_news(clean_symbol)
        if headlines:
            return headlines

    return fetch_headlines(asset.news_keywords)
