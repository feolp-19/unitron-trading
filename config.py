import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def get_secret(key: str, default: str = "") -> str:
    """Read from Streamlit secrets (cloud) or env vars (local)."""
    try:
        import streamlit as st
        if key in st.secrets.get("api_keys", {}):
            return st.secrets["api_keys"][key]
    except Exception:
        pass
    return os.getenv(key, default)


@dataclass
class Asset:
    ticker: str
    display_name: str
    news_keywords: list[str]
    asset_type: str  # index, commodity, stock, crypto
    category: str    # Indices, Råvaror, Svenska aktier, Amerikanska aktier, Krypto


CURATED_ASSETS: dict[str, list[Asset]] = {
    "Index": [
        Asset("^GDAXI", "DAX 40", ["DAX", "German economy", "Deutsche Börse"], "index", "Index"),
        Asset("^OMX", "OMXS30", ["OMXS30", "Stockholm börsen", "Swedish economy"], "index", "Index"),
        Asset("^GSPC", "S&P 500", ["S&P 500", "US economy", "Wall Street"], "index", "Index"),
        Asset("^IXIC", "Nasdaq", ["Nasdaq", "tech stocks", "US technology"], "index", "Index"),
        Asset("^FTSE", "FTSE 100", ["FTSE", "UK economy", "London Stock Exchange"], "index", "Index"),
        Asset("^N225", "Nikkei 225", ["Nikkei", "Japan economy", "Tokyo Stock Exchange"], "index", "Index"),
        Asset("^HSI", "Hang Seng", ["Hang Seng", "Hong Kong", "China economy"], "index", "Index"),
    ],
    "Råvaror": [
        Asset("GC=F", "Guld", ["gold price", "gold market", "precious metals"], "commodity", "Råvaror"),
        Asset("SI=F", "Silver", ["silver price", "silver market", "precious metals"], "commodity", "Råvaror"),
        Asset("CL=F", "Olja (WTI)", ["oil price", "crude oil", "WTI", "OPEC"], "commodity", "Råvaror"),
        Asset("NG=F", "Naturgas", ["natural gas price", "energy market"], "commodity", "Råvaror"),
        Asset("PL=F", "Platina", ["platinum price", "platinum market"], "commodity", "Råvaror"),
        Asset("HG=F", "Koppar", ["copper price", "copper market", "industrial metals"], "commodity", "Råvaror"),
    ],
    "Krypto": [
        Asset("BTC-USD", "Bitcoin", ["Bitcoin", "BTC", "cryptocurrency"], "crypto", "Krypto"),
        Asset("ETH-USD", "Ethereum", ["Ethereum", "ETH", "smart contracts"], "crypto", "Krypto"),
        Asset("SOL-USD", "Solana", ["Solana", "SOL", "crypto"], "crypto", "Krypto"),
        Asset("XRP-USD", "XRP", ["XRP", "Ripple", "crypto"], "crypto", "Krypto"),
    ],
}

ALL_ASSETS_FLAT: list[Asset] = [
    asset for assets in CURATED_ASSETS.values() for asset in assets
]


def get_asset_by_ticker(ticker: str) -> Asset | None:
    for asset in ALL_ASSETS_FLAT:
        if asset.ticker == ticker:
            return asset
    return None


def create_custom_asset(ticker: str) -> Asset:
    """Create an Asset for a user-typed ticker not in the curated list."""
    ticker_upper = ticker.upper().strip()

    if ticker_upper.endswith("-USD"):
        asset_type = "crypto"
        category = "Krypto"
        name = ticker_upper.replace("-USD", "")
    elif ticker_upper.endswith(".ST"):
        asset_type = "stock"
        category = "Svenska aktier"
        name = ticker_upper.replace(".ST", "").replace("-", " ")
    elif ticker_upper.startswith("^"):
        asset_type = "index"
        category = "Index"
        name = ticker_upper.lstrip("^")
    elif "=F" in ticker_upper:
        asset_type = "commodity"
        category = "Råvaror"
        name = ticker_upper.replace("=F", "")
    else:
        asset_type = "stock"
        category = "Amerikanska aktier"
        name = ticker_upper

    return Asset(
        ticker=ticker_upper,
        display_name=name,
        news_keywords=[name],
        asset_type=asset_type,
        category=category,
    )


# AI provider config
AI_PROVIDER = get_secret("AI_PROVIDER", "groq").lower()

AI_CONFIGS = {
    "groq": {
        "api_key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
    },
    "grok": {
        "api_key_env": "XAI_API_KEY",
        "base_url": "https://api.x.ai/v1",
        "model": "grok-4.1-fast",
    },
    "gemini": {
        "api_key_env": "GOOGLE_API_KEY",
        "base_url": None,
        "model": "gemini-2.5-flash-lite",
    },
}

FINNHUB_API_KEY = get_secret("FINNHUB_API_KEY", "")
MAX_WATCHLIST_SCANS = int(get_secret("MAX_WATCHLIST_SCANS_PER_DAY", "10"))


SEARCH_ALIASES = {
    "gold": "GC=F", "guld": "GC=F",
    "silver": "SI=F",
    "oil": "CL=F", "olja": "CL=F",
    "gas": "NG=F", "naturgas": "NG=F",
    "dax": "^GDAXI",
    "omx": "^OMX", "omxs30": "^OMX", "stockholm": "^OMX",
    "s&p": "^GSPC", "s&p 500": "^GSPC", "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "ftse": "^FTSE",
    "bitcoin": "BTC-USD", "btc": "BTC-USD",
    "ethereum": "ETH-USD", "eth": "ETH-USD",
    "apple": "AAPL",
    "tesla": "TSLA",
    "microsoft": "MSFT",
    "nvidia": "NVDA",
    "amazon": "AMZN",
    "ericsson": "ERIC-B.ST",
    "volvo": "VOLV-B.ST",
    "h&m": "HM-B.ST", "hm": "HM-B.ST",
    "seb": "SEB-A.ST",
    "investor": "INVE-B.ST",
}


def search_asset(query: str) -> Asset | None:
    """Find an asset by name, ticker, or alias."""
    q = query.strip().lower()
    if not q:
        return None

    if q in SEARCH_ALIASES:
        ticker = SEARCH_ALIASES[q]
        found = get_asset_by_ticker(ticker)
        if found:
            return found
        return create_custom_asset(ticker)

    for asset in ALL_ASSETS_FLAT:
        if q in asset.display_name.lower() or q == asset.ticker.lower():
            return asset

    return create_custom_asset(query.strip())
