import json

import streamlit as st

from config import Asset, MAX_WATCHLIST_SCANS
from data.market_data import fetch_ohlc, get_52_week_range
from data.news_data import get_news_for_asset
from analysis.technical import analyze as technical_analyze
from analysis.sentiment import analyze_sentiment, dict_to_signal
from analysis.synergy import decide, TradeDecision


@st.cache_data(ttl=3600, show_spinner=False)
def scan_single(asset_json: str) -> dict | None:
    """Run full pipeline on a single asset. Accepts JSON for cache compatibility."""
    asset_dict = json.loads(asset_json)
    asset = Asset(**asset_dict)

    df = fetch_ohlc(asset.ticker)
    if df.empty:
        return None

    tech = technical_analyze(df)
    if tech is None:
        return None

    headlines = get_news_for_asset(asset)
    sent_dict = analyze_sentiment(asset.display_name, json.dumps(headlines))
    sent = dict_to_signal(sent_dict.copy())

    week_range = get_52_week_range(df)
    w52_low = week_range[0] if week_range else None
    w52_high = week_range[1] if week_range else None

    decision = decide(
        tech=tech,
        sent=sent,
        week_52_low=w52_low,
        week_52_high=w52_high,
        is_crypto=asset.asset_type == "crypto",
    )

    return {
        "asset": asset_dict,
        "action": decision.action,
        "confidence": decision.confidence_score,
        "rsi": tech.rsi_value,
        "price": tech.current_price,
        "sma": tech.sma_value,
        "sentiment_direction": sent.direction,
        "sentiment_confidence": sent.confidence,
        "stop_loss": decision.stop_loss_price,
        "take_profit": decision.take_profit_price,
        "reasoning": decision.reasoning,
        "warnings": decision.warnings,
    }


def scan_watchlist(assets: list[Asset]) -> list[dict]:
    """Scan a list of assets and return sorted results."""
    capped = assets[:MAX_WATCHLIST_SCANS]
    results = []

    progress = st.progress(0, text="Skannar bevakningslistan...")
    for i, asset in enumerate(capped):
        asset_json = json.dumps(asset.__dict__)
        result = scan_single(asset_json)
        if result:
            results.append(result)
        progress.progress((i + 1) / len(capped), text=f"Analyserar {asset.display_name}...")

    progress.empty()

    actionable = [r for r in results if r["action"] != "NONE"]
    no_trade = [r for r in results if r["action"] == "NONE"]

    actionable.sort(key=lambda x: x["confidence"], reverse=True)

    return actionable + no_trade
