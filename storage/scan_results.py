"""Persistent daily scan results — survive browser close / session reset.

Saves the fully rendered scan data to a JSON file so it can be loaded
on next visit without re-running the scan. Automatically expires at midnight."""

import json
import os
from datetime import datetime, date
from dataclasses import asdict

_RESULTS_FILE = os.path.join(os.path.dirname(__file__), "latest_scan.json")


def _serialize_result(result: dict) -> dict:
    """Convert a single scan result (with dataclass objects) to JSON-safe dict."""
    asset = result["asset"]
    tech = result["tech"]
    ai_result = result.get("ai_result")
    trading_plan = result.get("trading_plan")
    verification = result.get("verification")
    decision = result.get("decision")

    sr = tech.support_resistance

    out = {
        "asset": {
            "ticker": asset.ticker,
            "display_name": asset.display_name,
            "news_keywords": asset.news_keywords,
            "asset_type": asset.asset_type,
            "category": asset.category,
        },
        "tech": {
            "current_price": tech.current_price,
            "rsi_value": tech.rsi_value,
            "sma_20": tech.sma_20,
            "sma_50": tech.sma_50,
            "sma_200": tech.sma_200,
            "sma_50w": tech.sma_50w,
            "atr_value": tech.atr_value,
            "price_vs_sma": tech.price_vs_sma,
            "price_vs_weekly_sma": tech.price_vs_weekly_sma,
            "sma_alignment": tech.sma_alignment,
            "rsi_trend_2d": tech.rsi_trend_2d,
            "atr_ratio": tech.atr_ratio,
            "volume_ratio": tech.volume_ratio,
            "vix_value": tech.vix_value,
            "vix_level": tech.vix_level,
            "near_resistance": tech.near_resistance,
            "near_support": tech.near_support,
            "supports": sr.supports,
            "resistances": sr.resistances,
        },
        "ai_result": ai_result,
        "headlines": result.get("headlines", []),
    }

    if trading_plan:
        out["trading_plan"] = {
            "entry_price": trading_plan.entry_price,
            "stop_loss": trading_plan.stop_loss,
            "stop_loss_method": trading_plan.stop_loss_method,
            "stop_loss_reasoning": trading_plan.stop_loss_reasoning,
            "take_profit": trading_plan.take_profit,
            "take_profit_method": trading_plan.take_profit_method,
            "take_profit_reasoning": trading_plan.take_profit_reasoning,
            "risk_reward_ratio": trading_plan.risk_reward_ratio,
            "risk_amount": trading_plan.risk_amount,
            "reward_amount": trading_plan.reward_amount,
            "trailing_stop_level": trading_plan.trailing_stop_level,
            "trailing_stop_reasoning": trading_plan.trailing_stop_reasoning,
        }
    else:
        out["trading_plan"] = None

    if verification:
        out["verification"] = {
            "consensus": verification.consensus,
            "second_ai_agrees": verification.second_ai_agrees,
            "second_ai_verdict": verification.second_ai_verdict,
            "second_ai_confidence": verification.second_ai_confidence,
            "second_ai_reasoning": verification.second_ai_reasoning,
            "second_ai_provider": verification.second_ai_provider,
            "disagreement_points": verification.disagreement_points,
            "devils_advocate_risk": verification.devils_advocate_risk,
            "devils_advocate_proceed": verification.devils_advocate_proceed,
            "counter_arguments": verification.counter_arguments,
            "biggest_risk": verification.biggest_risk,
            "devils_advocate_recommendation": verification.devils_advocate_recommendation,
            "risk_headlines": verification.risk_headlines,
            "verified": verification.verified,
        }
    else:
        out["verification"] = None

    if decision:
        out["decision"] = {
            "action": decision.action,
            "confidence_score": decision.confidence_score,
            "reasoning": decision.reasoning,
        }
    else:
        out["decision"] = None

    return out


def save_scan(scan_data: dict) -> None:
    """Save scan results + report + log to JSON file."""
    serialized_results = [_serialize_result(r) for r in scan_data.get("results", [])]

    payload = {
        "scan_date": date.today().isoformat(),
        "scan_time": datetime.now().strftime("%H:%M"),
        "results": serialized_results,
        "report": scan_data.get("report", []),
        "log": scan_data.get("log", []),
    }

    os.makedirs(os.path.dirname(_RESULTS_FILE), exist_ok=True)
    with open(_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_scan() -> dict | None:
    """Load today's scan results. Returns None if no scan exists or if it's from a previous day."""
    if not os.path.exists(_RESULTS_FILE):
        return None

    try:
        with open(_RESULTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

    if data.get("scan_date") != date.today().isoformat():
        return None

    return data
