"""Persistent API usage tracker. Stores daily call counts in a JSON file."""

import json
import os
from datetime import datetime

USAGE_FILE = os.path.join(os.path.dirname(__file__), "usage.json")

DAILY_LIMITS = {
    "groq": 14400,
    "gemini": 1000,
    "grok": 1000,
    "tavily": 33,
    "scans": 1,
}


def _load() -> dict:
    try:
        with open(USAGE_FILE, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    today = datetime.now().strftime("%Y-%m-%d")
    if data.get("date") != today:
        data = {
            "date": today,
            "groq": 0,
            "gemini": 0,
            "grok": 0,
            "tavily": 0,
            "scans": 0,
        }
        _save(data)
    return data


def _save(data: dict):
    try:
        os.makedirs(os.path.dirname(USAGE_FILE), exist_ok=True)
        with open(USAGE_FILE, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def track_call(provider: str, count: int = 1):
    """Record API calls for a provider."""
    data = _load()
    key = provider.lower()
    if key in data:
        data[key] = data.get(key, 0) + count
        _save(data)


def track_scan():
    """Record a completed scan."""
    data = _load()
    data["scans"] = data.get("scans", 0) + 1
    _save(data)


def get_usage() -> dict:
    """Get today's usage stats with limits."""
    data = _load()
    return {
        provider: {
            "used": data.get(provider, 0),
            "limit": DAILY_LIMITS[provider],
            "remaining": max(0, DAILY_LIMITS[provider] - data.get(provider, 0)),
            "pct": min(100, int(data.get(provider, 0) / DAILY_LIMITS[provider] * 100))
                  if DAILY_LIMITS[provider] > 0 else 0,
        }
        for provider in DAILY_LIMITS
    }


def get_scan_count() -> int:
    """Get today's scan count."""
    data = _load()
    return data.get("scans", 0)


def can_scan() -> bool:
    """Check if we have scans remaining today."""
    return get_scan_count() < DAILY_LIMITS["scans"]
