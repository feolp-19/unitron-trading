import json
import os
from datetime import datetime

LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")


def _ensure_logs_dir():
    os.makedirs(LOGS_DIR, exist_ok=True)


def save_recommendation(
    ticker: str,
    asset_name: str,
    action: str,
    confidence: float,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    reasoning: list[str],
) -> None:
    _ensure_logs_dir()
    today = datetime.now().strftime("%Y-%m-%d")
    filepath = os.path.join(LOGS_DIR, f"{today}.json")

    existing = []
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []

    entry = {
        "timestamp": datetime.now().isoformat(),
        "ticker": ticker,
        "asset_name": asset_name,
        "action": action,
        "confidence": confidence,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "reasoning": reasoning,
    }

    # Don't duplicate if same ticker already logged today
    existing = [e for e in existing if e.get("ticker") != ticker]
    existing.append(entry)

    with open(filepath, "w") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)


def load_history(days: int = 30) -> list[dict]:
    _ensure_logs_dir()
    all_entries = []

    files = sorted(os.listdir(LOGS_DIR), reverse=True)
    for filename in files[:days]:
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(LOGS_DIR, filename)
        try:
            with open(filepath, "r") as f:
                entries = json.load(f)
                all_entries.extend(entries)
        except (json.JSONDecodeError, IOError):
            continue

    return all_entries
