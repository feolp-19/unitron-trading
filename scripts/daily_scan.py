#!/usr/bin/env python3
"""Automated daily deep scan — runs via GitHub Actions cron job.

Executes the full 3-phase deep scan pipeline:
  Phase 1: Quantitative Sifting (Groq) — all assets
  Phase 2: Devil's Advocate (Gemini) — top 5
  Phase 3: Macro Synthesis (Gemini) — survivors

Saves results to storage/latest_scan.json for the Streamlit app.
Uses 8s delays between API calls for generous rate-limit headroom.
"""

import json
import os
import sys
from datetime import datetime, date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.deep_scan import run_deep_scan, DeepScanResult

_RESULTS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "storage", "latest_scan.json",
)


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _serialize_result(result: DeepScanResult) -> dict:
    """Convert DeepScanResult to a JSON-serializable dict for file storage."""

    def _ser_candidate(c):
        asset = c["asset"]
        tech = c["tech"]
        sr = tech.support_resistance if hasattr(tech, "support_resistance") else None

        asset_d = {
            "ticker": asset.ticker, "display_name": asset.display_name,
            "news_keywords": asset.news_keywords, "asset_type": asset.asset_type,
            "category": asset.category,
        } if hasattr(asset, "ticker") else asset

        tech_d = {
            "current_price": tech.current_price, "rsi_value": tech.rsi_value,
            "sma_20": tech.sma_20, "sma_50": tech.sma_50, "sma_200": tech.sma_200,
            "sma_50w": tech.sma_50w, "atr_value": tech.atr_value,
            "price_vs_sma": tech.price_vs_sma,
            "price_vs_weekly_sma": tech.price_vs_weekly_sma,
            "sma_alignment": tech.sma_alignment, "sma_bias": tech.sma_bias,
            "rsi_trend_2d": tech.rsi_trend_2d, "atr_ratio": tech.atr_ratio,
            "volume_ratio": tech.volume_ratio,
            "vix_value": tech.vix_value, "vix_level": tech.vix_level,
            "near_resistance": tech.near_resistance, "near_support": tech.near_support,
            "supports": sr.supports if sr else [],
            "resistances": sr.resistances if sr else [],
            "macd_value": tech.macd_value, "macd_signal": tech.macd_signal,
            "macd_histogram": tech.macd_histogram, "macd_cross": tech.macd_cross,
            "bb_upper": tech.bb_upper, "bb_lower": tech.bb_lower,
            "bb_middle": tech.bb_middle, "bb_position": tech.bb_position,
            "bb_width": tech.bb_width,
        } if hasattr(tech, "current_price") else tech

        out = {
            "asset": asset_d, "tech": tech_d,
            "phase1": c.get("phase1"), "phase2": c.get("phase2"),
            "phase3": c.get("phase3"), "headlines": c.get("headlines", []),
        }

        tp = c.get("trading_plan")
        if tp and not isinstance(tp, dict):
            out["trading_plan"] = {
                "entry_price": tp.entry_price, "stop_loss": tp.stop_loss,
                "stop_loss_method": tp.stop_loss_method,
                "stop_loss_reasoning": tp.stop_loss_reasoning,
                "take_profit": tp.take_profit,
                "take_profit_method": tp.take_profit_method,
                "take_profit_reasoning": tp.take_profit_reasoning,
                "risk_reward_ratio": tp.risk_reward_ratio,
                "risk_amount": tp.risk_amount, "reward_amount": tp.reward_amount,
                "trailing_stop_level": tp.trailing_stop_level,
                "trailing_stop_reasoning": tp.trailing_stop_reasoning,
            }
        else:
            out["trading_plan"] = tp

        if "final_verdict" in c:
            out["final_verdict"] = c["final_verdict"]
            out["final_confidence"] = c.get("final_confidence", 0)

        return out

    return {
        "scan_date": result.scan_date,
        "scan_time": result.scan_time,
        "vix_value": result.vix_value,
        "dxy_value": result.dxy_value,
        "global_sentiment": result.global_sentiment,
        "all_scores": result.all_scores,
        "top5": [_ser_candidate(c) for c in result.top5],
        "final_picks": [_ser_candidate(c) for c in result.final_picks],
        "log": result.log,
        "total_assets": result.total_assets,
        "phase1_calls": result.phase1_calls,
        "phase2_calls": result.phase2_calls,
        "phase3_calls": result.phase3_calls,
    }


def run_scan():
    log(f"Starting Unitron Deep Scan — 3-phase pipeline")

    def console_log(phase, msg):
        log(f"  [{phase}] {msg}")

    result = run_deep_scan(log_fn=console_log, max_top=5, api_delay=8)

    payload = _serialize_result(result)

    os.makedirs(os.path.dirname(_RESULTS_FILE), exist_ok=True)
    with open(_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    log(f"Results saved to {_RESULTS_FILE}")
    log(f"Summary: {result.total_assets} scanned → {len(result.top5)} deep-analyzed → {len(result.final_picks)} recommended")


if __name__ == "__main__":
    run_scan()
