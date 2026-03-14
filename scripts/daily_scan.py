#!/usr/bin/env python3
"""Automated daily 8-stage deep scan — runs via GitHub Actions cron job.

Executes the full 8-stage intelligence pipeline:
  Stage 0: Data Foundation
  Stage 1: Global Macro Anchor (Gemini)
  Stage 2: Multi-Lens Technical Scan (Groq)
  Stage 3-4: Ranking & Deep News
  Stage 5: High-Dimensional Deep Dive (Gemini)
  Stage 6: Devil's Advocate (Gemini)
  Stage 7: Cross-Validation (Groq)
  Stage 8: Yesterday's Accuracy Review

Saves results to storage/latest_scan.json for the Streamlit app.
Uses 8s delays between API calls for generous rate-limit headroom.
"""

import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.deep_scan import run_deep_scan, DeepScanResult

_RESULTS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "storage", "latest_scan.json",
)


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _serialize_result(result: DeepScanResult) -> dict:
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
            "stage2": c.get("stage2"),
            "stage5": c.get("stage5"),
            "stage6": c.get("stage6"),
            "stage7": c.get("stage7"),
            "synthesis": c.get("synthesis"),
            "headlines": c.get("headlines", []),
            "final_verdict": c.get("final_verdict", "NO_TRADE"),
            "final_confidence": c.get("final_confidence", 0),
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

        return out

    return {
        "scan_date": result.scan_date,
        "scan_time": result.scan_time,
        "vix_value": result.vix_value,
        "dxy_value": result.dxy_value,
        "us10y_value": result.us10y_value,
        "market_regime": result.market_regime,
        "regime_report": result.regime_report,
        "global_sentiment": result.global_sentiment,
        "all_scores": result.all_scores,
        "top5": [_ser_candidate(c) for c in result.top5],
        "final_picks": [_ser_candidate(c) for c in result.final_picks],
        "yesterday_review": result.yesterday_review,
        "log": result.log,
        "total_assets": result.total_assets,
        "stage_calls": result.stage_calls,
    }


def run_scan():
    log("Starting Unitron 8-Stage Intelligence Pipeline")

    def console_log(phase, msg):
        log(f"  [{phase}] {msg}")

    result = run_deep_scan(log_fn=console_log, max_top=5, api_delay=8)

    payload = _serialize_result(result)

    os.makedirs(os.path.dirname(_RESULTS_FILE), exist_ok=True)
    with open(_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    log(f"Results saved to {_RESULTS_FILE}")
    log(f"Summary: {result.total_assets} scanned -> {len(result.top5)} deep-analyzed -> {len(result.final_picks)} recommended")
    log(f"Market Regime: {result.market_regime}")


if __name__ == "__main__":
    run_scan()
