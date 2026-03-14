#!/usr/bin/env python3
"""Automated daily scan — runs via GitHub Actions cron job.

Performs the full 4-pass dual-AI analysis on all assets and saves
results to storage/latest_scan.json. The Streamlit app loads this
file on startup for instant results.

No rate limiting concerns since we use generous delays (10s between assets).
"""

import json
import os
import sys
import time
from datetime import datetime, date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ALL_ASSETS_FLAT
from data.market_data import fetch_ohlc, get_52_week_range
from data.news_data import get_news_for_asset
from analysis.technical import analyze as technical_analyze
from analysis.sentiment import (
    analyze_sentiment, dict_to_signal, run_analysis_with_provider,
    run_risk_assessment, run_macro_context, _format_sr_text,
)
from analysis.synergy import decide
from analysis.exit_strategy import generate_trading_plan
from analysis.verification import verify_recommendation

DELAY_BETWEEN_ASSETS = 10
DELAY_BETWEEN_PASSES = 3
MIN_CONFIDENCE = 0.55
MIN_RR_RATIO = 1.5

_RESULTS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "storage", "latest_scan.json",
)


def _serialize_result(result: dict) -> dict:
    """Convert scan result to JSON-safe dict."""
    asset = result["asset"]
    tech = result["tech"]
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
            "sma_bias": tech.sma_bias,
            "macd_value": tech.macd_value,
            "macd_signal": tech.macd_signal,
            "macd_histogram": tech.macd_histogram,
            "macd_cross": tech.macd_cross,
            "bb_upper": tech.bb_upper,
            "bb_lower": tech.bb_lower,
            "bb_middle": tech.bb_middle,
            "bb_position": tech.bb_position,
            "bb_width": tech.bb_width,
        },
        "ai_result": result.get("ai_result"),
        "headlines": result.get("headlines", []),
    }

    tp = result.get("trading_plan")
    if tp and not isinstance(tp, dict):
        out["trading_plan"] = {
            "entry_price": tp.entry_price,
            "stop_loss": tp.stop_loss,
            "stop_loss_method": tp.stop_loss_method,
            "stop_loss_reasoning": tp.stop_loss_reasoning,
            "take_profit": tp.take_profit,
            "take_profit_method": tp.take_profit_method,
            "take_profit_reasoning": tp.take_profit_reasoning,
            "risk_reward_ratio": tp.risk_reward_ratio,
            "risk_amount": tp.risk_amount,
            "reward_amount": tp.reward_amount,
            "trailing_stop_level": tp.trailing_stop_level,
            "trailing_stop_reasoning": tp.trailing_stop_reasoning,
        }
    else:
        out["trading_plan"] = tp

    verif = result.get("verification")
    if verif and not isinstance(verif, dict):
        out["verification"] = {
            "consensus": verif.consensus,
            "second_ai_agrees": verif.second_ai_agrees,
            "second_ai_verdict": verif.second_ai_verdict,
            "second_ai_confidence": verif.second_ai_confidence,
            "second_ai_reasoning": verif.second_ai_reasoning,
            "second_ai_provider": verif.second_ai_provider,
            "disagreement_points": verif.disagreement_points,
            "devils_advocate_risk": verif.devils_advocate_risk,
            "devils_advocate_proceed": verif.devils_advocate_proceed,
            "counter_arguments": verif.counter_arguments,
            "biggest_risk": verif.biggest_risk,
            "devils_advocate_recommendation": verif.devils_advocate_recommendation,
            "risk_headlines": verif.risk_headlines,
            "verified": verif.verified,
        }
    else:
        out["verification"] = verif

    decision = result.get("decision")
    if decision and not isinstance(decision, dict):
        out["decision"] = {
            "action": decision.action,
            "confidence_score": decision.confidence_score,
            "reasoning": decision.reasoning,
        }
    else:
        out["decision"] = decision

    return out


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def run_scan():
    log(f"Starting daily scan — {len(ALL_ASSETS_FLAT)} assets")
    scan_data = {"results": [], "report": [], "log": []}

    for i, asset in enumerate(ALL_ASSETS_FLAT):
        log(f"({i+1}/{len(ALL_ASSETS_FLAT)}) {asset.display_name}...")
        entry = {
            "name": asset.display_name,
            "ticker": asset.ticker,
            "status": "?", "rsi": "-", "vix": "-", "volume": "-",
            "sentiment": "-", "action": "-", "ai_verdict": "-", "reason": "",
        }

        try:
            df = fetch_ohlc(asset.ticker)
            if df.empty:
                entry["status"] = "Ingen data"
                scan_data["report"].append(entry)
                scan_data["log"].append(f"❌ {asset.display_name} — Ingen data")
                log(f"  → No data")
                continue

            tech = technical_analyze(df, ticker=asset.ticker)
            if tech is None:
                entry["status"] = "For lite data"
                scan_data["report"].append(entry)
                scan_data["log"].append(f"❌ {asset.display_name} — För lite data")
                log(f"  → Not enough data")
                continue

            sr = tech.support_resistance
            entry["rsi"] = f"{tech.rsi_value:.1f}"
            entry["vix"] = f"{tech.vix_value:.1f}" if tech.vix_value else "-"
            entry["volume"] = "N/A" if tech.volume_ratio < 0.01 else f"{tech.volume_ratio:.1f}x"

            headlines = get_news_for_asset(asset)
            headlines_json = json.dumps(headlines)
            sent_dict = analyze_sentiment(asset.display_name, headlines_json)
            sent = dict_to_signal(sent_dict.copy())
            entry["sentiment"] = f"{sent.direction} ({sent.confidence:.0%})"

            analysis_kwargs = dict(
                price=tech.current_price, sma_20=tech.sma_20, sma_50=tech.sma_50,
                sma_200=tech.sma_200, price_vs_sma=tech.price_vs_sma,
                sma_50w=tech.sma_50w, price_vs_weekly_sma=tech.price_vs_weekly_sma,
                sma_alignment=tech.sma_alignment, sma_bias=tech.sma_bias,
                rsi=tech.rsi_value, atr=tech.atr_value,
                rsi_trend=tech.rsi_trend_2d, atr_ratio=tech.atr_ratio,
                volume_ratio=tech.volume_ratio, vix_value=tech.vix_value,
                vix_level=tech.vix_level, supports_json=json.dumps(sr.supports),
                resistances_json=json.dumps(sr.resistances),
                near_resistance=tech.near_resistance, near_support=tech.near_support,
                headlines_json=headlines_json,
                macd_value=tech.macd_value, macd_signal=tech.macd_signal,
                macd_histogram=tech.macd_histogram, macd_cross=tech.macd_cross,
                bb_upper=tech.bb_upper, bb_lower=tech.bb_lower,
                bb_middle=tech.bb_middle, bb_position=tech.bb_position,
                bb_width=tech.bb_width,
            )

            sr_text = _format_sr_text(
                sr.supports, sr.resistances,
                tech.current_price, tech.near_resistance, tech.near_support,
            )
            headlines_text = "\n".join(
                f"{j+1}. {h['headline']}" for j, h in enumerate(headlines)
            ) if headlines else "No recent headlines available."

            # Pass 1: Groq full analysis
            log(f"  Pass 1: Groq analysis")
            groq_result = run_analysis_with_provider(
                provider="Groq", asset_name=asset.display_name, **analysis_kwargs,
            )
            time.sleep(DELAY_BETWEEN_PASSES)

            # Pass 2: Gemini independent analysis
            log(f"  Pass 2: Gemini analysis")
            gemini_result = run_analysis_with_provider(
                provider="Gemini", asset_name=asset.display_name, **analysis_kwargs,
            )
            time.sleep(DELAY_BETWEEN_PASSES)

            # Pass 3: Risk assessment (Gemini)
            log(f"  Pass 3: Risk assessment")
            risk_data = run_risk_assessment(
                asset_name=asset.display_name, price=tech.current_price,
                sma_alignment=tech.sma_alignment, rsi=tech.rsi_value,
                volume_ratio=tech.volume_ratio, vix_value=tech.vix_value,
                atr=tech.atr_value, atr_ratio=tech.atr_ratio,
                near_resistance=tech.near_resistance, near_support=tech.near_support,
                sr_text=sr_text, headlines_text=headlines_text,
            )
            time.sleep(DELAY_BETWEEN_PASSES)

            # Pass 4: Macro context (Gemini)
            log(f"  Pass 4: Macro context")
            macro_data = run_macro_context(
                asset_name=asset.display_name, asset_type=asset.asset_type,
                price=tech.current_price, sma_alignment=tech.sma_alignment,
                vix_value=tech.vix_value, headlines_text=headlines_text,
            )

            groq_verdict = groq_result.get("verdict", "NO_TRADE") if groq_result else "NO_TRADE"
            gemini_verdict = gemini_result.get("verdict", "NO_TRADE") if gemini_result else "NO_TRADE"
            groq_conf = groq_result.get("confidence", 0) if groq_result else 0
            gemini_conf = gemini_result.get("confidence", 0) if gemini_result else 0

            # Apply risk/macro modifiers
            risk_penalty = 0
            if risk_data:
                risk_level = risk_data.get("overall_risk", "MEDIUM")
                if risk_level == "CRITICAL":
                    risk_penalty = -0.20
                elif risk_level == "HIGH":
                    risk_penalty = -0.10
                elif risk_level == "LOW":
                    risk_penalty = 0.05
                if not risk_data.get("safe_to_trade", True):
                    risk_penalty = min(risk_penalty, -0.15)

            macro_modifier = 0
            if macro_data:
                macro_modifier = float(macro_data.get("recommendation_modifier", 0))
                macro_modifier = max(-0.15, min(0.15, macro_modifier))

            groq_conf = max(0, min(1, groq_conf + risk_penalty + macro_modifier))
            gemini_conf = max(0, min(1, gemini_conf + risk_penalty + macro_modifier))

            groq_ok = groq_result is not None
            gemini_ok = gemini_result is not None

            # Consensus logic
            if groq_ok and gemini_ok:
                same_direction = (groq_verdict == gemini_verdict and groq_verdict != "NO_TRADE")
                if same_direction:
                    avg_conf = (groq_conf + gemini_conf) / 2
                    ai_result = groq_result if groq_conf >= gemini_conf else gemini_result
                    ai_result["confidence"] = round(avg_conf, 2)
                    ai_result["provider"] = f"Groq ({groq_conf:.0%}) + Gemini ({gemini_conf:.0%})"
                    ai_result["risk_data"] = risk_data
                    ai_result["macro_data"] = macro_data
                    consensus_tag = "KONSENSUS"
                elif groq_verdict == "NO_TRADE" and gemini_verdict == "NO_TRADE":
                    ai_result = groq_result
                    ai_result["provider"] = "Groq + Gemini"
                    consensus_tag = "ENIGA: NO_TRADE"
                else:
                    ai_result = {
                        "verdict": "NO_TRADE", "confidence": 0,
                        "provider": "Groq + Gemini",
                        "analysis": f"Oense: Groq={groq_verdict}, Gemini={gemini_verdict}",
                        "key_factors": [], "risks": [],
                    }
                    consensus_tag = "OENIGA"
            elif groq_ok:
                ai_result = groq_result
                consensus_tag = "ENBART GROQ"
            elif gemini_ok:
                ai_result = gemini_result
                consensus_tag = "ENBART GEMINI"
            else:
                ai_result = None
                consensus_tag = "BÅDA MISSLYCKADES"

            risk_tag = f" | Risk: {risk_data.get('overall_risk', '?')}" if risk_data else ""
            macro_tag = ""
            if macro_data:
                macro_tag = f" | Makro: {macro_data.get('macro_bias', '?')} ({macro_data.get('recommendation_modifier', 0):+.2f})"

            if ai_result:
                verdict = ai_result.get("verdict", "NO_TRADE")
                confidence = ai_result.get("confidence", 0)
                entry["ai_verdict"] = verdict

                action = {"BUY_BULL": "BULL", "BUY_BEAR": "BEAR"}.get(verdict, "NONE")

                if action != "NONE" and confidence < MIN_CONFIDENCE:
                    entry["action"] = "NONE"
                    entry["status"] = "Svag signal"
                    entry["reason"] = f"Konfidens ({confidence:.0%}) under minimum"
                    scan_data["log"].append(
                        f"🟡 {asset.display_name} — {verdict} avvisad ({consensus_tag}{risk_tag}{macro_tag})"
                    )
                    log(f"  → {verdict} rejected (confidence {confidence:.0%} < 55%)")
                    scan_data["report"].append(entry)
                    time.sleep(DELAY_BETWEEN_ASSETS)
                    continue

                trading_plan = None
                if action != "NONE":
                    trading_plan = generate_trading_plan(tech, action)
                    if trading_plan:
                        rr_val = (trading_plan.reward_amount / trading_plan.risk_amount
                                  if trading_plan.risk_amount > 0 else 0)
                        if rr_val < MIN_RR_RATIO:
                            entry["action"] = "NONE"
                            entry["status"] = "Svag signal"
                            entry["reason"] = f"R/R {trading_plan.risk_reward_ratio} under minimum"
                            scan_data["log"].append(
                                f"🟡 {asset.display_name} — {verdict} avvisad (R/R < 1:1.5)"
                            )
                            log(f"  → {verdict} rejected (R/R too low)")
                            scan_data["report"].append(entry)
                            time.sleep(DELAY_BETWEEN_ASSETS)
                            continue

                entry["action"] = action
                if action != "NONE":
                    entry["status"] = "SIGNAL"
                    entry["reason"] = ai_result.get("analysis", "")[:120]
                    scan_data["results"].append({
                        "asset": asset, "ai_result": ai_result, "tech": tech,
                        "sent": sent, "trading_plan": trading_plan, "headlines": headlines,
                    })
                    scan_data["log"].append(
                        f"🟢 {asset.display_name} — **{verdict}** ({confidence:.0%}) ✅ {consensus_tag}{risk_tag}{macro_tag}"
                    )
                    log(f"  → ✅ CANDIDATE: {verdict} ({confidence:.0%}) [{consensus_tag}]")
                else:
                    entry["status"] = "Ingen signal"
                    entry["reason"] = ai_result.get("analysis", "")[:120]
                    scan_data["log"].append(
                        f"⚪ {asset.display_name} — NO_TRADE ({consensus_tag}{risk_tag}{macro_tag})"
                    )
                    log(f"  → NO_TRADE ({consensus_tag})")
            else:
                week_range = get_52_week_range(df)
                w52_low = week_range[0] if week_range else None
                decision = decide(
                    tech=tech, sent=sent,
                    week_52_low=w52_low,
                    is_crypto=asset.asset_type == "crypto",
                )
                entry["ai_verdict"] = "fallback"
                entry["action"] = "NONE"
                entry["status"] = "Ingen signal"
                entry["reason"] = decision.reasoning[0] if decision.reasoning else ""
                scan_data["log"].append(f"⚪ {asset.display_name} — Ingen signal (fallback)")
                log(f"  → No signal (fallback)")

        except Exception as e:
            entry["status"] = "Fel"
            entry["reason"] = str(e)[:80]
            scan_data["log"].append(f"🔴 {asset.display_name} — FEL: {str(e)[:50]}")
            log(f"  → ERROR: {e}")

        scan_data["report"].append(entry)
        time.sleep(DELAY_BETWEEN_ASSETS)

    # Sort candidates by confidence
    scan_data["results"].sort(
        key=lambda x: x.get("ai_result", {}).get("confidence", 0) if x.get("ai_result") else 0,
        reverse=True,
    )

    # Stage 2: Verify candidates
    if scan_data["results"]:
        n = len(scan_data["results"])
        log(f"\nStage 2: Verifying {n} candidate(s)...")
        scan_data["log"].append(f"\n#### ⚙️ Steg 2 — Djupanalys av {n} kandidat{'er' if n > 1 else ''}")

        verified = []
        for idx, result in enumerate(scan_data["results"]):
            asset = result["asset"]
            ai_result = result.get("ai_result", {})
            tech = result["tech"]
            tp = result.get("trading_plan")
            verdict = ai_result.get("verdict", "BUY_BULL") if ai_result else "BUY_BULL"

            sr = tech.support_resistance
            sr_text = _format_sr_text(
                sr.supports, sr.resistances,
                tech.current_price, tech.near_resistance, tech.near_support,
            )
            raw_headlines = result.get("headlines", [])
            headlines_text = "\n".join(
                f"- {h.get('headline', '')}" for h in raw_headlines
            ) if raw_headlines else "Inga rubriker."

            log(f"  Verifying {asset.display_name}...")
            scan_data["log"].append(f"🔍 Djupanalys {asset.display_name}...")

            try:
                verification = verify_recommendation(
                    asset_name=asset.display_name,
                    first_provider="Groq",
                    verdict=verdict,
                    confidence=ai_result.get("confidence", 0) if ai_result else 0,
                    tech=tech,
                    sr_text=sr_text,
                    headlines_text=headlines_text,
                    stop_loss=tp.stop_loss if tp else 0,
                    take_profit=tp.take_profit if tp else 0,
                    news_keywords=asset.news_keywords,
                )
            except Exception as e:
                verification = None
                log(f"  → Verification error: {e}")

            result["verification"] = verification

            if verification and verification.verified:
                verified.append(result)
                scan_data["log"].append(f"   ✅ {asset.display_name} — VERIFIERAD")
                log(f"  → ✅ VERIFIED")
            elif verification:
                result["rejected_reason"] = "verification_failed"
                verified.append(result)
                scan_data["log"].append(
                    f"   ⚠️ {asset.display_name} — EJ VERIFIERAD (risk: {verification.devils_advocate_risk})"
                )
                log(f"  → ⚠️ NOT VERIFIED (risk: {verification.devils_advocate_risk})")
            else:
                verified.append(result)
                scan_data["log"].append(f"   🔄 {asset.display_name} — Djupanalys misslyckades")
                log(f"  → Verification failed")

            time.sleep(DELAY_BETWEEN_PASSES)

        scan_data["results"] = verified

    # Summary
    total = len(ALL_ASSETS_FLAT)
    n_verified = len([r for r in scan_data["results"] if r.get("verification") and r["verification"].verified])
    n_unverified = len([r for r in scan_data["results"] if r.get("verification") and not r["verification"].verified])
    scan_data["log"].append(
        f"\n**Klart!** {total} tillgångar screenade → "
        f"{len(scan_data['results'])} kandidater → "
        f"{n_verified} verifierade, {n_unverified} ej verifierade"
    )
    log(f"\nDone! {total} scanned → {len(scan_data['results'])} candidates → {n_verified} verified")

    # Save results
    serialized = [_serialize_result(r) for r in scan_data["results"]]
    payload = {
        "scan_date": date.today().isoformat(),
        "scan_time": datetime.now().strftime("%H:%M"),
        "results": serialized,
        "report": scan_data["report"],
        "log": scan_data["log"],
    }

    os.makedirs(os.path.dirname(_RESULTS_FILE), exist_ok=True)
    with open(_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log(f"Results saved to {_RESULTS_FILE}")


if __name__ == "__main__":
    run_scan()
