"""Unitron Deep Scan Engine — 3-phase analysis pipeline.

Phase 1: Quantitative Sifting (Groq) — screen all assets, pick Top 5
Phase 2: Devil's Advocate (Groq/Gemini) — skeptical review of Top 5
Phase 3: Macro Synthesis (Gemini) — deep chain-of-thought with DXY, VIX, 14d news

Uses 5s delays between API calls to stay under free-tier RPM limits.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime

from config import ALL_ASSETS_FLAT, Asset
from data.market_data import fetch_ohlc
from data.news_data import get_news_for_asset
from analysis.technical import analyze as technical_analyze, TechnicalSignal
from analysis.exit_strategy import generate_trading_plan
from analysis.sentiment import (
    _call_specific_provider, _call_ai_with_fallback,
    _format_sr_text, _interpret_rsi, _interpret_volume,
    _interpret_vix, _interpret_volatility, _interpret_sma_alignment, _interpret_bb,
)

API_DELAY = 5


def _parse_json(raw: str) -> dict | None:
    if not raw:
        return None
    try:
        cleaned = raw.strip()
        if "```" in cleaned:
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
        return json.loads(cleaned)
    except Exception:
        return None


def _build_ohlcv_table(df, days: int = 60) -> str:
    """Format last N days of OHLCV as a compact text table for AI."""
    tail = df.tail(days)
    lines = ["Date       | Open     | High     | Low      | Close    | Volume"]
    lines.append("-" * 70)
    for date_idx, row in tail.iterrows():
        d = date_idx.strftime("%Y-%m-%d")
        lines.append(
            f"{d} | {row['Open']:>8.2f} | {row['High']:>8.2f} | "
            f"{row['Low']:>8.2f} | {row['Close']:>8.2f} | {int(row['Volume']):>10,}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PHASE 1: Quantitative Sifting
# ---------------------------------------------------------------------------

PHASE1_PROMPT = """You are a senior quantitative analyst. Perform a rigorous QUANTITATIVE AUDIT of {asset_name}.

=== 60-DAY OHLCV DATA ===
{ohlcv_table}

=== PRE-CALCULATED INDICATORS ===
- Current Price: {price}
- SMA 20: {sma_20} | SMA 50: {sma_50} | SMA 200: {sma_200}
- SMA Alignment: {sma_alignment} | Directional Bias: {sma_bias}
- RSI (14): {rsi} {rsi_interp}
- ATR (14): {atr} | Volatility: {atr_ratio}x vs 30d avg ({vol_interp})
- MACD: {macd_value} | Signal: {macd_signal} | Histogram: {macd_histogram}
- MACD Crossover: {macd_cross}
- Bollinger Bands: Upper {bb_upper} | Middle {bb_middle} | Lower {bb_lower}
- BB Position: {bb_position} | Band Width: {bb_width}% ({bb_interp})
- Volume vs 20d avg: {volume_ratio}x ({volume_interp})
- VIX: {vix_value} ({vix_interp})
- Weekly SMA 50: {sma_50w} (price {price_vs_weekly})

=== SUPPORT & RESISTANCE ===
{sr_text}

Your analysis must cover:
1. TREND STRUCTURE: Is the trend clearly bullish, bearish, or range-bound? Why?
2. MOMENTUM: Is momentum accelerating or decelerating? MACD histogram direction?
3. VOLATILITY SETUP: Is a Bollinger squeeze forming? Is volatility expanding or contracting?
4. VOLUME CONFIRMATION: Does volume support the current price action?
5. KEY LEVELS: What are the critical S/R levels that would define the trade?
6. SETUP QUALITY: Rate this setup from 1-10 for a SHORT-TERM (1-5 day) certificate trade.

Return ONLY valid JSON:
{{
  "setup_score": 1 to 10,
  "direction": "BULL" or "BEAR" or "NEUTRAL",
  "confidence": 0.0 to 1.0,
  "trend_analysis": "2-3 sentences on trend structure",
  "momentum_analysis": "1-2 sentences on momentum",
  "volatility_analysis": "1-2 sentences on volatility setup",
  "volume_analysis": "1 sentence on volume",
  "key_levels": {{"entry": 0.0, "stop_loss": 0.0, "target": 0.0}},
  "setup_summary": "1 sentence: why this is or isn't a good trade"
}}"""


def run_phase1(assets: list[Asset], log_fn=None, delay: int = API_DELAY):
    """Screen all assets with Groq. Returns list of scored results."""
    results = []

    for i, asset in enumerate(assets):
        if log_fn:
            log_fn(f"phase1", f"Kvantitativ granskning: {asset.display_name} ({i+1}/{len(assets)})")

        try:
            df = fetch_ohlc(asset.ticker)
            if df.empty or len(df) < 50:
                if log_fn:
                    log_fn("skip", f"⚪ {asset.display_name} — otillräcklig data")
                continue

            tech = technical_analyze(df, ticker=asset.ticker)
            if tech is None:
                if log_fn:
                    log_fn("skip", f"⚪ {asset.display_name} — teknisk analys misslyckades")
                continue

            sr = tech.support_resistance
            ohlcv_table = _build_ohlcv_table(df, days=60)

            prompt = PHASE1_PROMPT.format(
                asset_name=asset.display_name,
                ohlcv_table=ohlcv_table,
                price=f"{tech.current_price:,.2f}",
                sma_20=f"{tech.sma_20:,.2f}", sma_50=f"{tech.sma_50:,.2f}",
                sma_200=f"{tech.sma_200:,.2f}",
                sma_alignment=_interpret_sma_alignment(tech.sma_alignment),
                sma_bias=tech.sma_bias.upper(),
                rsi=f"{tech.rsi_value:.1f}", rsi_interp=_interpret_rsi(tech.rsi_value),
                atr=f"{tech.atr_value:,.2f}", atr_ratio=f"{tech.atr_ratio:.1f}",
                vol_interp=_interpret_volatility(tech.atr_ratio),
                macd_value=f"{tech.macd_value:.4f}", macd_signal=f"{tech.macd_signal:.4f}",
                macd_histogram=f"{tech.macd_histogram:+.4f}",
                macd_cross=tech.macd_cross.replace("_", " ").upper() if tech.macd_cross != "none" else "None",
                bb_upper=f"{tech.bb_upper:,.2f}", bb_middle=f"{tech.bb_middle:,.2f}",
                bb_lower=f"{tech.bb_lower:,.2f}",
                bb_position=tech.bb_position.replace("_", " "),
                bb_width=f"{tech.bb_width:.1f}", bb_interp=_interpret_bb(tech.bb_width),
                volume_ratio=f"{tech.volume_ratio:.1f}",
                volume_interp=_interpret_volume(tech.volume_ratio),
                vix_value=f"{tech.vix_value:.1f}" if tech.vix_value else "N/A",
                vix_interp=_interpret_vix(tech.vix_value, tech.vix_level),
                sma_50w=f"{tech.sma_50w:,.2f}" if tech.sma_50w else "N/A",
                price_vs_weekly=tech.price_vs_weekly_sma,
                sr_text=_format_sr_text(sr.supports, sr.resistances, tech.current_price,
                                        tech.near_resistance, tech.near_support),
            )

            from storage.usage_tracker import track_call
            raw = _call_specific_provider(prompt, "Groq")
            provider = "Groq"
            if not raw:
                raw = _call_specific_provider(prompt, "Gemini")
                provider = "Gemini"

            data = _parse_json(raw)
            if data:
                data["provider"] = provider
                headlines = get_news_for_asset(asset)
                results.append({
                    "asset": asset,
                    "tech": tech,
                    "df": df,
                    "phase1": data,
                    "headlines": headlines,
                })
                score = data.get("setup_score", 0)
                direction = data.get("direction", "NEUTRAL")
                conf = data.get("confidence", 0)
                emoji = "🟢" if score >= 6 else ("🟡" if score >= 4 else "⚪")
                if log_fn:
                    log_fn("result", f"{emoji} {asset.display_name} — Score: {score}/10, {direction} ({conf:.0%}) [{provider}]")
            else:
                if log_fn:
                    log_fn("skip", f"⚪ {asset.display_name} — AI-svar kunde inte tolkas")

        except Exception as e:
            if log_fn:
                log_fn("error", f"🔴 {asset.display_name} — FEL: {str(e)[:60]}")

        time.sleep(delay)

    results.sort(key=lambda r: r["phase1"].get("setup_score", 0), reverse=True)
    return results


# ---------------------------------------------------------------------------
# PHASE 2: Devil's Advocate
# ---------------------------------------------------------------------------

PHASE2_PROMPT = """You are a professional RISK ANALYST hired to PROTECT CAPITAL. Your job is to find every reason why this trade WILL FAIL.

The trading team wants to {trade_direction} {asset_name} based on this analysis:

=== QUANTITATIVE AUDIT (Phase 1) ===
- Direction: {direction} | Confidence: {confidence}
- Setup Score: {setup_score}/10
- Trend: {trend_analysis}
- Momentum: {momentum_analysis}
- Volatility: {volatility_analysis}
- Volume: {volume_analysis}
- Proposed Entry: {entry} | Stop-Loss: {stop_loss} | Target: {target}

=== CURRENT NEWS ===
{headlines_text}

=== MARKET CONTEXT ===
- VIX: {vix} | RSI: {rsi} | Volume: {volume_ratio}x avg
- SMA Alignment: {sma_alignment}

YOUR MISSION: Find exactly 3 strong, SPECIFIC reasons why this trade will FAIL. Think about:
1. BULL TRAPS / BEAR TRAPS: Is this a false breakout? Is there a hidden divergence?
2. VALUE TRAPS: Is the "cheap" price justified by deteriorating fundamentals?
3. MACRO HEADWINDS: Central bank policy, geopolitical risk, sector rotation away from this asset?
4. CROWDED TRADE: Is everyone on the same side? Contrarian warning signs?
5. TECHNICAL FAILURE POINTS: What specific price level invalidates the entire thesis?

Be SPECIFIC. Reference actual price levels, news events, and data points.

Return ONLY valid JSON:
{{
  "risk_rating": "LOW" or "MEDIUM" or "HIGH" or "CRITICAL",
  "failure_reasons": [
    {{"reason": "specific reason", "severity": "medium|high|critical", "type": "bull_trap|value_trap|macro|crowded|technical"}},
    {{"reason": "specific reason", "severity": "medium|high|critical", "type": "bull_trap|value_trap|macro|crowded|technical"}},
    {{"reason": "specific reason", "severity": "medium|high|critical", "type": "bull_trap|value_trap|macro|crowded|technical"}}
  ],
  "worst_case_scenario": "2-3 sentences: what happens if this goes wrong",
  "invalidation_level": 0.0,
  "should_proceed": true or false,
  "recommendation": "1 sentence: final recommendation to the trader"
}}"""


def run_phase2(top_candidates: list[dict], log_fn=None, delay: int = API_DELAY):
    """Run Devil's Advocate skeptical review on top candidates."""
    for i, candidate in enumerate(top_candidates):
        asset = candidate["asset"]
        p1 = candidate["phase1"]
        tech = candidate["tech"]
        headlines = candidate.get("headlines", [])

        if log_fn:
            log_fn("phase2", f"Djävulens Advokat: {asset.display_name} ({i+1}/{len(top_candidates)})")

        headlines_text = "\n".join(
            f"- {h.get('headline', '')}" for h in headlines
        ) if headlines else "Inga nyheter tillgängliga."

        trade_dir = "BUY (BULL)" if p1["direction"] == "BULL" else "SHORT (BEAR)"
        key_levels = p1.get("key_levels", {})

        prompt = PHASE2_PROMPT.format(
            trade_direction=trade_dir,
            asset_name=asset.display_name,
            direction=p1["direction"],
            confidence=f"{p1.get('confidence', 0):.0%}",
            setup_score=p1.get("setup_score", 0),
            trend_analysis=p1.get("trend_analysis", ""),
            momentum_analysis=p1.get("momentum_analysis", ""),
            volatility_analysis=p1.get("volatility_analysis", ""),
            volume_analysis=p1.get("volume_analysis", ""),
            entry=f"{key_levels.get('entry', 0):,.2f}",
            stop_loss=f"{key_levels.get('stop_loss', 0):,.2f}",
            target=f"{key_levels.get('target', 0):,.2f}",
            headlines_text=headlines_text,
            vix=f"{tech.vix_value:.1f}" if tech.vix_value else "N/A",
            rsi=f"{tech.rsi_value:.1f}",
            volume_ratio=f"{tech.volume_ratio:.1f}",
            sma_alignment=_interpret_sma_alignment(tech.sma_alignment),
        )

        raw = _call_specific_provider(prompt, "Gemini")
        provider = "Gemini"
        if not raw:
            raw = _call_specific_provider(prompt, "Groq")
            provider = "Groq"

        data = _parse_json(raw)
        if data:
            data["provider"] = provider
            candidate["phase2"] = data
            risk = data.get("risk_rating", "UNKNOWN")
            proceed = "✅ FORTSÄTT" if data.get("should_proceed") else "⛔ AVBRYT"
            if log_fn:
                log_fn("result", f"{'🟢' if data.get('should_proceed') else '🔴'} {asset.display_name} — Risk: {risk} — {proceed} [{provider}]")
        else:
            candidate["phase2"] = None
            if log_fn:
                log_fn("error", f"🟡 {asset.display_name} — Djävulens Advokat misslyckades")

        time.sleep(delay)

    return top_candidates


# ---------------------------------------------------------------------------
# PHASE 3: Macro Synthesis (Gemini chain-of-thought)
# ---------------------------------------------------------------------------

PHASE3_PROMPT = """Du är en senior makrostrateg på en svensk investmentbank. Skriv en DJUPANALYS (minst 200 ord) för {asset_name}.

Du MÅSTE resonera steg-för-steg (Chain-of-Thought). Visa ditt resonemang tydligt.

=== KVANTITATIV DATA (Fas 1 — Groq) ===
- Riktning: {direction} | Konfidens: {confidence}
- Setup-betyg: {setup_score}/10
- Trendanalys: {trend_analysis}
- Momentum: {momentum_analysis}
- Volatilitet: {volatility_analysis}
- Volym: {volume_analysis}

=== SKEPTISK GRANSKNING (Fas 2 — Djävulens Advokat) ===
- Riskbetyg: {risk_rating}
- Rekommendation: {da_recommendation}
{failure_reasons_text}
- Värsta scenario: {worst_case}

=== MARKNADSKONTEXT ===
- VIX (Rädsleindex): {vix}
- Dollar Index (DXY): {dxy}
- Globalt sentiment: {global_sentiment}

=== NYHETSÖVERSIKT (14 dagar) ===
{headlines_text}

=== TEKNISKA NIVÅER ===
{sr_text}
- MACD: {macd_histogram} (histogram), Crossover: {macd_cross}
- Bollinger: Position {bb_position}, Bandbredd {bb_width}%
- RSI: {rsi}

=== DIN UPPGIFT ===
Steg 1: TRENDBEKRÄFTELSE — Stämmer den tekniska bilden överens med makromiljön?
Steg 2: RISKBEDÖMNING — Är Djävulens Advokats invändningar giltiga eller överdrivna?
Steg 3: TIDPUNKT — Är timingen rätt för en entry nu, eller bör man vänta?
Steg 4: SLUTGILTIG DOM — Syntetisera allt till ett handlingbart beslut.

VIKTIGT: Du MÅSTE skriva minst 200 ord. Var specifik med prisnivåer.

Returnera ENBART giltig JSON:
{{
  "verdict": "BUY_BULL" eller "BUY_BEAR" eller "NO_TRADE",
  "final_confidence": 0.0 till 1.0,
  "chain_of_thought": "Minst 200 ord: Steg 1... Steg 2... Steg 3... Steg 4...",
  "entry_price": 0.0,
  "stop_loss": 0.0,
  "take_profit": 0.0,
  "risk_reward": "1:X.X",
  "key_catalyst": "1 mening: den viktigaste faktorn för denna trade",
  "biggest_risk": "1 mening: det som kan gå mest fel",
  "time_horizon": "1-3 dagar" eller "3-5 dagar",
  "exit_strategy": "2-3 meningar: exakt när och hur man stänger positionen"
}}"""


def fetch_dxy() -> float | None:
    """Fetch US Dollar Index (DXY) current value."""
    try:
        import yfinance as yf
        df = yf.download("DX-Y.NYB", period="5d", progress=False, auto_adjust=True)
        if df.empty:
            return None
        import pandas as pd
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return float(df["Close"].iloc[-1])
    except Exception:
        return None


def _assess_global_sentiment(vix: float | None, dxy: float | None) -> str:
    parts = []
    if vix is not None:
        if vix > 30:
            parts.append("Extrem rädsla (VIX > 30)")
        elif vix > 25:
            parts.append("Förhöjd rädsla (VIX > 25)")
        elif vix > 20:
            parts.append("Försiktighet (VIX > 20)")
        elif vix > 15:
            parts.append("Normalt marknadsläge")
        else:
            parts.append("Lugn marknad (VIX < 15)")

    if dxy is not None:
        if dxy > 105:
            parts.append("Stark dollar (DXY > 105) — negativt för råvaror/EM")
        elif dxy < 100:
            parts.append("Svag dollar (DXY < 100) — positivt för råvaror")
        else:
            parts.append("Neutral dollarstyrka")

    return ". ".join(parts) if parts else "Data ej tillgänglig"


def run_phase3(candidates: list[dict], log_fn=None, delay: int = API_DELAY):
    """Deep Gemini synthesis with chain-of-thought for each candidate."""
    dxy = fetch_dxy()
    if log_fn:
        log_fn("info", f"📊 DXY (Dollar Index): {dxy:.2f}" if dxy else "📊 DXY: Ej tillgänglig")

    for i, candidate in enumerate(candidates):
        asset = candidate["asset"]
        p1 = candidate["phase1"]
        p2 = candidate.get("phase2") or {}
        tech = candidate["tech"]
        headlines = candidate.get("headlines", [])

        if log_fn:
            log_fn("phase3", f"Makrosyntes: {asset.display_name} ({i+1}/{len(candidates)})")

        headlines_text = "\n".join(
            f"- {h.get('headline', '')}" for h in headlines
        ) if headlines else "Inga nyheter."

        sr = tech.support_resistance
        sr_text = _format_sr_text(sr.supports, sr.resistances, tech.current_price,
                                   tech.near_resistance, tech.near_support)

        failure_reasons = p2.get("failure_reasons", [])
        failure_text = "\n".join(
            f"  {j+1}. [{fr.get('type', '?')}] {fr.get('reason', '')} (Allvarlighet: {fr.get('severity', '?')})"
            for j, fr in enumerate(failure_reasons)
        ) if failure_reasons else "  Ingen skeptisk granskning tillgänglig."

        vix = tech.vix_value
        global_sent = _assess_global_sentiment(vix, dxy)

        prompt = PHASE3_PROMPT.format(
            asset_name=asset.display_name,
            direction=p1.get("direction", "NEUTRAL"),
            confidence=f"{p1.get('confidence', 0):.0%}",
            setup_score=p1.get("setup_score", 0),
            trend_analysis=p1.get("trend_analysis", ""),
            momentum_analysis=p1.get("momentum_analysis", ""),
            volatility_analysis=p1.get("volatility_analysis", ""),
            volume_analysis=p1.get("volume_analysis", ""),
            risk_rating=p2.get("risk_rating", "EJ GRANSKAT"),
            da_recommendation=p2.get("recommendation", "Ej tillgänglig"),
            failure_reasons_text=failure_text,
            worst_case=p2.get("worst_case_scenario", "Ej tillgänglig"),
            vix=f"{vix:.1f}" if vix else "N/A",
            dxy=f"{dxy:.2f}" if dxy else "N/A",
            global_sentiment=global_sent,
            headlines_text=headlines_text,
            sr_text=sr_text,
            macd_histogram=f"{tech.macd_histogram:+.4f}",
            macd_cross=tech.macd_cross.replace("_", " ") if tech.macd_cross != "none" else "Ingen",
            bb_position=tech.bb_position.replace("_", " "),
            bb_width=f"{tech.bb_width:.1f}",
            rsi=f"{tech.rsi_value:.1f}",
        )

        raw = _call_specific_provider(prompt, "Gemini")
        provider = "Gemini"
        if not raw:
            raw = _call_specific_provider(prompt, "Groq")
            provider = "Groq"

        data = _parse_json(raw)
        if data:
            data["provider"] = provider
            data["dxy"] = dxy
            data["global_sentiment"] = global_sent
            candidate["phase3"] = data

            verdict = data.get("verdict", "NO_TRADE")
            conf = data.get("final_confidence", 0)
            emoji = "📈" if verdict == "BUY_BULL" else ("📉" if verdict == "BUY_BEAR" else "⚪")
            if log_fn:
                log_fn("result", f"{emoji} {asset.display_name} — Slutgiltig dom: {verdict} ({conf:.0%}) [{provider}]")
        else:
            candidate["phase3"] = None
            if log_fn:
                log_fn("error", f"🟡 {asset.display_name} — Makrosyntes misslyckades")

        time.sleep(delay)

    return candidates


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

@dataclass
class DeepScanResult:
    scan_date: str
    scan_time: str
    vix_value: float | None
    dxy_value: float | None
    global_sentiment: str
    all_scores: list[dict]
    top5: list[dict]
    final_picks: list[dict]
    log: list[str]
    total_assets: int
    phase1_calls: int
    phase2_calls: int
    phase3_calls: int


def run_deep_scan(log_fn=None, max_top: int = 5, api_delay: int = API_DELAY) -> DeepScanResult:
    """Execute the full 3-phase deep scan pipeline."""
    log_lines = []

    def _log(phase, msg):
        log_lines.append(msg)
        if log_fn:
            log_fn(phase, msg)

    assets = ALL_ASSETS_FLAT
    _log("start", f"🔬 **Unitron Djupanalys** — {len(assets)} tillgångar, {datetime.now().strftime('%H:%M')}")
    _log("info", "")

    # Phase 1
    _log("header", "### 📊 Fas 1: Kvantitativ Granskning (Groq)")
    phase1_results = run_phase1(assets, log_fn=_log, delay=api_delay)

    # Build scores summary for all assets
    all_scores = []
    for r in phase1_results:
        p1 = r["phase1"]
        all_scores.append({
            "name": r["asset"].display_name,
            "ticker": r["asset"].ticker,
            "score": p1.get("setup_score", 0),
            "direction": p1.get("direction", "NEUTRAL"),
            "confidence": p1.get("confidence", 0),
            "summary": p1.get("setup_summary", ""),
            "provider": p1.get("provider", ""),
        })

    top_candidates = [r for r in phase1_results if r["phase1"].get("setup_score", 0) >= 5][:max_top]

    if not top_candidates:
        top_candidates = phase1_results[:max_top]

    _log("info", "")
    _log("info", f"**Fas 1 klar:** {len(phase1_results)} analyserade → Topp {len(top_candidates)} vidare")
    _log("info", "")

    # Phase 2
    _log("header", "### 😈 Fas 2: Djävulens Advokat")
    top_candidates = run_phase2(top_candidates, log_fn=_log, delay=api_delay)

    survivors = [c for c in top_candidates if c.get("phase2", {}).get("should_proceed", False)]
    _log("info", "")
    _log("info", f"**Fas 2 klar:** {len(survivors)}/{len(top_candidates)} klarade den skeptiska granskningen")

    # If devil's advocate killed everything, still pass top candidates through phase 3
    phase3_candidates = survivors if survivors else top_candidates[:3]
    _log("info", "")

    # Phase 3
    _log("header", "### 🌍 Fas 3: Makrosyntes (Gemini)")
    phase3_candidates = run_phase3(phase3_candidates, log_fn=_log, delay=api_delay)

    # Build final picks
    final_picks = []
    for c in phase3_candidates:
        p3 = c.get("phase3")
        if not p3:
            continue
        verdict = p3.get("verdict", "NO_TRADE")
        if verdict == "NO_TRADE":
            continue

        tech = c["tech"]
        action = "BULL" if verdict == "BUY_BULL" else "BEAR"
        trading_plan = generate_trading_plan(tech, action)

        final_picks.append({
            **c,
            "final_verdict": verdict,
            "final_confidence": p3.get("final_confidence", 0),
            "trading_plan": trading_plan,
        })

    final_picks.sort(key=lambda x: x.get("final_confidence", 0), reverse=True)

    _log("info", "")
    _log("info", f"**Analys klar!** {len(final_picks)} handelsrekommendation{'er' if len(final_picks) != 1 else ''}")

    dxy_val = phase3_candidates[0].get("phase3", {}).get("dxy") if phase3_candidates else None
    vix_val = phase1_results[0]["tech"].vix_value if phase1_results else None
    global_sent = phase3_candidates[0].get("phase3", {}).get("global_sentiment", "") if phase3_candidates else ""

    return DeepScanResult(
        scan_date=datetime.now().strftime("%Y-%m-%d"),
        scan_time=datetime.now().strftime("%H:%M"),
        vix_value=vix_val,
        dxy_value=dxy_val,
        global_sentiment=global_sent,
        all_scores=all_scores,
        top5=[{
            "asset": c["asset"],
            "tech": c["tech"],
            "phase1": c["phase1"],
            "phase2": c.get("phase2"),
            "phase3": c.get("phase3"),
            "headlines": c.get("headlines", []),
        } for c in phase3_candidates],
        final_picks=final_picks,
        log=log_lines,
        total_assets=len(assets),
        phase1_calls=len(phase1_results),
        phase2_calls=len(top_candidates),
        phase3_calls=len(phase3_candidates),
    )
