"""Unitron 8-Stage Intelligence Pipeline.

Stage 0: Data Foundation — fetch OHLCV, macro indicators, Finnhub news
Stage 1: Global Macro Anchor (Gemini) — define Market Regime
Stage 2: Multi-Lens Technical Scan (Groq) — 5 micro-prompts per asset, batched
Stage 3: Ranking & Filtering — select Top 5 finalists
Stage 4: Deep News (Tavily) — premium articles for Top 5
Stage 5: High-Dimensional Deep Dive (Gemini) — 4 research modules per finalist
Stage 6: Devil's Advocate (Gemini) — 3 data-driven rejection reasons
Stage 7: Cross-Validation (Groq) — flag hallucinations, logical flaws
Stage 8: Yesterday's Accuracy Review — learning brief

All API calls separated by configurable delay (default 5s).
"""

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta

from config import ALL_ASSETS_FLAT, Asset
from data.market_data import fetch_ohlc, fetch_macro_indicator
from data.news_data import get_news_for_screening, get_deep_news
from analysis.technical import analyze as technical_analyze, TechnicalSignal
from analysis.exit_strategy import generate_trading_plan
from analysis.sentiment import (
    _call_specific_provider,
    _format_sr_text, _interpret_rsi, _interpret_volume,
    _interpret_vix, _interpret_volatility, _interpret_sma_alignment, _interpret_bb,
)

API_DELAY = 5
_RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage")

NAME_ALIASES: dict[str, str] = {
    "gold": "Guld", "silver": "Silver", "oil": "Olja (WTI)", "wti": "Olja (WTI)",
    "crude oil": "Olja (WTI)", "natural gas": "Naturgas", "platinum": "Platina",
    "copper": "Koppar", "bitcoin": "Bitcoin", "ethereum": "Ethereum",
    "solana": "Solana", "xrp": "XRP", "ripple": "XRP",
    "guld": "Guld", "olja": "Olja (WTI)", "naturgas": "Naturgas",
    "platina": "Platina", "koppar": "Koppar",
}


def _match_asset_response(ai_name: str, batch: list[dict], already_matched: set) -> dict | None:
    """Match an AI-returned asset name to a batch item using ticker, display name, and aliases."""
    ai_lower = ai_name.lower().strip()
    ai_upper = ai_name.upper().strip()

    for item in batch:
        if item["asset"].ticker in already_matched:
            continue
        if item["asset"].ticker in ai_upper or item["asset"].ticker.upper() in ai_upper:
            return item

    for item in batch:
        if item["asset"].ticker in already_matched:
            continue
        dn = item["asset"].display_name.lower()
        if dn in ai_lower or ai_lower in dn:
            return item

    resolved = NAME_ALIASES.get(ai_lower)
    if resolved:
        for item in batch:
            if item["asset"].ticker in already_matched:
                continue
            if item["asset"].display_name == resolved:
                return item

    return None


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


def _load_yesterday_results() -> dict | None:
    filepath = os.path.join(_RESULTS_DIR, "latest_scan.json")
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        if data.get("scan_date") == yesterday:
            return data
        two_days = (date.today() - timedelta(days=2)).isoformat()
        if data.get("scan_date") == two_days:
            return data
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# STAGE 0: Data Foundation
# ---------------------------------------------------------------------------

@dataclass
class MacroContext:
    vix: float | None = None
    dxy: float | None = None
    us10y: float | None = None
    regime: str = ""
    regime_report: str = ""


def run_stage0(assets: list[Asset], log_fn=None) -> tuple[list[dict], MacroContext]:
    """Fetch all market data, technical indicators, and screening news."""
    if log_fn:
        log_fn("stage0", "Hämtar marknadsdata och makroindikatorer...")

    vix = fetch_macro_indicator("^VIX")
    dxy = fetch_macro_indicator("DX-Y.NYB")
    us10y = fetch_macro_indicator("^TNX")
    macro = MacroContext(vix=vix, dxy=dxy, us10y=us10y)

    if log_fn:
        log_fn("info", f"VIX: {vix:.1f}" if vix else "VIX: N/A")
        log_fn("info", f"DXY: {dxy:.2f}" if dxy else "DXY: N/A")
        log_fn("info", f"US 10Y: {us10y:.2f}%" if us10y else "US 10Y: N/A")

    asset_data = []
    for i, asset in enumerate(assets):
        if log_fn:
            log_fn("stage0", f"Data: {asset.display_name} ({i+1}/{len(assets)})")

        df = fetch_ohlc(asset.ticker)
        if df.empty or len(df) < 50:
            if log_fn:
                log_fn("skip", f"{asset.display_name} — otillräcklig data")
            continue

        tech = technical_analyze(df, ticker=asset.ticker)
        if tech is None:
            if log_fn:
                log_fn("skip", f"{asset.display_name} — teknisk analys misslyckades")
            continue

        headlines = get_news_for_screening(asset)

        asset_data.append({
            "asset": asset,
            "df": df,
            "tech": tech,
            "headlines": headlines,
        })

    if log_fn:
        log_fn("info", f"Steg 0 klart: {len(asset_data)}/{len(assets)} tillgångar redo")

    return asset_data, macro


# ---------------------------------------------------------------------------
# STAGE 1: Global Macro Anchor (Gemini)
# ---------------------------------------------------------------------------

STAGE1_PROMPT = """You are a senior macro strategist. Analyze the current global market environment.

=== MACRO DATA ===
- VIX (Volatility Index): {vix}
- DXY (US Dollar Index): {dxy}
- US 10-Year Treasury Yield: {us10y}

=== YOUR TASK ===
1. Define the current MARKET REGIME from these options:
   - RISK_ON: Low VIX (<18), falling yields, weak dollar — equities and risk assets favored
   - RISK_OFF: High VIX (>25), flight to safety — gold, bonds, USD favored
   - REFLATION: Rising yields + rising equities — commodities and cyclicals favored
   - STAGFLATION: Rising yields + falling equities + high inflation — gold, energy favored
   - TIGHTENING: Strong dollar, rising yields — bearish for EM, commodities, crypto
   - EASING: Falling yields, weak dollar — bullish for all risk assets
   - TRANSITION: Mixed signals, no clear regime

2. For each asset class, state the BIAS:
   - Equities (indices): bullish/bearish/neutral
   - Commodities (gold, oil, copper): bullish/bearish/neutral
   - Crypto: bullish/bearish/neutral
   - Safe havens: bullish/bearish/neutral

3. Identify the TOP 3 macro risks for the next 1-5 trading days.

Output ONLY the raw JSON object. No markdown, no code blocks, no explanation.
{{
  "regime": "one of the regime names above",
  "regime_description": "2-3 sentences explaining WHY this regime applies now",
  "asset_biases": {{
    "equities": "bullish/bearish/neutral",
    "commodities": "bullish/bearish/neutral",
    "crypto": "bullish/bearish/neutral",
    "safe_havens": "bullish/bearish/neutral"
  }},
  "macro_risks": [
    "risk 1 in 1 sentence",
    "risk 2 in 1 sentence",
    "risk 3 in 1 sentence"
  ],
  "key_levels": {{
    "vix_watch": "level to watch and why",
    "dxy_watch": "level to watch and why",
    "yield_watch": "level to watch and why"
  }}
}}"""


def run_stage1(macro: MacroContext, log_fn=None, delay: int = API_DELAY) -> MacroContext:
    """Stage 1: Define the Market Regime using Gemini."""
    if log_fn:
        log_fn("stage1", "Analyserar marknadsregim (Gemini)...")

    prompt = STAGE1_PROMPT.format(
        vix=f"{macro.vix:.1f}" if macro.vix else "N/A",
        dxy=f"{macro.dxy:.2f}" if macro.dxy else "N/A",
        us10y=f"{macro.us10y:.2f}%" if macro.us10y else "N/A",
    )

    raw = _call_specific_provider(prompt, "Gemini")
    if not raw:
        time.sleep(delay)
        raw = _call_specific_provider(prompt, "Groq")

    data = _parse_json(raw)
    if data:
        macro.regime = data.get("regime", "TRANSITION")
        macro.regime_report = json.dumps(data, indent=2, ensure_ascii=False)
        if log_fn:
            log_fn("result", f"Marknadsregim: {macro.regime}")
            desc = data.get("regime_description", "")
            if desc:
                log_fn("info", desc[:120])
    else:
        macro.regime = "TRANSITION"
        macro.regime_report = "{}"
        if log_fn:
            log_fn("error", "Marknadsregim-analys misslyckades — använder TRANSITION")

    time.sleep(delay)
    return macro


# ---------------------------------------------------------------------------
# STAGE 2: Multi-Lens Technical Scan (Groq) — 5 micro-prompts batched
# ---------------------------------------------------------------------------

MICRO_LENS_PROMPTS = {
    "trend": """Analyze the TREND STRUCTURE of these assets. For each, determine:
- Is the primary trend bullish, bearish, or ranging?
- SMA alignment (20/50/200) direction?
- Is price above or below the 50-week SMA?
Rate "score" from 1 (worst trend) to 10 (strongest, clearest trend).""",

    "reversal": """Analyze REVERSAL SAFETY for these assets. Look for:
- RSI divergences (price making new high/low but RSI not confirming)
- Double tops/bottoms in the recent 60 days
- Exhaustion candles or capitulation volume
Rate "score" from 1 (extreme reversal risk, very dangerous) to 10 (no reversal signals, very safe to trade).""",

    "momentum": """Analyze MOMENTUM for these assets. Focus on:
- MACD histogram direction and crossover status
- RSI level and trajectory (rising/falling)
- Volume trend (expanding or contracting with price)
Rate "score" from 1 (weakest momentum) to 10 (strongest, most confirmed momentum).""",

    "volatility": """Analyze VOLATILITY SETUP for these assets. Look for:
- Bollinger Band squeeze (narrowing bands = potential breakout)
- ATR trend (expanding = trend, contracting = consolidation)
- VIX correlation (is macro volatility aligned?)
Rate "score" from 1 (worst volatility setup) to 10 (best setup, e.g. breakout from squeeze).""",

    "risk_reward": """Analyze RISK/REWARD for these assets. Calculate:
- Distance to nearest support (downside risk)
- Distance to nearest resistance (upside potential)
- Current R:R ratio for a new position
Rate "score" from 1 (terrible R:R) to 10 (excellent R:R, e.g. 1:3+).""",
}

STAGE2_PROMPT = """You are a quantitative analyst. Analyze the following {batch_count} assets through a {lens_name} lens.

=== MARKET REGIME ===
{regime_context}

=== ASSETS ===
{assets_block}

{lens_instruction}

IMPORTANT: For "asset", use the EXACT identifier from each asset header, e.g. "DAX 40 (^GDAXI)".
Use the exact field name "score" for your 1-10 rating.

Output ONLY the raw JSON array. No markdown, no code blocks, no explanation.
[
  {{
    "asset": "exact header identifier",
    "score": 7,
    "direction": "BULL" or "BEAR" or "NEUTRAL",
    "analysis": "1-2 sentence specific analysis",
    "key_level": 0.0
  }}
]"""


def _build_asset_block(items: list[dict]) -> str:
    blocks = []
    for item in items:
        asset = item["asset"]
        tech = item["tech"]
        sr = tech.support_resistance

        block = f"""--- {asset.display_name} ({asset.ticker}) ---
Price: {tech.current_price:,.2f} | SMA20: {tech.sma_20:,.2f} | SMA50: {tech.sma_50:,.2f} | SMA200: {tech.sma_200:,.2f}
SMA Bias: {tech.sma_bias.upper()} | Alignment: {_interpret_sma_alignment(tech.sma_alignment)}
RSI: {tech.rsi_value:.1f} {_interpret_rsi(tech.rsi_value)}
MACD: {tech.macd_value:.4f} | Signal: {tech.macd_signal:.4f} | Hist: {tech.macd_histogram:+.4f} | Cross: {tech.macd_cross}
BB: Upper {tech.bb_upper:,.2f} | Lower {tech.bb_lower:,.2f} | Position: {tech.bb_position} | Width: {tech.bb_width:.1f}%
ATR: {tech.atr_value:,.2f} | Ratio: {tech.atr_ratio:.1f}x | Volume: {tech.volume_ratio:.1f}x avg
Supports: {', '.join(f'{s:,.2f}' for s in sr.supports[:3])} | Resistances: {', '.join(f'{r:,.2f}' for r in sr.resistances[:3])}
Weekly SMA50: {f'{tech.sma_50w:,.2f}' if tech.sma_50w else 'N/A'} ({tech.price_vs_weekly_sma})"""
        blocks.append(block)

    return "\n\n".join(blocks)


def run_stage2(asset_data: list[dict], macro: MacroContext, log_fn=None, delay: int = API_DELAY) -> list[dict]:
    """Stage 2: Run 5 micro-lens Groq prompts in batches of 4 assets."""
    batch_size = 4
    all_lens_scores = {item["asset"].ticker: {} for item in asset_data}

    regime_ctx = macro.regime_report if macro.regime_report != "{}" else f"Regime: {macro.regime}"

    for lens_name, lens_instruction in MICRO_LENS_PROMPTS.items():
        if log_fn:
            log_fn("stage2", f"Lins: {lens_name.upper()} ({len(asset_data)} tillgångar)")

        for batch_start in range(0, len(asset_data), batch_size):
            batch = asset_data[batch_start:batch_start + batch_size]
            assets_block = _build_asset_block(batch)

            prompt = STAGE2_PROMPT.format(
                batch_count=len(batch),
                lens_name=lens_name.upper(),
                regime_context=regime_ctx[:500],
                assets_block=assets_block,
                lens_instruction=lens_instruction,
            )

            raw = _call_specific_provider(prompt, "Groq")
            if not raw:
                time.sleep(delay)
                raw = _call_specific_provider(prompt, "Gemini")

            results = _parse_json(raw)
            if results and isinstance(results, list):
                matched_tickers = set()
                for idx, r in enumerate(results):
                    asset_name = r.get("asset", "")
                    matched_item = _match_asset_response(asset_name, batch, matched_tickers)
                    if not matched_item and idx < len(batch):
                        matched_item = batch[idx]
                    if matched_item:
                        matched_tickers.add(matched_item["asset"].ticker)
                        all_lens_scores[matched_item["asset"].ticker][lens_name] = {
                            "score": r.get("score", 5),
                            "direction": r.get("direction", "NEUTRAL"),
                            "analysis": r.get("analysis", ""),
                            "key_level": r.get("key_level", 0),
                        }

            time.sleep(delay)

    for item in asset_data:
        ticker = item["asset"].ticker
        lens_data = all_lens_scores.get(ticker, {})
        scores = [v.get("score", 5) for v in lens_data.values()]
        composite = sum(scores) / len(scores) if scores else 5.0

        directions = [v.get("direction", "NEUTRAL") for v in lens_data.values()]
        bull_count = directions.count("BULL")
        bear_count = directions.count("BEAR")
        if bull_count > bear_count and bull_count >= 3:
            consensus_dir = "BULL"
        elif bear_count > bull_count and bear_count >= 3:
            consensus_dir = "BEAR"
        else:
            consensus_dir = "NEUTRAL"

        item["stage2"] = {
            "lenses": lens_data,
            "composite_score": round(composite, 1),
            "consensus_direction": consensus_dir,
            "confidence": round(composite / 10, 2),
        }

        if log_fn:
            emoji = "+" if composite >= 6 else ("-" if composite < 4 else "~")
            log_fn("result", f"[{emoji}] {item['asset'].display_name}: {composite:.1f}/10 {consensus_dir}")

    return asset_data


# ---------------------------------------------------------------------------
# STAGE 3 & 4: Ranking + Deep News
# ---------------------------------------------------------------------------

def run_stage3_4(asset_data: list[dict], log_fn=None, max_top: int = 5, delay: int = API_DELAY) -> list[dict]:
    """Stage 3: Rank and select Top N. Stage 4: Fetch deep news for finalists."""
    ranked = sorted(
        asset_data,
        key=lambda x: x.get("stage2", {}).get("composite_score", 0),
        reverse=True,
    )

    for item in ranked:
        s2 = item.get("stage2", {})
        score = s2.get("composite_score", 0)
        direction = s2.get("consensus_direction", "NEUTRAL")
        if log_fn:
            log_fn("stage3", f"{item['asset'].display_name}: {score:.1f}/10 {direction}")

    qualified = [r for r in ranked if r.get("stage2", {}).get("composite_score", 0) >= 5]
    finalists = qualified[:max_top] if qualified else ranked[:max_top]

    if log_fn:
        log_fn("info", f"Topp {len(finalists)} vidare till djupanalys")
        log_fn("stage4", f"Hämtar premium-nyheter (Tavily) för {len(finalists)} finalister...")

    for item in finalists:
        deep_headlines = get_deep_news(item["asset"])
        item["deep_headlines"] = deep_headlines
        if log_fn:
            log_fn("info", f"{item['asset'].display_name}: {len(deep_headlines)} artiklar")
        time.sleep(1)

    return finalists


# ---------------------------------------------------------------------------
# STAGE 5: High-Dimensional Deep Dive (Gemini)
# ---------------------------------------------------------------------------

MODULE_A_PROMPT = """You are a financial historian. For {asset_name} (current price: {price}):

=== MARKET REGIME: {regime} ===
=== TECHNICAL CONTEXT ===
- SMA Bias: {sma_bias} | RSI: {rsi} | MACD Cross: {macd_cross}
- Current trend: {trend_analysis}

Search your knowledge for HISTORICAL ANALOGS — past periods where this asset (or similar assets)
showed a comparable technical and macro setup. Consider:
- 1970s stagflation periods
- 2008 financial crisis
- 2011 gold peak
- 2015 China devaluation
- 2020 COVID crash and recovery
- 2022 rate hike cycle

For each analog, state: What happened next? What was the outcome 5-20 trading days later?

Output ONLY the raw JSON object. No markdown, no code blocks, no explanation.
{{
  "analogs": [
    {{
      "period": "e.g. Q4 2022",
      "description": "1-2 sentences on the setup similarity",
      "outcome": "what happened — percentage move and timeframe",
      "relevance_score": 8
    }}
  ],
  "historical_verdict": "BULLISH" or "BEARISH" or "INCONCLUSIVE",
  "confidence_from_history": 0.5,
  "key_lesson": "1 sentence: the most important lesson from history"
}}"""

MODULE_B_PROMPT = """You are an inter-market analyst. For {asset_name} (current price: {price}):

=== MACRO CONTEXT ===
- DXY (Dollar Index): {dxy}
- US 10Y Yield: {us10y}
- VIX: {vix}
- Market Regime: {regime}

Analyze INTER-MARKET CONVERGENCE:
1. How does {asset_name} typically correlate with DXY? Is the current move CONFIRMED or CONTRADICTED by DXY?
2. Bond yield impact: Are rising/falling yields supporting or undermining this asset?
3. Currency effect: Is this a REAL move in the asset, or is it primarily a USD effect?
4. Cross-asset confirmation: Are related assets (e.g., gold vs silver, oil vs energy stocks) confirming the move?

Output ONLY the raw JSON object. No markdown, no code blocks, no explanation.
{{
  "dxy_correlation": "positive/negative/decorrelated",
  "dxy_verdict": "confirms/contradicts/neutral",
  "yield_impact": "supporting/undermining/neutral",
  "currency_adjusted": true or false,
  "real_move_confidence": 0.0 to 1.0,
  "cross_asset_confirmation": "strong/weak/absent",
  "inter_market_verdict": "CONFIRMED" or "DIVERGENT" or "INCONCLUSIVE",
  "analysis": "3-4 sentences of detailed inter-market reasoning"
}}"""

MODULE_C_PROMPT = """You are a commodity/sector fundamental analyst. For {asset_name}:

=== RECENT NEWS ===
{headlines_text}

=== MARKET REGIME: {regime} ===

Analyze SUPPLY/DEMAND FUNDAMENTALS and GEOPOLITICAL FACTORS:
1. SUPPLY side: Are there production disruptions, mining strikes, OPEC cuts, central bank buying/selling?
2. DEMAND side: Is industrial demand rising or falling? China PMI trends? EV adoption impact?
3. GEOPOLITICAL: Wars, sanctions, trade disputes — how do they affect this specific asset?
4. SEASONAL patterns: Is this a historically strong or weak period for this asset?
5. INVENTORY/STORAGE: Are inventories building or drawing down?

Be SPECIFIC. Reference actual events from the news headlines.

Output ONLY the raw JSON object. No markdown, no code blocks, no explanation.
{{
  "supply_pressure": "tight/balanced/oversupplied",
  "demand_trend": "rising/stable/falling",
  "geopolitical_risk": "LOW/MEDIUM/HIGH",
  "geopolitical_factors": ["factor 1", "factor 2"],
  "seasonal_bias": "bullish/bearish/neutral",
  "fundamental_verdict": "BULLISH" or "BEARISH" or "NEUTRAL",
  "confidence": 0.0 to 1.0,
  "key_driver": "1 sentence: the single most important supply/demand factor right now"
}}"""

MODULE_D_PROMPT = """You are a risk scenario planner. For {asset_name} (current price: {price}):

=== CURRENT POSITION ===
- Direction bias: {direction}
- Key support: {support}
- Key resistance: {resistance}
- ATR (14): {atr}

Simulate THREE stress scenarios and estimate the likely price impact on {asset_name}:

1. WAR ESCALATION: A major geopolitical conflict escalates (e.g., new sanctions, military action).
   - What happens to this asset in the first 48 hours?

2. SUDDEN FED PIVOT: The Federal Reserve unexpectedly signals a 50bp rate cut at the next meeting.
   - How does this asset respond in the first week?

3. LIQUIDITY CRUNCH / FLASH CRASH: A major hedge fund or bank faces a margin call, triggering forced liquidation.
   - What is the maximum drawdown risk for this asset?

For each scenario, provide a specific price target and percentage move.

Output ONLY the raw JSON object. No markdown, no code blocks, no explanation.
{{
  "scenarios": [
    {{
      "name": "War Escalation",
      "price_target": 0.0,
      "pct_move": "+/- X.X%",
      "timeframe": "48 hours",
      "reasoning": "2 sentences"
    }},
    {{
      "name": "Fed Pivot",
      "price_target": 0.0,
      "pct_move": "+/- X.X%",
      "timeframe": "1 week",
      "reasoning": "2 sentences"
    }},
    {{
      "name": "Liquidity Crunch",
      "price_target": 0.0,
      "pct_move": "+/- X.X%",
      "timeframe": "1-3 days",
      "reasoning": "2 sentences"
    }}
  ],
  "worst_case_price": 0.0,
  "max_drawdown_pct": "X.X%",
  "tail_risk_verdict": "MANAGEABLE" or "ELEVATED" or "EXTREME",
  "hedging_suggestion": "1 sentence"
}}"""


def run_stage5(finalists: list[dict], macro: MacroContext, log_fn=None, delay: int = API_DELAY) -> list[dict]:
    """Stage 5: 4 unique Gemini research modules per finalist."""
    modules = [
        ("A", "Historiska Analogier", MODULE_A_PROMPT),
        ("B", "Intermarknadsanalys", MODULE_B_PROMPT),
        ("C", "Utbud/Efterfrågan & Geopolitik", MODULE_C_PROMPT),
        ("D", "Scenariostresstest", MODULE_D_PROMPT),
    ]

    for item in finalists:
        asset = item["asset"]
        tech = item["tech"]
        sr = tech.support_resistance
        s2 = item.get("stage2", {})
        headlines = item.get("deep_headlines", item.get("headlines", []))

        item["stage5"] = {}

        for mod_key, mod_name, mod_prompt in modules:
            if log_fn:
                log_fn("stage5", f"Modul {mod_key}: {mod_name} — {asset.display_name}")

            headlines_text = "\n".join(
                f"- {h.get('headline', '')}" for h in headlines[:15]
            ) if headlines else "Inga nyheter tillgängliga."

            trend_analysis = ""
            lenses = s2.get("lenses", {})
            if "trend" in lenses:
                trend_analysis = lenses["trend"].get("analysis", "")

            prompt = mod_prompt.format(
                asset_name=asset.display_name,
                price=f"{tech.current_price:,.2f}",
                regime=macro.regime,
                sma_bias=tech.sma_bias.upper(),
                rsi=f"{tech.rsi_value:.1f}",
                macd_cross=tech.macd_cross,
                trend_analysis=trend_analysis,
                dxy=f"{macro.dxy:.2f}" if macro.dxy else "N/A",
                us10y=f"{macro.us10y:.2f}%" if macro.us10y else "N/A",
                vix=f"{macro.vix:.1f}" if macro.vix else "N/A",
                headlines_text=headlines_text,
                direction=s2.get("consensus_direction", "NEUTRAL"),
                support=f"{sr.supports[0]:,.2f}" if sr.supports else "N/A",
                resistance=f"{sr.resistances[0]:,.2f}" if sr.resistances else "N/A",
                atr=f"{tech.atr_value:,.2f}",
            )

            raw = _call_specific_provider(prompt, "Gemini")
            data = _parse_json(raw)
            if data:
                item["stage5"][mod_key] = data
                if log_fn:
                    if mod_key == "A":
                        v = data.get("historical_verdict", "?")
                        log_fn("result", f"  Historisk dom: {v}")
                    elif mod_key == "B":
                        v = data.get("inter_market_verdict", "?")
                        log_fn("result", f"  Intermarknadsdom: {v}")
                    elif mod_key == "C":
                        v = data.get("fundamental_verdict", "?")
                        log_fn("result", f"  Fundamental dom: {v}")
                    elif mod_key == "D":
                        v = data.get("tail_risk_verdict", "?")
                        log_fn("result", f"  Tail-risk: {v}")
            else:
                item["stage5"][mod_key] = None
                if log_fn:
                    log_fn("error", f"  Modul {mod_key} misslyckades")

            time.sleep(delay)

    return finalists


# ---------------------------------------------------------------------------
# STAGE 6: Devil's Advocate (Gemini)
# ---------------------------------------------------------------------------

STAGE6_PROMPT = """You are a RISK ANALYST hired to PROTECT CAPITAL. Find every reason why this trade WILL FAIL.

=== TRADE PROPOSAL ===
Asset: {asset_name} | Direction: {direction} | Composite Score: {composite}/10
Entry: ~{price} | Market Regime: {regime}

=== TECHNICAL SUMMARY ===
{tech_summary}

=== HIGH-DIMENSIONAL RESEARCH ===
- Historical Analogs: {hist_verdict} (confidence: {hist_conf})
- Inter-Market: {inter_verdict}
- Fundamentals: {fund_verdict}
- Tail Risk: {tail_verdict}

=== NEWS ===
{headlines_text}

YOUR MISSION: Find exactly 3 strong, SPECIFIC, data-driven reasons why this trade will FAIL.
For each reason, reference actual price levels, news events, or historical precedent.

Output ONLY the raw JSON object. No markdown, no code blocks, no explanation.
{{
  "risk_rating": "LOW" or "MEDIUM" or "HIGH" or "CRITICAL",
  "failure_reasons": [
    {{"reason": "specific reason with data", "severity": "medium|high|critical", "type": "bull_trap|value_trap|macro|crowded|technical"}},
    {{"reason": "specific reason with data", "severity": "medium|high|critical", "type": "bull_trap|value_trap|macro|crowded|technical"}},
    {{"reason": "specific reason with data", "severity": "medium|high|critical", "type": "bull_trap|value_trap|macro|crowded|technical"}}
  ],
  "worst_case_scenario": "2-3 sentences: what happens if this goes wrong",
  "invalidation_level": 0.0,
  "should_proceed": true or false,
  "recommendation": "1 sentence: final recommendation to the trader"
}}"""


def run_stage6(finalists: list[dict], macro: MacroContext, log_fn=None, delay: int = API_DELAY) -> list[dict]:
    """Stage 6: Devil's Advocate — find 3 reasons to reject each trade."""
    for item in finalists:
        asset = item["asset"]
        tech = item["tech"]
        s2 = item.get("stage2", {})
        s5 = item.get("stage5", {})
        headlines = item.get("deep_headlines", item.get("headlines", []))

        if log_fn:
            log_fn("stage6", f"Djävulens Advokat: {asset.display_name}")

        tech_summary = (
            f"Price: {tech.current_price:,.2f} | RSI: {tech.rsi_value:.1f} | "
            f"MACD Cross: {tech.macd_cross} | BB Position: {tech.bb_position} | "
            f"Volume: {tech.volume_ratio:.1f}x | SMA Bias: {tech.sma_bias}"
        )

        headlines_text = "\n".join(
            f"- {h.get('headline', '')}" for h in headlines[:10]
        ) if headlines else "Inga nyheter."

        mod_a = s5.get("A") or {}
        mod_b = s5.get("B") or {}
        mod_c = s5.get("C") or {}
        mod_d = s5.get("D") or {}

        prompt = STAGE6_PROMPT.format(
            asset_name=asset.display_name,
            direction=s2.get("consensus_direction", "NEUTRAL"),
            composite=s2.get("composite_score", 5),
            price=f"{tech.current_price:,.2f}",
            regime=macro.regime,
            tech_summary=tech_summary,
            hist_verdict=mod_a.get("historical_verdict", "N/A"),
            hist_conf=f"{mod_a.get('confidence_from_history', 0):.0%}",
            inter_verdict=mod_b.get("inter_market_verdict", "N/A"),
            fund_verdict=mod_c.get("fundamental_verdict", "N/A"),
            tail_verdict=mod_d.get("tail_risk_verdict", "N/A"),
            headlines_text=headlines_text,
        )

        raw = _call_specific_provider(prompt, "Gemini")
        data = _parse_json(raw)
        if data:
            item["stage6"] = data
            risk = data.get("risk_rating", "?")
            proceed = data.get("should_proceed", False)
            emoji = "+" if proceed else "X"
            if log_fn:
                log_fn("result", f"[{emoji}] {asset.display_name}: Risk {risk} — {'FORTSÄTT' if proceed else 'AVBRYT'}")
        else:
            item["stage6"] = None
            if log_fn:
                log_fn("error", f"{asset.display_name}: Devil's Advocate misslyckades")

        time.sleep(delay)

    return finalists


# ---------------------------------------------------------------------------
# STAGE 7: Cross-Validation (Groq)
# ---------------------------------------------------------------------------

STAGE7_PROMPT = """You are an AI AUDITOR. Your job is to review another AI's analysis for logical errors, hallucinations, and inconsistencies.

=== ASSET: {asset_name} ===
=== ORIGINAL AI ANALYSIS (Gemini) ===

Composite Score: {composite}/10 | Direction: {direction}

Stage 5 Research:
- Historical Analogs verdict: {hist_verdict}
- Inter-Market verdict: {inter_verdict}
- Fundamental verdict: {fund_verdict}
- Tail Risk verdict: {tail_verdict}

Stage 6 Devil's Advocate:
- Risk Rating: {risk_rating}
- Should Proceed: {should_proceed}
- Failure Reasons: {failure_reasons}

=== ACTUAL TECHNICAL DATA (ground truth) ===
Price: {price} | RSI: {rsi} | MACD: {macd_hist} | BB: {bb_position}
SMA20: {sma_20} | SMA50: {sma_50} | SMA200: {sma_200}
Volume: {volume}x avg | ATR: {atr}
Supports: {supports} | Resistances: {resistances}

CHECK FOR:
1. HALLUCINATIONS: Did the AI invent data points not supported by the technical data?
2. LOGICAL INCONSISTENCIES: Does the conclusion contradict the evidence?
3. OVERCONFIDENCE: Is the confidence justified by the data spread?
4. MISSING RISKS: Are there obvious risks the analysis ignored?

Output ONLY the raw JSON object. No markdown, no code blocks, no explanation.
{{
  "hallucinations_found": [],
  "logical_errors": [],
  "missing_risks": [],
  "overconfidence": true or false,
  "adjusted_confidence": 0.0 to 1.0,
  "validation_passed": true or false,
  "final_verdict": "APPROVE" or "REJECT" or "REDUCE_CONFIDENCE",
  "auditor_notes": "2-3 sentences of key findings"
}}"""


def run_stage7(finalists: list[dict], macro: MacroContext, log_fn=None, delay: int = API_DELAY) -> list[dict]:
    """Stage 7: Groq cross-validates Gemini's analysis."""
    for item in finalists:
        asset = item["asset"]
        tech = item["tech"]
        sr = tech.support_resistance
        s2 = item.get("stage2", {})
        s5 = item.get("stage5", {})
        s6 = item.get("stage6") or {}

        if log_fn:
            log_fn("stage7", f"Korsvalidering: {asset.display_name}")

        mod_a = s5.get("A") or {}
        mod_b = s5.get("B") or {}
        mod_c = s5.get("C") or {}
        mod_d = s5.get("D") or {}

        failure_reasons = s6.get("failure_reasons", [])
        fr_text = "; ".join(f.get("reason", "") for f in failure_reasons) if failure_reasons else "None"

        prompt = STAGE7_PROMPT.format(
            asset_name=asset.display_name,
            composite=s2.get("composite_score", 5),
            direction=s2.get("consensus_direction", "NEUTRAL"),
            hist_verdict=mod_a.get("historical_verdict", "N/A"),
            inter_verdict=mod_b.get("inter_market_verdict", "N/A"),
            fund_verdict=mod_c.get("fundamental_verdict", "N/A"),
            tail_verdict=mod_d.get("tail_risk_verdict", "N/A"),
            risk_rating=s6.get("risk_rating", "N/A"),
            should_proceed=s6.get("should_proceed", "N/A"),
            failure_reasons=fr_text[:300],
            price=f"{tech.current_price:,.2f}",
            rsi=f"{tech.rsi_value:.1f}",
            macd_hist=f"{tech.macd_histogram:+.4f}",
            bb_position=tech.bb_position,
            sma_20=f"{tech.sma_20:,.2f}",
            sma_50=f"{tech.sma_50:,.2f}",
            sma_200=f"{tech.sma_200:,.2f}",
            volume=f"{tech.volume_ratio:.1f}",
            atr=f"{tech.atr_value:,.2f}",
            supports=", ".join(f"{s:,.2f}" for s in sr.supports[:3]),
            resistances=", ".join(f"{r:,.2f}" for r in sr.resistances[:3]),
        )

        raw = _call_specific_provider(prompt, "Groq")
        if not raw:
            time.sleep(delay)
            raw = _call_specific_provider(prompt, "Gemini")

        data = _parse_json(raw)
        if data:
            item["stage7"] = data
            verdict = data.get("final_verdict", "?")
            adj_conf = data.get("adjusted_confidence", 0)
            emoji = "+" if verdict == "APPROVE" else ("-" if verdict == "REJECT" else "~")
            if log_fn:
                log_fn("result", f"[{emoji}] {asset.display_name}: {verdict} (justerad konfidens: {adj_conf:.0%})")
        else:
            item["stage7"] = None
            if log_fn:
                log_fn("error", f"{asset.display_name}: Korsvalidering misslyckades")

        time.sleep(delay)

    return finalists


# ---------------------------------------------------------------------------
# STAGE 8: Yesterday's Accuracy Review
# ---------------------------------------------------------------------------

STAGE8_PROMPT = """You are a trading performance analyst. Review yesterday's recommendations.

=== YESTERDAY'S PICKS ===
{picks_text}

=== TODAY'S PRICES ===
{prices_text}

For each pick:
1. Was the direction correct? (price moved in the recommended direction)
2. Did it hit the target, stop-loss, or neither?
3. What was the actual P/L percentage from entry?

Then write a LEARNING BRIEF (2-4 sentences):
- Why were we right or wrong?
- What patterns did we miss?
- How should this adjust our confidence today?

Output ONLY the raw JSON object. No markdown, no code blocks, no explanation.
{{
  "reviews": [
    {{
      "asset": "name",
      "yesterday_direction": "BULL/BEAR",
      "yesterday_entry": 0.0,
      "yesterday_target": 0.0,
      "yesterday_stop": 0.0,
      "today_price": 0.0,
      "pnl_pct": "+/- X.X%",
      "correct": true or false,
      "hit_target": true or false,
      "hit_stop": true or false
    }}
  ],
  "accuracy_pct": 0.0,
  "learning_brief": "2-4 sentences",
  "confidence_adjustment": "raise/lower/maintain"
}}"""


def run_stage8(log_fn=None, delay: int = API_DELAY) -> dict | None:
    """Stage 8: Compare yesterday's picks against today's prices."""
    yesterday_data = _load_yesterday_results()
    if not yesterday_data:
        if log_fn:
            log_fn("stage8", "Inga gårdagsresultat tillgängliga — hoppar över")
        return None

    picks = yesterday_data.get("final_picks", [])
    if not picks:
        if log_fn:
            log_fn("stage8", "Inga rekommendationer igår — hoppar över")
        return None

    if log_fn:
        log_fn("stage8", f"Granskar {len(picks)} rekommendationer från igår...")

    picks_lines = []
    prices_lines = []

    for pick in picks:
        asset = pick.get("asset", {})
        tp = pick.get("trading_plan") or {}
        syn = pick.get("synthesis") or {}
        ticker = asset.get("ticker", "")
        name = asset.get("display_name", "?")

        verdict = pick.get("final_verdict", syn.get("verdict", "?"))
        entry = tp.get("entry_price") or syn.get("entry_price", 0)
        target = tp.get("take_profit") or syn.get("take_profit", 0)
        stop = tp.get("stop_loss") or syn.get("stop_loss", 0)

        picks_lines.append(
            f"- {name} ({ticker}): {verdict}, Entry: {entry}, Target: {target}, Stop: {stop}"
        )

        today_price = fetch_macro_indicator(ticker) if ticker else None
        if today_price:
            prices_lines.append(f"- {name}: {today_price:,.2f}")
        else:
            df = fetch_ohlc(ticker)
            if not df.empty:
                prices_lines.append(f"- {name}: {float(df['Close'].iloc[-1]):,.2f}")
            else:
                prices_lines.append(f"- {name}: N/A")

    prompt = STAGE8_PROMPT.format(
        picks_text="\n".join(picks_lines),
        prices_text="\n".join(prices_lines),
    )

    raw = _call_specific_provider(prompt, "Gemini")
    if not raw:
        time.sleep(delay)
        raw = _call_specific_provider(prompt, "Groq")

    data = _parse_json(raw)
    if data:
        if log_fn:
            acc = data.get("accuracy_pct", 0)
            adj = data.get("confidence_adjustment", "maintain")
            log_fn("result", f"Gårdagens träffsäkerhet: {acc:.0f}% — Justering: {adj}")
            brief = data.get("learning_brief", "")
            if brief:
                log_fn("info", brief[:200])
        return data

    if log_fn:
        log_fn("error", "Gårdagsanalys misslyckades")
    return None


# ---------------------------------------------------------------------------
# FINAL SYNTHESIS & TRADING PLAN
# ---------------------------------------------------------------------------

SYNTHESIS_PROMPT = """Du är en senior portföljförvaltare på en svensk investmentbank. Fatta ett SLUTGILTIGT beslut om {asset_name}.

=== MARKNADSREGIM: {regime} ===

=== KVANTITATIV ANALYS (Steg 2) ===
Kompositbetyg: {composite}/10 | Riktning: {direction}
{lens_summary}

=== DJUPFORSKNING (Steg 5) ===
A) Historiska analogier: {hist_verdict} (konfidens: {hist_conf})
   Nyckellektion: {hist_lesson}
B) Intermarknadsanalys: {inter_verdict}
   Analys: {inter_analysis}
C) Fundamenta: {fund_verdict}
   Nyckelfaktor: {fund_driver}
D) Stresstest: {tail_verdict}
   Värsta pris: {worst_price}

=== DJÄVULENS ADVOKAT (Steg 6) ===
Risk: {risk_rating} | Fortsätt: {should_proceed}
{da_reasons}

=== KORSVALIDERING (Steg 7) ===
Dom: {cv_verdict} | Justerad konfidens: {cv_confidence}
Revisionsnoteringar: {cv_notes}

=== GÅRDAGENS LÄRDOMAR ===
{yesterday_brief}

INSTRUKTIONER:
1. Syntetisera ALLT ovan till ETT beslut
2. Om steg 7 säger REJECT — du MÅSTE ha extremt starka skäl för att åsidosätta
3. Konfidens under 0.55 = NO_TRADE
4. Skriv MINST 150 ord resonemang

Returnera ENBART rå JSON. Ingen markdown, inga kodblock, ingen förklaring utanför JSON.
{{
  "verdict": "BUY_BULL" eller "BUY_BEAR" eller "NO_TRADE",
  "final_confidence": 0.0 till 1.0,
  "chain_of_thought": "Minst 150 ord resonemang på svenska",
  "entry_price": 0.0,
  "stop_loss": 0.0,
  "take_profit": 0.0,
  "risk_reward": "1:X.X",
  "key_catalyst": "1 mening",
  "biggest_risk": "1 mening",
  "time_horizon": "1-3 dagar" eller "3-5 dagar",
  "exit_strategy": "2-3 meningar: exakt när och hur man stänger positionen"
}}"""


def run_final_synthesis(finalists: list[dict], macro: MacroContext, yesterday_review: dict | None,
                        log_fn=None, delay: int = API_DELAY) -> list[dict]:
    """Generate final verdict and trading plan for each finalist."""
    yesterday_brief = ""
    if yesterday_review:
        yesterday_brief = yesterday_review.get("learning_brief", "Ingen data.")
    else:
        yesterday_brief = "Första skanningen — ingen historik tillgänglig."

    final_picks = []

    for item in finalists:
        asset = item["asset"]
        tech = item["tech"]
        s2 = item.get("stage2", {})
        s5 = item.get("stage5", {})
        s6 = item.get("stage6") or {}
        s7 = item.get("stage7") or {}

        if log_fn:
            log_fn("synthesis", f"Slutgiltig syntes: {asset.display_name}")

        mod_a = s5.get("A") or {}
        mod_b = s5.get("B") or {}
        mod_c = s5.get("C") or {}
        mod_d = s5.get("D") or {}

        lenses = s2.get("lenses", {})
        lens_lines = []
        for lens_name, lens_data in lenses.items():
            lens_lines.append(f"  {lens_name}: {lens_data.get('score', '?')}/10 — {lens_data.get('analysis', '')[:80]}")
        lens_summary = "\n".join(lens_lines) if lens_lines else "Inga linsdata."

        failure_reasons = s6.get("failure_reasons", [])
        da_lines = []
        for fr in failure_reasons:
            da_lines.append(f"  [{fr.get('type', '?')}] {fr.get('reason', '')} ({fr.get('severity', '?')})")
        da_reasons = "\n".join(da_lines) if da_lines else "Inga invändningar."

        prompt = SYNTHESIS_PROMPT.format(
            asset_name=asset.display_name,
            regime=macro.regime,
            composite=s2.get("composite_score", 5),
            direction=s2.get("consensus_direction", "NEUTRAL"),
            lens_summary=lens_summary,
            hist_verdict=mod_a.get("historical_verdict", "N/A"),
            hist_conf=f"{mod_a.get('confidence_from_history', 0):.0%}",
            hist_lesson=mod_a.get("key_lesson", "N/A"),
            inter_verdict=mod_b.get("inter_market_verdict", "N/A"),
            inter_analysis=mod_b.get("analysis", "N/A")[:200],
            fund_verdict=mod_c.get("fundamental_verdict", "N/A"),
            fund_driver=mod_c.get("key_driver", "N/A"),
            tail_verdict=mod_d.get("tail_risk_verdict", "N/A"),
            worst_price=mod_d.get("worst_case_price", "N/A"),
            risk_rating=s6.get("risk_rating", "N/A"),
            should_proceed=s6.get("should_proceed", "N/A"),
            da_reasons=da_reasons,
            cv_verdict=s7.get("final_verdict", "N/A"),
            cv_confidence=f"{s7.get('adjusted_confidence', 0):.0%}",
            cv_notes=s7.get("auditor_notes", "N/A")[:200],
            yesterday_brief=yesterday_brief[:200],
        )

        raw = _call_specific_provider(prompt, "Gemini")
        if not raw:
            time.sleep(delay)
            raw = _call_specific_provider(prompt, "Groq")

        data = _parse_json(raw)
        if data:
            item["synthesis"] = data
            verdict = data.get("verdict", "NO_TRADE")
            conf = data.get("final_confidence", 0)

            if verdict != "NO_TRADE" and conf >= 0.55:
                action = "BULL" if verdict == "BUY_BULL" else "BEAR"
                trading_plan = generate_trading_plan(tech, action)
                item["trading_plan"] = trading_plan
                item["final_verdict"] = verdict
                item["final_confidence"] = conf
                final_picks.append(item)
                if log_fn:
                    log_fn("result", f"[+] {asset.display_name}: {verdict} ({conf:.0%})")
            else:
                item["final_verdict"] = "NO_TRADE"
                item["final_confidence"] = conf
                if log_fn:
                    log_fn("result", f"[-] {asset.display_name}: NO_TRADE ({conf:.0%})")
        else:
            item["synthesis"] = None
            item["final_verdict"] = "NO_TRADE"
            item["final_confidence"] = 0
            if log_fn:
                log_fn("error", f"{asset.display_name}: Syntes misslyckades")

        time.sleep(delay)

    final_picks.sort(key=lambda x: x.get("final_confidence", 0), reverse=True)
    return final_picks


# ---------------------------------------------------------------------------
# Full Pipeline Orchestrator
# ---------------------------------------------------------------------------

@dataclass
class DeepScanResult:
    scan_date: str
    scan_time: str
    vix_value: float | None
    dxy_value: float | None
    us10y_value: float | None
    market_regime: str
    regime_report: str
    global_sentiment: str
    all_scores: list[dict]
    top5: list[dict]
    final_picks: list[dict]
    yesterday_review: dict | None
    log: list[str]
    total_assets: int
    stage_calls: dict = field(default_factory=dict)


def run_deep_scan(log_fn=None, max_top: int = 5, api_delay: int = API_DELAY) -> DeepScanResult:
    """Execute the full 8-stage intelligence pipeline."""
    log_lines = []

    def _log(phase, msg):
        log_lines.append(msg)
        if log_fn:
            log_fn(phase, msg)

    assets = ALL_ASSETS_FLAT
    _log("start", f"**Unitron 8-Stegs Djupanalys** — {len(assets)} tillgångar, {datetime.now().strftime('%H:%M')}")
    _log("info", "")

    # --- STAGE 0: Data Foundation ---
    _log("header", "### Steg 0: Datafundament")
    asset_data, macro = run_stage0(assets, log_fn=_log)
    _log("info", "")

    # --- STAGE 8: Yesterday's Review (run early to inject learnings) ---
    _log("header", "### Steg 8: Gårdagens Utvärdering")
    yesterday_review = run_stage8(log_fn=_log, delay=api_delay)
    _log("info", "")

    # --- STAGE 1: Global Macro Anchor ---
    _log("header", "### Steg 1: Global Makroankare (Gemini)")
    macro = run_stage1(macro, log_fn=_log, delay=api_delay)
    _log("info", "")

    # --- STAGE 2: Multi-Lens Technical Scan ---
    _log("header", "### Steg 2: Flerlins Teknisk Skanning (Groq)")
    asset_data = run_stage2(asset_data, macro, log_fn=_log, delay=api_delay)
    _log("info", "")

    all_scores = []
    for item in asset_data:
        s2 = item.get("stage2", {})
        all_scores.append({
            "name": item["asset"].display_name,
            "ticker": item["asset"].ticker,
            "score": s2.get("composite_score", 0),
            "direction": s2.get("consensus_direction", "NEUTRAL"),
            "confidence": s2.get("confidence", 0),
            "summary": "",
            "lenses": s2.get("lenses", {}),
        })

    # --- STAGE 3 & 4: Ranking + Deep News ---
    _log("header", "### Steg 3-4: Ranking & Premium-Nyheter")
    finalists = run_stage3_4(asset_data, log_fn=_log, max_top=max_top, delay=api_delay)
    _log("info", f"**{len(finalists)} finalister vidare till djupanalys**")
    _log("info", "")

    # --- STAGE 5: High-Dimensional Deep Dive ---
    _log("header", "### Steg 5: Högdimensionell Djupanalys (Gemini)")
    finalists = run_stage5(finalists, macro, log_fn=_log, delay=api_delay)
    _log("info", "")

    # --- STAGE 6: Devil's Advocate ---
    _log("header", "### Steg 6: Djävulens Advokat (Gemini)")
    finalists = run_stage6(finalists, macro, log_fn=_log, delay=api_delay)
    _log("info", "")

    # --- STAGE 7: Cross-Validation ---
    _log("header", "### Steg 7: Korsvalidering (Groq)")
    finalists = run_stage7(finalists, macro, log_fn=_log, delay=api_delay)
    _log("info", "")

    # --- Final Synthesis ---
    _log("header", "### Slutgiltig Syntes & Handelsplan")
    final_picks = run_final_synthesis(finalists, macro, yesterday_review, log_fn=_log, delay=api_delay)
    _log("info", "")

    _log("info", f"**Analys klar!** {len(final_picks)} handelsrekommendation{'er' if len(final_picks) != 1 else ''}")

    global_sent = _assess_global_sentiment(macro)

    return DeepScanResult(
        scan_date=datetime.now().strftime("%Y-%m-%d"),
        scan_time=datetime.now().strftime("%H:%M"),
        vix_value=macro.vix,
        dxy_value=macro.dxy,
        us10y_value=macro.us10y,
        market_regime=macro.regime,
        regime_report=macro.regime_report,
        global_sentiment=global_sent,
        all_scores=all_scores,
        top5=[{
            "asset": c["asset"],
            "tech": c["tech"],
            "stage2": c.get("stage2"),
            "stage5": c.get("stage5"),
            "stage6": c.get("stage6"),
            "stage7": c.get("stage7"),
            "synthesis": c.get("synthesis"),
            "headlines": c.get("deep_headlines", c.get("headlines", [])),
            "final_verdict": c.get("final_verdict", "NO_TRADE"),
            "final_confidence": c.get("final_confidence", 0),
        } for c in finalists],
        final_picks=final_picks,
        yesterday_review=yesterday_review,
        log=log_lines,
        total_assets=len(assets),
        stage_calls={
            "stage0": len(asset_data),
            "stage1": 1,
            "stage2_batches": (len(asset_data) + 3) // 4 * 5,
            "stage5": len(finalists) * 4,
            "stage6": len(finalists),
            "stage7": len(finalists),
            "synthesis": len(finalists),
            "stage8": 1 if yesterday_review else 0,
        },
    )


def _assess_global_sentiment(macro: MacroContext) -> str:
    parts = []
    if macro.vix is not None:
        if macro.vix > 30:
            parts.append("Extrem rädsla (VIX > 30)")
        elif macro.vix > 25:
            parts.append("Förhöjd rädsla (VIX > 25)")
        elif macro.vix > 20:
            parts.append("Försiktighet (VIX > 20)")
        elif macro.vix > 15:
            parts.append("Normalt marknadsläge")
        else:
            parts.append("Lugn marknad (VIX < 15)")

    if macro.dxy is not None:
        if macro.dxy > 105:
            parts.append("Stark dollar (DXY > 105)")
        elif macro.dxy < 100:
            parts.append("Svag dollar (DXY < 100)")
        else:
            parts.append("Neutral dollarstyrka")

    if macro.us10y is not None:
        if macro.us10y > 4.5:
            parts.append(f"Hög ränta ({macro.us10y:.2f}%)")
        elif macro.us10y < 3.5:
            parts.append(f"Låg ränta ({macro.us10y:.2f}%)")
        else:
            parts.append(f"Neutral ränta ({macro.us10y:.2f}%)")

    if macro.regime:
        parts.append(f"Regim: {macro.regime}")

    return ". ".join(parts) if parts else "Data ej tillgänglig"
