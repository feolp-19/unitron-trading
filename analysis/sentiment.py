"""AI sentiment analysis with automatic provider fallback chain:
Groq -> Gemini -> Grok -> keyword fallback.

AI is the sole decision maker via FULL_ANALYSIS_PROMPT,
receiving ALL data (technicals, VIX, volume, SMAs, S/R, news)."""

import json
from dataclasses import dataclass, field

import streamlit as st

from config import AI_CONFIGS, get_secret


@dataclass
class HeadlineSentiment:
    headline: str
    sentiment: str
    relevance: float
    reasoning: str


@dataclass
class SentimentSignal:
    direction: str
    confidence: float
    relevant_count: int
    total_count: int
    low_data_quality: bool
    summary: str
    headline_details: list[HeadlineSentiment] = field(default_factory=list)
    herd_ratio: float = 0.0
    ai_provider_used: str = ""


SENTIMENT_PROMPT = """You are a financial sentiment analyst. Analyze these news headlines for their impact on the given asset.

Asset: {asset_name}

For each headline, return:
- sentiment: "positive", "negative", or "neutral"
- relevance: 0.0 to 1.0 (how relevant is this headline to the asset)
- reasoning: one short sentence explaining why

After analyzing all headlines, provide:
- overall_sentiment: "positive", "negative", or "neutral"
- confidence: 0.0 to 1.0
- summary: one sentence summarizing the macro outlook

Return ONLY valid JSON in this exact format:
{{
  "headlines": [
    {{"sentiment": "positive", "relevance": 0.8, "reasoning": "..."}},
    ...
  ],
  "overall_sentiment": "positive",
  "confidence": 0.75,
  "summary": "..."
}}

Headlines:
{headlines_text}"""


FULL_ANALYSIS_PROMPT = """You are a professional trader managing real money. You must decide: BUY a BULL certificate, BUY a BEAR certificate, or NO TRADE.

MINDSET: Default is NO_TRADE, but you SHOULD recommend trades when genuine opportunities exist. Look for setups where 3+ indicators align. Not every asset will have a trade, but good setups happen regularly across 17 assets.

You are scanning {asset_name}.

=== PRICE & TREND ===
- Current price: {price}
- 20-day SMA: {sma_20} | 50-day SMA: {sma_50} | 200-day SMA: {sma_200}
- Price vs 200-day SMA: {price_vs_sma}
- 50-week SMA: {sma_50w} → price is {price_vs_weekly_sma} the weekly trend
- SMA stack alignment: {sma_alignment}
- SMA directional bias: {sma_bias}
- Multi-timeframe: {timeframe_alignment}

=== MACD ===
- MACD: {macd_value} | Signal: {macd_signal} | Histogram: {macd_histogram}
- MACD crossover: {macd_cross}

=== BOLLINGER BANDS ===
- Upper: {bb_upper} | Middle: {bb_middle} | Lower: {bb_lower}
- Price position: {bb_position}
- Band width: {bb_width}% ({bb_interpretation})

=== SUPPORT & RESISTANCE ===
{sr_text}

=== MOMENTUM ===
- RSI (14): {rsi} {rsi_interpretation}
- RSI 2-day change: {rsi_trend} (momentum is {rsi_momentum})

=== VOLATILITY ===
- ATR (14): {atr} | Volatility ratio: {atr_ratio}x ({volatility_interpretation})

=== VOLUME ===
- Volume vs 20-day average: {volume_ratio}x ({volume_interpretation})

=== FEAR & GREED (VIX) ===
- VIX: {vix_value} — {vix_interpretation}

=== NEWS & MACRO ===
{headlines_text}

=== TRADE SETUP PATTERNS (look for these) ===
1. TREND CONTINUATION: Price above key SMAs + MACD bullish + RSI 40-65 = strong BULL
2. TREND CONTINUATION (BEAR): Price below key SMAs + MACD bearish + RSI 35-60 = strong BEAR
3. BREAKOUT: Price breaking above Bollinger upper band with high volume = momentum BULL
4. BREAKDOWN: Price breaking below Bollinger lower band with high volume = momentum BEAR
5. BOUNCE: Price near Bollinger lower + RSI < 35 + support nearby = reversal BULL
6. REJECTION: Price near Bollinger upper + RSI > 65 + resistance nearby = reversal BEAR
7. MACD CROSS: Recent bullish/bearish MACD crossover confirms direction change

=== MINIMUM REQUIREMENTS FOR A TRADE ===
At least 3 of these must be true:
1. SMA bias supports the direction (bullish bias for BULL, bearish for BEAR)
2. MACD supports (histogram positive for BULL, negative for BEAR, or recent cross)
3. RSI supports (not contradicting — not > 70 for BULL, not < 30 for BEAR)
4. News sentiment is not contradicting the direction
5. Bollinger position supports (not at extreme opposite band)
6. Volume > 1.0x average (some conviction behind the move)
7. No major S/R obstacle within 2% in the trade direction

If fewer than 3 conditions are met → NO_TRADE.

=== CONFIDENCE SCORING RUBRIC ===
Start at 0.40 (base) and adjust:
+0.12 if SMA alignment is perfect stack (bullish_stack or bearish_stack)
+0.07 if SMA bias supports but not perfect stack (e.g. 2 of 3 SMAs aligned)
+0.08 if MACD histogram supports direction AND recent crossover
+0.05 if MACD histogram supports but no recent crossover
+0.08 if daily AND weekly timeframes aligned
-0.10 if daily and weekly timeframes CONFLICT
+0.05 if RSI supports (50-65 for BULL, 35-50 for BEAR)
+0.07 if RSI in extreme reversal zone (< 30 oversold bounce, > 70 overbought sell)
-0.08 if RSI contradicts direction
+0.08 if volume > 1.5x average (strong conviction)
+0.03 if volume 1.0-1.5x (normal)
-0.05 if volume < 0.7x (weak)
+0.08 if news strongly supports
+0.04 if news mildly supports
-0.07 if news contradicts
+0.05 if Bollinger position supports (e.g. below lower for bounce, in band trending)
-0.05 if price at opposite Bollinger extreme
+0.05 if clear S/R runway (> 3% to next obstacle)
-0.03 if VIX > 25
-0.07 if VIX > 30

Final confidence MUST be 0.30-0.90. Round to 2 decimals.
Below 0.55 → change verdict to NO_TRADE.
Show your math in the analysis field.

Return ONLY valid JSON:
{{
  "verdict": "BUY_BULL" or "BUY_BEAR" or "NO_TRADE",
  "confidence": 0.0 to 1.0,
  "analysis": "2-4 sentences with confidence math, referencing specific indicators and S/R levels",
  "key_factors": ["factor 1", "factor 2", "factor 3"],
  "risks": ["specific risk 1", "specific risk 2"],
  "stop_loss_reasoning": "1 sentence: where to place stop-loss and why",
  "take_profit_reasoning": "1 sentence: where to take profit and why",
  "outlook": "1 sentence: what to watch next"
}}"""


RISK_ASSESSMENT_PROMPT = """You are a professional risk analyst. Your ONLY job is to find ALL risks for {asset_name} right now.
Think like someone who WILL LOSE MONEY if you miss a risk.

=== MARKET DATA ===
- Price: {price}, SMA alignment: {sma_alignment}
- RSI: {rsi}, Volume: {volume_ratio}x avg, VIX: {vix_value}
- ATR: {atr} (volatility ratio: {atr_ratio}x)
- Near resistance: {near_resistance}, Near support: {near_support}

=== SUPPORT & RESISTANCE ===
{sr_text}

=== NEWS ===
{headlines_text}

Analyze EVERY risk dimension:
1. TECHNICAL RISKS: Exhausted trend? Overbought/oversold? Divergences? Volume declining?
2. MACRO RISKS: Interest rates? Inflation? Central bank decisions? Geopolitical tension?
3. SECTOR RISKS: Industry-specific headwinds? Regulatory changes? Competition?
4. TIMING RISKS: End of quarter? Options expiry? Earnings season? Holiday trading?
5. SENTIMENT RISKS: Too crowded trade? Herd mentality? Contrarian signals?
6. LIQUIDITY RISKS: Low volume? Wide spreads? Illiquid instrument?

Return ONLY valid JSON:
{{
  "overall_risk": "LOW" or "MEDIUM" or "HIGH" or "CRITICAL",
  "risk_score": 0.0 to 1.0 (1.0 = maximum risk),
  "risks": [
    {{"category": "technical|macro|sector|timing|sentiment|liquidity", "description": "specific risk", "severity": "low|medium|high|critical"}},
    ...
  ],
  "biggest_threat": "1 sentence: the single most dangerous risk right now",
  "safe_to_trade": true or false,
  "reasoning": "2-3 sentences explaining your overall risk assessment"
}}"""


MACRO_CONTEXT_PROMPT = """You are a macro strategist analyzing {asset_name} in the broader market context.

=== ASSET DATA ===
- Asset: {asset_name} (type: {asset_type})
- Price: {price}, SMA alignment: {sma_alignment}
- VIX: {vix_value}

=== NEWS HEADLINES ===
{headlines_text}

Analyze the MACRO environment for this specific asset:
1. How does the current global economic environment affect this asset?
2. Are there upcoming events (central bank meetings, data releases, elections) that create risk?
3. What is the prevailing institutional consensus on this asset/sector?
4. Is this asset correlated with other markets that are showing warning signs?
5. What is the best-case AND worst-case scenario for the next 1-5 trading days?

Return ONLY valid JSON:
{{
  "macro_bias": "bullish" or "bearish" or "neutral",
  "macro_confidence": 0.0 to 1.0,
  "key_events": ["upcoming event 1", "event 2"],
  "correlations": "1 sentence: relevant cross-market signals",
  "best_case": "1 sentence",
  "worst_case": "1 sentence",
  "institutional_view": "1 sentence: what smart money is likely doing",
  "recommendation_modifier": -0.15 to +0.15
}}"""


def _call_groq(prompt: str) -> str | None:
    try:
        from groq import Groq
        from storage.usage_tracker import track_call
        api_key = get_secret("GROQ_API_KEY")
        if not api_key:
            return None
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=AI_CONFIGS["groq"]["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=4000,
        )
        track_call("groq")
        return response.choices[0].message.content
    except Exception:
        return None


def _call_gemini(prompt: str) -> str | None:
    try:
        from google import genai
        from storage.usage_tracker import track_call
        api_key = get_secret("GOOGLE_API_KEY")
        if not api_key:
            return None
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=AI_CONFIGS["gemini"]["model"],
            contents=prompt,
            config={"max_output_tokens": 4096, "temperature": 0.1},
        )
        track_call("gemini")
        return response.text
    except Exception:
        return None


def _call_grok(prompt: str) -> str | None:
    try:
        from openai import OpenAI
        from storage.usage_tracker import track_call
        api_key = get_secret("XAI_API_KEY")
        if not api_key:
            return None
        client = OpenAI(api_key=api_key, base_url=AI_CONFIGS["grok"]["base_url"])
        response = client.chat.completions.create(
            model=AI_CONFIGS["grok"]["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=4000,
        )
        track_call("grok")
        return response.choices[0].message.content
    except Exception:
        return None


def _call_ai_with_fallback(prompt: str) -> tuple[str | None, str]:
    """Try AI providers in order: Groq -> Gemini -> Grok."""
    providers = [
        ("Groq", _call_groq),
        ("Gemini", _call_gemini),
        ("Grok", _call_grok),
    ]
    for name, fn in providers:
        result = fn(prompt)
        if result:
            return result, name
    return None, "none"


def _call_specific_provider(prompt: str, provider: str) -> str | None:
    """Call a specific AI provider by name."""
    fns = {"Groq": _call_groq, "Gemini": _call_gemini, "Grok": _call_grok}
    fn = fns.get(provider)
    return fn(prompt) if fn else None


def _keyword_fallback(headlines: list[dict]) -> SentimentSignal:
    positive_words = {"surge", "rally", "gain", "rise", "up", "bull", "growth", "profit",
                      "beat", "record", "high", "boost", "strong", "positive", "recovery"}
    negative_words = {"crash", "fall", "drop", "decline", "loss", "bear", "recession", "fear",
                      "crisis", "down", "weak", "miss", "cut", "sell", "tariff", "war", "risk"}

    pos_count = neg_count = 0
    details = []

    for h in headlines:
        text = h["headline"].lower()
        p = sum(1 for w in positive_words if w in text)
        n = sum(1 for w in negative_words if w in text)
        sentiment = "positive" if p > n else ("negative" if n > p else "neutral")
        if sentiment == "positive":
            pos_count += 1
        elif sentiment == "negative":
            neg_count += 1
        details.append(HeadlineSentiment(
            headline=h["headline"], sentiment=sentiment,
            relevance=0.5, reasoning="Nyckelordsanalys (AI ej tillganglig)",
        ))

    total = len(headlines)
    if pos_count > neg_count:
        direction, confidence = "POSITIVE", pos_count / total if total else 0
    elif neg_count > pos_count:
        direction, confidence = "NEGATIVE", neg_count / total if total else 0
    else:
        direction, confidence = "NEUTRAL", 0.3

    dominant = max(pos_count, neg_count)

    return SentimentSignal(
        direction=direction,
        confidence=round(confidence, 2),
        relevant_count=total,
        total_count=total,
        low_data_quality=total < 3,
        summary="Nyckelordsbaserad analys — AI ej tillganglig",
        headline_details=details,
        herd_ratio=round(dominant / total, 2) if total else 0,
        ai_provider_used="keyword_fallback",
    )


def _parse_ai_response(raw: str, headlines: list[dict], provider: str) -> SentimentSignal | None:
    try:
        cleaned = raw.strip()
        if "```" in cleaned:
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        data = json.loads(cleaned)

        details = []
        relevant_count = pos_count = neg_count = 0

        for i, item in enumerate(data.get("headlines", [])):
            headline_text = headlines[i]["headline"] if i < len(headlines) else ""
            hs = HeadlineSentiment(
                headline=headline_text,
                sentiment=item.get("sentiment", "neutral"),
                relevance=float(item.get("relevance", 0.5)),
                reasoning=item.get("reasoning", ""),
            )
            details.append(hs)
            if hs.relevance > 0.3:
                relevant_count += 1
                if hs.sentiment == "positive":
                    pos_count += 1
                elif hs.sentiment == "negative":
                    neg_count += 1

        overall = data.get("overall_sentiment", "neutral").upper()
        direction = "POSITIVE" if overall == "POSITIVE" else ("NEGATIVE" if overall == "NEGATIVE" else "NEUTRAL")

        dominant = max(pos_count, neg_count)

        return SentimentSignal(
            direction=direction,
            confidence=round(float(data.get("confidence", 0.5)), 2),
            relevant_count=relevant_count,
            total_count=len(headlines),
            low_data_quality=relevant_count < 3,
            summary=data.get("summary", ""),
            headline_details=details,
            herd_ratio=round(dominant / relevant_count, 2) if relevant_count else 0,
            ai_provider_used=provider,
        )
    except Exception:
        return None


@st.cache_data(ttl=14400, show_spinner=False)
def analyze_sentiment(asset_name: str, headlines_json: str) -> dict:
    """Analyze sentiment with automatic AI fallback. Returns dict for caching."""
    headlines = json.loads(headlines_json)

    if not headlines:
        return SentimentSignal(
            direction="NEUTRAL", confidence=0.0,
            relevant_count=0, total_count=0,
            low_data_quality=True, summary="Inga nyhetsrubriker hittades",
            ai_provider_used="none",
        ).__dict__

    prompt = SENTIMENT_PROMPT.format(
        asset_name=asset_name,
        headlines_text="\n".join(f"{i+1}. {h['headline']}" for i, h in enumerate(headlines)),
    )

    raw_response, provider = _call_ai_with_fallback(prompt)

    if raw_response:
        result = _parse_ai_response(raw_response, headlines, provider)
        if result:
            return result.__dict__

    return _keyword_fallback(headlines).__dict__


def _interpret_rsi(rsi: float) -> str:
    if rsi < 30:
        return "(OVERSOLD — potential bounce up)"
    if rsi < 40:
        return "(approaching oversold)"
    if rsi > 70:
        return "(OVERBOUGHT — potential pullback)"
    if rsi > 60:
        return "(approaching overbought)"
    return "(neutral zone)"


def _interpret_volume(vol_ratio: float) -> str:
    if vol_ratio < 0.01:
        return "volume data unavailable for this asset — ignore volume factor"
    if vol_ratio > 2.0:
        return "very high volume — strong conviction"
    if vol_ratio > 1.5:
        return "above average — confirming move"
    if vol_ratio > 0.7:
        return "average volume"
    return "low volume — weak conviction"


def _interpret_vix(vix: float | None, level: str) -> str:
    if vix is None:
        return "VIX data unavailable"
    labels = {
        "low_fear": f"{vix:.1f} — Low fear/complacency. Markets calm.",
        "normal": f"{vix:.1f} — Normal market conditions.",
        "elevated": f"{vix:.1f} — Elevated fear. Use tighter stops.",
        "extreme_fear": f"{vix:.1f} — EXTREME FEAR. Very high risk.",
    }
    return labels.get(level, f"{vix:.1f}")


def _interpret_volatility(atr_ratio: float) -> str:
    if atr_ratio > 2.0:
        return "extremely volatile — widen stops"
    if atr_ratio > 1.3:
        return "above normal volatility"
    if atr_ratio > 0.7:
        return "normal volatility"
    return "very low volatility — potential breakout coming"


def _interpret_sma_alignment(alignment: str) -> str:
    labels = {
        "bullish_stack": "BULLISH STACK (Price > SMA20 > SMA50 > SMA200) — strong uptrend",
        "bearish_stack": "BEARISH STACK (Price < SMA20 < SMA50 < SMA200) — strong downtrend",
        "mixed": "MIXED — no clear trend alignment, be cautious",
    }
    return labels.get(alignment, alignment)


def _format_sr_text(
    supports: list[float], resistances: list[float],
    price: float, near_resistance: bool, near_support: bool,
) -> str:
    lines = []
    if resistances:
        for i, r in enumerate(resistances):
            dist_pct = ((r - price) / price) * 100
            lines.append(f"- Resistance {i+1}: {r:,.2f} ({dist_pct:+.1f}% from current price)")
    else:
        lines.append("- No significant resistance levels identified above current price")

    if supports:
        for i, s in enumerate(supports):
            dist_pct = ((s - price) / price) * 100
            lines.append(f"- Support {i+1}: {s:,.2f} ({dist_pct:+.1f}% from current price)")
    else:
        lines.append("- No significant support levels identified below current price")

    if near_resistance:
        lines.append("- ⚠️ WARNING: Price is within 2% of nearest resistance — risk of rejection!")
    if near_support:
        lines.append("- ⚠️ NOTE: Price is within 2% of nearest support — potential bounce zone")

    return "\n".join(lines)


def _interpret_bb(bb_width: float) -> str:
    if bb_width < 2:
        return "very tight squeeze — breakout likely imminent"
    if bb_width < 4:
        return "narrow bands — low volatility, potential breakout"
    if bb_width > 8:
        return "very wide bands — high volatility, trend in motion"
    return "normal band width"


def _build_prompt_kwargs(
    asset_name, price, sma_20, sma_50, sma_200, price_vs_sma,
    sma_50w, price_vs_weekly_sma, sma_alignment, sma_bias,
    rsi, atr, rsi_trend, atr_ratio, volume_ratio,
    vix_value, vix_level,
    supports_json, resistances_json, near_resistance, near_support,
    headlines_json,
    macd_value, macd_signal, macd_histogram, macd_cross,
    bb_upper, bb_lower, bb_middle, bb_position, bb_width,
) -> dict:
    """Build the common format kwargs for the analysis prompt."""
    headlines = json.loads(headlines_json)
    supports = json.loads(supports_json)
    resistances = json.loads(resistances_json)
    headlines_text = "\n".join(
        f"{i+1}. {h['headline']}" for i, h in enumerate(headlines)
    ) if headlines else "No recent headlines available."

    if price_vs_sma == price_vs_weekly_sma and price_vs_sma != "at":
        timeframe_alignment = f"ALIGNED — price is {price_vs_sma} both daily and weekly trends (strong)"
    elif price_vs_weekly_sma == "unavailable":
        timeframe_alignment = "Weekly data unavailable — rely on daily trend only"
    elif price_vs_sma == "at" or price_vs_weekly_sma == "at":
        timeframe_alignment = "One timeframe is neutral — signal is moderate"
    else:
        timeframe_alignment = f"CONFLICTING — daily: {price_vs_sma}, weekly: {price_vs_weekly_sma} (weak)"

    sr_text = _format_sr_text(supports, resistances, price, near_resistance, near_support)

    return dict(
        asset_name=asset_name,
        price=f"{price:,.2f}",
        sma_20=f"{sma_20:,.2f}",
        sma_50=f"{sma_50:,.2f}",
        sma_200=f"{sma_200:,.2f}",
        price_vs_sma=price_vs_sma,
        sma_50w=f"{sma_50w:,.2f}" if sma_50w else "unavailable",
        price_vs_weekly_sma=price_vs_weekly_sma if price_vs_weekly_sma != "unavailable" else "N/A",
        sma_alignment=_interpret_sma_alignment(sma_alignment),
        sma_bias=sma_bias.upper(),
        timeframe_alignment=timeframe_alignment,
        sr_text=sr_text,
        rsi=f"{rsi:.1f}",
        rsi_interpretation=_interpret_rsi(rsi),
        rsi_trend=f"{rsi_trend:+.1f}",
        rsi_momentum="accelerating upward" if rsi_trend > 3 else ("accelerating downward" if rsi_trend < -3 else "stable"),
        atr=f"{atr:,.2f}",
        atr_ratio=f"{atr_ratio:.1f}",
        volatility_interpretation=_interpret_volatility(atr_ratio),
        volume_ratio=f"{volume_ratio:.1f}",
        volume_interpretation=_interpret_volume(volume_ratio),
        vix_value=f"{vix_value:.1f}" if vix_value else "unavailable",
        vix_interpretation=_interpret_vix(vix_value, vix_level),
        headlines_text=headlines_text,
        macd_value=f"{macd_value:.4f}",
        macd_signal=f"{macd_signal:.4f}",
        macd_histogram=f"{macd_histogram:+.4f}",
        macd_cross=macd_cross.replace("_", " ").upper() if macd_cross != "none" else "No recent crossover",
        bb_upper=f"{bb_upper:,.2f}",
        bb_lower=f"{bb_lower:,.2f}",
        bb_middle=f"{bb_middle:,.2f}",
        bb_position=bb_position.replace("_", " "),
        bb_width=f"{bb_width:.1f}",
        bb_interpretation=_interpret_bb(bb_width),
    )


_ANALYSIS_EXTRA_PARAMS = [
    "sma_bias", "macd_value", "macd_signal", "macd_histogram", "macd_cross",
    "bb_upper", "bb_lower", "bb_middle", "bb_position", "bb_width",
]


@st.cache_data(ttl=14400, show_spinner=False)
def run_full_analysis(
    asset_name: str, price: float,
    sma_20: float, sma_50: float, sma_200: float, price_vs_sma: str,
    sma_50w: float | None, price_vs_weekly_sma: str, sma_alignment: str,
    sma_bias: str,
    rsi: float, atr: float, rsi_trend: float, atr_ratio: float,
    volume_ratio: float, vix_value: float | None, vix_level: str,
    supports_json: str, resistances_json: str,
    near_resistance: bool, near_support: bool, headlines_json: str,
    macd_value: float = 0, macd_signal: float = 0, macd_histogram: float = 0,
    macd_cross: str = "none",
    bb_upper: float = 0, bb_lower: float = 0, bb_middle: float = 0,
    bb_position: str = "in_band", bb_width: float = 0,
) -> dict | None:
    """Run comprehensive AI analysis — the primary decision maker."""
    kwargs = _build_prompt_kwargs(
        asset_name, price, sma_20, sma_50, sma_200, price_vs_sma,
        sma_50w, price_vs_weekly_sma, sma_alignment, sma_bias,
        rsi, atr, rsi_trend, atr_ratio, volume_ratio,
        vix_value, vix_level,
        supports_json, resistances_json, near_resistance, near_support,
        headlines_json,
        macd_value, macd_signal, macd_histogram, macd_cross,
        bb_upper, bb_lower, bb_middle, bb_position, bb_width,
    )
    prompt = FULL_ANALYSIS_PROMPT.format(**kwargs)

    raw, provider = _call_ai_with_fallback(prompt)
    if not raw:
        return None

    try:
        cleaned = raw.strip()
        if "```" in cleaned:
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        data = json.loads(cleaned)
        data["provider"] = provider
        return data
    except Exception:
        return None


@st.cache_data(ttl=14400, show_spinner=False)
def run_analysis_with_provider(
    provider: str, asset_name: str, price: float,
    sma_20: float, sma_50: float, sma_200: float, price_vs_sma: str,
    sma_50w: float | None, price_vs_weekly_sma: str, sma_alignment: str,
    sma_bias: str,
    rsi: float, atr: float, rsi_trend: float, atr_ratio: float,
    volume_ratio: float, vix_value: float | None, vix_level: str,
    supports_json: str, resistances_json: str,
    near_resistance: bool, near_support: bool, headlines_json: str,
    macd_value: float = 0, macd_signal: float = 0, macd_histogram: float = 0,
    macd_cross: str = "none",
    bb_upper: float = 0, bb_lower: float = 0, bb_middle: float = 0,
    bb_position: str = "in_band", bb_width: float = 0,
) -> dict | None:
    """Run full analysis forcing a specific AI provider."""
    kwargs = _build_prompt_kwargs(
        asset_name, price, sma_20, sma_50, sma_200, price_vs_sma,
        sma_50w, price_vs_weekly_sma, sma_alignment, sma_bias,
        rsi, atr, rsi_trend, atr_ratio, volume_ratio,
        vix_value, vix_level,
        supports_json, resistances_json, near_resistance, near_support,
        headlines_json,
        macd_value, macd_signal, macd_histogram, macd_cross,
        bb_upper, bb_lower, bb_middle, bb_position, bb_width,
    )
    prompt = FULL_ANALYSIS_PROMPT.format(**kwargs)

    raw = _call_specific_provider(prompt, provider)
    if not raw:
        return None

    try:
        cleaned = raw.strip()
        if "```" in cleaned:
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        data = json.loads(cleaned)
        data["provider"] = provider
        return data
    except Exception:
        return None


@st.cache_data(ttl=14400, show_spinner=False)
def run_risk_assessment(
    asset_name: str, price: float, sma_alignment: str,
    rsi: float, volume_ratio: float, vix_value: float | None,
    atr: float, atr_ratio: float,
    near_resistance: bool, near_support: bool,
    sr_text: str, headlines_text: str,
) -> dict | None:
    """Dedicated risk analysis pass — finds all reasons NOT to trade."""
    prompt = RISK_ASSESSMENT_PROMPT.format(
        asset_name=asset_name,
        price=f"{price:,.2f}",
        sma_alignment=_interpret_sma_alignment(sma_alignment),
        rsi=f"{rsi:.1f}",
        volume_ratio=f"{volume_ratio:.1f}",
        vix_value=f"{vix_value:.1f}" if vix_value else "unavailable",
        atr=f"{atr:,.2f}",
        atr_ratio=f"{atr_ratio:.1f}",
        near_resistance=near_resistance,
        near_support=near_support,
        sr_text=sr_text,
        headlines_text=headlines_text,
    )
    raw = _call_specific_provider(prompt, "Gemini")
    if not raw:
        raw = _call_specific_provider(prompt, "Groq")
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


@st.cache_data(ttl=14400, show_spinner=False)
def run_macro_context(
    asset_name: str, asset_type: str,
    price: float, sma_alignment: str,
    vix_value: float | None, headlines_text: str,
) -> dict | None:
    """Macro/sector context analysis — bigger picture view."""
    prompt = MACRO_CONTEXT_PROMPT.format(
        asset_name=asset_name,
        asset_type=asset_type,
        price=f"{price:,.2f}",
        sma_alignment=_interpret_sma_alignment(sma_alignment),
        vix_value=f"{vix_value:.1f}" if vix_value else "unavailable",
        headlines_text=headlines_text,
    )
    raw = _call_specific_provider(prompt, "Gemini")
    if not raw:
        raw = _call_specific_provider(prompt, "Groq")
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


def dict_to_signal(d: dict) -> SentimentSignal:
    """Convert cached dict back to SentimentSignal."""
    details = [
        HeadlineSentiment(**h) if isinstance(h, dict) else h
        for h in d.pop("headline_details", [])
    ]
    return SentimentSignal(**d, headline_details=details)
