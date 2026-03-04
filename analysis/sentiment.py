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


FULL_ANALYSIS_PROMPT = """You are an expert financial analyst and professional trader. You must decide whether to BUY a BULL certificate (betting price goes UP), BUY a BEAR certificate (betting price goes DOWN), or NO TRADE.

Your job is to synthesize ALL the data below and make ONE clear trading decision. Be decisive — only say NO_TRADE if the data is truly conflicting or insufficient.

=== ASSET ===
{asset_name}

=== PRICE & TREND (SMAs) ===
- Current price: {price}
- 20-day SMA: {sma_20} (short-term trend)
- 50-day SMA: {sma_50} (medium-term trend)
- 200-day SMA: {sma_200} → price is {price_vs_sma} the 200-day SMA
- 50-week SMA: {sma_50w} → price is {price_vs_weekly_sma} the weekly trend
- SMA alignment: {sma_alignment}
- Multi-timeframe: {timeframe_alignment}

=== SUPPORT & RESISTANCE ===
{sr_text}

=== MOMENTUM ===
- RSI (14): {rsi} {rsi_interpretation}
- RSI 2-day change: {rsi_trend} (momentum is {rsi_momentum})

=== VOLATILITY ===
- ATR (14): {atr}
- Volatility ratio: {atr_ratio}x vs 30-day average ({volatility_interpretation})

=== VOLUME ===
- Current volume vs 20-day average: {volume_ratio}x ({volume_interpretation})

=== FEAR & GREED (VIX) ===
- VIX: {vix_value} — {vix_interpretation}

=== NEWS & MACRO ===
{headlines_text}

=== DECISION RULES ===
1. SMA ALIGNMENT: If Price > SMA20 > SMA50 > SMA200 = "bullish stack" (strong BULL). If reversed = "bearish stack" (strong BEAR). Mixed = weaker signal.
2. SUPPORT/RESISTANCE: If price is near a major resistance, be cautious about BULL entries (risk of rejection). If near support, be cautious about BEAR entries. USE these levels for stop-loss and take-profit reasoning.
3. MOMENTUM: RSI 30-70 is normal. Below 30 = oversold (potential bounce). Above 70 = overbought (potential pullback).
4. VOLUME: Volume > 1.5x average confirms the current move. Low volume means weak conviction.
5. VIX: High VIX (>25) = fear/uncertainty, be cautious. Low VIX (<15) = complacency.
6. NEWS: Headlines must support the technical direction. Contradicting news weakens the signal.

=== CONFIDENCE SCORING RUBRIC (you MUST follow this) ===
Start at 0.50 and adjust:
+0.10 if SMA alignment is bullish_stack (for BULL) or bearish_stack (for BEAR)
+0.05 if SMA alignment is mixed but price is on the right side of SMA200
+0.10 if daily AND weekly timeframes are aligned (same direction)
-0.10 if daily and weekly timeframes CONFLICT
+0.05 if RSI supports the direction (RSI < 50 for BEAR, RSI > 50 for BULL)
+0.05 if RSI is in extreme zone favoring the trade (< 30 for bounce BULL, > 70 for reversal BEAR)
-0.10 if RSI contradicts (e.g. RSI > 65 for BULL = overbought risk)
+0.10 if volume > 1.5x average (strong conviction)
-0.05 if volume < 0.7x average (weak conviction)
+0.10 if news sentiment clearly supports the direction
-0.10 if news sentiment contradicts the direction
+0.05 if no S/R obstacle within 3% of current price in the trade direction
-0.10 if price is within 2% of a major resistance (for BULL) or support (for BEAR)
-0.05 if VIX > 25 (elevated fear)
-0.10 if VIX > 30 (extreme fear)

Final confidence MUST be between 0.30 and 0.95. Round to 2 decimals.
Each asset WILL score differently — do NOT default to the same number.
Show your math in the analysis field (e.g. "Base 0.50 + SMA aligned +0.10 + volume confirms +0.10 - near resistance -0.10 = 0.60").

Return ONLY valid JSON:
{{
  "verdict": "BUY_BULL" or "BUY_BEAR" or "NO_TRADE",
  "confidence": 0.0 to 1.0,
  "analysis": "2-4 sentences: explain WHY, referencing specific data points including S/R levels",
  "key_factors": ["factor 1", "factor 2", "factor 3"],
  "risks": ["specific risk 1", "specific risk 2"],
  "stop_loss_reasoning": "1 sentence: where to place stop-loss and why (reference S/R or ATR)",
  "take_profit_reasoning": "1 sentence: where to take profit and why (reference S/R levels)",
  "outlook": "1 sentence: what event or level to watch next"
}}"""


def _call_groq(prompt: str) -> str | None:
    try:
        from groq import Groq
        api_key = get_secret("GROQ_API_KEY")
        if not api_key:
            return None
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=AI_CONFIGS["groq"]["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000,
        )
        return response.choices[0].message.content
    except Exception:
        return None


def _call_gemini(prompt: str) -> str | None:
    try:
        from google import genai
        api_key = get_secret("GOOGLE_API_KEY")
        if not api_key:
            return None
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=AI_CONFIGS["gemini"]["model"],
            contents=prompt,
        )
        return response.text
    except Exception:
        return None


def _call_grok(prompt: str) -> str | None:
    try:
        from openai import OpenAI
        api_key = get_secret("XAI_API_KEY")
        if not api_key:
            return None
        client = OpenAI(api_key=api_key, base_url=AI_CONFIGS["grok"]["base_url"])
        response = client.chat.completions.create(
            model=AI_CONFIGS["grok"]["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000,
        )
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


@st.cache_data(ttl=86400, show_spinner=False)
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


@st.cache_data(ttl=86400, show_spinner=False)
def run_full_analysis(
    asset_name: str,
    price: float,
    sma_20: float,
    sma_50: float,
    sma_200: float,
    price_vs_sma: str,
    sma_50w: float | None,
    price_vs_weekly_sma: str,
    sma_alignment: str,
    rsi: float,
    atr: float,
    rsi_trend: float,
    atr_ratio: float,
    volume_ratio: float,
    vix_value: float | None,
    vix_level: str,
    supports_json: str,
    resistances_json: str,
    near_resistance: bool,
    near_support: bool,
    headlines_json: str,
) -> dict | None:
    """Run comprehensive AI analysis — the primary decision maker."""
    headlines = json.loads(headlines_json)
    supports = json.loads(supports_json)
    resistances = json.loads(resistances_json)
    headlines_text = "\n".join(
        f"{i+1}. {h['headline']}" for i, h in enumerate(headlines)
    ) if headlines else "No recent headlines available."

    # Multi-timeframe alignment
    if price_vs_sma == price_vs_weekly_sma and price_vs_sma != "at":
        timeframe_alignment = f"ALIGNED — price is {price_vs_sma} both daily and weekly trends (strong)"
    elif price_vs_weekly_sma == "unavailable":
        timeframe_alignment = "Weekly data unavailable — rely on daily trend only"
    elif price_vs_sma == "at" or price_vs_weekly_sma == "at":
        timeframe_alignment = "One timeframe is neutral — signal is moderate"
    else:
        timeframe_alignment = f"CONFLICTING — daily: {price_vs_sma}, weekly: {price_vs_weekly_sma} (weak)"

    sr_text = _format_sr_text(supports, resistances, price, near_resistance, near_support)

    prompt = FULL_ANALYSIS_PROMPT.format(
        asset_name=asset_name,
        price=f"{price:,.2f}",
        sma_20=f"{sma_20:,.2f}",
        sma_50=f"{sma_50:,.2f}",
        sma_200=f"{sma_200:,.2f}",
        price_vs_sma=price_vs_sma,
        sma_50w=f"{sma_50w:,.2f}" if sma_50w else "unavailable",
        price_vs_weekly_sma=price_vs_weekly_sma if price_vs_weekly_sma != "unavailable" else "N/A",
        sma_alignment=_interpret_sma_alignment(sma_alignment),
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
    )

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


def dict_to_signal(d: dict) -> SentimentSignal:
    """Convert cached dict back to SentimentSignal."""
    details = [
        HeadlineSentiment(**h) if isinstance(h, dict) else h
        for h in d.pop("headline_details", [])
    ]
    return SentimentSignal(**d, headline_details=details)
