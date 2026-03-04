"""AI sentiment analysis with automatic provider fallback chain:
Groq -> Gemini -> Grok -> keyword fallback.

Tier 1 upgrade: AI is now the sole decision maker via FULL_ANALYSIS_PROMPT,
receiving ALL data (technicals, VIX, volume, weekly SMA, news)."""

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

=== PRICE & TREND ===
- Current price: {price}
- 200-day SMA: {sma_200} → price is {price_vs_sma} the 200-day SMA
- 50-week SMA: {sma_50w} → price is {price_vs_weekly_sma} the weekly trend
- Multi-timeframe alignment: {timeframe_alignment}

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
1. TREND: If price is above BOTH daily and weekly SMAs, the trend favors BULL. If below both, favors BEAR. Mixed = weaker signal.
2. MOMENTUM: RSI 30-70 is normal. Below 30 = oversold (potential bounce up). Above 70 = overbought (potential pullback).
3. VOLUME: Volume > 1.5x average confirms the current move. Low volume means weak conviction.
4. VIX: High VIX (>25) means fear/uncertainty — be more cautious. Low VIX (<15) means complacency — watch for reversals.
5. NEWS: Headlines must support the technical direction. If news contradicts technicals, reduce confidence or NO_TRADE.
6. CONFIDENCE: Only give confidence > 0.7 if MULTIPLE factors align (trend + momentum + volume + news).

Return ONLY valid JSON:
{{
  "verdict": "BUY_BULL" or "BUY_BEAR" or "NO_TRADE",
  "confidence": 0.0 to 1.0,
  "analysis": "2-4 sentences: explain WHY you made this decision, referencing specific data points",
  "key_factors": ["factor 1 that drove the decision", "factor 2", "factor 3"],
  "risks": ["specific risk 1", "specific risk 2"],
  "stop_loss_reasoning": "1 sentence: where to place stop-loss and why",
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
    """Try AI providers in order: Groq -> Gemini -> Grok. Returns (response, provider_name)."""
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
    """Basic keyword-based sentiment when all AI providers fail."""
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
        "low_fear": f"{vix:.1f} — Low fear/complacency. Markets calm, watch for sudden spikes.",
        "normal": f"{vix:.1f} — Normal market conditions.",
        "elevated": f"{vix:.1f} — Elevated fear. Markets uncertain, use tighter stops.",
        "extreme_fear": f"{vix:.1f} — EXTREME FEAR. Very high risk environment, be extra cautious.",
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


@st.cache_data(ttl=86400, show_spinner=False)
def run_full_analysis(
    asset_name: str,
    price: float,
    sma_200: float,
    price_vs_sma: str,
    sma_50w: float | None,
    price_vs_weekly_sma: str,
    rsi: float,
    atr: float,
    rsi_trend: float,
    atr_ratio: float,
    volume_ratio: float,
    vix_value: float | None,
    vix_level: str,
    headlines_json: str,
) -> dict | None:
    """Run comprehensive AI analysis — this is the primary decision maker."""
    headlines = json.loads(headlines_json)
    headlines_text = "\n".join(
        f"{i+1}. {h['headline']}" for i, h in enumerate(headlines)
    ) if headlines else "No recent headlines available."

    # Multi-timeframe alignment
    if price_vs_sma == price_vs_weekly_sma and price_vs_sma != "at":
        timeframe_alignment = f"ALIGNED — price is {price_vs_sma} both daily and weekly trends (strong signal)"
    elif price_vs_weekly_sma == "unavailable":
        timeframe_alignment = "Weekly data unavailable — rely on daily trend only"
    elif price_vs_sma == "at" or price_vs_weekly_sma == "at":
        timeframe_alignment = "One timeframe is neutral — signal is moderate"
    else:
        timeframe_alignment = f"CONFLICTING — daily says {price_vs_sma}, weekly says {price_vs_weekly_sma} (weaker signal)"

    prompt = FULL_ANALYSIS_PROMPT.format(
        asset_name=asset_name,
        price=f"{price:,.2f}",
        sma_200=f"{sma_200:,.2f}",
        price_vs_sma=price_vs_sma,
        sma_50w=f"{sma_50w:,.2f}" if sma_50w else "unavailable",
        price_vs_weekly_sma=price_vs_weekly_sma if price_vs_weekly_sma != "unavailable" else "N/A (data unavailable)",
        timeframe_alignment=timeframe_alignment,
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
