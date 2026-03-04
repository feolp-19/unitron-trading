"""AI sentiment analysis with automatic provider fallback chain:
Groq -> Gemini -> keyword fallback."""

import json
import os
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


FULL_ANALYSIS_PROMPT = """You are an expert financial analyst. Provide a comprehensive trading analysis for {asset_name}.

TECHNICAL DATA:
- Current price: {price}
- 200-day SMA: {sma} (price is {price_vs_sma} the SMA)
- RSI (14): {rsi}
- ATR (14): {atr}
- RSI 2-day trend: {rsi_trend}
- Volatility ratio: {atr_ratio}x normal

NEWS HEADLINES:
{headlines_text}

Based on ALL of this data, provide your analysis in this JSON format:
{{
  "verdict": "BUY_BULL" or "BUY_BEAR" or "NO_TRADE",
  "confidence": 0.0 to 1.0,
  "analysis": "2-3 sentences explaining your reasoning, combining technicals and news",
  "risks": ["risk 1", "risk 2"],
  "outlook": "1 sentence on what to watch next"
}}

Return ONLY valid JSON."""


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


@st.cache_data(ttl=86400, show_spinner=False)
def run_full_analysis(asset_name: str, price: float, sma: float, price_vs_sma: str,
                       rsi: float, atr: float, rsi_trend: float, atr_ratio: float,
                       headlines_json: str) -> dict | None:
    """Run comprehensive AI analysis combining technicals and news."""
    headlines = json.loads(headlines_json)
    headlines_text = "\n".join(
        f"{i+1}. {h['headline']}" for i, h in enumerate(headlines)
    ) if headlines else "No recent headlines available."

    prompt = FULL_ANALYSIS_PROMPT.format(
        asset_name=asset_name,
        price=f"{price:,.2f}",
        sma=f"{sma:,.2f}",
        price_vs_sma=price_vs_sma,
        rsi=f"{rsi:.1f}",
        atr=f"{atr:,.2f}",
        rsi_trend=f"{rsi_trend:+.1f}",
        atr_ratio=f"{atr_ratio:.1f}",
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
