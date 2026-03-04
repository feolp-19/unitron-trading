import json
import os
from dataclasses import dataclass, field

import streamlit as st

from config import AI_PROVIDER, AI_CONFIGS, get_secret


@dataclass
class HeadlineSentiment:
    headline: str
    sentiment: str       # positive, negative, neutral
    relevance: float     # 0.0-1.0
    reasoning: str


@dataclass
class SentimentSignal:
    direction: str                  # POSITIVE, NEGATIVE, NEUTRAL
    confidence: float               # 0.0-1.0
    relevant_count: int
    total_count: int
    low_data_quality: bool
    summary: str
    headline_details: list[HeadlineSentiment] = field(default_factory=list)
    herd_ratio: float = 0.0        # % of headlines with dominant sentiment


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


def _build_prompt(asset_name: str, headlines: list[dict]) -> str:
    headlines_text = "\n".join(
        f"{i+1}. {h['headline']}" for i, h in enumerate(headlines)
    )
    return SENTIMENT_PROMPT.format(
        asset_name=asset_name,
        headlines_text=headlines_text,
    )


def _call_groq(prompt: str) -> str | None:
    try:
        from groq import Groq
        cfg = AI_CONFIGS["groq"]
        client = Groq(api_key=get_secret(cfg["api_key_env"]))
        response = client.chat.completions.create(
            model=cfg["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.warning(f"Groq API-fel: {e}")
        return None


def _call_grok(prompt: str) -> str | None:
    try:
        from openai import OpenAI
        cfg = AI_CONFIGS["grok"]
        client = OpenAI(
            api_key=get_secret(cfg["api_key_env"]),
            base_url=cfg["base_url"],
        )
        response = client.chat.completions.create(
            model=cfg["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.warning(f"Grok API-fel: {e}")
        return None


def _call_gemini(prompt: str) -> str | None:
    try:
        from google import genai
        cfg = AI_CONFIGS["gemini"]
        client = genai.Client(api_key=get_secret(cfg["api_key_env"]))
        response = client.models.generate_content(
            model=cfg["model"],
            contents=prompt,
        )
        return response.text
    except Exception as e:
        st.warning(f"Gemini API-fel: {e}")
        return None


def _call_ai(prompt: str) -> str | None:
    providers = {
        "groq": _call_groq,
        "grok": _call_grok,
        "gemini": _call_gemini,
    }
    call_fn = providers.get(AI_PROVIDER, _call_groq)
    return call_fn(prompt)


def _keyword_fallback(headlines: list[dict]) -> SentimentSignal:
    """Basic keyword-based sentiment when AI is unavailable."""
    positive_words = {"surge", "rally", "gain", "rise", "up", "bull", "growth", "profit", "beat", "record", "high", "boost", "strong"}
    negative_words = {"crash", "fall", "drop", "decline", "loss", "bear", "recession", "fear", "crisis", "down", "weak", "miss", "cut", "sell"}

    pos_count = 0
    neg_count = 0
    details = []

    for h in headlines:
        text = h["headline"].lower()
        p = sum(1 for w in positive_words if w in text)
        n = sum(1 for w in negative_words if w in text)
        if p > n:
            sentiment = "positive"
            pos_count += 1
        elif n > p:
            sentiment = "negative"
            neg_count += 1
        else:
            sentiment = "neutral"
        details.append(HeadlineSentiment(
            headline=h["headline"],
            sentiment=sentiment,
            relevance=0.5,
            reasoning="Nyckelordsbaserad analys (AI ej tillgänglig)",
        ))

    total = len(headlines)
    if pos_count > neg_count:
        direction = "POSITIVE"
        confidence = pos_count / total if total else 0
    elif neg_count > pos_count:
        direction = "NEGATIVE"
        confidence = neg_count / total if total else 0
    else:
        direction = "NEUTRAL"
        confidence = 0.3

    dominant = max(pos_count, neg_count)
    herd_ratio = dominant / total if total else 0

    return SentimentSignal(
        direction=direction,
        confidence=round(confidence, 2),
        relevant_count=total,
        total_count=total,
        low_data_quality=total < 5,
        summary="Nyckelordsbaserad analys — AI-leverantör ej tillgänglig",
        headline_details=details,
        herd_ratio=round(herd_ratio, 2),
    )


def _parse_ai_response(raw: str, headlines: list[dict]) -> SentimentSignal | None:
    """Parse the JSON response from the AI model."""
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]

        data = json.loads(cleaned)

        details = []
        relevant_count = 0
        pos_count = 0
        neg_count = 0

        ai_headlines = data.get("headlines", [])
        for i, item in enumerate(ai_headlines):
            headline_text = headlines[i]["headline"] if i < len(headlines) else ""
            hs = HeadlineSentiment(
                headline=headline_text,
                sentiment=item.get("sentiment", "neutral"),
                relevance=float(item.get("relevance", 0.5)),
                reasoning=item.get("reasoning", ""),
            )
            details.append(hs)
            if hs.relevance > 0.5:
                relevant_count += 1
                if hs.sentiment == "positive":
                    pos_count += 1
                elif hs.sentiment == "negative":
                    neg_count += 1

        overall = data.get("overall_sentiment", "neutral").upper()
        if overall == "POSITIVE":
            direction = "POSITIVE"
        elif overall == "NEGATIVE":
            direction = "NEGATIVE"
        else:
            direction = "NEUTRAL"

        dominant = max(pos_count, neg_count)
        herd_ratio = dominant / relevant_count if relevant_count else 0

        return SentimentSignal(
            direction=direction,
            confidence=round(float(data.get("confidence", 0.5)), 2),
            relevant_count=relevant_count,
            total_count=len(headlines),
            low_data_quality=relevant_count < 5,
            summary=data.get("summary", ""),
            headline_details=details,
            herd_ratio=round(herd_ratio, 2),
        )
    except (json.JSONDecodeError, KeyError, IndexError):
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def analyze_sentiment(asset_name: str, headlines_json: str) -> dict:
    """Analyze sentiment. Accepts JSON string for caching compatibility.
    Returns dict representation of SentimentSignal."""
    headlines = json.loads(headlines_json)

    if not headlines:
        return SentimentSignal(
            direction="NEUTRAL",
            confidence=0.0,
            relevant_count=0,
            total_count=0,
            low_data_quality=True,
            summary="Inga nyhetsrubriker hittades",
            headline_details=[],
            herd_ratio=0.0,
        ).__dict__

    prompt = _build_prompt(asset_name, headlines)
    raw_response = _call_ai(prompt)

    if raw_response:
        result = _parse_ai_response(raw_response, headlines)
        if result:
            return result.__dict__

    fallback = _keyword_fallback(headlines)
    return fallback.__dict__


def dict_to_signal(d: dict) -> SentimentSignal:
    """Convert cached dict back to SentimentSignal."""
    details = [
        HeadlineSentiment(**h) if isinstance(h, dict) else h
        for h in d.pop("headline_details", [])
    ]
    return SentimentSignal(**d, headline_details=details)
