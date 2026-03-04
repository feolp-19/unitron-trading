"""Stage 2 Verification: second AI opinion, devil's advocate, and deeper news search.

Only runs on candidates that passed Stage 1 quality gates.
Uses a DIFFERENT AI provider than Stage 1 to ensure independent opinions."""

import json
from dataclasses import dataclass, field

import streamlit as st

from config import get_secret, AI_CONFIGS
from data.news_data import fetch_tavily


SECOND_OPINION_PROMPT = """You are an independent financial analyst reviewing a trade recommendation.
Another analyst has recommended {verdict} on {asset_name}. You must INDEPENDENTLY decide if you agree.

Do NOT blindly agree. Analyze the data yourself and reach your own conclusion.
Your default should be skepticism — prove the trade is worth taking.

=== MARKET DATA ===
- Current price: {price}
- SMA 20/50/200: {sma_20} / {sma_50} / {sma_200} (alignment: {sma_alignment})
- 50-week SMA: {sma_50w} (price is {price_vs_weekly_sma} weekly trend)
- RSI (14): {rsi}
- ATR (14): {atr}
- Volume vs average: {volume_ratio}x
- VIX: {vix_value}

=== SUPPORT & RESISTANCE ===
{sr_text}

=== NEWS ===
{headlines_text}

=== ADDITIONAL RISK RESEARCH ===
{risk_news_text}

Do you AGREE with the {verdict} recommendation? Analyze independently.

Return ONLY valid JSON:
{{
  "agree": true or false,
  "verdict": "BUY_BULL" or "BUY_BEAR" or "NO_TRADE",
  "confidence": 0.0 to 1.0,
  "reasoning": "2-3 sentences explaining your independent assessment",
  "disagreement_points": ["point 1", "point 2"] or [] if you agree
}}"""


DEVILS_ADVOCATE_PROMPT = """You are a risk analyst whose ONLY job is to find reasons NOT to take this trade.
Be aggressive in finding problems. Think like someone whose money is on the line.

A trade recommendation has been made: {verdict} on {asset_name} at {price}.
Stop-loss: {stop_loss}, Take-profit: {take_profit}

=== MARKET DATA ===
- SMA alignment: {sma_alignment}
- RSI: {rsi}
- Volume: {volume_ratio}x average
- VIX: {vix_value}
- ATR: {atr}

=== SUPPORT & RESISTANCE ===
{sr_text}

=== NEWS (original scan) ===
{headlines_text}

=== ADDITIONAL RISK NEWS ===
{risk_news_text}

Find every reason this trade could FAIL. Consider:
1. Is the trend exhausted? (How long has this trend been running? Late entry risk?)
2. Are there hidden risks in the news? (Upcoming events, policy changes, earnings?)
3. Is the stop-loss realistic given volatility?
4. Is the market too crowded on this side? (Everyone bullish = contrarian risk)
5. Are there macro headwinds? (USD strength, interest rates, geopolitics?)
6. Is the R/R actually good enough after fees and slippage?

Return ONLY valid JSON:
{{
  "risk_level": "LOW" or "MEDIUM" or "HIGH" or "CRITICAL",
  "should_proceed": true or false,
  "counter_arguments": ["argument 1", "argument 2", "argument 3"],
  "biggest_risk": "1 sentence: the single most dangerous thing about this trade",
  "recommendation": "1 sentence: proceed, reduce size, or avoid entirely"
}}"""


@dataclass
class VerificationResult:
    consensus: bool
    second_ai_agrees: bool
    second_ai_verdict: str
    second_ai_confidence: float
    second_ai_reasoning: str
    second_ai_provider: str
    disagreement_points: list[str]
    devils_advocate_risk: str
    devils_advocate_proceed: bool
    counter_arguments: list[str]
    biggest_risk: str
    devils_advocate_recommendation: str
    risk_headlines: list[dict] = field(default_factory=list)
    verified: bool = False


def _pick_second_provider(first_provider: str) -> tuple:
    """Select a different AI provider for independent second opinion."""
    from analysis.sentiment import _call_groq, _call_gemini, _call_grok

    providers = {
        "Groq": ("Gemini", _call_gemini),
        "Gemini": ("Groq", _call_groq),
        "Grok": ("Gemini", _call_gemini),
    }

    if first_provider in providers:
        name, fn = providers[first_provider]
        api_key_env = AI_CONFIGS[name.lower()]["api_key_env"]
        if get_secret(api_key_env):
            return name, fn

    for name, fn in [("Groq", _call_groq), ("Gemini", _call_gemini), ("Grok", _call_grok)]:
        if name != first_provider:
            api_key_env = AI_CONFIGS[name.lower()]["api_key_env"]
            if get_secret(api_key_env):
                return name, fn

    return None, None


def _fetch_risk_news(asset_name: str, keywords: list[str]) -> list[dict]:
    """Search for negative/risk news about the asset."""
    risk_query = f"{asset_name} risk warning concern decline drop"
    return fetch_tavily(risk_query, max_results=5)


def _parse_json_response(raw: str) -> dict | None:
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


def verify_recommendation(
    asset_name: str,
    first_provider: str,
    verdict: str,
    confidence: float,
    tech,
    sr_text: str,
    headlines_text: str,
    stop_loss: float,
    take_profit: float,
    news_keywords: list[str],
) -> VerificationResult | None:
    """Run Stage 2 verification on a Stage 1 candidate.

    1. Fetch additional risk-focused news
    2. Get independent second AI opinion
    3. Run devil's advocate analysis
    4. Return consensus result
    """
    risk_headlines = _fetch_risk_news(asset_name, news_keywords)
    risk_news_text = "\n".join(
        f"- {h['headline']}" for h in risk_headlines
    ) if risk_headlines else "No additional risk news found."

    second_name, second_fn = _pick_second_provider(first_provider)
    if not second_fn:
        return None

    second_prompt = SECOND_OPINION_PROMPT.format(
        verdict=verdict,
        asset_name=asset_name,
        price=f"{tech.current_price:,.2f}",
        sma_20=f"{tech.sma_20:,.2f}",
        sma_50=f"{tech.sma_50:,.2f}",
        sma_200=f"{tech.sma_200:,.2f}",
        sma_alignment=tech.sma_alignment,
        sma_50w=f"{tech.sma_50w:,.2f}" if tech.sma_50w else "N/A",
        price_vs_weekly_sma=tech.price_vs_weekly_sma,
        rsi=f"{tech.rsi_value:.1f}",
        atr=f"{tech.atr_value:,.2f}",
        volume_ratio=f"{tech.volume_ratio:.1f}",
        vix_value=f"{tech.vix_value:.1f}" if tech.vix_value else "N/A",
        sr_text=sr_text,
        headlines_text=headlines_text,
        risk_news_text=risk_news_text,
    )

    second_raw = second_fn(second_prompt)
    second_data = _parse_json_response(second_raw) if second_raw else None

    # If second provider failed, try all other providers
    if not second_data:
        from analysis.sentiment import _call_groq, _call_gemini, _call_grok
        fallback_providers = [
            ("Groq", _call_groq), ("Gemini", _call_gemini), ("Grok", _call_grok),
        ]
        for fb_name, fb_fn in fallback_providers:
            if fb_name != first_provider and fb_name != second_name:
                fb_raw = fb_fn(second_prompt)
                if fb_raw:
                    second_data = _parse_json_response(fb_raw)
                    if second_data:
                        second_name = fb_name
                        break

    if not second_data:
        return None

    second_agrees = second_data.get("agree", False)
    second_verdict = second_data.get("verdict", "NO_TRADE")

    da_prompt = DEVILS_ADVOCATE_PROMPT.format(
        verdict=verdict,
        asset_name=asset_name,
        price=f"{tech.current_price:,.2f}",
        stop_loss=f"{stop_loss:,.2f}",
        take_profit=f"{take_profit:,.2f}",
        sma_alignment=tech.sma_alignment,
        rsi=f"{tech.rsi_value:.1f}",
        volume_ratio=f"{tech.volume_ratio:.1f}",
        vix_value=f"{tech.vix_value:.1f}" if tech.vix_value else "N/A",
        atr=f"{tech.atr_value:,.2f}",
        sr_text=sr_text,
        headlines_text=headlines_text,
        risk_news_text=risk_news_text,
    )

    da_fn = second_fn
    da_raw = da_fn(da_prompt)
    da_data = _parse_json_response(da_raw) if da_raw else None

    if not da_data:
        da_data = {
            "risk_level": "UNKNOWN",
            "should_proceed": True,
            "counter_arguments": [],
            "biggest_risk": "Kunde inte köra djupanalys",
            "recommendation": "Fortsätt med försiktighet",
        }

    da_proceed = da_data.get("should_proceed", True)
    consensus = second_agrees and da_proceed

    return VerificationResult(
        consensus=consensus,
        second_ai_agrees=second_agrees,
        second_ai_verdict=second_verdict,
        second_ai_confidence=float(second_data.get("confidence", 0)),
        second_ai_reasoning=second_data.get("reasoning", ""),
        second_ai_provider=second_name,
        disagreement_points=second_data.get("disagreement_points", []),
        devils_advocate_risk=da_data.get("risk_level", "UNKNOWN"),
        devils_advocate_proceed=da_proceed,
        counter_arguments=da_data.get("counter_arguments", []),
        biggest_risk=da_data.get("biggest_risk", ""),
        devils_advocate_recommendation=da_data.get("recommendation", ""),
        risk_headlines=risk_headlines,
        verified=consensus,
    )
