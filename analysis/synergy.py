"""Synergy Engine — fallback decision logic when AI analysis fails.
Uses trend-following with RSI guard, S/R awareness, and bias filters."""

from dataclasses import dataclass

from analysis.technical import TechnicalSignal
from analysis.sentiment import SentimentSignal


@dataclass
class TradeDecision:
    action: str                     # BULL, BEAR, NONE
    confidence_score: float         # 0.0-1.0
    current_price: float
    stop_loss_price: float
    take_profit_price: float
    reasoning: list[str]
    warnings: list[str]
    uncertainty_factors: list[str]


def _derive_tech_direction(tech: TechnicalSignal) -> str:
    if tech.price_vs_sma == "above" and tech.rsi_value < 65:
        return "BULL"
    if tech.price_vs_sma == "below" and tech.rsi_value > 35:
        return "BEAR"
    return "NEUTRAL"


def _compute_confidence(tech: TechnicalSignal, sent: SentimentSignal) -> float:
    sma_distance = abs(tech.current_price - tech.sma_200) / tech.sma_200
    sma_score = min(sma_distance / 0.10, 1.0)

    sent_score = sent.confidence

    coverage_ratio = sent.relevant_count / max(sent.total_count, 1)
    coverage_penalty = max(coverage_ratio, 0.3)

    volume_bonus = 0.1 if tech.volume_ratio > 1.5 else 0.0
    timeframe_bonus = 0.1 if tech.price_vs_sma == tech.price_vs_weekly_sma else 0.0
    alignment_bonus = 0.1 if tech.sma_alignment in ("bullish_stack", "bearish_stack") else 0.0

    raw = (sma_score * 0.2 + sent_score * 0.3 + coverage_penalty * 0.1
           + volume_bonus + timeframe_bonus + alignment_bonus)
    return round(min(raw, 1.0), 2)


def _compute_uncertainty_factors(tech: TechnicalSignal, sent: SentimentSignal) -> list[str]:
    factors = []

    if 25 <= tech.rsi_value <= 35 or 65 <= tech.rsi_value <= 75:
        factors.append(
            f"RSI ligger nära gränszonen ({tech.rsi_value:.1f}) — "
            f"signalen kan snabbt ändras"
        )

    if tech.price_vs_sma == "at":
        factors.append(
            "Priset ligger nära 200-dagars SMA — "
            "potentiell korsning kan ändra riktningen"
        )

    if sent.relevant_count < 10:
        factors.append(
            f"Tunn rubrikstäckning ({sent.relevant_count} av {sent.total_count} "
            f"rubriker relevanta) — sentimentet kan vara opålitligt"
        )

    if 0.4 <= sent.herd_ratio <= 0.6:
        factors.append(
            "Blandad sentiment — rubrikerna pekar åt olika håll"
        )

    if tech.atr_ratio > 1.5:
        factors.append(
            f"Hög volatilitet (ATR {tech.atr_ratio:.1f}x över genomsnittet) — "
            f"ökad risk för falska signaler"
        )

    if tech.vix_value and tech.vix_value > 25:
        factors.append(
            f"VIX på {tech.vix_value:.1f} — marknadens rädsla är förhöjd"
        )

    if tech.price_vs_sma != tech.price_vs_weekly_sma and tech.price_vs_weekly_sma != "unavailable":
        factors.append(
            "Daglig och veckovis trend pekar åt olika håll — svagare signal"
        )

    if tech.sma_alignment == "mixed":
        factors.append(
            "SMA-linjering saknas (20/50/200 inte i ordning) — trendstyrkan är osäker"
        )

    return factors


def decide(
    tech: TechnicalSignal,
    sent: SentimentSignal,
    week_52_low: float | None = None,
    week_52_high: float | None = None,
    is_crypto: bool = False,
) -> TradeDecision:
    """Fallback rule-based Synergy Engine when AI is unavailable."""
    reasoning = []
    warnings = []
    action = "NONE"

    tech_direction = _derive_tech_direction(tech)

    # --- Trend-following AND gate ---
    if tech_direction == "BULL" and sent.direction == "POSITIVE":
        action = "BULL"
        reasoning.append(f"Priset ({tech.current_price:,.2f}) ligger över SMA ({tech.sma_200:,.2f})")
        reasoning.append(f"RSI på {tech.rsi_value:.1f} — trend uppåt utan överköpt")
        reasoning.append(f"AI-sentiment är positivt ({sent.confidence:.0%} konfidens)")
        if tech.sma_alignment == "bullish_stack":
            reasoning.append("SMA-linjering: Pris > SMA20 > SMA50 > SMA200 (stark upptrend)")
    elif tech_direction == "BEAR" and sent.direction == "NEGATIVE":
        action = "BEAR"
        reasoning.append(f"Priset ({tech.current_price:,.2f}) ligger under SMA ({tech.sma_200:,.2f})")
        reasoning.append(f"RSI på {tech.rsi_value:.1f} — trend nedåt utan översålt")
        reasoning.append(f"AI-sentiment är negativt ({sent.confidence:.0%} konfidens)")
        if tech.sma_alignment == "bearish_stack":
            reasoning.append("SMA-linjering: Pris < SMA20 < SMA50 < SMA200 (stark nedtrend)")
    else:
        reasoning.append(
            f"Teknisk signal: {tech_direction} | Sentiment: {sent.direction} — "
            f"signalerna är inte synkroniserade"
        )

    # --- Anti-Value Trap Filter ---
    if action == "BULL" and week_52_low is not None:
        proximity_to_low = (tech.current_price - week_52_low) / week_52_low
        if proximity_to_low < 0.05 and sent.direction == "NEGATIVE":
            action = "NONE"
            warnings.append(
                "Värdefälla-varning: Priset ligger nära 52-veckorslägsta "
                "men sentimentet är negativt — undvik köp"
            )

    # --- Anchoring Bias: near resistance ---
    if action == "BULL" and tech.near_resistance:
        sr = tech.support_resistance
        res_val = sr.resistances[0] if sr.resistances else 0
        warnings.append(
            f"Ankrings-bias: Priset ligger nära motstånd ({res_val:,.2f}). "
            f"Risk att priset vänder — överväg att vänta på bekräftat utbrott."
        )

    # --- Anchoring Bias: near support on BEAR ---
    if action == "BEAR" and tech.near_support:
        sr = tech.support_resistance
        sup_val = sr.supports[0] if sr.supports else 0
        warnings.append(
            f"Ankrings-bias: Priset ligger nära stöd ({sup_val:,.2f}). "
            f"Risk att priset studsar — överväg att vänta på bekräftat genombrott."
        )

    # --- Data Quality Warning ---
    if sent.low_data_quality and action != "NONE":
        warnings.append(
            f"Låg datakvalitet: Endast {sent.relevant_count} relevanta rubriker hittades — "
            f"sentimentet kan vara opålitligt, men signalen bibehålls"
        )

    # --- Crypto warning ---
    if is_crypto:
        warnings.append(
            "Krypto-varning: Marknaden handlas 24/7 med högre volatilitet — "
            "signaler är mindre tillförlitliga"
        )

    confidence = _compute_confidence(tech, sent) if action != "NONE" else 0.0

    # --- ATR-based stop-loss & take-profit ---
    atr = tech.atr_value
    sr = tech.support_resistance
    if action == "BULL":
        if sr.supports:
            stop_loss = round(sr.supports[0] - 0.5 * atr, 2)
        else:
            stop_loss = round(tech.current_price - 2 * atr, 2)
        if sr.resistances:
            take_profit = round(sr.resistances[0], 2)
        else:
            take_profit = round(tech.current_price + 3 * atr, 2)
    elif action == "BEAR":
        if sr.resistances:
            stop_loss = round(sr.resistances[0] + 0.5 * atr, 2)
        else:
            stop_loss = round(tech.current_price + 2 * atr, 2)
        if sr.supports:
            take_profit = round(sr.supports[0], 2)
        else:
            take_profit = round(tech.current_price - 3 * atr, 2)
    else:
        stop_loss = 0.0
        take_profit = 0.0

    uncertainty_factors = _compute_uncertainty_factors(tech, sent)

    return TradeDecision(
        action=action,
        confidence_score=confidence,
        current_price=tech.current_price,
        stop_loss_price=stop_loss,
        take_profit_price=take_profit,
        reasoning=reasoning,
        warnings=warnings,
        uncertainty_factors=uncertainty_factors,
    )
