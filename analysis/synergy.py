from dataclasses import dataclass, field

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


def _compute_confidence(tech: TechnicalSignal, sent: SentimentSignal) -> float:
    """Composite confidence from RSI extremity, SMA distance, and sentiment strength."""
    # RSI extremity: how far past the 30/70 threshold (0-1 scale)
    if tech.rsi_value < 30:
        rsi_score = (30 - tech.rsi_value) / 30
    elif tech.rsi_value > 70:
        rsi_score = (tech.rsi_value - 70) / 30
    else:
        rsi_score = 0.0
    rsi_score = min(rsi_score, 1.0)

    # SMA distance: how far price is from SMA (capped at 10% = 1.0)
    sma_distance = abs(tech.current_price - tech.sma_value) / tech.sma_value
    sma_score = min(sma_distance / 0.10, 1.0)

    # Sentiment conviction
    sent_score = sent.confidence

    # Data coverage penalty
    coverage_ratio = sent.relevant_count / max(sent.total_count, 1)
    coverage_penalty = max(coverage_ratio, 0.3)

    raw = (rsi_score * 0.3 + sma_score * 0.2 + sent_score * 0.4 + coverage_penalty * 0.1)
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

    return factors


def decide(
    tech: TechnicalSignal,
    sent: SentimentSignal,
    week_52_low: float | None = None,
    week_52_high: float | None = None,
    is_crypto: bool = False,
) -> TradeDecision:
    """Run the Synergy Engine: strict AND gate + filters."""
    reasoning = []
    warnings = []
    action = "NONE"

    # --- Strict AND gate ---
    if tech.direction == "BULL" and sent.direction == "POSITIVE":
        action = "BULL"
        reasoning.append(f"RSI på {tech.rsi_value} indikerar översålt (< 30)")
        reasoning.append(f"Priset ({tech.current_price}) ligger över 200-dagars SMA ({tech.sma_value})")
        reasoning.append(f"AI-sentiment är positivt ({sent.confidence:.0%} konfidens)")
    elif tech.direction == "BEAR" and sent.direction == "NEGATIVE":
        action = "BEAR"
        reasoning.append(f"RSI på {tech.rsi_value} indikerar överköpt (> 70)")
        reasoning.append(f"Priset ({tech.current_price}) ligger under 200-dagars SMA ({tech.sma_value})")
        reasoning.append(f"AI-sentiment är negativt ({sent.confidence:.0%} konfidens)")
    else:
        reasoning.append(
            f"Teknisk signal: {tech.direction} | Sentiment: {sent.direction} — "
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

    # --- Data Quality Gate ---
    if sent.low_data_quality:
        if action != "NONE":
            warnings.append(
                f"Låg datakvalitet: Endast {sent.relevant_count} relevanta rubriker hittades — "
                f"sentimentet kan vara otillförlitligt"
            )
            action = "NONE"

    # --- Crypto warning ---
    if is_crypto:
        warnings.append(
            "Krypto-varning: Marknaden handlas 24/7 med högre volatilitet — "
            "signaler är mindre tillförlitliga"
        )

    # --- Compute confidence ---
    confidence = _compute_confidence(tech, sent) if action != "NONE" else 0.0

    # --- ATR-based stop-loss & take-profit ---
    atr = tech.atr_value
    if action == "BULL":
        stop_loss = round(tech.current_price - 2 * atr, 2)
        take_profit = round(tech.current_price + 3 * atr, 2)
    elif action == "BEAR":
        stop_loss = round(tech.current_price + 2 * atr, 2)
        take_profit = round(tech.current_price - 3 * atr, 2)
    else:
        stop_loss = 0.0
        take_profit = 0.0

    # --- Uncertainty factors ---
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
