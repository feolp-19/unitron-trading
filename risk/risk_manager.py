from dataclasses import dataclass

from analysis.technical import TechnicalSignal
from analysis.sentiment import SentimentSignal


@dataclass
class RiskAssessment:
    bias_warnings: list[str]
    exit_plan: dict


def assess_risks(
    tech: TechnicalSignal,
    sent: SentimentSignal,
    action: str,
    is_crypto: bool = False,
) -> RiskAssessment:
    bias_warnings = []

    # Herd mentality: >90% of relevant headlines same direction
    if sent.herd_ratio > 0.9 and sent.relevant_count >= 5:
        bias_warnings.append(
            "Flockmentalitet: Över 90% av rubrikerna pekar åt samma håll. "
            "Var försiktig — marknaden kan redan ha prissatt detta."
        )

    # Recency bias: RSI moved sharply in 2 days
    if abs(tech.rsi_trend_2d) > 15:
        direction_text = "uppåt" if tech.rsi_trend_2d > 0 else "nedåt"
        bias_warnings.append(
            f"Recency Bias: RSI har rört sig {abs(tech.rsi_trend_2d):.1f} punkter {direction_text} "
            f"på 2 dagar. Undvik att fatta beslut baserat på kortsiktiga rörelser."
        )

    # VIX-based warning
    if tech.vix_value and tech.vix_value > 25 and action != "NONE":
        bias_warnings.append(
            f"VIX-varning: VIX på {tech.vix_value:.1f} indikerar förhöjd marknadsrädsla. "
            f"Överväg mindre positionsstorlek och tightare stop-loss."
        )

    # Low volume warning
    if tech.volume_ratio < 0.5 and action != "NONE":
        bias_warnings.append(
            f"Låg volym ({tech.volume_ratio:.1f}x genomsnitt): "
            f"Svag övertygelse bakom rörelsen — ökad risk för falskt utbrott."
        )

    # Multi-timeframe conflict
    if (tech.price_vs_sma != tech.price_vs_weekly_sma
            and tech.price_vs_weekly_sma != "unavailable"
            and action != "NONE"):
        bias_warnings.append(
            "Tidsramskonflikt: Daglig och veckovis trend pekar åt olika håll. "
            "Kortare hållperiod rekommenderas."
        )

    # Loss aversion reminder
    if action != "NONE":
        bias_warnings.append(
            "Förlustaversion: Håll dig till den förutbestämda exit-planen. "
            "Flytta inte stop-loss i hopp om återhämtning."
        )

    # Crypto-specific
    if is_crypto:
        bias_warnings.append(
            "Kryptomarknaden handlas 24/7. Prisrörelser kan ske medan du sover — "
            "överväg att använda automatiska stop-loss-ordrar."
        )

    # Exit plan
    atr = tech.atr_value
    if action == "BULL":
        exit_plan = {
            "entry": tech.current_price,
            "stop_loss": round(tech.current_price - 2 * atr, 2),
            "take_profit": round(tech.current_price + 3 * atr, 2),
            "risk_reward": "1:1.5",
            "strategy": (
                "Sälj om priset når take-profit eller stop-loss. "
                "Flytta INTE stop-loss nedåt."
            ),
        }
    elif action == "BEAR":
        exit_plan = {
            "entry": tech.current_price,
            "stop_loss": round(tech.current_price + 2 * atr, 2),
            "take_profit": round(tech.current_price - 3 * atr, 2),
            "risk_reward": "1:1.5",
            "strategy": (
                "Sälj om priset når take-profit eller stop-loss. "
                "Flytta INTE stop-loss uppåt."
            ),
        }
    else:
        exit_plan = {
            "entry": None,
            "stop_loss": None,
            "take_profit": None,
            "risk_reward": "-",
            "strategy": "Ingen handel rekommenderas idag.",
        }

    return RiskAssessment(
        bias_warnings=bias_warnings,
        exit_plan=exit_plan,
    )
