from dataclasses import dataclass

from analysis.technical import TechnicalSignal
from analysis.sentiment import SentimentSignal
from analysis.exit_strategy import TradingPlan, generate_trading_plan


@dataclass
class RiskAssessment:
    bias_warnings: list[str]
    exit_plan: dict
    trading_plan: TradingPlan | None


def assess_risks(
    tech: TechnicalSignal,
    sent: SentimentSignal,
    action: str,
    is_crypto: bool = False,
) -> RiskAssessment:
    bias_warnings = []

    if sent.herd_ratio > 0.9 and sent.relevant_count >= 5:
        bias_warnings.append(
            "Flockmentalitet: Över 90% av rubrikerna pekar åt samma håll. "
            "Var försiktig — marknaden kan redan ha prissatt detta."
        )

    if abs(tech.rsi_trend_2d) > 15:
        direction_text = "uppåt" if tech.rsi_trend_2d > 0 else "nedåt"
        bias_warnings.append(
            f"Recency Bias: RSI har rört sig {abs(tech.rsi_trend_2d):.1f} punkter {direction_text} "
            f"på 2 dagar. Undvik att fatta beslut baserat på kortsiktiga rörelser."
        )

    if tech.vix_value and tech.vix_value > 25 and action != "NONE":
        bias_warnings.append(
            f"VIX-varning: VIX på {tech.vix_value:.1f} indikerar förhöjd marknadsrädsla. "
            f"Överväg mindre positionsstorlek och tightare stop-loss."
        )

    if tech.volume_ratio < 0.5 and action != "NONE":
        bias_warnings.append(
            f"Låg volym ({tech.volume_ratio:.1f}x genomsnitt): "
            f"Svag övertygelse bakom rörelsen — ökad risk för falskt utbrott."
        )

    if (tech.price_vs_sma != tech.price_vs_weekly_sma
            and tech.price_vs_weekly_sma != "unavailable"
            and action != "NONE"):
        bias_warnings.append(
            "Tidsramskonflikt: Daglig och veckovis trend pekar åt olika håll. "
            "Kortare hållperiod rekommenderas."
        )

    # Anchoring bias: near resistance on BULL
    if action == "BULL" and tech.near_resistance:
        sr = tech.support_resistance
        res_val = sr.resistances[0] if sr.resistances else 0
        bias_warnings.append(
            f"Ankrings-bias: Priset ({tech.current_price:,.2f}) är nära motstånd "
            f"({res_val:,.2f}). Undvik att ankra till gamla prisnivåer — "
            f"motstånd kan orsaka vändning."
        )

    # Anchoring bias: near support on BEAR
    if action == "BEAR" and tech.near_support:
        sr = tech.support_resistance
        sup_val = sr.supports[0] if sr.supports else 0
        bias_warnings.append(
            f"Ankrings-bias: Priset ({tech.current_price:,.2f}) är nära stöd "
            f"({sup_val:,.2f}). Stödnivåer kan orsaka studs uppåt."
        )

    if action != "NONE":
        bias_warnings.append(
            "Förlustaversion: Håll dig till den förutbestämda exit-planen. "
            "Flytta inte stop-loss i hopp om återhämtning."
        )

    if is_crypto:
        bias_warnings.append(
            "Kryptomarknaden handlas 24/7. Prisrörelser kan ske medan du sover — "
            "överväg att använda automatiska stop-loss-ordrar."
        )

    # Generate the full trading plan using the exit strategy engine
    trading_plan = generate_trading_plan(tech, action)

    # Build legacy exit_plan dict from trading plan or ATR fallback
    if trading_plan:
        exit_plan = {
            "entry": trading_plan.entry_price,
            "stop_loss": trading_plan.stop_loss,
            "take_profit": trading_plan.take_profit,
            "risk_reward": trading_plan.risk_reward_ratio,
            "strategy": trading_plan.trailing_stop_reasoning,
        }
    elif action == "BULL":
        atr = tech.atr_value
        exit_plan = {
            "entry": tech.current_price,
            "stop_loss": round(tech.current_price - 2 * atr, 2),
            "take_profit": round(tech.current_price + 3 * atr, 2),
            "risk_reward": "1:1.5",
            "strategy": "Sälj om priset når take-profit eller stop-loss.",
        }
    elif action == "BEAR":
        atr = tech.atr_value
        exit_plan = {
            "entry": tech.current_price,
            "stop_loss": round(tech.current_price + 2 * atr, 2),
            "take_profit": round(tech.current_price - 3 * atr, 2),
            "risk_reward": "1:1.5",
            "strategy": "Sälj om priset når take-profit eller stop-loss.",
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
        trading_plan=trading_plan,
    )
