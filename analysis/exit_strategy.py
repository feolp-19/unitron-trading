"""Exit Strategy Engine — generates a complete 'Handelsplan' (Trading Plan)
for every BULL or BEAR recommendation.

Uses ATR-based stops, support/resistance levels, and risk/reward calculations.
Take-profit is capped at 5x ATR to keep targets realistic for certificate trading."""

from dataclasses import dataclass
from analysis.technical import TechnicalSignal


@dataclass
class TradingPlan:
    entry_price: float
    stop_loss: float
    stop_loss_method: str
    stop_loss_reasoning: str
    take_profit: float
    take_profit_method: str
    take_profit_reasoning: str
    risk_reward_ratio: str
    risk_amount: float
    reward_amount: float
    trailing_stop_level: float
    trailing_stop_reasoning: str


def generate_trading_plan(tech: TechnicalSignal, action: str) -> TradingPlan | None:
    if action not in ("BULL", "BEAR"):
        return None

    entry = tech.current_price
    atr = tech.atr_value
    sr = tech.support_resistance

    if action == "BULL":
        return _bull_plan(entry, atr, sr.supports, sr.resistances)
    else:
        return _bear_plan(entry, atr, sr.supports, sr.resistances)


def _bull_plan(
    entry: float, atr: float,
    supports: list[float], resistances: list[float],
) -> TradingPlan:
    # --- STOP-LOSS ---
    atr_stop = round(entry - 2 * atr, 2)

    if supports:
        support_stop = round(supports[0] - 0.5 * atr, 2)
        if support_stop > atr_stop and support_stop < entry:
            stop_loss = support_stop
            sl_method = "Stöd-baserad"
            sl_reasoning = (
                f"Placerad strax under närmaste stödnivå ({supports[0]:,.2f}) "
                f"minus 0.5x ATR som buffert. Skyddar kapital vid oväntad nedgång."
            )
        else:
            stop_loss = atr_stop
            sl_method = "ATR-baserad"
            sl_reasoning = (
                f"2x ATR ({atr:,.2f}) under ingångspriset. "
                f"Ger tillräckligt utrymme för normal volatilitet."
            )
    else:
        stop_loss = atr_stop
        sl_method = "ATR-baserad"
        sl_reasoning = (
            f"2x ATR ({atr:,.2f}) under ingångspriset. "
            f"Inga tydliga stödnivåer identifierade."
        )

    # --- TAKE-PROFIT ---
    risk_distance = entry - stop_loss
    min_target = round(entry + 2 * risk_distance, 2)
    max_target = round(entry + 5 * atr, 2)  # cap at 5x ATR

    if resistances:
        resistance_target = resistances[0]
        # Only use resistance if it's within a realistic range (max 5x ATR)
        if resistance_target <= max_target and resistance_target >= min_target:
            take_profit = round(resistance_target, 2)
            tp_method = "Motstånd-baserad"
            tp_reasoning = (
                f"Närmaste motståndsnivå på {resistance_target:,.2f}. "
                f"Historiskt motstånd — hög sannolikhet för vändning."
            )
        elif resistance_target < min_target:
            take_profit = min_target
            tp_method = "Risk/Reward-baserad"
            tp_reasoning = (
                f"Minst 2:1 risk/reward-kvot krävs. Närmaste motstånd "
                f"({resistance_target:,.2f}) ger för liten vinst."
            )
        else:
            take_profit = max_target
            tp_method = "ATR-baserad"
            tp_reasoning = (
                f"Målkurs satt till 5x ATR ({atr:,.2f}). "
                f"Närmaste motstånd ({resistance_target:,.2f}) ligger för långt bort "
                f"för realistisk certifikathandel."
            )
    else:
        take_profit = min(min_target, max_target)
        tp_method = "Risk/Reward-baserad"
        tp_reasoning = (
            f"2:1 risk/reward-kvot baserad på ATR. "
            f"Inga tydliga motståndsnivåer identifierade."
        )

    # --- TRAILING STOP ---
    trailing_stop = round(entry + 1 * atr, 2)
    trailing_reasoning = (
        f"När priset passerar {trailing_stop:,.2f} (+1x ATR), "
        f"flytta stop-loss till ingångspriset ({entry:,.2f}) för att eliminera risk. "
        f"Fortsätt flytta stop-loss uppåt med 1x ATR för varje ny toppnotering."
    )

    reward_distance = take_profit - entry
    rr = reward_distance / risk_distance if risk_distance > 0 else 0

    return TradingPlan(
        entry_price=entry,
        stop_loss=stop_loss,
        stop_loss_method=sl_method,
        stop_loss_reasoning=sl_reasoning,
        take_profit=take_profit,
        take_profit_method=tp_method,
        take_profit_reasoning=tp_reasoning,
        risk_reward_ratio=f"1:{rr:.1f}",
        risk_amount=round(risk_distance, 2),
        reward_amount=round(reward_distance, 2),
        trailing_stop_level=trailing_stop,
        trailing_stop_reasoning=trailing_reasoning,
    )


def _bear_plan(
    entry: float, atr: float,
    supports: list[float], resistances: list[float],
) -> TradingPlan:
    # --- STOP-LOSS ---
    atr_stop = round(entry + 2 * atr, 2)

    if resistances:
        resistance_stop = round(resistances[0] + 0.5 * atr, 2)
        if resistance_stop < atr_stop and resistance_stop > entry:
            stop_loss = resistance_stop
            sl_method = "Motstånd-baserad"
            sl_reasoning = (
                f"Placerad strax ovanför närmaste motståndsnivå ({resistances[0]:,.2f}) "
                f"plus 0.5x ATR som buffert. Skyddar kapital vid oväntad uppgång."
            )
        else:
            stop_loss = atr_stop
            sl_method = "ATR-baserad"
            sl_reasoning = (
                f"2x ATR ({atr:,.2f}) ovanför ingångspriset. "
                f"Ger tillräckligt utrymme för normal volatilitet."
            )
    else:
        stop_loss = atr_stop
        sl_method = "ATR-baserad"
        sl_reasoning = (
            f"2x ATR ({atr:,.2f}) ovanför ingångspriset. "
            f"Inga tydliga motståndsnivåer identifierade."
        )

    # --- TAKE-PROFIT ---
    risk_distance = stop_loss - entry
    min_target = round(entry - 2 * risk_distance, 2)
    max_target_distance = 5 * atr
    max_target = round(entry - max_target_distance, 2)  # cap at 5x ATR below

    if supports:
        support_target = supports[0]
        # Only use support if it's within a realistic range (max 5x ATR)
        if support_target >= max_target and support_target <= min_target:
            take_profit = round(support_target, 2)
            tp_method = "Stöd-baserad"
            tp_reasoning = (
                f"Närmaste stödnivå på {support_target:,.2f}. "
                f"Historiskt stöd — hög sannolikhet för studs."
            )
        elif support_target > min_target:
            take_profit = min_target
            tp_method = "Risk/Reward-baserad"
            tp_reasoning = (
                f"Minst 2:1 risk/reward-kvot krävs. Närmaste stöd "
                f"({support_target:,.2f}) ger för liten vinst."
            )
        else:
            take_profit = max_target
            tp_method = "ATR-baserad"
            tp_reasoning = (
                f"Målkurs satt till 5x ATR ({atr:,.2f}). "
                f"Närmaste stöd ({support_target:,.2f}) ligger för långt bort "
                f"för realistisk certifikathandel."
            )
    else:
        take_profit = max(min_target, max_target)
        tp_method = "Risk/Reward-baserad"
        tp_reasoning = (
            f"2:1 risk/reward-kvot baserad på ATR. "
            f"Inga tydliga stödnivåer identifierade."
        )

    # --- TRAILING STOP ---
    trailing_stop = round(entry - 1 * atr, 2)
    trailing_reasoning = (
        f"När priset faller under {trailing_stop:,.2f} (-1x ATR), "
        f"flytta stop-loss till ingångspriset ({entry:,.2f}) för att eliminera risk. "
        f"Fortsätt flytta stop-loss nedåt med 1x ATR för varje ny bottennotering."
    )

    reward_distance = entry - take_profit
    rr = reward_distance / risk_distance if risk_distance > 0 else 0

    return TradingPlan(
        entry_price=entry,
        stop_loss=stop_loss,
        stop_loss_method=sl_method,
        stop_loss_reasoning=sl_reasoning,
        take_profit=take_profit,
        take_profit_method=tp_method,
        take_profit_reasoning=tp_reasoning,
        risk_reward_ratio=f"1:{rr:.1f}",
        risk_amount=round(risk_distance, 2),
        reward_amount=round(reward_distance, 2),
        trailing_stop_level=trailing_stop,
        trailing_stop_reasoning=trailing_reasoning,
    )
