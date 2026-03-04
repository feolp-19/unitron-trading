from dataclasses import dataclass

import pandas as pd


@dataclass
class TechnicalSignal:
    direction: str          # BULL, BEAR, NEUTRAL
    rsi_value: float
    sma_value: float
    atr_value: float
    current_price: float
    price_vs_sma: str       # above, below, at
    rsi_trend_2d: float     # RSI change over last 2 days (for recency bias detection)
    atr_ratio: float        # current ATR / 30-day avg ATR (volatility spike detection)


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_sma(series: pd.Series, period: int = 200) -> pd.Series:
    effective_period = min(period, len(series) - 1)
    return series.rolling(window=effective_period, min_periods=1).mean()


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - close).abs(),
        (low - close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def analyze(df: pd.DataFrame) -> TechnicalSignal | None:
    """Run full technical analysis on OHLC DataFrame. Returns None if insufficient data."""
    if df.empty or len(df) < 50:
        return None

    close = df["Close"]
    sma_period = min(200, len(close) - 1)

    rsi_series = compute_rsi(close)
    sma_series = compute_sma(close, sma_period)
    atr_series = compute_atr(df)

    current_price = float(close.iloc[-1])
    rsi_value = float(rsi_series.iloc[-1])
    sma_value = float(sma_series.iloc[-1])
    atr_value = float(atr_series.iloc[-1])

    rsi_2d_ago = float(rsi_series.iloc[-3]) if len(rsi_series) >= 3 else rsi_value
    rsi_trend_2d = rsi_value - rsi_2d_ago

    atr_30d_avg = float(atr_series.tail(30).mean())
    atr_ratio = atr_value / atr_30d_avg if atr_30d_avg > 0 else 1.0

    sma_distance_pct = ((current_price - sma_value) / sma_value) * 100
    if sma_distance_pct > 1:
        price_vs_sma = "above"
    elif sma_distance_pct < -1:
        price_vs_sma = "below"
    else:
        price_vs_sma = "at"

    # Direction logic: trend-following with RSI guard
    # BULL: price in uptrend (above SMA) and not overbought (RSI < 65)
    # BEAR: price in downtrend (below SMA) and not oversold (RSI > 35)
    # NEUTRAL: price near SMA (no clear trend) or RSI at extremes against trend
    if price_vs_sma == "above" and rsi_value < 65:
        direction = "BULL"
    elif price_vs_sma == "below" and rsi_value > 35:
        direction = "BEAR"
    else:
        direction = "NEUTRAL"

    return TechnicalSignal(
        direction=direction,
        rsi_value=round(rsi_value, 2),
        sma_value=round(sma_value, 2),
        atr_value=round(atr_value, 2),
        current_price=round(current_price, 2),
        price_vs_sma=price_vs_sma,
        rsi_trend_2d=round(rsi_trend_2d, 2),
        atr_ratio=round(atr_ratio, 2),
    )
