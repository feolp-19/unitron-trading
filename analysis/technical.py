"""Technical Analysis Module — full bottom-up analysis.

Computes: SMA 20/50/200, RSI(14), ATR(14), volume ratio, VIX,
50-week SMA (multi-timeframe), and support/resistance levels
from local price peaks and troughs."""

from dataclasses import dataclass, field
import pandas as pd
import streamlit as st
import yfinance as yf


@dataclass
class SupportResistance:
    supports: list[float]       # nearest 2-3 support levels below price
    resistances: list[float]    # nearest 2-3 resistance levels above price


@dataclass
class TechnicalSignal:
    current_price: float
    rsi_value: float
    sma_20: float
    sma_50: float
    sma_200: float
    sma_50w: float | None
    atr_value: float
    price_vs_sma: str               # above, below, at  (vs 200-day)
    price_vs_weekly_sma: str         # above, below, at, unavailable
    sma_alignment: str               # bullish_stack, bearish_stack, mixed
    rsi_trend_2d: float
    atr_ratio: float
    volume_ratio: float
    vix_value: float | None
    vix_level: str
    support_resistance: SupportResistance
    near_resistance: bool            # True if price within 2% of nearest resistance
    near_support: bool               # True if price within 2% of nearest support


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


def find_support_resistance(df: pd.DataFrame, window: int = 20, n_levels: int = 3) -> SupportResistance:
    """Identify support/resistance from local peaks and troughs.

    Uses a rolling window to find local minima (supports) and maxima (resistances),
    then clusters nearby levels and returns the nearest ones to current price.
    """
    if len(df) < window * 2:
        return SupportResistance(supports=[], resistances=[])

    high = df["High"].values
    low = df["Low"].values
    close_last = float(df["Close"].iloc[-1])

    peaks = []
    troughs = []

    for i in range(window, len(df) - window):
        if high[i] == max(high[i - window:i + window + 1]):
            peaks.append(float(high[i]))
        if low[i] == min(low[i - window:i + window + 1]):
            troughs.append(float(low[i]))

    peaks = _cluster_levels(peaks, close_last)
    troughs = _cluster_levels(troughs, close_last)

    resistances = sorted([p for p in peaks if p > close_last * 1.001])[:n_levels]
    supports = sorted([t for t in troughs if t < close_last * 0.999], reverse=True)[:n_levels]

    return SupportResistance(
        supports=[round(s, 2) for s in supports],
        resistances=[round(r, 2) for r in resistances],
    )


def _cluster_levels(levels: list[float], reference_price: float, tolerance_pct: float = 1.5) -> list[float]:
    """Cluster nearby price levels into single representative levels."""
    if not levels:
        return []

    sorted_levels = sorted(levels)
    tolerance = reference_price * (tolerance_pct / 100)
    clusters: list[list[float]] = [[sorted_levels[0]]]

    for level in sorted_levels[1:]:
        if abs(level - clusters[-1][-1]) <= tolerance:
            clusters[-1].append(level)
        else:
            clusters.append([level])

    # Clusters with more touches are stronger, weight them higher
    result = []
    for cluster in clusters:
        avg = sum(cluster) / len(cluster)
        result.append((avg, len(cluster)))

    result.sort(key=lambda x: x[1], reverse=True)
    return [r[0] for r in result]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_vix() -> float | None:
    try:
        df = yf.download("^VIX", period="5d", progress=False, auto_adjust=True)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return float(df["Close"].iloc[-1])
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_weekly_sma(ticker: str) -> float | None:
    try:
        df = yf.download(ticker, period="2y", interval="1wk", progress=False, auto_adjust=True)
        if df.empty or len(df) < 20:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        sma = df["Close"].rolling(window=min(50, len(df) - 1), min_periods=1).mean()
        return float(sma.iloc[-1])
    except Exception:
        return None


def _classify_vix(vix: float | None) -> str:
    if vix is None:
        return "unavailable"
    if vix < 15:
        return "low_fear"
    if vix < 20:
        return "normal"
    if vix < 30:
        return "elevated"
    return "extreme_fear"


def _classify_sma_alignment(price: float, sma20: float, sma50: float, sma200: float) -> str:
    """Determine SMA stacking order: bullish (20>50>200), bearish (200>50>20), or mixed."""
    if price > sma20 > sma50 > sma200:
        return "bullish_stack"
    if price < sma20 < sma50 < sma200:
        return "bearish_stack"
    return "mixed"


def analyze(df: pd.DataFrame, ticker: str = "") -> TechnicalSignal | None:
    """Run full technical analysis."""
    if df.empty or len(df) < 50:
        return None

    close = df["Close"]

    rsi_series = compute_rsi(close)
    sma_20_series = compute_sma(close, 20)
    sma_50_series = compute_sma(close, 50)
    sma_200_series = compute_sma(close, min(200, len(close) - 1))
    atr_series = compute_atr(df)

    current_price = float(close.iloc[-1])
    rsi_value = float(rsi_series.iloc[-1])
    sma_20 = float(sma_20_series.iloc[-1])
    sma_50 = float(sma_50_series.iloc[-1])
    sma_200 = float(sma_200_series.iloc[-1])
    atr_value = float(atr_series.iloc[-1])

    rsi_2d_ago = float(rsi_series.iloc[-3]) if len(rsi_series) >= 3 else rsi_value
    rsi_trend_2d = rsi_value - rsi_2d_ago

    atr_30d_avg = float(atr_series.tail(30).mean())
    atr_ratio = atr_value / atr_30d_avg if atr_30d_avg > 0 else 1.0

    # Volume
    if "Volume" in df.columns and df["Volume"].sum() > 0:
        current_vol = float(df["Volume"].iloc[-1])
        avg_vol = float(df["Volume"].tail(20).mean())
        volume_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
    else:
        volume_ratio = 1.0

    # Price vs 200-day SMA
    sma_distance_pct = ((current_price - sma_200) / sma_200) * 100
    if sma_distance_pct > 1:
        price_vs_sma = "above"
    elif sma_distance_pct < -1:
        price_vs_sma = "below"
    else:
        price_vs_sma = "at"

    # SMA alignment
    sma_alignment = _classify_sma_alignment(current_price, sma_20, sma_50, sma_200)

    # VIX
    vix_value = fetch_vix()
    vix_level = _classify_vix(vix_value)

    # Weekly SMA
    sma_50w = fetch_weekly_sma(ticker) if ticker else None
    if sma_50w is not None:
        w_distance = ((current_price - sma_50w) / sma_50w) * 100
        if w_distance > 1:
            price_vs_weekly_sma = "above"
        elif w_distance < -1:
            price_vs_weekly_sma = "below"
        else:
            price_vs_weekly_sma = "at"
    else:
        price_vs_weekly_sma = "unavailable"

    # Support & Resistance
    sr = find_support_resistance(df)
    near_resistance = bool(
        sr.resistances
        and abs(current_price - sr.resistances[0]) / current_price < 0.02
    )
    near_support = bool(
        sr.supports
        and abs(current_price - sr.supports[0]) / current_price < 0.02
    )

    return TechnicalSignal(
        current_price=round(current_price, 2),
        rsi_value=round(rsi_value, 2),
        sma_20=round(sma_20, 2),
        sma_50=round(sma_50, 2),
        sma_200=round(sma_200, 2),
        sma_50w=round(sma_50w, 2) if sma_50w else None,
        atr_value=round(atr_value, 2),
        price_vs_sma=price_vs_sma,
        price_vs_weekly_sma=price_vs_weekly_sma,
        sma_alignment=sma_alignment,
        rsi_trend_2d=round(rsi_trend_2d, 2),
        atr_ratio=round(atr_ratio, 2),
        volume_ratio=round(volume_ratio, 2),
        vix_value=round(vix_value, 2) if vix_value else None,
        vix_level=vix_level,
        support_resistance=sr,
        near_resistance=near_resistance,
        near_support=near_support,
    )
