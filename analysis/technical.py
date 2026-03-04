from dataclasses import dataclass
import pandas as pd
import streamlit as st
import yfinance as yf


@dataclass
class TechnicalSignal:
    current_price: float
    rsi_value: float
    sma_200: float
    sma_50w: float | None       # 50-week SMA (multi-timeframe)
    atr_value: float
    price_vs_sma: str           # above, below, at
    price_vs_weekly_sma: str    # above, below, at, unavailable
    rsi_trend_2d: float
    atr_ratio: float            # current ATR vs 30-day avg
    volume_ratio: float         # current volume vs 20-day avg
    vix_value: float | None     # VIX fear/greed
    vix_level: str              # low_fear, normal, elevated, extreme_fear


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


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_vix() -> float | None:
    """Fetch current VIX value."""
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
    """Fetch 50-week SMA for multi-timeframe analysis."""
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


def analyze(df: pd.DataFrame, ticker: str = "") -> TechnicalSignal | None:
    """Run full technical analysis including VIX, volume, and multi-timeframe."""
    if df.empty or len(df) < 50:
        return None

    close = df["Close"]

    rsi_series = compute_rsi(close)
    sma_series = compute_sma(close, min(200, len(close) - 1))
    atr_series = compute_atr(df)

    current_price = float(close.iloc[-1])
    rsi_value = float(rsi_series.iloc[-1])
    sma_200 = float(sma_series.iloc[-1])
    atr_value = float(atr_series.iloc[-1])

    rsi_2d_ago = float(rsi_series.iloc[-3]) if len(rsi_series) >= 3 else rsi_value
    rsi_trend_2d = rsi_value - rsi_2d_ago

    atr_30d_avg = float(atr_series.tail(30).mean())
    atr_ratio = atr_value / atr_30d_avg if atr_30d_avg > 0 else 1.0

    # Volume analysis
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

    # VIX
    vix_value = fetch_vix()
    vix_level = _classify_vix(vix_value)

    # Weekly SMA (multi-timeframe)
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

    return TechnicalSignal(
        current_price=round(current_price, 2),
        rsi_value=round(rsi_value, 2),
        sma_200=round(sma_200, 2),
        sma_50w=round(sma_50w, 2) if sma_50w else None,
        atr_value=round(atr_value, 2),
        price_vs_sma=price_vs_sma,
        price_vs_weekly_sma=price_vs_weekly_sma,
        rsi_trend_2d=round(rsi_trend_2d, 2),
        atr_ratio=round(atr_ratio, 2),
        volume_ratio=round(volume_ratio, 2),
        vix_value=round(vix_value, 2) if vix_value else None,
        vix_level=vix_level,
    )
