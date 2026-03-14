import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ohlc(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Fetch daily OHLC data from yfinance. Returns empty DataFrame on failure."""
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return pd.DataFrame()

        df = df[required].copy()
        df.index.name = "Date"
        df = df.dropna(subset=["Close"])
        return df
    except Exception:
        return pd.DataFrame()


def get_current_price(df: pd.DataFrame) -> float | None:
    if df.empty:
        return None
    return float(df["Close"].iloc[-1])


def get_52_week_range(df: pd.DataFrame) -> tuple[float, float] | None:
    if len(df) < 20:
        return None
    last_252 = df["Close"].tail(252)
    return float(last_252.min()), float(last_252.max())


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_macro_indicator(ticker: str) -> float | None:
    """Fetch a single macro indicator (DXY, VIX, US10Y, etc.)."""
    try:
        df = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return float(df["Close"].iloc[-1])
    except Exception:
        return None
